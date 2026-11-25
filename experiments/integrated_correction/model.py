"""Exact-token generation, causal replay, and the layer-20 residual hook."""

import re
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .storage import digest, sha256

LAYER = 19
HIDDEN_SIZE = 1536
MAX_NEW = {"gsm8k": 1024, "math": 2048}
TEMPERATURE = .7
TOP_P = .95
HOOK = "model.model.layers[19] output residual, before final normalization"
GSM_SYSTEM = (
    "You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer "
    "with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write "
    "nothing after this final line. Do not omit this line. The final answer must contain only a number, without "
    "units, currency symbols, Markdown emphasis, or LaTeX."
)
GSM_USER = (
    "Problem:\n{question}\n\nSolve the problem step by step. After your reasoning, write the answer on the LAST "
    "line using exactly this format:\n#### <number>\nReplace <number> with the numerical answer. Do not write any "
    "text after that line."
)
MATH_SYSTEM = (
    "You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer "
    r"with a separate final line formatted exactly as Final answer: \boxed{<answer>}. "
    "Write nothing after this final line. Do not omit this line."
)
MATH_USER = (
    "Problem:\n{question}\n\nSolve the problem step by step. After your reasoning, write the answer on the LAST "
    "line using exactly this format:\nFinal answer: \\boxed{{<answer>}}\n"
    "Replace <answer> with the answer. Do not write any text after that line."
)


@dataclass(frozen=True)
class Problem:
    id: str
    dataset: str
    question: str

    @classmethod
    def from_row(cls, row):
        return cls(row["id"], row["dataset"], row["question"])


def response_seed(seed, problem_id, role, sample_index):
    return int(digest([seed, problem_id, role, sample_index])[:16], 16) % (2**63)


def response_id(problem_id, role, sample_index):
    return f"{problem_id}/{role}/{sample_index}"


def prompt_messages(problem):
    system, user = ((GSM_SYSTEM, GSM_USER) if problem.dataset == "gsm8k" else (MATH_SYSTEM, MATH_USER))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(question=problem.question)}]


class BudgetExhausted(Exception):
    pass


class Engine:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        eoses = model.generation_config.eos_token_id
        self.eos_ids = set(eoses if isinstance(eoses, list) else [eoses])
        if not self.eos_ids or None in self.eos_ids:
            raise ValueError("Model EOS configuration is missing")
        self.layer = model.model.layers[LAYER]
        self.forward_tokens = self.generated_tokens = 0
        self.limit = None

    @classmethod
    def load(cls, config):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model_file = hf_hub_download(config["model"], "model.safetensors",
                                         revision=config["model_revision"], local_files_only=True)
        except OSError:
            model_file = hf_hub_download(config["model"], "model.safetensors", revision=config["model_revision"])
        if sha256(Path(model_file)) != config["model_weights_sha256"]:
            raise ValueError("Model weights checksum mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["model_revision"], torch_dtype=getattr(torch, config["dtype"]),
            attn_implementation="sdpa", use_safetensors=True).to(config["device"])
        if model.config.hidden_size != HIDDEN_SIZE:
            raise ValueError("Unexpected model hidden size")
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
        return cls(model, tokenizer, config["device"])

    def start_cost(self, limit=None):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.forward_tokens = self.generated_tokens = 0
        self.limit = limit
        self.started = time.perf_counter()

    def cost(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"forward_tokens": self.forward_tokens, "generated_tokens": self.generated_tokens,
                "seconds": time.perf_counter() - self.started, "peak_gpu_bytes":
                torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0}

    def count_forward(self, tensor):
        if self.limit is not None and self.forward_tokens + tensor.numel() > self.limit:
            raise BudgetExhausted
        self.forward_tokens += tensor.numel()

    def prompt(self, problem):
        return self.tokenizer.apply_chat_template(prompt_messages(problem), add_generation_prompt=True, tokenize=True)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def record(self, problem, prompt, tokens, role, sample_index, seed, sampled=False,
               prefix_length=0, batch_size=1, use_cache=False):
        return {"id": response_id(problem.id, role, sample_index), "problem_id": problem.id,
                "role": role, "sample_index": sample_index, "seed": response_seed(seed, problem.id, role, sample_index),
                "prompt_ids": prompt, "output_ids": tokens, "text": self.decode(tokens),
                "finish": "eos" if tokens and tokens[-1] in self.eos_ids else "length",
                "generation": {"algorithm": "top_p" if sampled else "greedy",
                               "temperature": TEMPERATURE if sampled else 0., "top_p": TOP_P if sampled else 1.,
                               "max_new_tokens": MAX_NEW[problem.dataset], "prefix_length": prefix_length,
                               "batch_size": batch_size, "use_kv_cache": use_cache}}

    def generate_many(self, problem, prompt, role, indices, seed, sampled, prefix=(), **options):
        return self.generate_requests(problem, prompt, role, [(index, prefix) for index in indices], seed, sampled, **options)

    @torch.no_grad()
    def generate_requests(self, problem, prompt, role, requests, seed, sampled=True, *,
                          temperature=TEMPERATURE, top_p=TOP_P):
        """Left-pad shared batches, with a separate RNG stream and token cap for each request."""
        batch = len(requests)
        if not batch:
            return []
        if len({index for index, _ in requests}) != batch:
            raise ValueError("Duplicate generation request")
        limit = MAX_NEW[problem.dataset]
        tokens = [list(prefix) for _, prefix in requests]
        if any(len(t) >= limit for t in tokens):
            raise ValueError("Continuation prefix exhausts its token limit")
        width = max(map(len, tokens)) + len(prompt)
        eos = min(self.eos_ids)
        padded = [[eos] * (width - len(prompt) - len(t)) + prompt + t for t in tokens]
        current = torch.tensor(padded, device=self.device)
        positions = torch.arange(width, device=self.device)
        left = torch.tensor([width - len(prompt) - len(t) for t in tokens], device=self.device)
        attention_mask = (positions[None, :] >= left[:, None]).long()
        position_ids = (positions[None, :] - left[:, None]).clamp_min(0)
        uniforms = (torch.as_tensor(np.stack([
            np.random.default_rng(response_seed(seed, problem.id, role, index)).random(limit, dtype=np.float32)
            for index, _ in requests], axis=1), device=self.device) if sampled else None)

        done = [False] * batch
        cache = None
        for step in range(limit):
            self.count_forward(current)
            output = self.model(input_ids=current, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = output.past_key_values
            scores = output.logits[:, -1]
            sampled_ids = sample_top_p(scores, uniforms[step], temperature, top_p) if sampled else scores.argmax(-1)
            next_ids = sampled_ids.tolist()
            for i, token in enumerate(next_ids):
                if done[i]:
                    next_ids[i] = eos
                    continue
                tokens[i].append(token)
                self.generated_tokens += 1
                done[i] = token in self.eos_ids or len(tokens[i]) == limit
            if all(done):
                break
            current = torch.tensor(next_ids, device=self.device).unsqueeze(-1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
            position_ids = position_ids[:, -1:] + 1
        records = [self.record(problem, prompt, ids, role, index, seed, sampled, len(prefix), batch, True)
                   for ids, (index, prefix) in zip(tokens, requests)]
        if sampled:
            for record in records:
                record["generation"].update(temperature=temperature, top_p=top_p)
        return records

    @torch.no_grad()
    def replay(self, ids, positions, patch=None, readout=None):
        """Return selected layer-20 outputs. logits at position i predict token i+1."""
        captured = None

        def hook(_layer, _inputs, output):
            nonlocal captured
            hidden = output[0]
            if patch is not None:
                position, delta = patch
                hidden = hidden.clone()
                position = position if position >= 0 else hidden.shape[1] + position
                hidden[:, position] = (hidden[:, position].float() + delta).to(hidden.dtype)
            captured = hidden[0, positions].float()
            if patch is not None:
                return (hidden, *output[1:])
            return None

        inputs = torch.tensor([ids], device=self.device)
        self.count_forward(inputs)
        handle = self.layer.register_forward_hook(hook)
        try:
            result = self.model.model(input_ids=inputs, use_cache=False)
        finally:
            handle.remove()
        if captured is None:
            raise RuntimeError("Layer-20 hook did not fire")
        final = result.last_hidden_state[0, readout] if readout is not None else None
        return captured, final

    @torch.no_grad()
    def next_token(self, ids, patch=None):
        _, hidden = self.replay(ids, [], patch=patch, readout=-1)
        token = int(self.model.lm_head(hidden).argmax())
        self.generated_tokens += 1
        return token

    @torch.no_grad()
    def features(self, problem, response, confidence=False):
        prompt, output = response["prompt_ids"], response["output_ids"]
        stop = reasoning_length(problem.dataset, output, self.eos_ids, self.decode)
        start = len(prompt)
        positions = [start - 1, *range(start, start + stop)]
        window = slice(start - 1, start - 1 + stop) if confidence else None
        hidden, final = self.replay(prompt + output[:stop], positions, readout=window)
        q_end = hidden[0]
        r_mean = hidden[1:].mean(0) if stop else q_end
        vectors = {"q_end": q_end.cpu().numpy(), "r_mean": r_mean.cpu().numpy()}
        if confidence:
            quality = []
            for offset in range(0, stop, 32):
                logits = self.model.lm_head(final[offset:offset + 32]).float()
                ids = torch.tensor(output[offset:offset + len(logits)], device=self.device)
                logp = logits.log_softmax(-1)
                top = logp.topk(2).values
                quality.append(torch.stack((-logp.gather(1, ids[:, None]).flatten(),
                                            -(logp.exp() * logp).sum(-1), top[:, 0] - top[:, 1]), dim=1))
            metrics = torch.cat(quality) if quality else torch.zeros((1, 3), device=self.device)
            vectors["confidence"] = np.array([*metrics.mean(0).tolist(), float(metrics[:, 0].max()), len(output)])
        return vectors, stop


def sample_top_p(scores, uniforms, temperature=TEMPERATURE, top_p=TOP_P):
    logits = scores.float() / temperature
    normalizer = logits.logsumexp(-1, keepdim=True)
    best, order = logits.topk(min(64, logits.shape[-1]), sorted=True)
    cdf = (best - normalizer).exp().cumsum(-1)
    if cdf.shape[-1] < logits.shape[-1] and (cdf[:, -1] < top_p).any():
        best, order = logits.sort(descending=True)
        cdf = (best - normalizer).exp().cumsum(-1)
    last = (cdf < top_p).sum(-1).clamp_max(cdf.shape[-1] - 1)
    mass = cdf.gather(1, last[:, None])
    selected = (cdf <= uniforms[:, None] * mass).sum(-1).minimum(last)
    return order.gather(1, selected[:, None]).squeeze(-1)


def reasoning_length(dataset, output, eos_ids, decode):
    end = next((i for i, value in enumerate(output) if value in eos_ids), len(output))
    tokens = output[:end]
    text = decode(tokens)
    if dataset == "gsm8k":
        marker = text.rfind("####")
    else:
        matches = list(re.finditer(r"Final answer:|\\boxed\s*\{", text, flags=re.IGNORECASE))
        marker = matches[-1].start() if matches else -1
    if marker < 0:
        return end
    line = text.rfind("\n", 0, marker) + 1
    # Keep exactly the generated tokens ending before the answer line, including
    # when a token joins the newline and the first character of the answer.
    return bisect_right(range(end + 1), line, key=lambda n: len(decode(tokens[:n]))) - 1


def prefix_positions(length):
    return sorted({length * fraction // 40 * 4 for fraction in range(1, 9)} - {0})
