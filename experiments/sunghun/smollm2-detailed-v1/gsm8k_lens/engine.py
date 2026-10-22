import importlib.metadata
import platform
import random
import time

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          StoppingCriteria, StoppingCriteriaList)

from .answers import parse_answer


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "pandas", "pyarrow",
                 "safetensors", "huggingface-hub", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gib": (torch.cuda.get_device_properties(0).total_memory / 2**30
                               if torch.cuda.is_available() else None)}


def configure_runtime(cfg):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(cfg["model"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this config; no CPU fallback is performed")
        torch.cuda.set_device(device)
        if cfg["model"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable. Use a separately named FP32 run")
        torch.cuda.manual_seed_all(cfg["seed"])
    return device


def load_model(cfg, revision):
    device = configure_runtime(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=revision,
                                               use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], revision=revision, trust_remote_code=False,
        use_safetensors=True, torch_dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attention_backend"],
    ).to(device).eval()
    expected = {"num_hidden_layers": "expected_layers", "hidden_size": "expected_hidden_size",
                "vocab_size": "expected_vocab_size"}
    for actual, configured in expected.items():
        if getattr(model.config, actual) != cfg["model"][configured]:
            raise ValueError(f"Unexpected model architecture: {actual}")
    if model.config.max_position_embeddings < cfg["generation"]["max_sequence_tokens"]:
        raise ValueError("Configured sequence budget exceeds model context")
    return model, tokenizer


def make_generation_config(model, tokenizer, cfg):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Missing EOS token")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=cfg["generation"]["max_new_tokens"],
        repetition_penalty=1.0, eos_token_id=eos, pad_token_id=pad,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
    )


class FinalAnswerLineStop(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.finished = False

    def __call__(self, input_ids, scores=None, **kwargs):
        last_piece = self.tokenizer.decode([int(input_ids[0, -1])], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        stop = False
        if last_piece.endswith(("\n", "\r")):
            text = self.tokenizer.decode(input_ids[0, self.prompt_length:].tolist(),
                                         skip_special_tokens=True, clean_up_tokenization_spaces=False)
            stop = parse_answer(text)["status"] == "ok"
        self.finished = self.finished or stop
        return torch.tensor([stop], device=input_ids.device, dtype=torch.bool)


@torch.inference_mode()
def generate_example(model, tokenizer, question, cfg, record_quality=False):
    messages = [{"role": "system", "content": cfg["prompt"]["system"]},
                {"role": "user", "content": cfg["prompt"]["user_template"].replace("{question}", question)}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) > cfg["generation"]["max_prompt_tokens"]:
        raise ValueError("Prompt exceeds max_prompt_tokens; refusing silent truncation")
    gen_config = make_generation_config(model, tokenizer, cfg)
    inputs = torch.tensor([ids], device=model.device)
    final_line_stop = FinalAnswerLineStop(tokenizer, len(ids))
    start = time.perf_counter()
    output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                            generation_config=gen_config, return_dict_in_generate=True,
                            output_logits=record_quality,
                            stopping_criteria=StoppingCriteriaList([final_line_stop]))
    continuation = output.sequences[0, len(ids):].tolist()
    eos = gen_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    terminated = bool(continuation and continuation[-1] in eos_ids)
    record = {"prompt_token_ids": ids, "generated_token_ids": continuation,
              "text": tokenizer.decode(continuation, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False),
              "prompt_tokens": len(ids), "generated_tokens": len(continuation),
              "truncated": not (terminated or final_line_stop.finished),
              "stop_reason": ("eos" if terminated else "final_answer_line" if final_line_stop.finished else "max_new_tokens"),
              "effective_stopping_rule": "eos_or_completed_final_answer_line_v1",
              "generation_seconds": time.perf_counter() - start,
              "effective_generation_config": gen_config.to_dict()}
    if record_quality:
        if len(output.logits) != len(continuation):
            raise ValueError("Generation logits/token length mismatch")
        record["generation_quality"] = {
            "top1": [int(logit[0].argmax()) for logit in output.logits],
            "generated_logprob": [float(logit[0].float().log_softmax(-1)[token])
                                  for logit, token in zip(output.logits, continuation)],
        }
    return record


def prediction_positions(prompt_length, generated_length):
    if prompt_length < 1 or generated_length < 1:
        raise ValueError("Prompt and generation must be nonempty")
    return torch.arange(prompt_length - 1, prompt_length + generated_length - 1)


@torch.inference_mode()
def replay(model, prompt_ids, generated_ids):
    positions = prediction_positions(len(prompt_ids), len(generated_ids))
    residuals = [None] * len(model.model.layers)
    handles = []

    def capture(index):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residuals[index] = hidden[0, positions.to(hidden.device)].detach().to("cpu")
        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(capture(index)))
    try:
        ids = torch.tensor([prompt_ids + generated_ids], device=model.device)
        result = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, return_dict=True)
        native = result.last_hidden_state[0, positions.to(model.device)].cpu()
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in residuals):
        raise RuntimeError("A decoder hook did not fire")
    return torch.stack(residuals), native, positions
