"""Documented PyTorch 2.3 top-p compatibility; preserve existing greedy batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from transformers import TopPLogitsWarper


def cpu_cumsum_top_p(self, input_ids, scores):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # CUDA softmax stays unchanged; only the unsupported prefix sum moves to CPU.
    cumulative_probs = sorted_logits.softmax(dim=-1).cpu().cumsum(dim=-1).to(scores.device)
    remove = cumulative_probs <= (1 - self.top_p)
    remove[..., -self.min_tokens_to_keep:] = 0
    indices_to_remove = remove.scatter(1, sorted_indices, remove)
    return scores.masked_fill(indices_to_remove, self.filter_value)


def sampler_qa():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    inputs = torch.ones((3, 1), dtype=torch.long)
    scores = torch.randn(3, 49152)
    warper = TopPLogitsWarper(.95)
    expected = warper(inputs, scores)
    actual = cpu_cumsum_top_p(warper, inputs, scores)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    device_scores = scores.cuda()
    processed = cpu_cumsum_top_p(warper, inputs.cuda(), device_scores)
    torch.manual_seed(77)
    first = torch.multinomial(processed.softmax(-1), 1)
    torch.manual_seed(77)
    second = torch.multinomial(processed.softmax(-1), 1)
    assert torch.equal(first, second)
    return dict(cpu_matches_original_exactly=True, cuda_strict_determinism_smoke_passed=True,
                repeated_seed_draw_equal=True, torch=torch.__version__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--qa-only", action="store_true")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = p.parse_args()
    root = args.root.resolve()
    from common import write_json
    if args.qa_only:
        result = sampler_qa()
        print(json.dumps(result))
        return
    if not args.worker:
        old = json.loads((root / "workflow_status.json").read_text())
        try:
            os.kill(old["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Refuse duplicate live worker")
        qa = sampler_qa()
        original = json.loads((root / "RUN_MANIFEST.json").read_text())
        write_json(root / "runtime_patch.json", dict(
            source="src/runtime_compat.py", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_fingerprint=original["fingerprint"], added_unix=time.time(), qa=qa,
            reason="PyTorch 2.3 CUDA cumsum disallowed with strict deterministic algorithms",
            change="Only TopPLogitsWarper cumulative sum uses CPU. GPU sort/softmax, requested T/top_p, model and feature code unchanged.",
            numerical_caveat="CPU versus CUDA cumulative sum can differ at floating-point threshold boundaries; this policy is explicit provenance, not a claim of bitwise equality to an unpatched GPU run.",
            existing_greedy_examples_unchanged=32))
        env = dict(os.environ, HF_HOME=str(root / "cache"), HF_HUB_DISABLE_XET="1",
                   TMPDIR=str(root / "tmp"), CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
        with (root / "job.log").open("a") as stream:
            process = subprocess.Popen([str(root / "venv/bin/python"), "-u", __file__, "--worker", "--root", str(root)],
                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True)
        print(json.dumps(dict(pid=process.pid, resumed=True, preserved_examples=32, qa=qa)))
        return
    TopPLogitsWarper.__call__ = cpu_cumsum_top_p
    import sys
    sys.argv = [sys.argv[0], "--root", str(root)]
    from run import main as run_main
    try:
        run_main()
    except Exception as error:
        write_json(root / "workflow_status.json", dict(state="failed", pid=os.getpid(), error=repr(error), updated_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
