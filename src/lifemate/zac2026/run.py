"""Offline LifeMate evaluation and timing entry point."""

import argparse
import json
import os
import random
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from lifemate.solution.agent import run_sample
from lifemate.solution.skills import AbstentionSkill
from .asr_retry import accepts_retry, needs_retry, pending_slot
from .audio import GipFormerASR
from .guard import Guard, opt_whitelist
from .prompt import FewShotIndex
from .qwen import MODEL_ALIAS, server_command, server_environment
from .vnnum import itn_time

SEED = 42
ASR_REPO = "TuanPhat/GipFormer-1.5-LifeMate-v4"
LLM_REPO = "TuanPhat/Qwen3.5-LifeMate-v4"
MODEL_REVISIONS = {
    ASR_REPO: "2570e079084df37cb152095e010f71f7d1bfc779",
    LLM_REPO: "cc4b5f58d5f771ad86ddbea114066e41bc4b97a4",
}
ASR_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
LLM_FILES = ["*.safetensors", "*.json", "chat_template.jinja", "merges.txt", "vocab.txt", "tokenizer.model"]


def model_path(local_path, repo, revision, files):
    if local_path is not None:
        return local_path.resolve()
    from huggingface_hub import snapshot_download

    revision = revision or MODEL_REVISIONS.get(repo)
    return Path(snapshot_download(repo, revision=revision, allow_patterns=files))



def seed_everything():
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_cases(root, split, limit=0):
    rows = _load_jsonl(root / split / f"{split}.jsonl")
    return rows[:limit] if limit else rows


def _transcribe_sample(asr, split_dir, row):
    paths = [row["audio"]]
    paths.extend(turn["audio"] for turn in row["history"] if turn["role"] == "user" and turn.get("audio"))
    paths = list(dict.fromkeys(paths))
    raw = asr.transcribe_batch([split_dir / path for path in paths])
    texts = {path: itn_time(text) for path, text in zip(paths, raw, strict=True)}
    path = row["audio"]
    slot = pending_slot(row["history"])
    if slot is not None and needs_retry(slot, raw[0], texts[path]):
        retry, = asr.transcribe_batch([split_dir / path])
        normalized = itn_time(retry)
        if accepts_retry(slot, raw[0], retry, normalized):
            texts[path] = normalized
    return texts


@contextmanager
def _chat_backend(model, port, work):
    import httpx
    from openai import APITimeoutError, OpenAI

    url = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    with (work / "sglang.log").open("wb") as log:
        proc = subprocess.Popen(
            server_command(model, port), env=server_environment(),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"SGLang exited with code {proc.returncode}. See {log.name}")
                try:
                    with opener.open(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"SGLang startup timed out. See {log.name}")
            with OpenAI(base_url=url + "/v1", api_key="EMPTY", max_retries=0,
                        http_client=httpx.Client(trust_env=False)) as client:
                def chat(messages, tools, stop_at):
                    remaining = stop_at - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    try:
                        return client.chat.completions.create(
                            model=MODEL_ALIAS, messages=messages, tools=tools, tool_choice="auto",
                            temperature=0, max_tokens=1024, timeout=remaining,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    except APITimeoutError:
                        raise TimeoutError from None
                yield chat
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


@contextmanager
def load_resources(root, split, train_root, asr_path, model, work, port=30004):
    started = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    split_dir = root / split
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv, catalog_tools

    def catalog(path):
        return catalog_tools(json.loads(path.read_text(encoding="utf-8")))

    train = _load_jsonl(train_root / "train.jsonl")
    train_specs = catalog(train_root / "train_tools_catalog.json")
    specs = catalog(split_dir / f"{split}_tools_catalog.json")
    fewshot = FewShotIndex(train)
    guard = partial(Guard, whitelist=opt_whitelist(train, train_specs))
    abstention = AbstentionSkill(specs)
    asr = GipFormerASR({"gipformer_path": asr_path, "cpu_threads": 4})
    try:
        with _chat_backend(model, port, work) as chat:
            print(f"model_load_s={time.monotonic() - started:.3f}", flush=True)

            def predict(row):
                start = time.monotonic()
                env = ToolEnv(split, row["id"], root=root)
                try:
                    texts = _transcribe_sample(asr, split_dir, row)
                except Exception as exc:
                    result = {"id": row["id"], "tool_calls": env.get_tool_calls(),
                              "state_hash": env.get_state_hash(), "error": repr(exc)}
                else:
                    remaining = 30 - (time.monotonic() - start)
                    result = run_sample(row, env, specs=specs, transcript=texts.__getitem__, fewshot=fewshot,
                                        guard_factory=guard, abstention=abstention, chat=chat,
                                        fewshot_count=2, seconds=remaining)
                elapsed = time.monotonic() - start
                result["time"] = int(elapsed * 1000)
                if elapsed >= 30:
                    result["timeout"] = True
                return result

            yield predict
    finally:
        asr.close()
        sys.path.remove(str(split_dir))


def write_results(predict, rows, output):
    from tqdm.auto import tqdm

    start = time.monotonic()
    errors, timeouts, max_time = {}, [], 0
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the result inode when the grader mounts the output file itself.
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError(f"Expected a regular output file: {output}")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        os.chmod(output, 0o644)
        for row in tqdm(rows, desc="LifeMate"):
            result = predict(row)
            record = {k: result[k] for k in ("id", "tool_calls", "state_hash", "time")}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            max_time = max(max_time, result["time"])
            if "error" in result:
                errors[row["id"]] = result["error"]
            if result.get("timeout"):
                timeouts.append(row["id"])
        stream.flush()
        os.fsync(stream.fileno())
    stats = {"rows": len(rows), "errors": errors, "timeouts": timeouts, "max_sample_ms": max_time,
             "inference_s": round(time.monotonic() - start, 3)}
    print(json.dumps(stats), flush=True)
    return stats


def confirm_output(output):
    info = output.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise RuntimeError(f"Expected a nonempty regular file: {output}")
    print(f"output={output.absolute()} bytes={info.st_size} mode={stat.S_IMODE(info.st_mode):04o}", flush=True)


def default_split(root):
    return "private_test" if (root / "private_test/private_test.jsonl").is_file() else "public_test"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing evaluation splits")
    parser.add_argument("--split", choices=("private_test", "public_test"))
    parser.add_argument("--train-root", type=Path, required=True, help="Directory containing train JSONL and tool catalog")
    parser.add_argument("--asr", type=Path, help="Local GipFormer INT8 directory")
    parser.add_argument("--asr-repo", default=ASR_REPO, help="Hugging Face ASR repo used when --asr is omitted")
    parser.add_argument("--asr-revision", help="Hugging Face revision for the ASR model")
    parser.add_argument("--model", type=Path, help="Local QwenTune directory")
    parser.add_argument("--model-repo", default=LLM_REPO, help="Hugging Face LLM repo used when --model is omitted")
    parser.add_argument("--model-revision", help="Hugging Face revision for the LLM")
    parser.add_argument("--out", type=Path, default=Path("submission.jsonl"))
    parser.add_argument("--port", type=int, default=30004)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    split = args.split or default_split(args.root)
    seed_everything()
    rows = read_cases(args.root, split, args.limit)
    if not rows:
        parser.error(f"No cases in {split}")
    asr = model_path(args.asr, args.asr_repo, args.asr_revision, ASR_FILES)
    model = model_path(args.model, args.model_repo, args.model_revision, LLM_FILES)
    # Keep server logs off the grader's result mount.
    with TemporaryDirectory(prefix="lifemate-") as work:
        try:
            with load_resources(args.root, split, args.train_root, asr, model,
                                Path(work), args.port) as predict:
                write_results(predict, rows, args.out)
        except Exception:
            log = Path(work) / "sglang.log"
            if log.is_file():
                print("\n".join(log.read_text(errors="replace").splitlines()[-12:]), file=sys.stderr)
            raise
        confirm_output(args.out)
    return 0
