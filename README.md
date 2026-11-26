# LifeMate - Vietnamese Voice Agent

LifeMate is a Vietnamese voice agent developed for Zalo AI Challenge 2026. GipFormer 1.5 transcribes audio, and Qwen3.5 LifeMate uses retrieved examples to decide which tools to call. Guard V3 validates arguments before the agent invokes a tool in ToolEnv. The agent can look up references and ask for clarification when information is missing.

## Public Test Results

| Experiment (oldest to newest) | Score |
| --- | ---: |
| 1 | 3.15000 |
| 2 | 7.70000 |
| 3 | 13.11000 |
| 4 | 32.15000 |
| 5 | 32.35000 |
| 6 | 38.25000 |
| 7 | 47.18000 |
| Latest v4 update | **49.18000** |

Each score reflects the full pipeline on the public test set.

## v4 Models

| Model | Role |
| --- | --- |
| [GipFormer 1.5 LifeMate v4](https://huggingface.co/TuanPhat/GipFormer-1.5-LifeMate-v4) | Fine-tuned from GipFormer 1.5 for Vietnamese speech recognition. |
| [Qwen3.5 LifeMate v4](https://huggingface.co/TuanPhat/Qwen3.5-LifeMate-v4) | Fine-tuned from Qwen3.5-4B for tool calling in LifeMate. |

## Run the Pipeline

Use Python 3.12 on Linux x86-64 with an NVIDIA GPU and a CUDA toolkit compatible with SGLang and PyTorch. If CUDA is installed in a custom location, set `CUDA_HOME` and add `$CUDA_HOME/bin` to `PATH`.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference]'
python -m lifemate --help
~~~

Use the official challenge dataset. Pass the parent directory of `public_test/` or `private_test/` to `--root`. Each split contains JSONL records, the tool catalog, audio files, and the organizers' `zaic_tools` package. Point `--train-root` to the directory containing `train.jsonl` and `train_tools_catalog.json` for retrieval and Guard V3.

~~~bash
python -m lifemate \
  --root /path/to/splits \
  --split public_test \
  --train-root /path/to/train \
  --out submission.jsonl \
  --limit 3
~~~

The first run downloads the two checkpoints from the Hugging Face revisions pinned in the code. To run offline, pass `--asr /path/to/gipformer` and `--model /path/to/qwen`. To select other Hub revisions, use `--asr-revision` and `--model-revision`. Each output JSONL record contains `id`, `tool_calls`, `state_hash`, and `time` in milliseconds.

## Run the Tests

~~~bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
~~~

## License

The LifeMate source code in this repository is licensed under the Apache License 2.0; see [LICENSE](LICENSE). The GipFormer checkpoint retains the upstream MIT license from G-Group AI Lab, and the Qwen checkpoint retains the upstream Apache-2.0 license from Qwen. Those model licenses remain separate from the source code license.
