# LifeMate - Vietnamese Voice Agent

LifeMate là trợ lý giọng nói tiếng Việt được phát triển cho Zalo AI Challenge 2026. Pipeline nhận dạng audio bằng GipFormer 1.5 INT8, truy xuất ví dụ để tạo prompt cho Qwen3.5 LifeMate và dùng Guard V3 để kiểm tra đối số trước khi gọi công cụ trong ToolEnv. Qwen có thể tra cứu để giải quyết tham chiếu và hỏi lại khi thiếu thông tin.

## Kết quả public test

| Phiên bản theo thứ tự từ cũ đến mới | Điểm |
| --- | ---: |
| 1 | 3.15000 |
| 2 | 7.70000 |
| 3 | 13.11000 |
| 4 | 32.15000 |
| 5 | 32.35000 |
| 6 | 38.25000 |
| 7 | 47.18000 |
| Bản cập nhật v4 mới nhất | **49.18000** |

Điểm là kết quả của toàn pipeline trên public test. Các số thứ tự trong bảng biểu thị thứ tự thử nghiệm.

## Checkpoint v4

| Model | Inference |
| --- | --- |
| [GipFormer 1.5 LifeMate v4](https://huggingface.co/TuanPhat/GipFormer-1.5-LifeMate-v4/tree/2570e079084df37cb152095e010f71f7d1bfc779) | ASR trên CPU qua sherpa-onnx. Hub cung cấp checkpoint raw-2000 ở dạng SafeTensors và bộ ONNX INT8 dùng khi chấm. |
| [Qwen3.5 LifeMate v4](https://huggingface.co/TuanPhat/Qwen3.5-LifeMate-v4/tree/cc4b5f58d5f771ad86ddbea114066e41bc4b97a4) | QwenTune dựa trên Qwen3.5-4B, checkpoint đã merge ở dạng SafeTensors, chạy bằng SGLang. |

## Chạy pipeline

Dùng Python 3.12 trên Linux x86-64, GPU NVIDIA và CUDA toolkit tương thích với SGLang/PyTorch. Nếu CUDA cài ở đường dẫn riêng, đặt `CUDA_HOME` và thêm `$CUDA_HOME/bin` vào `PATH`.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference]'
python -m lifemate --help
~~~

Chuẩn bị bộ dữ liệu chính thức. `--root` là thư mục cha của `public_test/` hoặc `private_test/`. Thư mục split chứa JSONL, catalog, audio và package `zaic_tools` của ban tổ chức. `--train-root` trỏ vào thư mục có `train.jsonl` và `train_tools_catalog.json` để chạy retrieval và Guard V3.

~~~bash
python -m lifemate \
  --root /path/to/splits \
  --split public_test \
  --train-root /path/to/train \
  --out submission.jsonl \
  --limit 3
~~~

Lần chạy đầu tải hai checkpoint Hugging Face theo revision v4 đã ghim trong mã. Để chạy offline, truyền `--asr /path/to/gipformer` và `--model /path/to/qwentune`. Dùng `--asr-revision` và `--model-revision` để chọn revision cụ thể khi tải từ Hub. Mỗi dòng JSONL đầu ra có `id`, `tool_calls`, `state_hash` và `time` (mili giây).

## Kiểm thử

~~~bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
~~~

## Giấy phép

Phần mã LifeMate do nhóm viết hiện chưa được cấp giấy phép riêng. Checkpoint GipFormer kế thừa MIT từ G-Group AI Lab và checkpoint Qwen kế thừa Apache-2.0 từ Qwen.
