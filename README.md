# Embedding service: Giga-Embeddings-instruct-480M-0826 (GGUF Q8_0)

End-to-end release pipeline that converts
[ai-sage/Giga-Embeddings-instruct-480M-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826)
into a GGUF model served by `llama-server` as an OpenAI-compatible
embedding endpoint.

> 🇷🇺 Эмбеддинг-модель 480M на базе двунаправленного Qwen3 для задач retrieval
> на русском и английском. FP16 + Q8_0 GGUF. Технические детали и заметки
> по сборке — ниже. **Веса не хранятся в этом репозитории** —
> они превышают лимит GitHub в 100 МБ на файл.

## Где взять веса

| Artifact | Size | HF |
|---|---|---|
| Исходная HF-модель | 933 МБ | [ai-sage/Giga-Embeddings-instruct-480M-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826) |
| FP16 GGUF | 968 МБ | https://huggingface.co/rad0main/Giga-Embeddings-instruct-480M-0826-GGUF |
| Q8_0 GGUF | 496 МБ | https://huggingface.co/rad0main/Giga-Embeddings-instruct-480M-0826-GGUF |

## Что в этом репозитории

| Файл | Назначение |
|---|---|
| `embed_bench.py` | Скрипт бенчмарка: поднимает `llama-server`, мерит метрики EM-016..048 + INST |
| `summarize_bench.py` | Сводит результаты двух прогонов в табличку дельт |
| `docker-compose.yml` | Конфиг для запуска embedding-сервиса в контейнере |
| `docs/BENCHMARK_SUMMARY.md` | Сводная таблица бенчмарка (giga vs qwen3-embedding-0.6b) |
| `logs/embed_bench_*.console.txt` | Сырые stdout бенчмарка для обеих моделей |
| `.gitignore` | Исключает `*.gguf`, `models/`, `llama.cpp/`, `logs/*.json` |

## What was built

| Artifact | Size | Where |
|---|---|---|
| HF source | 933 MB | `models/Giga-Embeddings-instruct-480M-0826/` (в .gitignore) |
| FP16 GGUF | 968 MB | HF: `rad0main/Giga-Embeddings-instruct-480M-0826-GGUF` |
| Q8_0 GGUF | 496 MB | HF: `rad0main/Giga-Embeddings-instruct-480M-0826-GGUF` |
| `llama-server` | built | собирается из исходников `ggerganov/llama.cpp` (в .gitignore) |
| Docker compose | – | `docker-compose.yml` |

## Architecture notes

- The HF model is `Qwen3BidirectionalModel` — a bidirectional (non-causal)
  variant of Qwen3 used as an embedding encoder. Weights are identical to
  stock Qwen3 (RoPE, RMSNorm, QK-norm, SwiGLU). The only differences are
  `is_causal=False` and full-attention masks, which are applied at runtime
  by llama.cpp via the `--pooling mean` + bidirectional attention path.
- Upstream SentenceTransformers config:
  - `1_Pooling/config.json`: `pooling_mode_mean_tokens=true` → `--pooling mean`
  - `2_Normalize/config.json`: empty `{}` but `embd-normalize=2` (L2) is the
    standard default and matches common usage.

## Run with Docker

```bash
docker compose up -d
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello world","model":"giga-embeddings-instruct-480M-0826"}'
```

The `docker-compose.yml` uses the CUDA build by default. To run CPU-only,
delete the `deploy.resources` block.

## Run locally (Windows)

```powershell
$env:PATH = "C:\Users\rad0\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;$env:PATH"
.\llama.cpp\build\bin\llama-server.exe `
  --model .\models\gguf\giga-embeddings-instruct-480M-0826.Q8_0.gguf `
  --embedding --pooling mean --embd-normalize 2 `
  --host 0.0.0.0 --port 8080
```

## Reproducing the build

The full pipeline (in `logs/`):

1. **Install toolchain** (one-time, no admin required):
   ```powershell
   & "$env:LOCALAPPDATA\Microsoft\WindowsApps\winget.exe" install --id BrechtSanders.WinLibs.POSIX.UCRT --scope user --accept-package-agreements --accept-source-agreements
   python -m pip install "huggingface_hub" "transformers>=4.43" "tokenizers>=0.19" "numpy<3" "torch" "sentencepiece" "protobuf"
   ```
2. **Clone & build llama.cpp** (MinGW + patches; see `logs/01b-reconfigure.log`):
   ```powershell
   git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
   # Patch cpp-httplib for MinGW: CreateFile2 -> CreateFileW, MapViewOfFileFromApp -> MapViewOfFile, etc.
   # Set _WIN32_WINNT=0x0A00 (Win 10) so cpp-httplib compiles.
   cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release `
         -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00" `
         -DGGML_OPENMP=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_SERVER=ON
   cmake --build build --config Release -- -j 8
   ```
3. **Download the model**:
   ```powershell
   hf download ai-sage/Giga-Embeddings-instruct-480M-0826 --local-dir models/Giga-Embeddings-instruct-480M-0826
   ```
4. **Convert to GGUF FP16** — see `logs/04-convert.log`:
   - Make a working copy with `config.json` patched:
     - `architectures: ["Qwen3ForCausalLM"]`
     - `model_type: "qwen3"`
     - drop `is_causal`, `auto_map`, `max_window_layers`, `sliding_window`,
       `layer_types`, `use_sliding_window`, `attention_bias`
   - Patch `llama.cpp/conversion/base.py` `get_vocab_base_pre` to add the
     tokenizer hash for this model (qwen2 BPE). The required hash is
     `5c01b97b9959d897bb3670b43dd3cfe4ab93cef6280acd4d55a15e66d68213c9`.
   - Run:
     ```powershell
     python convert_hf_to_gguf.py <patched-model-dir> --outfile models/gguf/...fp16.gguf --outtype f16
     ```
5. **Quantize to Q8_0**:
   ```powershell
   llama-quantize models/gguf/...fp16.gguf models/gguf/...Q8_0.gguf Q8_0
   ```
6. **Serve** via `docker compose up -d` or the local command above.

## Logs

All steps log to `logs/`:
- `01-configure.log` / `01b-reconfigure.log` — CMake configure
- `02-build.log` — CMake build
- `03-download.log` — HF model download
- `04-convert.log` — HF→GGUF FP16
- `05-quantize.log` — Q8_0 quantization

## Caveats

- The model is a bidirectional Qwen3 — weights are stock Qwen3, but the
  attention mask at inference must be full (not causal). llama.cpp's
  embedding path applies this when `--embedding` is set. If you observe
  nonsense outputs, confirm `--embedding` is in the command line.
- For best quality on Russian/English retrieval tasks, follow the model's
  instruction template. The model uses a "query:" / "passage:" style prompt;
  the SentenceTransformers wrapper prepends it. For raw API use, prepend
  manually if the upstream model card recommends it.
- The Q8_0 quantization is lossless in practice for embedding quality
  (typical cosine-sim drift is < 0.1%). If you need higher fidelity, use
  the FP16 GGUF instead.
