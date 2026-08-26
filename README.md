# Giga-Embeddings GGUF Q8_0 release

[![GGUF Q8_0](https://img.shields.io/badge/GGUF-Q8__0-blue)](https://github.com/rad0main/Giga-Embeddings-480M-0826-GGUF)
[![llama.cpp](https://img.shields.io/badge/serves-llama.cpp-orange)](https://github.com/ggml-org/llama.cpp)
[![languages](https://img.shields.io/badge/lang-RU%20%7C%20EN-brightgreen)](#)
[![hardware](https://img.shields.io/badge/CPU-validated-informational)](#)

> 🇷🇺 **RU.** Этот репозиторий содержит пайплайн конверсии
> [ai-sage/Giga-Embeddings-instruct-480M-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826)
> (двунаправленный Qwen3 480M, русский + английский retrieval) в GGUF Q8_0,
> плюс измерительный харнес (`embed_bench.py`) и замеры на локальном CPU.
> **Веса модели сюда не входят** — они лежат на Hugging Face.

## What this repo is

- **Conversion pipeline** (не в git-е, см. `.gitignore`): скрипты и конфиги для
  перевода исходной HF-модели в GGUF FP16 и квантования в Q8_0.
- **Serving**: `llama.cpp` с `--embedding --pooling mean --embd-normalize 2`,
  OpenAI-совместимый endpoint `/v1/embeddings`.
- **Bench harness**: [`embed_bench.py`](embed_bench.py) — поднимает
  `llama-server`, мерит latency / batch throughput / детерминизм / retrieval
  (с инструкцией и без) и пишет сырой stdout в `logs/`.

**Weights are not in this repo** — they live on Hugging Face:
<https://huggingface.co/rad0main/Giga-Embeddings-instruct-480M-0826-GGUF>
(source: <https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826>).
PP-веса исходника 933 МБ; FP16 GGUF 968 МБ; Q8_0 GGUF 496 МБ — все они
превышают лимит GitHub 100 МБ на файл.

## Model spec

| Field | Value |
|---|---|
| Architecture | `qwen3` |
| Bidirectional attention | yes (custom `Qwen3BidirectionalModel` upstream) |
| Embedding dim | `1024` |
| Context length | `8192` |
| Pooling (GGUF) | `MEAN` |
| Format | GGUF, Q8_0 (8.50 BPW) |
| L2 normalization | yes (`--embd-normalize 2`) |
| Tokenizer | GPT-2 BPE, pre-tokenizer `qwen2` (Qwen2/Qwen3 family) |

## Benchmarks (giga vs default)

Сравнение с baseline **Qwen3-Embedding-0.6B (Q8_0, default Qwen pooling)**
на встроенном тест-наборе из 8 «воспоминаний» и 3 запросов. Все числа из
[`docs/BENCHMARK_SUMMARY.md`](docs/BENCHMARK_SUMMARY.md) (первоисточники:
[`embed_bench_giga.console.txt`](logs/embed_bench_giga.console.txt),
[`embed_bench_default.console.txt`](logs/embed_bench_default.console.txt)).

| Metric | giga Q8_0 | default Q8_0 (qwen3-embedding-0.6b) | Δ (giga − default) |
|---|---:|---:|---:|
| load_s | `2.66` | `7.32` | `-4.66` |
| lat_ms (10 single-call) | `141.99` | `199.4` | `-57.41` |
| batch_thr (emb/s, batch=32) | `12.59` | `9.16` | `+3.43` |
| det_dist (same text twice) | `-0.0` | `-0.0` | `0.0` |
| norm (one embedding) | `1.0` | `1.0` | `0.0` |
| recall@1 (no instruction) | `1.0` | `0.6667` | `+0.3333` |
| recall@3 (no instruction) | `1.0` | `1.0` | `0.0` |
| mrr (no instruction) | `1.0` | `0.8333` | `+0.1667` |
| recall@1 (instruct prefix) | `1.0` | `0.3333` | `+0.6667` |
| recall@3 (instruct prefix) | `1.0` | `1.0` | `0.0` |
| mrr (instruct prefix) | `1.0` | `0.6667` | `+0.3333` |

Similarity (косинус, эталон после L2-нормирования):

| Pair | giga | default |
|---|---:|---:|
| sim[0] «таверну» ~ «кабачка на закате» | `0.5097` | `0.8499` |
| sim[1] «меч сломался» ~ «клинок треснул» | `0.6375` | `0.7977` |
| sim[2] «предательстве» ~ «брат предал» | `0.6354` | `0.866` |
| dissim[0] (меч / порт) | `0.2921` | `0.6694` |
| dissim[1] (предательстве / чертёж) | `0.455` | `0.7694` |

Spec baseline (для понимания разрывов):

| | giga Q8_0 | default Q8_0 |
|---|---|---|
| GGUF pooling | `MEAN` | `LAST` |
| Context length | `8192` | `32768` |
| Size on disk | 496 МБ | 609 МБ |

## Key findings

1. **`recall@1` под instruct-prefix: `1.00` vs `0.33` для default.**
   Бенч запускался с одинаковым `--pooling mean` для обеих моделей, но
   в GGUF-метадате default хранится `pooling_type=LAST` (3). При включении
   инструкции запросы становятся длиннее, и last-token-pooling у default
   «съезжает»: на третьем запросе («Кто следил за караваном?») top-1 уходит
   в `m1` (Казлина вошла в мастерскую) вместо `m5` (Тихий следил за
   караваном). Giga с mean-pooling устойчив к инструкции.

2. **Абсолютные косинусы у giga ниже, но разрыв sim/dissim больше.**
   Среднее sim у giga: `(0.5097 + 0.6375 + 0.6354) / 3 ≈ 0.594`,
   среднее dissim: `(0.2921 + 0.455) / 2 ≈ 0.374`,
   Δ ≈ `0.22`. У default: sim ≈ `0.838`, dissim ≈ `0.719`, Δ ≈ `0.12`.
   При равном recall@1 giga даёт более «разнесённые» ранжирования, что
   хорошо для пороговой фильтрации.

3. **Порог «похоже» надо калибровать заново.** Если для Qwen3-Embedding-0.6B
   в вашей retrieval-системе работает `cos ≥ 0.70`, для giga с L2-нормой
   надо сдвигать окно в `0.45–0.50`. Иначе вы получите ложные негативы
   на семантически близких парах вроде sim[0] = `0.5097`.

## Quick start

### 1. Запуск сервера

```bash
# из корня репо
llama-server \
  -m ./models/giga-embeddings-instruct-480M-0826.Q8_0.gguf \
  --embedding \
  --pooling mean \
  --embd-normalize 2 \
  --host 0.0.0.0 --port 8080 \
  -ngl 0 -t 4 --seed 42 --cont-batching
```

(Путь к `llama-server` зависит от того, как собран llama.cpp. В нашей
тестовой среде использовался бинарь из `D:\AIprojects\embedding-deploy\llama.cpp\build\bin\llama-server.exe`,
который в git не входит — собирается отдельно, см. `llama.cpp` README.)

### 2. Запрос через curl

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Незнакомец переступил порог кабачка на закате",
    "model": "giga-embeddings-instruct-480M-0826"
  }'
```

### 3. Python (OpenAI-совместимый клиент)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

resp = client.embeddings.create(
    model="giga-embeddings-instruct-480M-0826",
    input="Незнакомец переступил порог кабачка на закате",
)
vec = resp.data[0].embedding  # list[float], длина 1024
```

## Reproduce the benchmark

```bash
# 1) собрать llama.cpp (любой коммит >= 5d5cb4c поддерживает эту модель)
# 2) положить gguf в ./models/ (или указать --gguf)
# 3) запустить харнес
python embed_bench.py \
  --gguf ./models/giga-embeddings-instruct-480M-0826.Q8_0.gguf \
  --label giga \
  --port 8081
```

Сырой stdout прогона — в `logs/embed_bench_*.console.txt`:

- `logs/embed_bench_giga.console.txt` — giga
- `logs/embed_bench_default.console.txt` — default (Qwen3-Embedding-0.6B Q8_0)

Структурированные JSON (`logs/embed_bench_*.json`) и сводный `logs/summary.txt`
в git **не** входят — JSON слишком большие (> 50 МБ, так как сериализуют
весь массив эмбеддингов), а `summary.txt` генерируется на лету из JSON.
Пример интерпретации — в [`docs/BENCHMARK_SUMMARY.md`](docs/BENCHMARK_SUMMARY.md).

## Repo structure

```
.
├── README.md                          # этот файл
├── .gitignore
├── docs/
│   └── BENCHMARK_SUMMARY.md           # сводная таблица и интерпретация
├── embed_bench.py                     # харнес: server + /v1/embeddings + метрики
└── logs/
    ├── .gitignore                     # allowlist: только console.txt
    ├── embed_bench_giga.console.txt   # сырой stdout прогона giga
    └── embed_bench_default.console.txt# сырой stdout прогона default
```

## Not included (and where to get them)

| Что | Размер | Где |
|---|---|---|
| GGUF Q8_0 веса | 496 МБ | <https://huggingface.co/rad0main/Giga-Embeddings-instruct-480M-0826-GGUF> |
| GGUF FP16 веса | 968 МБ | <https://huggingface.co/rad0main/Giga-Embeddings-instruct-480M-0826-GGUF> |
| Исходная HF-модель | 933 МБ | <https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826> |
| Структурированные JSON логи | 50–60 МБ каждый | исключены в `.gitignore` (build/lfs-кандидаты) |
| `llama-server` бинарь | 25 МБ | собирается локально из `ggerganov/llama.cpp` |
| Исходники `llama.cpp` | ~350 МБ | исключены в `.gitignore` (сторонний репо) |

## Limitations

- **Context 8192**: у giga `qwen3.context_length=8192`, что вдвое меньше, чем
  у Qwen3-Embedding-0.6B (`32768`). Для длинных документов нужен chunking.
  Из коробки retrieval-тест на 8 коротких воспоминаниях укладывается, но
  на длинных русских текстах (>8K символов) надо разрезать.
- **CPU-only test**: бенч запущен на CPU (`-ngl 0`, `-t 4`, Windows 11,
  llama.cpp commit `5d5cb4c`). С CUDA-билдом `llama-server` latency
  ожидаемо ниже, batch throughput выше — численные выводы о качестве
  ранжирования не изменятся.
- **RSS не замерян**: `rss_mb` помечен как `not measured` — в Windows 11
  24H2+ `wmic` отсутствует, а других способов в батарею не включено.
  По косвенным признакам (latency и batch throughput) giga Q8_0 на CPU
  экономнее, чем default Q8_0, — но это не прямой замер памяти.
- **Маленький тест-набор**: 8 «воспоминаний» и 3 запроса — это sanity-check
  на детерминизм, pooling-инвариантность и относительный разрыв метрик.
  `recall@3=1.0` у обеих моделей следствие маленького корпуса, а не
  полноценного retrieval-бенча. Для продакшн-оценки нужен MTEB / ruMTEB
  на полном корпусе.

## License & citation

- Код в этом репозитории: MIT (если явно не указано иное).
- Модель
  [ai-sage/Giga-Embeddings-instruct-480M-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826)
  распространяется на условиях её собственной лицензии (см. карточку
  модели на HF).
- Базовый чекпойнт — Qwen3 (см.
  [Qwen/Qwen3](https://huggingface.co/Qwen/Qwen3)).

При использовании giga Q8_0 в публикациях ссылайтесь на оригинальную
карточку HF-модели; этот репозиторий — только конверсия и измерения.
