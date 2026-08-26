#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
embed_bench.py — батарея тестов над локальной GGUF-моделью через llama.cpp server.

Метрики (EM-ид из ТЗ):
  [EM-016] load_s      — время до готовности сервера (до первого 200 на /health или /v1/models)
  [EM-012] dim         — размерность вектора
  [EM-027] norm        — L2-норма (для L2-normalized должна быть ~1)
  [EM-017] lat_ms      — средняя латентность 10 коротких фраз (одна за раз)
  [EM-018] batch_thr   — эмбеддингов/с на батче из 32 фраз
  [EM-028] det         — косинус-дистанция между двумя запросами одного и того же текста
  [EM-032/033] sim/dissim — косинусы пар A (похожие) и B (непохожие)
  [EM-044..048] retrieval — Recall@1, Recall@3, MRR по корпусу C и запросам D
  [INST]              — retrieval без инструкции и с instruction-префиксом

Использование:
  python embed_bench.py --gguf PATH --label giga --port 8080 [--instruct-prefix PREFIX]
  python embed_bench.py --gguf PATH --label giga --port 8080 --no-server
      (если сервер уже запущен)
"""
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import requests

# ---------- test data (НЕ менять) ----------
PAIRS_SIM = [
    ("Я видел, как он вошёл в таверну",
     "Незнакомец переступил порог кабачка на закате"),
    ("Меч сломался в бою",
     "Клинок треснул, отражая удар орка"),
    ("Она рассказала о предательстве",
     "Исповедь о том, как брат предал её семью"),
]
PAIRS_DISSIM = [
    ("Меч сломался в бою",
     "Северный порт зимой. Прибывает пароход без знаков."),
    ("Она рассказала о предательстве",
     "На полу — обронённый чертёж."),
]
CORPUS = [
    "Казлина вошла в мастерскую и услышала, как кап ржавой меди",                # m1
    "Йорен предложил союз против Братства",                                       # m2
    "На столе лежал обронённый чертёж шестерни",                                  # m3
    "Северный порт зимой. Прибывает пароход без знаков.",                         # m4
    "Тихий следил за караваном два дня",                                          # m5
    "Казлина договорилась о грузе с контрабандистом",                             # m6
    "Шестерня механизма была повреждена",                                         # m7
    "Первая улика: половинка медальона",                                          # m8
]
QUERIES = [
    ("Что случилось с механизмом в мастерской?", [6, 2]),   # 0-indexed: m7, m3
    ("С кем Казлина договаривалась о грузе?",   [5]),       # m6
    ("Кто следил за караваном?",                  [4]),    # m5
]
INSTR_PREFIX_DEFAULT = "Instruct: найди релевантное воспоминание\nQuery: "

LAT_PROMPTS = [
    "привет",
    "как дела?",
    "что нового?",
    "расскажи сказку",
    "погода сегодня хорошая",
    "который час?",
    "где библиотека",
    "хочу пить",
    "спокойной ночи",
    "до завтра",
]
BATCH_PROMPTS = LAT_PROMPTS * 3 + [   # 30
    "история про дракона", "про погоду", "что такое ИИ", "рецепт борща",
]  # 34, обрежем до 32


# ---------- helpers ----------
def now() -> float:
    return time.monotonic()


def _as_int(v):
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    try:
        return int(v)
    except Exception:
        return None


def parse_pooling(meta: dict) -> str:
    v = _as_int(meta.get("pooling_type"))
    if v is None:
        return "UNKNOWN"
    return {0: "NONE", 1: "MEAN", 2: "CLS", 3: "LAST", 4: "RANK"}.get(v, f"UNK({v})")


def read_gguf_meta(path: str) -> dict:
    """Read GGUF metadata via the gguf python package shipped with llama.cpp."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py"))
    from gguf import GGUFReader  # type: ignore
    r = GGUFReader(path)
    out = {}
    for k, field in r.fields.items():
        try:
            if field.types and field.types[0].name == "STRING":
                v = "".join(chr(b) for b in field.parts[field.data[0]])
            else:
                v = field.parts[field.data[0]].tolist() if len(field.data) == 1 else \
                    [field.parts[d].tolist() for d in field.data]
            out[k] = v
        except Exception:
            pass
    return out


def start_server(gguf: str, port: int, log_path: str, extra_args: List[str]) -> Tuple[subprocess.Popen, float]:
    """Start llama-server in background, return (proc, start_monotonic)."""
    # Try several known locations for the binary
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "llama.cpp", "build", "bin", "llama-server.exe"),
        os.path.join(here, "..", "llama.cpp", "build", "bin", "llama-server.exe"),
        r"D:\AIprojects\embedding-deploy\llama.cpp\build\bin\llama-server.exe",
    ]
    bin_path = None
    for c in candidates:
        c_abs = os.path.abspath(c)
        if os.path.isfile(c_abs):
            bin_path = c_abs
            break
    if bin_path is None:
        # last fallback: which-style lookup
        for p in os.environ.get("PATH", "").split(os.pathsep):
            cand = os.path.join(p, "llama-server.exe")
            if os.path.isfile(cand):
                bin_path = cand
                break
    if bin_path is None:
        raise FileNotFoundError("llama-server.exe not found in any known location")
    cmd = [
        bin_path,
        "-m", gguf,
        "--embedding",
        "--pooling", "mean",
        "--embd-normalize", "2",
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", "0",
        "-t", "4",
        "--seed", "42",
        "--cont-batching",
    ] + extra_args
    env = os.environ.copy()
    # add MinGW bin to PATH
    mingw = r"C:\Users\rad0\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
    env["PATH"] = mingw + ";" + env.get("PATH", "")
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"

    logf = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return proc, now()


def wait_for_server(port: int, timeout: float = 180) -> Optional[float]:
    """Poll /health or /v1/models until 200, return elapsed seconds. None on timeout."""
    urls = [f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/models"]
    t0 = now()
    while now() - t0 < timeout:
        for u in urls:
            try:
                r = requests.get(u, timeout=2)
                if r.status_code == 200:
                    return now() - t0
            except Exception:
                pass
        time.sleep(0.5)
    return None


def get_rss_mb(pid: int) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", f"ProcessId={pid}", "get", "WorkingSetSize", "/value"],
            timeout=5
        ).decode("utf-8", "ignore")
        m = re.search(r"WorkingSetSize=(\d+)", out)
        if m:
            return round(int(m.group(1)) / 1024 / 1024, 1)
    except Exception:
        return None
    return None


# ---------- embedding client ----------
class EmbedClient:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def embed(self, texts: List[str], model: str = "bench") -> List[List[float]]:
        """Call /v1/embeddings. If len(texts)==1, do per-text calls (single).
        Else do batched call."""
        if len(texts) == 1:
            r = self.session.post(
                f"{self.base}/v1/embeddings",
                json={"input": texts[0], "model": model},
                timeout=120,
            )
            r.raise_for_status()
            j = r.json()
            return [j["data"][0]["embedding"]]
        # batch
        r = self.session.post(
            f"{self.base}/v1/embeddings",
            json={"input": texts, "model": model},
            timeout=300,
        )
        r.raise_for_status()
        j = r.json()
        j["data"].sort(key=lambda x: x["index"])
        return [d["embedding"] for d in j["data"]]


# ---------- metrics ----------
def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cos(a, b)


def recall_at_k(rels: List[int], ranked: List[int], k: int) -> float:
    """rels: list of gold indices. ranked: ranked list of indices."""
    top = ranked[:k]
    return 1.0 if any(r in top for r in rels) else 0.0


def mrr(rels: List[int], ranked: List[int]) -> float:
    for i, idx in enumerate(ranked):
        if idx in rels:
            return 1.0 / (i + 1)
    return 0.0


# ---------- main ----------
def run_bench(gguf: str, label: str, port: int, no_server: bool,
              log_path: str, instruct_prefix: str, no_instruct: bool) -> dict:
    out: dict = {"label": label, "gguf": gguf}
    # metadata
    out["meta"] = read_gguf_meta(gguf)
    arch = out["meta"].get("general.architecture", "?")
    ctx = _as_int(out["meta"].get(f"{arch}.context_length"))
    emb_len = _as_int(out["meta"].get(f"{arch}.embedding_length"))
    pool = parse_pooling(out["meta"])
    out["arch"] = str(arch)
    out["context_length"] = ctx
    out["embedding_length"] = emb_len
    out["pooling"] = pool
    print(f"[META] {label}: arch={arch} dim={emb_len} ctx={ctx} pooling={pool}", flush=True)

    proc = None
    try:
        if not no_server:
            proc, t0 = start_server(gguf, port, log_path, [])
            print(f"[META] server pid={proc.pid} starting...", flush=True)
            t = wait_for_server(port, timeout=180)
            if t is None:
                print(f"[EM-016] load_s: SKIPPED (server failed to start in 180s)", flush=True)
                out["load_s"] = None
                return out
            out["load_s"] = round(t, 2)
            print(f"[EM-016] load_s: {out['load_s']}", flush=True)
            out["pid"] = proc.pid
            rss = get_rss_mb(proc.pid)
            if rss is not None:
                print(f"[RSS] rss_mb: {rss}", flush=True)
                out["rss_mb"] = rss
        else:
            out["load_s"] = 0.0

        cli = EmbedClient(port)

        # ----- dim/norm/det using the corpus
        e0 = np.array(cli.embed([CORPUS[0]])[0], dtype=np.float64)
        out["dim"] = int(e0.shape[0])
        out["norm"] = round(float(np.linalg.norm(e0)), 4)
        print(f"[EM-012] dim: {out['dim']}", flush=True)
        print(f"[EM-027] norm: {out['norm']}", flush=True)

        # det — дважды один текст
        e1 = np.array(cli.embed([CORPUS[0]])[0], dtype=np.float64)
        d_det = cos_dist(e0, e1)
        out["det_dist"] = round(d_det, 6)
        print(f"[EM-028] det: {out['det_dist']}", flush=True)

        # ----- latency: 10 single-call
        lat = []
        for p in LAT_PROMPTS:
            t1 = now()
            cli.embed([p])
            lat.append((now() - t1) * 1000.0)
        out["lat_ms"] = round(float(np.mean(lat)), 2)
        out["lat_ms_p50"] = round(float(np.median(lat)), 2)
        out["lat_ms_per_call"] = [round(x, 2) for x in lat]
        print(f"[EM-017] lat_ms: {out['lat_ms']} (p50={out['lat_ms_p50']})", flush=True)

        # ----- batch throughput: 32 фраз за один вызов
        batch = BATCH_PROMPTS[:32]
        t1 = now()
        emb_b = cli.embed(batch)
        dt = now() - t1
        out["batch_dt_s"] = round(dt, 4)
        out["batch_thr"] = round(len(batch) / dt, 2)
        out["batch_dim"] = int(len(emb_b[0]))
        print(f"[EM-018] batch_thr: {out['batch_thr']} emb/s (32 in {out['batch_dt_s']}s)", flush=True)

        # ----- sim pairs A
        sim_cos = []
        for a, b in PAIRS_SIM:
            ea, eb = cli.embed([a, b])
            sim_cos.append(cos(np.array(ea), np.array(eb)))
        out["sim"] = [round(x, 4) for x in sim_cos]
        print(f"[EM-032] sim: {out['sim']}", flush=True)

        # ----- dissim pairs B
        dis_cos = []
        for a, b in PAIRS_DISSIM:
            ea, eb = cli.embed([a, b])
            dis_cos.append(cos(np.array(ea), np.array(eb)))
        out["dissim"] = [round(x, 4) for x in dis_cos]
        print(f"[EM-033] dissim: {out['dissim']}", flush=True)

        # ----- retrieval: эмбеддинги корпуса один раз, потом по запросам
        corpus_emb = [np.array(x, dtype=np.float64) for x in cli.embed(CORPUS)]

        def retrieval(query_embs, query_texts, gold_lists):
            r1, r3, mrrs = [], [], []
            for qe, qtxt, gold in zip(query_embs, query_texts, gold_lists):
                ranked = sorted(
                    range(len(corpus_emb)),
                    key=lambda i: -cos(qe, corpus_emb[i]),
                )
                r1.append(recall_at_k(gold, ranked, 1))
                r3.append(recall_at_k(gold, ranked, 3))
                mrrs.append(mrr(gold, ranked))
                top1 = ranked[0]
                ok1 = "OK" if top1 in gold else "MISS"
                print(f"  Q='{qtxt[:50]}' gold={gold} top1={top1} ({ok1}) top5={ranked[:5]}", flush=True)
            return {
                "recall@1": round(float(np.mean(r1)), 4),
                "recall@3": round(float(np.mean(r3)), 4),
                "mrr": round(float(np.mean(mrrs)), 4),
            }

        # без инструкции
        q_texts = [q[0] for q in QUERIES]
        q_golds = [q[1] for q in QUERIES]
        q_embs = [np.array(x, dtype=np.float64) for x in cli.embed(q_texts)]
        out["retrieval_no_instr"] = retrieval(q_embs, q_texts, q_golds)
        print(f"[EM-044] no-instr: recall@1={out['retrieval_no_instr']['recall@1']} "
              f"recall@3={out['retrieval_no_instr']['recall@3']} "
              f"mrr={out['retrieval_no_instr']['mrr']}", flush=True)

        # с инструкцией
        q_texts_i = [instruct_prefix + q[0] for q in QUERIES]
        q_embs_i = [np.array(x, dtype=np.float64) for x in cli.embed(q_texts_i)]
        out["retrieval_instr"] = retrieval(q_embs_i, q_texts_i, q_golds)
        print(f"[EM-044] instr:    recall@1={out['retrieval_instr']['recall@1']} "
              f"recall@3={out['retrieval_instr']['recall@3']} "
              f"mrr={out['retrieval_instr']['mrr']}", flush=True)

    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--log-dir", default=r"D:\AIprojects\embedding-deploy\logs")
    ap.add_argument("--no-server", action="store_true",
                    help="use already running server on --port")
    ap.add_argument("--instruct-prefix", default=INSTR_PREFIX_DEFAULT)
    ap.add_argument("--no-instruct", action="store_true",
                    help="skip instruction-prefixed retrieval")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    log_path = os.path.join(args.log_dir, f"server_{args.label}.log")
    result = run_bench(
        args.gguf, args.label, args.port, args.no_server, log_path,
        args.instruct_prefix, args.no_instruct,
    )
    json_path = args.json_out or os.path.join(
        args.log_dir, f"embed_bench_{args.label}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OUT] {json_path}", flush=True)
    print(f"[LOG] {log_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
