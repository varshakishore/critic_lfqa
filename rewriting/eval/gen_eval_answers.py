"""Behavioural eval, step 1: generate DR Tulu traces + answers from each checkpoint.

Runs the real DR Tulu tool-calling workflow (AutoReasonSearchWorkflow, same as
dr-tulu/agent/workflows/query_direct.py) on the DPO test-split questions, once per
model. Nothing in dr-tulu is modified: the workflow is imported and pointed at
eval/drtulu_eval.yaml, a copy of auto_search_sft_s2_only_hamish.yaml whose
search_agent_model_name is "policy". Each checkpoint is served by vLLM under
--served-model-name policy, so one YAML covers every model.

Per model: start vLLM -> generate (async, in-process) -> kill vLLM. The MCP search
backend is started once and shared. Output is resumable: rows already present
(by record_idx, sample_idx) are skipped, and failures are not written, so a re-run
retries them.

    S2_API_KEY=... python eval/gen_eval_answers.py                 # all default models
    python eval/gen_eval_answers.py --models local=dpo_out_local_b0.05_ep1 --n-samples 1
"""
import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DR_TULU_AGENT = "/weka/nora-default/varshak/dr-tulu/agent"

DEFAULT_TEST = os.path.join(
    ROOT, "samples_1000",
    "drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod_fixed_dpo_pairs_full_test.jsonl")
DEFAULT_MODELS = [
    "base=rl-research/DR-Tulu-8B",
    # The DPO data's original traces came from the SFT checkpoint (also DR Tulu's RL
    # init), so this is the on-policy base for DPO runs started from SFT.
    "base_sft=rl-research/DR-Tulu-SFT-8B",
    f"full_b0.05_ep2={ROOT}/dpo_out_full_b0.05_ep2",
    f"local_b0.05_ep1={ROOT}/dpo_out_local_b0.05_ep1",
]
SERVED_NAME = "policy"  # must match search_agent_model_name in drtulu_eval.yaml


def wait_for_port(port, proc=None, host="localhost", timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process for port {port} exited with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"port {port} not ready after {timeout}s")


def port_open(port, host="localhost"):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def launch(cmd, log_path):
    log = open(log_path, "a")
    return subprocess.Popen(cmd, shell=True, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)


def kill(proc):
    if proc is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            pass


def load_questions(path):
    """One row per test record, ordered by record_idx."""
    seen = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            seen.setdefault(r["record_idx"], r["question"])
    return sorted(seen.items())


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["record_idx"], r["sample_idx"]))
    return done


async def generate_all(jobs, out_path, model_name, model_path, args):
    sys.path.insert(0, DR_TULU_AGENT)
    from workflows.auto_search_sft import AutoReasonSearchWorkflow

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    n_ok = n_fail = 0

    async def one(record_idx, question, sample_idx):
        nonlocal n_ok, n_fail
        async with sem:
            t0 = time.time()
            try:
                workflow = AutoReasonSearchWorkflow(configuration=args.config)
                result = await asyncio.wait_for(
                    workflow(problem=question, dataset_name="sqav2", verbose=False),
                    timeout=args.timeout)
            except Exception as e:
                n_fail += 1
                print(f"  [fail] rec {record_idx} s{sample_idx}: {type(e).__name__}: {e}", flush=True)
                return
            ft = result.get("full_traces")
            trace = ft.model_dump() if ft is not None and hasattr(ft, "model_dump") else ft
            row = {
                "model": model_name,
                "model_path": model_path,
                "record_idx": record_idx,
                "sample_idx": sample_idx,
                "question": question,
                "final_response": result["final_response"],
                "full_traces": trace["generated_text"] if isinstance(trace, dict) else trace,
                "seconds": round(time.time() - t0, 1),
            }
            async with write_lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            n_ok += 1
            print(f"  [ok] rec {record_idx} s{sample_idx} ({row['seconds']}s)  "
                  f"{n_ok + n_fail}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    return n_ok, n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="name=path_or_hf_id pairs, run sequentially")
    ap.add_argument("--test-file", default=DEFAULT_TEST)
    ap.add_argument("--config", default=os.path.join(HERE, "drtulu_eval.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "generations"))
    ap.add_argument("--n-samples", type=int, default=3,
                    help="samples per question (temperature 0.7 from the YAML)")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per generation")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for vLLM")
    ap.add_argument("--vllm-port", type=int, default=30001,
                    help="must match search_agent_base_url in the config")
    ap.add_argument("--mcp-port", type=int, default=8000)
    args = ap.parse_args()

    if "S2_API_KEY" not in os.environ:
        sys.exit("S2_API_KEY must be set (the MCP search backend needs it)")
    if port_open(args.vllm_port):
        sys.exit(f"port {args.vllm_port} is already in use; stop that vLLM server first, "
                 "or generations would come from whatever model it serves")

    questions = load_questions(args.test_file)[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"{len(questions)} questions x {args.n_samples} samples, {len(args.models)} models")

    mcp = vllm = None
    try:
        if port_open(args.mcp_port):
            print(f"[mcp] reusing server on port {args.mcp_port}")
        else:
            print("[mcp] starting...")
            mcp = launch(f"cd {DR_TULU_AGENT} && python -m dr_agent.mcp_backend.main "
                         f"--port {args.mcp_port} --host 0.0.0.0",
                         os.path.join(log_dir, "mcp.log"))
            wait_for_port(args.mcp_port, mcp)

        for spec in args.models:
            name, path = spec.split("=", 1)
            out_path = os.path.join(args.out_dir, f"{name}.jsonl")
            done = load_done(out_path)
            jobs = [(ri, q, s) for ri, q in questions for s in range(args.n_samples)
                    if (ri, s) not in done]
            print(f"\n=== {name}  ({path})  {len(done)} done, {len(jobs)} to generate -> {out_path}")
            if not jobs:
                continue

            print("[vllm] starting...")
            vllm = launch(f"CUDA_VISIBLE_DEVICES={args.gpu} vllm serve {path} "
                          f"--served-model-name {SERVED_NAME} --dtype auto "
                          f"--port {args.vllm_port} --max-model-len 40960 --host 0.0.0.0",
                          os.path.join(log_dir, f"{name}_vllm.log"))
            wait_for_port(args.vllm_port, vllm)
            print("[vllm] ready")

            t0 = time.time()
            n_ok, n_fail = asyncio.run(generate_all(jobs, out_path, name, path, args))
            print(f"=== {name}: {n_ok} ok, {n_fail} failed in {(time.time() - t0) / 60:.1f} min")

            kill(vllm)
            vllm = None
            while port_open(args.vllm_port):
                time.sleep(2)
    finally:
        kill(vllm)
        kill(mcp)


if __name__ == "__main__":
    main()
