import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
main_p = ROOT / "results" / "official_compare.json"
fix_p = ROOT / "results" / "official_compare_fixed.json"
md_p = ROOT / "results" / "official_compare.md"


def f(v, p=4):
    return "-" if v is None else f"{v:.{p}f}"


def ratio(a, b):
    return None if a is None or b is None else a / b


def tables(res):
    old = {(r["batch"], r["d_inner"], r["T"], r["dtype"]): r for r in res["scan"]}
    L = []
    for dt in ("float32", "bfloat16"):
        rows = [r for r in res["scan_fixed"] if r["dtype"] == dt]
        L.append(f"### {dt}: one fixed CUDA setting (16 channels per block, split 2) for every shape, idle check before every row")
        L.append("")
        L.append("Warm = CUDA graph of 5 calls divided by 5 (inputs stay in L2). Cold L2 = a 64 MB buffer is written before every call and the call is timed with CUDA events behind that write (so launch latency is hidden); the warm-event column is the same event method without the flush. Speedup = official / ours.")
        L.append("")
        L.append("| B | E | T | ours fixed (ms) | ours tuned (ms, earlier run) | official (ms) | vLLM (ms) | speedup fixed | speedup tuned | ours cold | official cold | speedup cold | idle met |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            k = (r["batch"], r["d_inner"], r["T"], r["dtype"])
            o = r["graph_ms"].get("official_scan")
            c = r["graph_ms"].get("ours_cuda_fixed")
            v = r["graph_ms"].get("vllm_scan")
            ot = old.get(k)
            tuned = ot["graph_ms"].get("ours_cuda_tuned") if ot and dt == "float32" else None
            otuned = ot["graph_ms"].get("official_scan") if ot and dt == "float32" else None
            cc = r.get("cold_l2_ms", {}).get("ours_cuda_fixed")
            oc = r.get("cold_l2_ms", {}).get("official_scan")
            L.append(f"| {r['batch']} | {r['d_inner']} | {r['T']} | {f(c)} | {f(tuned)} | {f(o)} | {f(v)} | {f(ratio(o, c), 2)} | {f(ratio(otuned, tuned), 2)} | {f(cc)} | {f(oc)} | {f(ratio(oc, cc), 2)} | {r.get('idle_gate_met')} |")
        L.append("")
    return L


def main():
    res = json.loads(main_p.read_text())
    fx = json.loads(fix_p.read_text())
    res["scan_fixed"] = fx["scan"]
    res["meta"]["scan_fixed"] = fx["meta"]
    res["errors"] = [e for e in res["errors"] if not e.startswith("fixed-config rerun")] + ["fixed-config rerun: " + e for e in fx["errors"]]
    main_p.write_text(json.dumps(res, indent=1))
    md = md_p.read_text()
    cut = md.find("## Fixed-config rerun")
    if cut >= 0:
        md = md[:cut]
    extra = ["## Fixed-config rerun (one CUDA setting for every shape, idle check before every row, cold-L2 variant)", ""] + tables(res)
    md_p.write_text(md.rstrip("\n") + "\n\n" + "\n".join(extra) + "\n")
    out = ROOT / "compare_official" / "fixed_tables.md"
    out.write_text("\n".join(tables(res)) + "\n")
    print("merged", len(res["scan_fixed"]), "rows")


if __name__ == "__main__":
    main()
