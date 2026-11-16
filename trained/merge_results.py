import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "results" / "trained_parts"
ORDER = ["R0s", "R0", "R1", "R2", "R3", "R4", "R4b", "R5", "R6", "R6b", "R6c", "R6d", "R7", "R8", "R9"]


def fmt(v, f="{:.2f}"):
    return f.format(v) if isinstance(v, (int, float)) else "-"


def main():
    tags = sorted({p.name.split("_R")[0] for p in PARTS.glob("*_R*_*.json")})
    out, md = {}, ["# Trained-detector optimization ladder", "",
                   "Test set: 10,905 face images (5,413 real / 5,492 fake), threshold 0.5. Latency = median of CUDA-event timings of the model forward on a normalized fp32 batch, including the channels_last conversion. J/img = mean GPU board power during a sustained 20 s run divided by img/s; raw and idle-subtracted (net). Peak MB = torch.cuda.max_memory_allocated at batch 64 including weights. An asterisk after img/s marks a sustained run that was not clean (throughput dipped inside the window or idle power read high because another process shared the GPU); latency is the best of 3 trials.", ""]
    for tag in tags:
        rows = []
        md += [f"## {tag}", "", "| Rung | Description | lat b1 / b8 / b64 (ms) | img/s b1 / b64 | peak MB b64 | J/img b64 raw / net | J/img b1 raw / net | Acc | AUC | flips vs R0s |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for key in ORDER + ["R0ref"]:
            a = PARTS / f"{tag}_{key}_acc.json"
            p = PARTS / f"{tag}_{key}_perf.json"
            A = json.loads(a.read_text()) if a.exists() else {}
            P = json.loads(p.read_text()) if p.exists() else {}
            if not A and not P:
                continue
            rec = {"rung": key, "acc": A, "perf": P}
            rows.append(rec)
            name = A.get("name") or P.get("name") or key
            if P.get("failed") or A.get("failed"):
                md.append(f"| {key} | {name} | FAILED: {(P.get('failed') or A.get('failed'))[:120]} | | | | | {fmt(A.get('accuracy'), '{:.4f}')} | {fmt(A.get('auc'), '{:.4f}')} | |")
                continue
            L, S, M = P.get("latency_ms", {}), P.get("sustained", {}), P.get("peak_mem_mb", {})
            g = lambda d, k: d.get(str(k), d.get(k, {}))
            lat = " / ".join(fmt(g(L, b)) if not isinstance(g(L, b), dict) else "-" for b in (1, 8, 64))
            flag = lambda d: "" if (d.get("stable") and d.get("idle_clean")) or not d else "*"
            ips = f"{fmt(g(S, 1).get('img_per_s'), '{:.0f}')}{flag(g(S, 1))} / {fmt(g(S, 64).get('img_per_s'), '{:.0f}')}{flag(g(S, 64))}" if S else "-"
            mem = fmt(g(M, 64) if not isinstance(g(M, 64), dict) else None, "{:.0f}")
            j64 = f"{fmt(g(S, 64).get('J_per_img_raw'), '{:.4f}')} / {fmt(g(S, 64).get('J_per_img_idle_subtracted'), '{:.4f}')}" if S else "-"
            j1 = f"{fmt(g(S, 1).get('J_per_img_raw'), '{:.4f}')} / {fmt(g(S, 1).get('J_per_img_idle_subtracted'), '{:.4f}')}" if S else "-"
            md.append(f"| {key} | {name} | {lat} | {ips} | {mem} | {j64} | {j1} | {fmt(A.get('accuracy'), '{:.4f}')} | {fmt(A.get('auc'), '{:.4f}')} | {A.get('pred_flips_vs_R0s', '-')} |")
        out[tag] = rows
        e = PARTS / f"{tag}_R10_e2e.json"
        if e.exists():
            E = json.loads(e.read_text())
            md += ["", f"### R10 end-to-end incl. JPEG decode ({tag}, rung {E['rung']}, {E['images']} images, {E['workers']} workers)", "",
                   "| Pipeline | end-to-end img/s | decode+prep only img/s | Acc | AUC |", "|---|---|---|---|---|",
                   f"| model only (GPU-resident input) | {E['model_only_img_per_s']:.0f} | - | - | - |"]
            for k, lab in (("cpu", "CPU PIL decode + resize, 4 workers"), ("gpu", "GPU nvjpeg decode + resize")):
                q = E["pipelines"][k]
                md.append(f"| {lab} | {q['end_to_end_median']:.0f} | {q['decode_prep_only_median']:.0f} | {q['accuracy']:.4f} | {q['auc']:.4f} |")
            rows.append({"rung": "R10", "e2e": E})
        md.append("")
    (ROOT / "results" / "trained_ladder.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results" / "trained_ladder.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
