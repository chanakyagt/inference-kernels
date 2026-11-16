"""Sum ncu_results/*.csv into per-frame DRAM bytes and kernel time; writes ncu_summary.json and prints a table."""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ncu_results"
UNIT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "nsecond": 1, "usecond": 1e3, "msecond": 1e6}


def parse(path):
    lines = path.read_text(errors="replace").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return None
    k = defaultdict(dict)
    for r in csv.DictReader(lines[start:]):
        v = float(r["Metric Value"].replace(",", "")) * UNIT.get(r["Metric Unit"], 1)
        k[r["ID"]][r["Metric Name"]] = v
        k[r["ID"]]["name"] = r["Kernel Name"]
    rd = sum(x.get("dram__bytes_read.sum", 0) for x in k.values())
    wr = sum(x.get("dram__bytes_write.sum", 0) for x in k.values())
    t = sum(x.get("gpu__time_duration.sum", 0) for x in k.values())
    top = sorted(k.values(), key=lambda x: -(x.get("dram__bytes_read.sum", 0) + x.get("dram__bytes_write.sum", 0)))[:5]
    return {"kernels": len(k), "dram_read_MB": rd / 1e6, "dram_write_MB": wr / 1e6, "dram_total_MB": (rd + wr) / 1e6,
            "kernel_time_sum_ms": t / 1e6,
            "top5_by_bytes": [{"name": x["name"][:80], "MB": (x.get("dram__bytes_read.sum", 0) + x.get("dram__bytes_write.sum", 0)) / 1e6} for x in top]}


res = {}
for p in sorted(OUT.glob("*.csv")):
    res[p.stem] = parse(p)
(HERE / "ncu_summary.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    if v:
        print(f"{k:45s} kernels {v['kernels']:4d}  read {v['dram_read_MB']:8.2f} MB  write {v['dram_write_MB']:7.2f} MB  total {v['dram_total_MB']:8.2f} MB  ktime {v['kernel_time_sum_ms']:.3f} ms")
    else:
        print(k, "NO DATA")
