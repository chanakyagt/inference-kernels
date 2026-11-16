"""Analytical model: EfficientNet-B0 (torchvision config) at 256x256, GPU traffic vs fused-MBConv accelerator.
All shapes derived from torchvision efficientnet_b0 (width 1.0, depth 1.0). No torch needed."""
import math, json, sys

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 256
# torchvision MBConvConfig(expand, k, stride, in, out, layers)
CFG = [(1,3,1,32,16,1),(6,3,2,16,24,2),(6,5,2,24,40,2),(6,3,2,40,80,3),
       (6,5,1,80,112,3),(6,5,2,112,192,4),(6,3,1,192,320,1)]
NCLS = 1   # binary linear head (assumed 1 logit; 2 changes nothing visible)

def out_hw(h, s): return math.ceil(h / s)

L = []   # layer list: dict(name, kind, block, H,W (out), P (out pixels), cin, cout, k, macs, wparams, in_elems, out_elems)
def add(**d): L.append(d)

h = out_hw(RES, 2)
add(name="stem", kind="stem", block="stem", P=h*h, cin=3, cout=32, k=3, macs=h*h*32*3*9,
    w=32*3*9, b=32, in_e=RES*RES*3, out_e=h*h*32, H=h)
cin = 32; bi = 0
blocks = []
for si, (e, k, s, ci, co, n) in enumerate(CFG, start=1):
    for li in range(n):
        stride = s if li == 0 else 1
        bin_ = cin
        cexp = bin_ * e
        Hin = h; Hout = out_hw(h, stride)
        sq = max(1, bin_ // 4)
        tag = f"s{si}b{li+1}"
        blk = dict(tag=tag, stage=si, cin=bin_, cexp=cexp, cout=co, k=k, stride=stride, Hin=Hin, Hout=Hout,
                   res=(stride == 1 and bin_ == co), sq=sq)
        blocks.append(blk)
        if e != 1:
            add(name=tag+".expand", kind="expand 1x1", block=tag, stage=si, P=Hin*Hin, cin=bin_, cout=cexp, k=1,
                macs=Hin*Hin*bin_*cexp, w=bin_*cexp, b=cexp, in_e=Hin*Hin*bin_, out_e=Hin*Hin*cexp, H=Hin)
        add(name=tag+".dw", kind="depthwise", block=tag, stage=si, P=Hout*Hout, cin=cexp, cout=cexp, k=k,
            macs=Hout*Hout*cexp*k*k, w=cexp*k*k, b=cexp, in_e=Hin*Hin*cexp, out_e=Hout*Hout*cexp, H=Hout)
        add(name=tag+".se", kind="SE", block=tag, stage=si, P=1, cin=cexp, cout=cexp, k=1,
            macs=2*cexp*sq + Hout*Hout*cexp,  # 2 FCs + the gate multiply (pool adds not counted)
            w=2*cexp*sq, b=sq+cexp, in_e=Hout*Hout*cexp, out_e=Hout*Hout*cexp, H=Hout)
        add(name=tag+".project", kind="project 1x1", block=tag, stage=si, P=Hout*Hout, cin=cexp, cout=co, k=1,
            macs=Hout*Hout*cexp*co, w=cexp*co, b=co, in_e=Hout*Hout*cexp, out_e=Hout*Hout*co, H=Hout,
            res=blk["res"])
        cin = co; h = Hout
add(name="head", kind="head conv", block="head", P=h*h, cin=320, cout=1280, k=1, macs=h*h*320*1280,
    w=320*1280, b=1280, in_e=h*h*320, out_e=h*h*1280, H=h)
add(name="fc", kind="pool+fc", block="head", P=1, cin=1280, cout=NCLS, k=1, macs=1280*NCLS, w=1280*NCLS, b=NCLS,
    in_e=h*h*1280, out_e=NCLS, H=1)

tot_mac = sum(l["macs"] for l in L)
tot_w = sum(l["w"] for l in L); tot_b = sum(l["b"] for l in L)
if RES != 256:
    print(f"RES {RES}: MACs {tot_mac/1e6:.1f} M, params {tot_w+tot_b:,}")
    sys.exit()

# ---------------- 1. per-layer traffic, GPU-like (TensorRT kernel graph as observed in layer_profile.json) --------
# Kernels observed: conv(+SiLU fused) for stem/expand/project(+add); depthwise conv then a SEPARATE SiLU kernel
# (__myl_CastMulTanhMul... in the depthwise group), stem also has a separate SiLU kernel; SE = pool kernel + gate
# kernel (read x, write x*g). Model each kernel as reading its inputs from DRAM and writing outputs to DRAM.
def gpu_bytes(l, ab):  # ab = activation bytes/elem, weights same width
    kd = l["kind"]; wb = (l["w"] + l["b"]) * ab
    if kd == "stem":
        return l["in_e"]*ab + l["out_e"]*ab*3 + wb      # conv write, SiLU read+write
    if kd == "depthwise":
        return l["in_e"]*ab + l["out_e"]*ab*3 + wb      # conv, separate SiLU pass
    if kd == "SE":
        return l["in_e"]*ab + l["in_e"]*ab + l["out_e"]*ab + wb   # pool read, gate read+write
    if kd == "project 1x1":
        return l["in_e"]*ab + l["out_e"]*ab*(2 if l.get("res") else 1) + wb  # + residual read
    return l["in_e"]*ab + l["out_e"]*ab + wb
def min_bytes(l, ab):  # one round trip: read input once, write output once, weights once
    return (l["in_e"] + l["out_e"] + l["w"] + l["b"]) * ab

print(f"Total MACs/image {tot_mac/1e6:.1f} M ; weights {tot_w:,} params (+{tot_b:,} bias)")
print("\nPer-layer (top by GPU-like fp16 bytes/image):")
rows = []
for l in L:
    rows.append((l["name"], l["kind"], l["macs"], (l["w"]+l["b"]), l["in_e"], l["out_e"], gpu_bytes(l, 2), gpu_bytes(l, 1),
                 l["macs"]/max(1, min_bytes(l, 2))))
rows.sort(key=lambda r: -r[6])
print(f"{'layer':16s}{'kind':13s}{'MMAC':>7s}{'wKB8':>7s}{'inKB8':>8s}{'outKB8':>8s}{'GPUfp16KB':>10s}{'MAC/B fp16':>11s}")
for r in rows[:14]:
    print(f"{r[0]:16s}{r[1]:13s}{r[2]/1e6:7.2f}{r[3]/1e3:7.1f}{r[4]/1e3:8.1f}{r[5]/1e3:8.1f}{r[6]/1e3:10.1f}{r[8]:11.1f}")
g16 = sum(gpu_bytes(l, 2) for l in L); g8 = sum(gpu_bytes(l, 1) for l in L)
m16 = sum(min_bytes(l, 2) for l in L)
wbytes8 = tot_w + tot_b
print(f"\nGPU-like DRAM bytes/image, fp16: {g16/1e6:.2f} MB (of which weights {2*wbytes8/1e6:.2f} MB); int8: {g8/1e6:.2f} MB")
print(f"one-round-trip-per-layer lower bound fp16: {m16/1e6:.2f} MB")
act16 = g16 - 2*wbytes8
print(f"activation part fp16 {act16/1e6:.2f} MB/image -> at b64 {64*act16/1e6:.0f} MB per batch")

# by kind
kinds = {}
for l in L:
    k = kinds.setdefault(l["kind"], dict(macs=0, gpu16=0, gpu16_act=0, w=0, min_act=0))
    k["min_act"] += min_bytes(l, 2) - (l["w"]+l["b"])*2
    k["macs"] += l["macs"]; k["gpu16"] += gpu_bytes(l, 2); k["w"] += (l["w"]+l["b"])*2
    k["gpu16_act"] += gpu_bytes(l, 2) - (l["w"]+l["b"])*2

# ---------------- 2. GPU achieved bandwidth from the measured profile (batch 64 fp16, TensorRT) -------------------
prof = json.load(open(r"C:\Users\chana\Projects\inference-kernels\results\trt\layer_profile.json"))
bk = prof["by_kind"]
B = 64; READ_BW = 272.4e9; PEAK = 45.1e12
print("\nGPU (TensorRT fp16, b64) per kind: modelled DRAM bytes / measured time  [weights once per batch]")
print(f"{'kind':14s}{'TRT ms':>8s}{'GB/s':>8s}{'%272':>6s}{'TFLOP/s':>9s}{'%peak':>7s}")
tot_t = 0
for kd, v in kinds.items():
    t = bk.get(kd, {}).get("trt_ms")
    if not t: continue
    by = B*v["gpu16_act"] + v["w"]
    fl = 2*B*v["macs"]
    tot_t += t
    byl = B*v["min_act"] + v["w"]
    print(f"{kd:14s}{t:8.2f}{byl/(t*1e-3)/1e9:6.0f}-{by/(t*1e-3)/1e9:<4.0f}{100*by/(t*1e-3)/READ_BW:6.0f}{fl/(t*1e-3)/1e12:9.2f}{100*fl/(t*1e-3)/PEAK:7.1f}")
print(f"whole model b64: {2*B*tot_mac/15.06e-3/1e12:.2f} TFLOP/s = {100*2*B*tot_mac/15.06e-3/PEAK:.1f}% of 45.1; "
      f"b1: {2*tot_mac/0.413e-3/1e12:.2f} TFLOP/s = {100*2*tot_mac/0.413e-3/PEAK:.1f}%")
print(f"b64 modelled bytes/batch {(B*act16+2*wbytes8)/1e6:.0f} MB in 15.06 ms -> {(B*act16+2*wbytes8)/15.06e-3/1e9:.0f} GB/s avg")
print(f"b1: modelled bytes {g16/1e6:.1f} MB (fits the 4060 Ti's 32 MB L2) ; 0.413 ms / {len(prof['trt_layers'])} TRT layers "
      f"= {413/len(prof['trt_layers']):.1f} us per kernel; DRAM-time floor if all from DRAM {g16/272.4e9*1e6:.0f} us")

# ---------------- 3. fused MBConv accelerator ----------------
# On-chip buffers for the SE-barrier schedule:
#   D buffer: full depthwise output of the current block (SE pool needs all of it before the gate is known)
#   IO ping-pong: block input (needed for residual + streamed into expand) and block output
#   line buffer: k rows of the expanded tensor (input width) feeding the depthwise unit
def sram_need(ab):
    D = max(b["Hout"]**2*b["cexp"] for b in blocks)*ab
    # stem fused into s1b1 (expand ratio 1, no residual): stem rows stream straight into the dw line buffer
    io = 0
    for b in blocks:
        inb = 0 if b["tag"] == "s1b1" else b["Hin"]**2*b["cin"]
        io = max(io, (inb + b["Hout"]**2*b["cout"])*ab)
    lb = max(b["k"]*b["Hin"]*b["cexp"] for b in blocks)*ab
    lbt = [(b["tag"], b["k"]*b["Hin"]*b["cexp"]*ab) for b in blocks]
    return D, io, lb
for ab, nm in ((1, "int8"), (2, "int16 act (W8A16)")):
    D, io, lb = sram_need(ab)
    print(f"\nSRAM ({nm}): D buffer {D/1e3:.0f} KB, IO ping-pong {io/1e3:.0f} KB, line buffer {lb/1e3:.0f} KB, "
          f"acc RF 16 KB -> activation SRAM {(D+io+lb+16384)/1e6:.2f} MB; weights int8 {wbytes8/1e6:.2f} MB")
Dblk = max(blocks, key=lambda b: b["Hout"]**2*b["cexp"]); print("  largest D block", Dblk["tag"], Dblk["Hout"], Dblk["cexp"])
# recompute variant: pass 1 = expand+dw for pool only, pass 2 = recompute -> no D buffer
rec_mac = sum(l["macs"] for l in L if l["kind"] in ("expand 1x1", "depthwise"))
print(f"recompute variant (no D buffer): +{rec_mac/1e6:.0f} M MACs (+{100*rec_mac/tot_mac:.0f}%)")

def accel(R=32, C=64, f=1.0e9, dw_lanes=32, Tpix=64, wbw_Bpc=128, taps=9):
    """Weight-stationary-per-tile GEMM array R(in) x C(out); dw unit: dw_lanes channels, 1 output pixel/cycle/lane
    for k=3 (9 mults/lane), k=5 takes ceil(25/9)=3 cycles. Returns cycles and op/traffic counts per image."""
    cyc = 0; sram = dict(act=0, w=0, psum=0, lb=0, D=0); detail = []
    util_num = 0; util_den = 0
    def gemm(P, cin, cout):
        tiles = math.ceil(cin/R)*math.ceil(cout/C)
        load = R*C/wbw_Bpc                    # cycles to load one weight tile (double buffered -> overlap)
        ptiles = math.ceil(P/Tpix)
        c = tiles*ptiles*max(Tpix if P >= Tpix else P, load)
        return c, tiles
    for b in blocks:
        tag = b["tag"]
        Pin, Pout = b["Hin"]**2, b["Hout"]**2
        if tag == "s1b1":
            cA_g, _ = gemm(Pin, 27, 32)        # stem as im2col GEMM, K=27
        else:
            cA_g, _ = gemm(Pin, b["cin"], b["cexp"]) if b["cexp"] != b["cin"] else (0, 0)
        cA_d = math.ceil(b["cexp"]/dw_lanes)*(b["Hin"]+b["k"]-1)**2*math.ceil(b["k"]**2/taps)
        # dw engine consumes one padded input pixel per cycle per lane (the mamba-scan-accelerator formula (H+2p)^2 per channel group,
        # stride 2 is input bound) -> cycles = groups * (H+2p)^2 * ceil(k^2/9)
        cA = max(cA_g, cA_d)                    # expand and dw pipelined row by row
        cSE = math.ceil(b["cexp"]/C)*math.ceil(b["sq"]/R)*2*16 + 64   # two tiny FCs + sigmoid, ~ load bound
        cC, _ = gemm(Pout, b["cexp"], b["cout"])
        cyc += cA + cSE + cC
        detail.append((tag, cA_g, cA_d, cSE, cC))
        accel.g = getattr(accel, "g", 0)
    hc, _ = gemm(64, 320, 1280); cyc += hc + 1280/C*2
    detail.append(("head", hc, 0, 0, 0))
    return cyc, detail

def sram_traffic(ab=1, C=64, Tpix=64):
    """SRAM bytes per image: GEMM input re-read once per output tile of C channels, output written once,
    weights read once per pixel tile, psums kept in the array accumulators (Tpix x C int32 RF);
    expanded rows: written to line buffer once, read k times; dw output written to D once, read once."""
    act = 0; w = 0; lb = 0; D = 0
    for l in L:
        kd = l["kind"]
        if kd in ("expand 1x1", "project 1x1", "head conv", "stem"):
            src = l["in_e"] if kd != "stem" else l["P"]*27
            act += src*ab*math.ceil(l["cout"]/C) + (0 if kd == "expand 1x1" else l["out_e"]*ab)
            if l.get("res"): act += l["out_e"]*ab       # residual read
            w += (l["w"]+l["b"])*max(1, math.ceil(l["P"]/Tpix))
        elif kd == "depthwise":
            lb += l["in_e"]*ab*(1 + l["k"])            # write once, read k rows
            D += l["out_e"]*ab*2                         # write D, read D in phase C
            w += l["w"]+l["b"]
        elif kd == "SE":
            w += l["w"]+l["b"]
        else:
            act += l["in_e"]*0                          # pooled on the fly
            w += l["w"]+l["b"]
    return dict(act=act, w=w, lb=lb, D=D)

print("\n--- accelerator sizing scenarios (batch 1) ---")
GPU_LAT_B1 = 0.413e-3; GPU_E_B1 = 0.0326; GPU_E_B64 = 0.0281; GPU_TP = 4233
# Horowitz ISSCC 2014, 45 nm 0.9 V (read from the paper's figure): 8b mult 0.2 pJ, 32b add 0.1 pJ,
# 8 KB SRAM 64-bit read 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ; DRAM 64-bit 1.3-2.6 nJ.
E45 = dict(mac8=0.2+0.1, mac16=0.45+0.1, sram_small=10/8, sram_bank=20/8, sram_big=100/8)
res = {}
for (R, C, f, lanes, taps, label) in [(32, 64, 1.0e9, 32, 9, "A0: 2048 MAC, dw 32x9, 1 GHz"),
                                (32, 64, 1.0e9, 64, 25, "A: 2048 MAC, dw 64x25, 1 GHz"), (64, 64, 1.0e9, 64, 25, "B: 4096 MAC, dw 64x25, 1 GHz"),
                                (32, 64, 0.5e9, 64, 25, "A @ 500 MHz"), (16, 16, 0.0908e9, 16, 9, "SKY130: 256 MAC, 90.8 MHz")]:
    cyc, det = accel(R, C, f, lanes, taps=taps)
    gsum = sum(d[1] + d[4] + d[3] for d in det); dsum = sum(d[2] for d in det)
    print(f"   pipelined-across-images throughput (GEMM busy {gsum/1e3:.0f}k, dw busy {dsum/1e3:.0f}k cycles): {f/max(gsum, dsum):,.0f} img/s")
    lat = cyc/f
    ideal = tot_mac/(R*C)
    res[label] = lat
    print(f"{label:28s} cycles {cyc/1e3:8.1f} k  latency {lat*1e3:7.3f} ms  GEMM-array util {100*ideal/cyc:4.0f}%  "
          f"latency X vs GPU b1 {GPU_LAT_B1/lat:5.2f}  img/s {1/lat:,.0f}")
cycA, detA = accel(32, 64, 1e9, 64, taps=25)
print("  biggest phases (A: expand-gemm, dw, SE, project):")
for d in sorted(detA, key=lambda d: -(max(d[1], d[2])+d[3]+d[4]))[:6]:
    print("   ", d[0], [int(x) for x in d[1:]])

# ---------------- energy per image ----------------
def energy(ab, e_mac, e_sram_act, e_sram_w, e_lb, e_dram_B, overhead, w_on_chip, leak_W=0.0, lat=0.0, scale_logic=1.0, scale_sram=1.0):
    t = sram_traffic(ab)
    E_mac = tot_mac*e_mac*scale_logic
    E_sram = (t["act"]+t["D"])*e_sram_act*scale_sram + t["w"]*e_sram_w*scale_sram + t["lb"]*e_lb*scale_sram
    dram = RES*RES*3 + (0 if w_on_chip else wbytes8)
    E_dram = dram*e_dram_B
    core = (E_mac + E_sram)*overhead
    return dict(mac=E_mac, sram=E_sram, dram=E_dram, overhead=(E_mac+E_sram)*(overhead-1), leak=leak_W*lat,
                total=core + E_dram + leak_W*lat, traffic=t, dram_bytes=dram)
t8 = sram_traffic(1)
print(f"\nSRAM traffic/image int8: act {t8['act']/1e6:.2f} MB, D {t8['D']/1e6:.2f} MB, line buf {t8['lb']/1e6:.2f} MB, weights {t8['w']/1e6:.2f} MB")
latA = res["A: 2048 MAC, dw 64x25, 1 GHz"]
scen = {
 # name: (ab, mac, sram_act, sram_w, lb, dram pJ/B, overhead, w_on_chip, leak W, logic scale, sram scale)
 "45nm best (W on chip, banked 32KB SRAM, 1.5x ovh)":      (1, E45["mac8"], E45["sram_bank"], E45["sram_bank"], E45["sram_small"], 80, 1.5, True, 0.0, 1, 1),
 "45nm central (W on chip, 2x ovh, 0.1 W leak)":           (1, E45["mac8"], E45["sram_bank"], E45["sram_big"], E45["sram_bank"], 80, 2.0, True, 0.1, 1, 1),
 "45nm worst (W from DRAM 20pJ/b, 1MB-SRAM cost, 3x)":     (1, E45["mac8"], E45["sram_big"], E45["sram_big"], E45["sram_bank"], 160, 3.0, False, 0.3, 1, 1),
 "45nm W8A16 central":                                    (2, E45["mac16"], E45["sram_bank"], E45["sram_big"], E45["sram_bank"], 80, 2.0, True, 0.1, 1, 1),
 "7nm-class assumed (logic x0.25, SRAM x0.5), central":  (1, E45["mac8"], E45["sram_bank"], E45["sram_big"], E45["sram_bank"], 80, 2.0, True, 0.1, 0.25, 0.5),
}
print(f"\nEnergy per image (mJ) vs GPU measured b1 {GPU_E_B1*1e3:.1f} mJ, b64 {GPU_E_B64*1e3:.1f} mJ:")
out = {}
for nm, p in scen.items():
    ab, em, ea, ew, el, ed, ov, won, lk, sl, ss = p
    E = energy(ab, em*1e-12, ea*1e-12, ew*1e-12, el*1e-12, ed*1e-12, ov, won, lk, latA, sl, ss)
    out[nm] = E["total"]
    print(f"  {nm:52s} total {E['total']*1e3:6.3f}  [MAC {E['mac']*1e3:.3f} SRAM {E['sram']*1e3:.3f} ovh {E['overhead']*1e3:.3f} "
          f"DRAM {E['dram']*1e3:.3f} ({E['dram_bytes']/1e6:.2f} MB) leak {E['leak']*1e3:.3f}]  X_b1 {GPU_E_B1/E['total']:5.0f}  X_b64 {GPU_E_B64/E['total']:5.0f}")

# ---------------- sensitivity: what makes X_energy fall below e.g. 10 ? ----------------
print("\nSensitivity (central 45nm, one knob at a time):")
base = list(scen["45nm central (W on chip, 2x ovh, 0.1 W leak)"])
def run(p, lat=latA):
    ab, em, ea, ew, el, ed, ov, won, lk, sl, ss = p
    return energy(ab, em*1e-12, ea*1e-12, ew*1e-12, el*1e-12, ed*1e-12, ov, won, lk, lat, sl, ss)["total"]
knobs = [("overhead x5 (control/clock/ leakage-heavy design)", 6, 5.0), ("weights from DRAM @ 40 pJ/b (DDR3-era I/O)", None, None),
         ("all SRAM at 1 MB-macro cost (12.5 pJ/B)", "sram", None), ("activation re-read x4 (bad tiling)", "reread", None),
         ("leakage 1 W", 8, 1.0), ("W8A16 instead of W8A8", "a16", None)]
for nm, idx, val in knobs:
    p = list(base)
    if idx == "sram": p[2] = p[3] = p[4] = E45["sram_big"]
    elif idx == "a16": p[0] = 2; p[1] = E45["mac16"]
    elif idx == "reread": p[2] = p[2]*4; p[4] = p[4]*4
    elif idx is None: p[7] = False; p[5] = 320
    else: p[idx] = val
    e = run(p); print(f"  {nm:52s} {e*1e3:6.3f} mJ  X_b1 {GPU_E_B1/e:5.0f}")
# break-even: overhead factor needed for X=1 and X=10
e_core = run(base[:6] + [1.0] + base[7:8] + [0.0] + base[9:])
print(f"  core (no overhead, no leak) {e_core*1e3:.3f} mJ -> overhead factor for X=10: {GPU_E_B1/10/e_core:.0f}x, for X=1: {GPU_E_B1/e_core:.0f}x")

# SKY130 anchor
print("\nSKY130 anchor (mamba-scan-accelerator synth, gemm_core_WxW_w8a16, no P&R): 16x16 1.907 mm2 = 7,449 um2/MAC, 90.6 MHz")
for n in (2048, 4096):
    print(f"  {n} MACs at sky130 density: {n*7449e-6:.1f} mm2 logic ; at {90.6} MHz: {tot_mac/(n*90.6e6)*1e3:.2f} ms ideal")
json.dump(dict(tot_mac=tot_mac, wbytes8=wbytes8, g16=g16, g8=g8, lat=res, energy=out), open("results.json", "w"), indent=1)
