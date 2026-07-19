#!/usr/bin/env python3
"""Prefetch value simulation for hot-resident (RETEST follow-up, July 2026).

Answers, from clean certified trace (c4_trace2.txt) + clean usage profile:
  1. True skew (top-N coverage of routed slots) with certified reads
  2. Static hot-N residency: cold slot fraction per token (N = 64..224)
  3. Incremental catch of COLD slots by: lag-1, id-table (x acceptance), both
  4. Adaptive (decayed-counts) resident set vs static — Sleepybear's LRU claim
  5. Bandwidth reality check: bytes/token to dynamically prefetch cold candidates
     vs just computing them on CPU (hot-resident dual-chain).

Usage: python3 simulate_prefetch.py <trace> <hot_cache_clean.json>
"""
import sys, json
from collections import defaultdict, Counter

trace_f = sys.argv[1] if len(sys.argv) > 1 else '/tmp/c4_trace2.txt'
prof_f  = sys.argv[2] if len(sys.argv) > 2 else '/mnt/raid0/mtp-prefetch/results/hot_cache_clean.json'
L, E, ACT = 40, 256, 8
ACC = 0.66                      # measured MTP draft acceptance
EXPERT_MB = 2.5                 # Q4_K_M expert (gate+up+down 2048x768) ~MB
PCIE_GBS = 22.0                 # effective h2d GB/s (3090 gen4 x16 measured-ish)

rows = []
last_pos = -1; seg = 0
for line in open(trace_f):
    p = line.split()
    if len(p) != 2 + L: continue
    pos, tok = int(p[0]), int(p[1])
    if pos < last_pos: seg += 1
    last_pos = pos
    tops = []
    for s in p[2:]:
        t8 = [int(x) for x in s.split(',')]
        tops.append(t8 if (len(t8) == 8 and all(0 <= x < E for x in t8)) else None)
    rows.append((seg, tok, tops))
half = len(rows) // 2

prof = json.load(open(prof_f))
counts = {int(k): v for k, v in prof['counts'].items()}

# 1) skew from clean profile
tot_all = sum(sum(counts[l]) for l in range(L))
for N in (64, 128, 224):
    cov = sum(sum(sorted(counts[l], reverse=True)[:N]) for l in range(L))
    print('skew: top-%-3d coverage = %.1f%%' % (N, 100.0 * cov / tot_all))

# id-table from train half of trace
table = defaultdict(lambda: [Counter() for _ in range(L)])
for sgm, tok, tops in rows[:half]:
    for l in range(L):
        if tops[l]: table[tok][l].update(tops[l])
tbl = {t: [set(e for e, _ in c[l].most_common(8)) for l in range(L)] for t, c in table.items()}

hdr = '%-6s %-10s %-12s %-12s %-12s %-12s %-10s' % (
    'N', 'cold/tok', 'lag1_catch', 'tbl*acc', 'both*acc', 'adapt_cold', 'pfetch_MB')
print(); print(hdr)
for N in (64, 128, 192, 224):
    hot = [set(sorted(range(E), key=lambda e: -counts[l][e])[:N]) for l in range(L)]
    # adaptive: decayed counts updated online over eval half (tau ~ 500 tokens)
    ad_cnt = [[float(counts[l][e]) for e in range(E)] for l in range(L)]
    ad_hot = [set(sorted(range(E), key=lambda e: -ad_cnt[l][e])[:N]) for l in range(L)]
    decay = 0.998
    cold = lag1c = tblc = bothc = adcold = 0
    slots = 0; pf_bytes = 0.0; ntok = 0
    for k in range(half, len(rows) - 1):
        sgm, tok, tops = rows[k]
        sgm2, tok2, tops2 = rows[k + 1]
        if sgm2 != sgm: continue
        ntok += 1
        pf_set_sz = 0
        for l in range(L):
            if tops2[l] is None: continue
            lag = set(tops[l]) if tops[l] else set()
            tb = tbl.get(tok2)
            tbs = tb[l] if tb else set()
            # dynamic prefetch candidates = predicted MINUS resident
            pf_set_sz += len((lag | tbs) - hot[l])
            for e in tops2[l]:
                slots += 1
                if e not in hot[l]:
                    cold += 1
                    if e in lag: lag1c += 1
                    if e in tbs: tblc += 1
                    if e in lag or e in tbs: bothc += 1
                if e not in ad_hot[l]: adcold += 1
        pf_bytes += pf_set_sz * EXPERT_MB
        # online update of adaptive counts (cheap background op in real impl)
        if ntok % 100 == 0:
            for l in range(L):
                if tops2[l] is None: continue
                for e in range(E): ad_cnt[l][e] *= decay ** 100
            ad_hot = [set(sorted(range(E), key=lambda e: -ad_cnt[l][e])[:N]) for l in range(L)]
        for l in range(L):
            if tops2[l]:
                for e in tops2[l]: ad_cnt[l][e] += 1.0
    coldpt = cold / max(ntok, 1)                      # cold expert-slots per token (of 320)
    print('%-6d %-10.2f %-12s %-12s %-12s %-12.2f %-10.0f' % (
        N, coldpt,
        '%.1f%%' % (100.0 * lag1c / max(cold, 1)),
        '%.1f%%' % (100.0 * ACC * tblc / max(cold, 1)),
        '%.1f%%' % (100.0 * (lag1c + ACC * (bothc - lag1c)) / max(cold, 1)),
        adcold / max(ntok, 1),
        pf_bytes / max(ntok, 1)))

print("""
kolone: cold/tok = prosjek cold expert-slotova po tokenu (od 320 = 40L x 8);
lag1/tbl/both = udio COLD slotova koje bi prefetch signal pokrio;
adapt_cold = cold/tok uz adaptivni (decay) resident set istog N;
pfetch_MB = MB/token koje bi dinamicki prefetch morao prenijeti preko PCIe.
Reality check: PCIe %.0f GB/s => 1 ms prenese ~%.0f MB; CPU GEMV cold experta
cita ~%.1f MB iz RAM-a @ ~60 GB/s => ~0.05 ms/expert + dispatch.""" % (
    PCIE_GBS, PCIE_GBS, EXPERT_MB))
