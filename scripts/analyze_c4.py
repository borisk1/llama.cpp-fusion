#!/usr/bin/env python3
"""C4: is deep-layer routing a function of TOKEN IDENTITY?
Trace line: pos token_id L0e0,..,L0e7 L1e0,... (40 layers x 8 experts)
For token ids with >=2 occurrences: mean pairwise top-8 overlap per layer
(within-id), vs cross-id baseline (random pairs of different ids).
If within-id >> cross-id in deep layers => offline token->expert table +
MTP-predicted id would BE a usable prefetch signal (bounded by acceptance).
If within-id ~ cross-id => routing depends on hidden state/context => closed.
"""
import sys, random
from collections import defaultdict

f = sys.argv[1] if len(sys.argv) > 1 else '/tmp/c4_trace.txt'
occ = defaultdict(list)   # token_id -> list of [40][8] routings (layer may be None = rejected read)
n_lines = 0
for line in open(f):
    parts = line.split()
    if len(parts) != 2 + 40: continue
    tok = int(parts[1])
    tops = []
    n_valid = 0
    for s in parts[2:]:
        t8 = [int(x) for x in s.split(',')]
        if len(t8) != 8 or t8[0] < 0 or any(x < 0 or x > 4095 for x in t8):
            tops.append(None)          # rejected/corrupt read for this layer
        else:
            tops.append(t8); n_valid += 1
    if n_valid < 10: continue          # need a usable share of layers
    occ[tok].append(tops)
    n_lines += 1

def overlap8(a, b):
    return sum(1 for x in a if x in b) / 8.0

rng = random.Random(42)
L = 40
win_hits = [0.0]*L; win_ns = [0]*L
for tok, lst in occ.items():
    if len(lst) < 2: continue
    # up to 30 random pairs per token id (avoid quadratic blowup on frequent ids)
    idx_pairs = [(i, j) for i in range(len(lst)) for j in range(i+1, len(lst))]
    if len(idx_pairs) > 30: idx_pairs = rng.sample(idx_pairs, 30)
    for i, j in idx_pairs:
        for l in range(L):
            if lst[i][l] is None or lst[j][l] is None: continue
            win_hits[l] += overlap8(lst[i][l], lst[j][l]); win_ns[l] += 1

# cross-id baseline: random pairs of occurrences with different ids
all_occ = [(tok, t) for tok, lst in occ.items() for t in lst]
cross_hits = [0.0]*L; cross_ns = [0]*L
for _ in range(min(20000, len(all_occ)*2)):
    (t1, a), (t2, b) = rng.sample(all_occ, 2)
    if t1 == t2: continue
    for l in range(L):
        if a[l] is None or b[l] is None: continue
        cross_hits[l] += overlap8(a[l], b[l]); cross_ns[l] += 1

multi = sum(1 for lst in occ.values() if len(lst) >= 2)
print('trace: %d tokens, %d unique ids, %d ids with >=2 occ, win pairs L20=%d, cross pairs L20=%d'
      % (n_lines, len(occ), multi, win_ns[20], cross_ns[20]))
print('%-6s %-12s %-12s' % ('layer', 'within-id', 'cross-id'))
for l in [0,1,2,3,4,5,10,20,30,39]:
    print('L%-5d %-12.4f %-12.4f' % (l, win_hits[l]/max(win_ns[l],1), cross_hits[l]/max(cross_ns[l],1)))
deep_w = sum(win_hits[3:]) / max(sum(win_ns[3:]), 1)
deep_c = sum(cross_hits[3:]) / max(sum(cross_ns[3:]), 1)
print('L3-39 avg: within-id=%.4f cross-id=%.4f (chance=8/256=0.0312)' % (deep_w, deep_c))
all_w = sum(win_hits) / max(sum(win_ns), 1)
print('ALL-layer avg within-id=%.4f  <- this is the ceiling for MTP-id+table prefetch (x acceptance)' % all_w)
