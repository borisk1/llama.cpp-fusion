#!/bin/bash
# bench decode t/s: 1 warmup + 3 mjerenja x 128 tokena, ispis median
for i in 0 1 2 3; do
  T=$(curl -s http://127.0.0.1:18080/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"The detailed history of European architecture spans many centuries and","n_predict":128,"temperature":0}' \
    | python3 -c "import json,sys; print(round(json.load(sys.stdin)['timings']['predicted_per_second'],2))")
  [ $i -gt 0 ] && echo "$T"
done | sort -n | sed -n 2p
