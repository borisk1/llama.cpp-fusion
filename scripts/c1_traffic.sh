#!/bin/bash
# C1 traffic generator: diverse prompts, greedy, 512 tokens each
PROMPTS=(
 "Photosynthesis converts light energy into"
 "The stock market crash of 1929 triggered"
 "class BinarySearchTree:\n    def insert(self, val):"
 "Deep in the ocean trenches, scientists discovered"
 "Machine translation systems historically evolved from"
 "The human immune system responds to viral infections by"
 "CREATE TABLE inventory (id SERIAL PRIMARY KEY,"
 "Hrvatska književnost devetnaestog stoljeća obilježena je"
 "Distributed consensus algorithms like Raft ensure"
 "Die Geschichte der Berliner Mauer beginnt im Jahr"
 "Le système solaire contient huit planètes dont"
 "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):"
 "Thermodynamics second law states that entropy"
 "The Silk Road connected ancient China with"
 "Kubernetes orchestrates containerized applications by"
 "Beethoven composed his ninth symphony while"
)
TOTAL_DRAFT=0; TOTAL_ACC=0
for i in "${!PROMPTS[@]}"; do
  R=$(curl -s http://127.0.0.1:18080/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\": $(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "${PROMPTS[$i]}"), \"n_predict\": 512, \"temperature\": 0}")
  DN=$(echo "$R" | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); print(t.get('draft_n',0), t.get('draft_n_accepted',0), round(t.get('predicted_per_second',0),1))")
  echo "prompt $i: draft_n/acc/tps = $DN"
  TOTAL_DRAFT=$((TOTAL_DRAFT + $(echo $DN | cut -d' ' -f1)))
  TOTAL_ACC=$((TOTAL_ACC + $(echo $DN | cut -d' ' -f2)))
done
echo "TOTAL draft_n=$TOTAL_DRAFT accepted=$TOTAL_ACC"
python3 -c "print('acceptance rate: %.4f' % ($TOTAL_ACC/$TOTAL_DRAFT))"
