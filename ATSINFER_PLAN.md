Implementacioni plan — ATSInfer potpuna implementacija
======================================================

Korak 1: DP knapsack sa switching cost (Algoritam 1)
- Trenutni: greedy po benefit/size
- Novi: DP O(n*M) sa:
  * switching cost c_i = activation_size / PCIe_bandwidth
  * max Σ(r_i * GPU_i) - Σ(c_i * switch_i)
  * MoE podjela: experti posebno od non-expert

Korak 2: Temporary GPU buffer pool
- N GPU buffera za privremene tezine
- Van schedulerovog allocatora (ne resetira se)
- Buffer reuse: isti slot za razlicite tenzore u razlicitim koracima

Korak 3: Dynamic Transfer DP (Algoritam 2)
- O(n²) DP za odlucivanje koji CPU tenzori se privremeno prebacuju na GPU
- Uzima u obzir: exposed transfer time = max(w_i - overlap_window, 0)
- Pre-compute seg(j,i) = execution + activation transfer na default putu

Korak 4: BuildRuntime
- Konstruise execution plan: transfer stream + compute stream + sync
- Za svaki tenzor: gdje se izvrsava, sta se prenosi, sta se ceka

Korak 5: Load-aware Re-scheduling (Algoritam 3)
- Performance monitor: mjeri stvarno CPU/GPU/PCIe vrijeme
- Poredi sa ocekivanim (devijacija)
- Trigger: kad devijacija > 15% i proslo dovoljno vremena
