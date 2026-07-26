## Inference-time algorithm: theoretical cost analysis

### Which algorithm, and why

I selected the **RMT-style inference procedure**: the transformer backbone (ESM-2) is left untouched, but the input sequence is split into fixed-size blocks,

and a **small set of learned "memory tokens" is carried across blocks to pass information forward**. I wanted to compare this inference method with the **full-sequence baseline** (plain self-attention over the whole protein at once) in the sense of cost analysis.

I do the theoretical mathematical cost analysis here for both methods, then will do the runtime cost analysis using Python libraries and the actual protein sequence data, which is driven from ProteinGym, select a portion of it, and make the actual analysis. 

This database is publicly available here: https://proteingym.org/ 

---

Why RMT over the alternatives from the literature reading list:

- **vs. full-sequence attention**: full attention is exact but scales quadratically in sequence length. That's the bottleneck I'm trying to fix.
- **vs. Block-Recurrent Transformer**: more expressive (LSTM-gated recurrent cell inside a full transformer layer), but it requires a custom recurrent cell trained from scratch. That's out of scope for a 3-week, 3-CFU project. No time or enough resources (data/GPU/etc) to train an architecture, only to retrofit one.
- **vs. Universal Transformer**: recurrence over depth (weight-tying across layers), not over sequence position. It addresses parameter count, not the sequence-length cost I want to target rn, so it's not a fair substitute here.
- **vs. sparse/linear attention (Longformer, Performer, sliding-window)**: these are legitimate alternatives worth a mention, but most require either a custom attention kernel or fine-tuning to adapt the pretrained weights to the new attention pattern. RMT's appeal is that it's a bolt-on: freeze or lightly adapt the pretrained ESM-2 weights and add memory-token machinery around it. :)))

#### **Pros of RMT here**:
retrofittable onto a pretrained model (fits the timeframe), memory footprint independent of total sequence length, straightforward to implement and monitor.

#### **Cons**:
it's an approximation. Long-range dependencies must be compressed through a fixed-size memory bottleneck, so some accuracy loss vs. exact full attention is expected. Segments must be processed in order (memory state at block *t* depends on block *t−1*), so it's not embarrassingly parallel across the sequence the way full attention is.

Plus, the splitting method could be inspired by biological aspects to make more sense, and potentially, reach better accuracy. Even if I test all the combinations, I will lose the interpretability of why some sorts of splittings work better than others.

---

### Setup / notation

- `n` = protein sequence length
- `d` = hidden dimension
- `L` = number of transformer layers
- `b` = block size
- `m` = number of memory tokens (`m ≪ b`)
- `s = ⌈n / b⌉` = number of segments

---

### Instruction-by-instruction breakdown

**Algorithm** `RMT_Inference(X, M, b, m)`

```
1.  Split X into segments X_1 ... X_s,  s = ⌈n / b⌉
2.  Mem_0 ← init memory state (size m × d)
3.  for t = 1 .. s:
3a.     Z_t ← concat(Mem_{t-1}, X_t)          # length b + m
3b.     for l = 1 .. L:
            Q, K, V ← Z_t W_Q, Z_t W_K, Z_t W_V
            A ← softmax(Q Kᵀ / √d)
            O ← A V
            Z_t ← FFN(OutProj(O) + Z_t)       # + residual
3c.     Mem_t, Y_t ← split(Z_t)               # recover updated memory + segment output
3d.     store Y_t; carry Mem_t forward
4.  Y ← concat(Y_1 ... Y_s)
```

**Per-instruction cost:**

| Step | Operation | Cost |
|---|---|---|
| 1 | split sequence into segments | O(n) |
| 2 | init memory | O(md) |
| 3a | concat memory + segment | O((b+m)d) |
| QKV projections | 3 matmuls, (b+m)×d by d×d | O((b+m)d²) |
| QKᵀ | (b+m)×d by d×(b+m) | O((b+m)²d) |
| softmax | elementwise over (b+m)×(b+m) | O((b+m)²) |
| A·V | (b+m)×(b+m) by (b+m)×d | O((b+m)²d) |
| out-proj + FFN | (b+m)×d by d×d (×~2, absorbing expansion factor) | O((b+m)d²) |
| — per layer, summed | | **O((b+m)²d + (b+m)d²)** |
| — per segment, ×L layers | | O(L·((b+m)²d + (b+m)d²)) |
| 3c/3d | split + store | O((b+m)d) |
| 4 | concat outputs | O(n) |

Outer loop (step 3) runs `s = n/b` times, so total:

```
T_RMT(n) = O( (n/b) · L · ((b+m)²d + (b+m)d²) )
```

Since `m ≪ b`, `(b+m) ≈ b`, giving:

```
T_RMT(n) = O( L·n·b·d + L·n·d² )  =  O(n)   for fixed b, d, L
```

**Linear in sequence length.**

---

### Comparison against the full-sequence baseline

For full attention, `s = 1`, block size = `n`:

```
T_full(n) = O( L·(n²d + n·d²) )  =  O(n²)   for fixed d, L
```

| | Full-sequence attention | RMT block-recurrent |
|---|---|---|
| **Time (total FLOPs)** | O(L(n²d + nd²)) | O(L(nbd + nd²)) → **O(n)**, fixed b |
| **Peak activation memory** | O(n²) (attention matrix over whole sequence) | O(b²) (attention matrix per block) → **independent of n** |
| **Parallelism / latency** | one forward pass, fully parallel across n | s sequential segment passes (memory dependency chain) — not parallel across the whole sequence |
| **Accuracy** | exact global context | lossy: long-range info must pass through an m-token bottleneck |

This is the actual trade-off the project is measuring: RMT converts the quadratic-in-n attention cost into a linear one and caps peak memory at a constant independent of protein length, at the price of (a) forced sequential processing across segments, and (b) a compression bottleneck that can hurt fitness-prediction accuracy on long proteins, which is exactly what the accuracy-vs-cost concept is important to pay attention to.

---

*important: `n` (protein length) is the variable we let grow, and that's what I'm measuring cost against in this analysis.*

*`d`, `b`, `L` are fixed model/hyperparameter choices, not functions of input length, so they're treated as constants in here.*

*My reason is that here I want to compare RMT-style with full-sequence-length methods of inference, so the key parameter to take care of would be n==sequence length.*

---

**This is actually the ideal situation for cost analysis, where all the instructions are considered as used. However, in reality this gonna be changed a bit. I need to do the actual cost analysis using Python libraries like "fvcore," introduced by Meta, and the selected dataset.**
