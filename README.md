# APAI-project-work
Transformer-based model - inference time cost analysis for the protein fitness prediction task


### Project goal:
Run a general-purpose protein transformer (baseline, full sequence, no block/recurrence) on a subset of ProteinGym, and measure two things: (1) compute cost (MACs and memory) as sequence length grows, (2) accuracy on fitness prediction. Then try to use the block strategy and/or optimization techniques like sequence compression with an autoencoder, teach a lighter model using knowledge distillation, etc.
What I did during the first week:
Read provided papers and try to understand their major points and gain better insight into selecting the base model.

### These papers are:
- 1.1 Recurrent Memory Transformer (RMT): adds memory tokens and segment-level recurrence to an existing Transformer without changing its architecture. Main takeaway: this can be retrofitted onto an already-pretrained model, which fits our timeframe better than training a new architecture from scratch.

- 1.2 Block-Recurrent Transformers: applies a full transformer layer recurrently over blocks with LSTM-style gating, giving linear complexity. More powerful than RMT but needs a custom recurrent cell trained from scratch, so I'm parking it as a possible extension rather than the main approach.

- 1.3 UNIVERSAL TRANSFORMERS (Most famous one :D): recurrence over depth (weight-tying across layers) instead of over sequence position. Useful context, but a different lever (parameter count) than what we need for the sequence-length cost problem.

### Select the base model architecture
- Base model: ESM-2 (facebook/esm2_t12_35M_UR50D, with esm2_t30_150M_UR50D as a fallback if the small model is too weak a baseline).
- Why? pretrained protein language model, available directly through Hugging Face, runs in plain Python/PyTorch (general transformer requirement from the plan), and already has published zero-shot fitness numbers on ProteinGym I can sanity-check against.
Small enough (35M – 150M params) to profile and iterate on quickly for a 3-CFU project, and to later distill into an even smaller student model.
### Select dataset and domain
- Domain: protein fitness prediction.
- Dataset: ProteinGym DMS substitutions benchmark (217 assays, ~2.7M measured missense variants), loaded via the Hugging Face dataset OATML-Markslab/ProteinGym_v1 (config “DMS_substitutions”) -  avoids handling the full raw zip archive.
For now, I don’t run all 217 assays. I picked a stratified subset (~15–20 assays) covering short (<200 aa), medium (200–500 aa), and long (>500 aa) proteins, so the cost/accuracy trade-off can be analyzed as a function of sequence length, not just in aggregate.
### Define the cost analysis methodology
- MACs: measured with fvcore.nn.FlopCountAnalysis, at a sweep of sequence lengths (40, 101, 248, 448, 770, 3423) (actual protein sequence sizes).
- Memory: peak GPU memory via torch.cuda.max_memory_allocated(), reset between runs.
- Repeat each cost measurement 3x and average to reduce noise.
