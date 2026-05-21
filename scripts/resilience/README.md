# Resilience Calibration

Generate the Tune Insight CKKS noise metadata:

```bash
python -m scripts.resilience.export_ckks_noise_estimator \
  --out profiles/ckks_noise_estimator_64k_s40.json
```

Run ResNetSiLU64k waterline calibration through Orbit and the Lattigo plaintext
noise simulator:

```bash
python -m scripts.resilience.resnet_ckks_noise_calibrate \
  --run-orbit \
  --samples 50 \
  --trials 3 \
  --search-mode waterline,const,groups,upscale-objective \
  --correctness-gate exact-top1 \
  --selection-policy pareto-middle \
  --drift-metric margin-retention \
  --candidate-scales 40,38,36,34,32,30,28,26,24,22,20,18,16,14 \
  --constant-scales 40,38,36,34,32,30,28,26,24,22,20,18,16 \
  --upscale-objective-weights 0,0.025,0.05,0.1,0.2,0.35,0.5 \
  --fhe-binary backend/lattigo/lowering/fhe_binary
```

The wrapper first finds a SiLU waterline floor, then sweeps constant operand
floors, ILP upscale proxy weights, and additional named groups
(`residual_add`, `downsample`, `final_pool`, `final_linear`) only when noisy
plaintext predictions preserve the clean baseline. Passing candidates are ranked
on a Pareto frontier of latency reduction and q05 top-1 margin retention; the
emitted profile uses the knee point rather than the fastest passing point.

The calibration expects `input_constants/ResNetSiLU64k_hecate.cst` and the
first 50 `input_data/64k/resnet/silu` samples. If they are absent and
`frontend/dacapo` is installed and patched, the script downloads
`resnet20.silu.model` and generates the missing ResNet-only artifacts.
