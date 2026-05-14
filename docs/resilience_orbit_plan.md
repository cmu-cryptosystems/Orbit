# Resilience-Guided Orbit Placement

This branch adds the first Orbit-side compatibility layer for the
noise-resilience profiler. Orbit can now consume either:

- a `ckks-robustness-profiler` summary JSON with `layers[*].ckks_hint`, or
- a compact Orbit-native constraints JSON with `constraints[*].min_scale`.

The loaded constraints become local scale waterlines on matched TDAG nodes. A
resilient edge can therefore use a lower local `min_scale` than the global `Sw`,
which gives Orbit's existing ILP more room to delay scale refresh and
bootstrapping. Fragile edges can use a higher `min_scale` than `Sw`, which keeps
maintenance closer to the edge.

## How Matching Works

For profiler JSON, Orbit builds match tokens from each layer's `target_id`,
`name`, and `module_names`. A TDAG node matches when one of those tokens appears
in the MLIR/TDAG comment. Orbit-native constraint files can match directly by:

- exact TDAG node label: `match.node`
- node regex: `match.node_regex`
- comment substring: `match.comment_contains`
- comment regex: `match.comment_regex`

The direct Orbit format is useful while model-edge IDs and MLIR comments are
still being aligned.

## CLI Usage

```bash
python3 -m scripts.optimizer.orbit.optimizer \
  --inputfile mlirs_input/ResNetReLU64k.mlir \
  --outputfile mlirs_output/orbit/resilience_demo.mlir \
  --costjson cost_models/profiled_LATTIGONEW_CPU64k_3_16.json \
  --maxlevel 16 \
  --waterscale 40 \
  --resilience-profile examples/resilience_constraints_example.json
```

The wrapper also accepts the same flag:

```bash
python3 scripts/optimizer/orbit/run_orbit.py \
  --model ResNet \
  --act ReLU \
  --n 64 \
  --Lm 16 \
  --Sw 40 \
  --resilience-profile examples/resilience_constraints_example.json
```

## Relationship To Noise Profiling

The profiler should learn a tolerance per model edge. Your proposed first
implementation can be:

1. Run one or more reference inputs through the plaintext model and save
   reference outputs or logits.
2. Attach learnable noise tensors or per-layer noise standard deviations.
3. Optimize noise magnitudes with gradient descent while penalizing output loss:

   ```text
   maximize   sum_e sigma_e^2
   subject to loss(model_with_noise(x), reference_output) <= loss_budget
   ```

   A practical unconstrained form is:

   ```text
   minimize   output_loss + lambda * max(0, output_loss - budget)^2
            - alpha * sum_e log(sigma_e + eps)
   ```

4. Reduce learned `sigma_e` into `tau_abs[e]`, `min_log2_scale[e]`, and a
   fragility label.
5. Export profiler JSON. Orbit consumes that JSON and maps `min_log2_scale` to
   local TDAG scale bounds.

Multiple calibration inputs should replace the single-reference setup before
claiming robustness. Group and simultaneous noise validation are still needed
because independent per-edge tolerances can overestimate joint safety.

## Current Boundary

This branch does not add CKKS error variables to Orbit's ILP yet. It uses
profiled tolerance as a local precision/waterline constraint, which is the
lowest-risk bridge into the existing bootstrap-placement machinery. The next
step is to add explicit edge error state:

```text
E_out <= E_in + E_rescale/modswitch/keyswitch/compute
E_out = E_bootstrap after bootstrap
E_out <= tau_abs[edge]  # only with --resilience-constraint-policy hard-tau
```

For speedup experiments, the default policy is `relax-only`: learned tolerances
lower the guided global waterline and local waterlines but do not impose
stricter `tau_abs` feasibility constraints than the baseline compile. This
keeps the robustness profile from accidentally adding bootstraps when the
current CKKS error proxy is conservative.
