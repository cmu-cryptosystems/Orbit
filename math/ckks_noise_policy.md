# CKKS Noise Policy Math

Orbit's built-in finalist gate uses the average-case CKKS noise model from Bossuat,
Costache, Mouchet, Nurnberger, and Troncoso-Pastoriza, ePrint 2024/853, and mirrors
the component-wise contract used by `tuneinsight/ckks-noise-estimator`.

## State

For each encrypted TDAG value, the estimator tracks coefficient-domain standard
deviations for the component-wise noise

```text
n(ct) = (e0, e1)
```

before decryption. It also tracks the current scale `Delta = 2^scale_bits`, level, and a
coefficient-domain RMS proxy for the plaintext value. The final decoded error bound is
computed only at the end:

```text
B = alpha * sqrt(sigma0^2 + N * sigma1^2 * sigma_s^2) / Delta
precision_bits = -log2(B)
```

Equivalently, Orbit reports

```text
estimated_noise_bits = log2(alpha * sqrt(sigma0^2 + N * sigma1^2 * sigma_s^2))
estimated_precision_bits = scale_bits - estimated_noise_bits
```

The bound holds under the paper's Gaussian and independence assumptions with

```text
Pr[||decoded error||_infty <= B] = erf(alpha / sqrt(2))^N
```

where `N` is the CKKS polynomial degree. The default `alpha = 14` is intentionally
conservative for finalist gating.

## Operation Rules

The implementation composes standard deviations by adding independent variances.

Fresh secret-key inputs:

```text
sigma(ctsk) = (sigma0, 0)
```

with `sigma0 = 3.2` unless configured otherwise.

Plaintext encoding and rescale rounding use uniform rounding noise:

```text
sigma_round^2 = 1/12
```

Rescale by a modulus factor approximated as `2^Sf`:

```text
sigma_i' = sqrt((sigma_i / 2^Sf)^2 + 1/12)
```

Ciphertext addition at aligned scales:

```text
sigma_i' = sqrt(sum_j sigma_{i,j}^2)
```

Plaintext addition affects only the first ciphertext component:

```text
sigma_0' = sqrt(sigma_0^2 + sigma_pt^2)
sigma_1' = sigma_1
```

Ciphertext-plaintext multiplication uses the paper's plaintext product form:

```text
sigma_i' = sqrt(N * sigma_i^2 * (sigma_pt_value^2 + sigma_pt_noise^2))
```

Ciphertext-ciphertext multiplication first applies the tensor-product variance formulas,
then adds a relinearization key-switch term. Rotation uses the same key-switch addend on
the second component. The key-switch RNS decomposition term is collapsed to a
conservative one-prime Gaussian envelope because the Python finalist request does not
carry the full Lattigo decomposition basis.

Bootstrapping is treated as a circuit refresh envelope: fresh public-key noise, one carried
input-error term, and one rescale rounding term. Full bootstrapping circuit estimation can
still be supplied through `--noise-estimator-binary`, which preserves the same JSON
contract and may call Tune Insight's Lattigo estimator directly.

## Policy Gate

The finalist gate accepts a candidate only when:

```text
estimated_precision_bits - precision_reserve_bits >= noise_estimator_min_output_margin_bits
```

and no hard unsupported condition was reported. The default reserve is 2 bits and the
default required output margin is 2 bits.
