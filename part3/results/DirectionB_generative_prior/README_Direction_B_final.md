# Direction B Final Summary: Generative Prior with Frequency-Constrained Fusion

## Goal

Direction B investigates whether Stable Diffusion + ControlNet-Tile can provide a useful generative prior for video super-resolution.

The exploration contains two stages:

1. Direct ControlNet-Tile enhancement.
2. Frequency-constrained residual fusion, where only high-frequency residuals from ControlNet are fused back into FUGR-C.

## Direct ControlNet-Tile Result

On the expanded 40-frame benchmark:

| Method | Frames | PSNR | SSIM | Sharpness |
|---|---:|---:|---:|---:|
| FUGR-C | 40 | 29.5043 | 0.8149 | 0.001287 |
| BasicVSR | 40 | 29.4878 | 0.8147 | 0.001284 |
| VSRGAN | 40 | 25.8719 | 0.7462 | 0.005246 |
| ControlNet-FUGR | 40 | 22.4517 | 0.6105 | 0.000903 |
| ControlNet-Basic | 40 | 22.4141 | 0.6095 | 0.000909 |

Direct ControlNet enhancement severely reduces full-reference fidelity and worsens temporal consistency.

## Frequency-Constrained Fusion

Best configuration:

RGB-HF-b0.30-s0.5

| Method | Frames | PSNR | SSIM | Sharpness | TDE |
|---|---:|---:|---:|---:|---:|
| Frequency Fusion | 40 | 29.5052 | 0.8143 | 0.001175 | 0.033674 |
| FUGR-C | 40 | 29.4976 | 0.8143 | 0.001287 | 0.033661 |
| ControlNet-FUGR | 40 | 22.4514 | 0.6101 | 0.000900 | 0.076655 |

Frequency Fusion gives a small PSNR gain over FUGR-C, but SSIM, sharpness, and TDE do not improve consistently.

## Per-sequence and Per-frame Delta

Frequency Fusion improves:

- 6 out of 10 sequences.
- 22 out of 40 frames.

Frame-level delta statistics:

- mean PSNR delta: +0.0030 dB
- median PSNR delta: +0.00067 dB
- bootstrap 95% CI: [-0.00027, 0.00652]

## Final Conclusion

Direction B shows that generative priors must be strongly constrained for high-fidelity VSR.

Direct ControlNet-Tile enhancement is harmful because it introduces structure and temporal drift. However, using ControlNet only as a high-frequency residual source can recover fidelity and produce a small PSNR gain on the expanded benchmark.

The gain is marginal and not statistically strong, so Direction B is best treated as an exploratory generative-prior branch. Direction C remains the final robust high-fidelity method.
