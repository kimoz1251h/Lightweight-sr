| model              | params | GMACs@720p (learned convs) | CPU ms@720p (end-to-end, 4T, median) | BSDS100 PSNR/SSIM | Set14 PSNR/SSIM | Set5 PSNR/SSIM | manga109 PSNR/SSIM | urban100 PSNR/SSIM | avg PSNR | avg SSIM |
|--------------------|--------|----------------------------|--------------------------------------|-------------------|-----------------|----------------|--------------------|--------------------|----------|----------|
| bicubic            | 0      | 0.000                      | 15                                   | 26.05 / 0.6721    | 26.22 / 0.7088  | 28.63 / 0.8137 | 25.07 / 0.7904     | 23.24 / 0.6615     | 25.84    | 0.7293   |
| fsrcnn_x4          | 24,683 | 1.402                      | 40                                   | 26.57 / 0.7008    | 26.95 / 0.7383  | 29.72 / 0.8428 | 26.13 / 0.8187     | 23.87 / 0.6954     | 26.65    | 0.7592   |
| efficient_c48b6_x4 | 53,520 | 3.011                      | 122                                  | 27.07 / 0.7185    | 27.83 / 0.7626  | 30.92 / 0.8744 | 28.23 / 0.8716     | 24.75 / 0.7365     | 27.76    | 0.7927   |

Note: GMACs count only the learned conv / transposed-conv layers. The bicubic skip (and the bicubic baseline's interpolation) is NOT counted in GMACs, but IS included in the end-to-end CPU runtime, so runtime can differ between models with equal GMACs.
