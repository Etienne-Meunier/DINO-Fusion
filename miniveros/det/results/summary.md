# Results (last-year mean T, water cells, K)

Network: per-run RMSE averaged over 4 initialisation sets of 5 members; hyper-parameters by the one-standard-error rule unless stated.

**Mean RMSE over the held-out runs**

| split | held out | mlp | gp_s | gp | rbf | nearest | mean |
|---|---|---|---|---|---|---|---|
| scattered | 10 | **0.017** | 0.019 | 0.022 | 0.021 | 0.038 | 0.161 |
| band | 30 | **0.038** | 0.045 | 0.051 | 0.057 | 0.118 | 0.204 |
| centre5 | 25 | **0.025** | 0.029 | 0.035 | 0.037 | 0.080 | 0.162 |
| centre7 | 49 | **0.029** | 0.035 | 0.069 | 0.068 | 0.116 | 0.203 |
| outer51 | 51 | **0.067** | 0.074 | 0.081 | 0.093 | 0.134 | 0.265 |
| outer75 | 75 | **0.114** | 0.125 | 0.131 | 0.138 | 0.159 | 0.226 |
| top | 10 | **0.049** | 0.079 | 0.083 | 0.084 | 0.176 | 0.482 |
| four | 96 | 0.147 | **0.120** | 0.235 | 0.146 | 0.181 | 0.260 |
| loo | 100 | **0.024** | 0.028 | 0.032 | 0.034 | 0.085 | 0.221 |

**Median RMSE over the held-out runs**

| split | held out | mlp | gp_s | gp | rbf | nearest | mean |
|---|---|---|---|---|---|---|---|
| scattered | 10 | **0.013** | 0.016 | 0.018 | 0.019 | 0.021 | 0.171 |
| band | 30 | **0.033** | 0.042 | 0.046 | 0.046 | 0.086 | 0.146 |
| centre5 | 25 | **0.019** | 0.026 | 0.035 | 0.036 | 0.068 | 0.166 |
| centre7 | 49 | **0.026** | 0.033 | 0.064 | 0.066 | 0.083 | 0.219 |
| outer51 | 51 | **0.027** | 0.032 | 0.037 | 0.055 | 0.060 | 0.111 |
| outer75 | 75 | 0.041 | **0.038** | 0.044 | 0.066 | 0.062 | 0.086 |
| top | 10 | **0.033** | 0.034 | 0.059 | 0.070 | 0.143 | 0.268 |
| four | 96 | 0.136 | 0.074 | 0.212 | 0.111 | **0.064** | 0.233 |
| loo | 100 | **0.017** | 0.019 | 0.027 | 0.027 | 0.048 | 0.169 |

Network mean RMSE / nearest run's: scattered 0.44, band 0.32, centre5 0.31, centre7 0.25, outer51 0.50, outer75 0.72, top 0.28, four 0.81, loo 0.28

**Network (one-standard-error rule, reported) against the GP on s** (per-run RMSE difference, network minus GP; negative = network better; Holm correction over the 8 evaluations other than four runs)

| split | configuration | network mean | range over seed sets | GP on s | difference | 95 % bootstrap interval | Wilcoxon p | Holm p | network wins |
|---|---|---|---|---|---|---|---|---|---|
| scattered | relu 0.01 | 0.0169 | 0.0008 | 0.0194 | -0.0025 | [-0.0053, -0.0005] | 0.037 | 0.074 | 7/10 |
| band | relu 0.01 | 0.0378 | 0.0010 | 0.0453 | -0.0075 | [-0.0132, -0.0020] | 0.015 | 0.044 | 23/30 |
| centre5 | relu 0.01 | 0.0246 | 0.0004 | 0.0291 | -0.0045 | [-0.0071, -0.0019] | 0.003 | 0.013 | 17/25 |
| centre7 | relu 0.01 | 0.0287 | 0.0006 | 0.0345 | -0.0058 | [-0.0096, -0.0021] | 0.003 | 0.013 | 34/49 |
| outer51 | relu 0.01 | 0.0669 | 0.0005 | 0.0744 | -0.0076 | [-0.0123, -0.0032] | 0.002 | 0.013 | 36/51 |
| outer75 | relu 0.1 | 0.1137 | 0.0009 | 0.1248 | -0.0111 | [-0.0170, -0.0059] | < 0.001 | 0.004 | 51/75 |
| top | relu 0.01 | 0.0491 | 0.0028 | 0.0788 | -0.0297 | [-0.0768, +0.0001] | 0.193 | 0.193 | 7/10 |
| four | relu 1 | 0.1469 | 0.0345 | 0.1196 | +0.0273 | [+0.0131, +0.0430] | 0.003 | - | 39/96 |
| loo | relu 0.01 | 0.0238 | 0.0004 | 0.0284 | -0.0046 | [-0.0086, -0.0015] | < 0.001 | < 0.001 | 67/100 |

**Network (minimum CV error) against the GP on s** (per-run RMSE difference, network minus GP; negative = network better; Holm correction over the 8 evaluations other than four runs)

| split | configuration | network mean | range over seed sets | GP on s | difference | 95 % bootstrap interval | Wilcoxon p | Holm p | network wins |
|---|---|---|---|---|---|---|---|---|---|
| scattered | relu 0.001 | 0.0164 | 0.0006 | 0.0194 | -0.0030 | [-0.0057, -0.0009] | 0.049 | 0.195 | 7/10 |
| band | relu 0.001 | 0.0369 | 0.0025 | 0.0453 | -0.0083 | [-0.0143, -0.0027] | 0.011 | 0.068 | 21/30 |
| centre5 | relu 0.0001 | 0.0259 | 0.0021 | 0.0291 | -0.0032 | [-0.0062, -0.0003] | 0.022 | 0.110 | 19/25 |
| centre7 | relu 0.001 | 0.0355 | 0.0035 | 0.0345 | +0.0010 | [-0.0034, +0.0057] | 0.745 | 1.000 | 21/49 |
| outer51 | relu 0.001 | 0.0673 | 0.0030 | 0.0744 | -0.0072 | [-0.0122, -0.0027] | 0.009 | 0.066 | 33/51 |
| outer75 | relu 0.01 | 0.1216 | 0.0020 | 0.1248 | -0.0032 | [-0.0102, +0.0036] | 0.594 | 1.000 | 38/75 |
| top | relu 0.0001 | 0.0583 | 0.0065 | 0.0788 | -0.0204 | [-0.0671, +0.0068] | 1.000 | 1.000 | 5/10 |
| four | relu 1 | 0.1469 | 0.0345 | 0.1196 | +0.0273 | [+0.0131, +0.0430] | 0.003 | - | 39/96 |
| loo | relu 0.001 | 0.0232 | 0.0009 | 0.0284 | -0.0052 | [-0.0092, -0.0023] | < 0.001 | < 0.001 | 66/100 |

**Corner run ck0.8_eps0.0875** (when held out)

| split | mlp | gp_s | gp | rbf | nearest | mean |
|---|---|---|---|---|---|---|
| outer51 | 0.664 | 0.703 | 0.693 | 0.726 | 0.950 | 1.424 |
| outer75 | 1.062 | 1.176 | 1.183 | 1.094 | 1.268 | 1.451 |
| top | 0.157 | 0.373 | 0.255 | 0.217 | 0.357 | 1.391 |
| four | 0.831 | 0.620 | 1.196 | 0.584 | 0.695 | 1.280 |
| loo | 0.168 | 0.306 | 0.220 | 0.178 | 0.357 | 1.366 |

**Network: hyper-parameters and ensemble spread** (CV RMSE: per-run RMSE of the 3 x 5-fold cross-validation, averaged over the repeats and the runs; standard error: std over the runs of the best configuration's per-run CV RMSE / sqrt(runs); spread: one-standard-error choice)

| split | training runs | one-SE rule | its CV RMSE | minimum rule | best CV RMSE | standard error | spread |
|---|---|---|---|---|---|---|---|
| scattered | 90 | relu 0.01 | 0.0264 | relu 0.001 | 0.0254 | 0.0024 | 0.003 |
| band | 70 | relu 0.01 | 0.0278 | relu 0.001 | 0.0275 | 0.0032 | 0.007 |
| centre5 | 75 | relu 0.01 | 0.0273 | relu 0.0001 | 0.0265 | 0.0029 | 0.005 |
| centre7 | 51 | relu 0.01 | 0.0338 | relu 0.001 | 0.0332 | 0.0040 | 0.009 |
| outer51 | 49 | relu 0.01 | 0.0255 | relu 0.001 | 0.0251 | 0.0027 | 0.008 |
| outer75 | 25 | relu 0.1 | 0.0295 | relu 0.01 | 0.0280 | 0.0030 | 0.008 |
| top | 90 | relu 0.01 | 0.0246 | relu 0.0001 | 0.0239 | 0.0026 | 0.011 |
| four | 4 | relu 1 | 0.5393 | relu 1 | 0.5393 | 0.1461 | 0.023 |
| loo | 99 | relu 0.01 | 0.0255 | relu 0.001 | 0.0249 | 0.0022 | 0.004 |
