# Synthesis audit of keras-hexagdly 0.7.2

Produced by `audit.py report` (see its docstring). First stage (C simulation, Vitis HLS 2024.2, RTL co-simulation) and second stage (Vivado 2024.2 logic synthesis, out of context, no place and route) of every case; the cases of the fused kernel in the Latency strategy with a reuse factor above 1 were run again after the last change of that kernel.

81 of 81 cases. Vitis HLS 2024.2 then Vivado 2024.2 logic synthesis out of context, xc7k160tffg676-2, 4 ns, ap_fixed<24,12>, 4 -> 8 channels.

| Case | Cycles | Cycles/token | HLS clock | Vivado path | Levels | LUT (HLS / Vivado) | DSP (HLS / Vivado) | Flags |
|---|---|---|---|---|---|---|---|---|
| conv2d_k1_s1_none_rf1_latency_fused | 193 | 1.3 | 3.455 | 2.587 | 3 | 12778 / 7238 | 0 / 0 |  |
| conv2d_k1_s1_ring_rf1_latency_fused | 194 | 1.3 | 3.455 | 2.574 | 3 | 5773 / 3111 | 0 / 0 |  |
| conv2d_k1_s1_none_rf2_latency_fused | 377 | 2.6 | 4.001 | 3.425 | 9 | 13125 / 6379 | 18 / 18 |  |
| conv2d_k1_s1_ring_rf2_latency_fused | 382 | 2.7 | 4.147 | 2.46 | 10 | 7791 / 3009 | 14 / 7 |  |
| conv2d_k1_s1_none_rf4_latency_fused | 745 | 5.2 | 3.164 | 3.499 | 0 | 7712 / 2936 | 56 / 98 |  |
| conv2d_k1_s1_ring_rf4_latency_fused | 745 | 5.2 | 4.147 | 2.716 | 3 | 6189 / 2099 | 32 / 16 |  |
| conv2d_k1_s1_none_rf8_latency_fused | 1472 | 10.2 | 3.164 | 2.786 | 3 | 4869 / 2294 | 32 / 32 |  |
| conv2d_k1_s1_ring_rf8_latency_fused | 1472 | 10.2 | 4.147 | 2.572 | 3 | 4789 / 1864 | 16 / 8 |  |
| conv2d_k1_s1_none_rf1_latency_dataflow | 195 | 1.4 | 3.046 | 2.892 | 8 | 13717 / 8270 | 0 / 0 |  |
| conv2d_k1_s1_ring_rf1_latency_dataflow | 195 | 1.4 | 3.046 | 2.951 | 5 | 6270 / 4208 | 0 / 0 |  |
| conv2d_k1_s1_none_rf2_latency_dataflow | 304 | 2.1 | 3.943 | 3.12 | 8 | 13375 / 7003 | 18 / 18 |  |
| conv2d_k1_s1_ring_rf2_latency_dataflow | 307 | 2.1 | 3.427 | 2.49 | 3 | 7567 / 3801 | 14 / 7 |  |
| conv2d_k1_s1_none_rf4_latency_dataflow | 594 | 4.1 | 3.046 | 3.499 | 0 | 7769 / 3488 | 56 / 98 |  |
| conv2d_k1_s1_ring_rf4_latency_dataflow | 594 | 4.1 | 3.427 | 2.609 | 4 | 6246 / 3040 | 32 / 16 |  |
| conv2d_k1_s1_none_rf8_latency_dataflow | 1170 | 8.1 | 3.046 | 2.59 | 3 | 4981 / 3210 | 32 / 32 |  |
| conv2d_k1_s1_ring_rf8_latency_dataflow | 1169 | 8.1 | 3.427 | 2.477 | 3 | 4845 / 2780 | 16 / 8 |  |
| conv2d_k1_s1_none_rf1_resource_fused | 198 | 1.4 | 2.919 | 2.725 | 2 | 12847 / 7052 | 0 / 0 |  |
| conv2d_k1_s1_ring_rf1_resource_fused | 198 | 1.4 | 2.906 | 2.725 | 2 | 6014 / 3265 | 0 / 0 |  |
| conv2d_k1_s1_none_rf2_resource_fused | 2945 | 20.5 | 2.919 | 2.546 | 7 | 13408 / 6437 | 18 / 18 |  |
| conv2d_k1_s1_ring_rf2_resource_fused | 3127 | 21.7 | 3.427 | 2.52 | 7 | 7241 / 3423 | 14 / 7 |  |
| conv2d_k1_s1_none_rf4_resource_fused | 3127 | 21.7 | 2.887 | 2.635 | 3 | 4984 / 3537 | 56 / 56 |  |
| conv2d_k1_s1_ring_rf4_resource_fused | 3309 | 23.0 | 3.427 | 2.56 | 3 | 5861 / 2836 | 32 / 16 |  |
| conv2d_k1_s1_none_rf8_resource_fused | 3855 | 26.8 | 2.913 | 2.619 | 3 | 5181 / 3225 | 32 / 32 |  |
| conv2d_k1_s1_ring_rf8_resource_fused | 3855 | 26.8 | 3.427 | 2.374 | 8 | 4460 / 2599 | 16 / 8 |  |
| conv2d_k1_s1_none_rf1_resource_dataflow | 195 | 1.4 | 3.046 | 2.892 | 8 | 13717 / 8270 | 0 / 0 |  |
| conv2d_k1_s1_ring_rf1_resource_dataflow | 195 | 1.4 | 3.046 | 2.951 | 5 | 6270 / 4208 | 0 / 0 |  |
| conv2d_k1_s1_none_rf2_resource_dataflow | 304 | 2.1 | 3.943 | 3.12 | 8 | 13375 / 7003 | 18 / 18 |  |
| conv2d_k1_s1_ring_rf2_resource_dataflow | 307 | 2.1 | 3.427 | 2.49 | 3 | 7567 / 3801 | 14 / 7 |  |
| conv2d_k1_s1_none_rf4_resource_dataflow | 594 | 4.1 | 3.046 | 3.499 | 0 | 7769 / 3488 | 56 / 98 |  |
| conv2d_k1_s1_ring_rf4_resource_dataflow | 594 | 4.1 | 3.427 | 2.609 | 4 | 6246 / 3040 | 32 / 16 |  |
| conv2d_k1_s1_none_rf8_resource_dataflow | 1170 | 8.1 | 3.046 | 2.59 | 3 | 4981 / 3210 | 32 / 32 |  |
| conv2d_k1_s1_ring_rf8_resource_dataflow | 1169 | 8.1 | 3.427 | 2.477 | 3 | 4845 / 2780 | 16 / 8 |  |
| conv2d_k2_s1_none_rf1_latency_fused | 223 | 1.5 | 3.455 | 2.56 | 3 | 34620 / 19505 | 0 / 0 |  |
| conv2d_k2_s1_ring_rf1_latency_fused | 224 | 1.6 | 3.455 | 2.556 | 3 | 10598 / 5219 | 0 / 0 |  |
| conv2d_k2_s1_none_rf4_latency_fused | 862 | 6.0 | 3.164 | 3.679 | 3 | 12418 / 6072 | 278 / 290 |  |
| conv2d_k2_s1_ring_rf4_latency_fused | 858 | 6.0 | 3.164 | 2.702 | 3 | 8573 / 3797 | 48 / 24 |  |
| conv2d_k2_s1_none_rf1_latency_dataflow | 224 | 1.6 | 3.046 | 2.719 | 9 | 37069 / 23107 | 0 / 0 |  |
| conv2d_k2_s1_ring_rf1_latency_dataflow | 226 | 1.6 | 3.046 | 2.944 | 5 | 11425 / 8379 | 0 / 0 |  |
| conv2d_k2_s1_none_rf4_latency_dataflow | 597 | 4.1 | 3.046 | 3.557 | 3 | 12125 / 8449 | 278 / 290 |  |
| conv2d_k2_s1_ring_rf4_latency_dataflow | 596 | 4.1 | 3.128 | 2.607 | 4 | 10271 / 6916 | 48 / 24 |  |
| conv2d_k2_s1_none_rf1_resource_fused | 228 | 1.6 | 2.919 | 2.469 | 7 | 34583 / 18381 | 0 / 0 |  |
| conv2d_k2_s1_ring_rf1_resource_fused | 229 | 1.6 | 2.919 | 2.443 | 7 | 11145 / 5497 | 0 / 0 |  |
| conv2d_k2_s1_none_rf4_resource_fused | 4655 | 32.3 | 3.07 | 2.567 | 1 | 12135 / 7569 | 227 / 227 |  |
| conv2d_k2_s1_ring_rf4_resource_fused | 4655 | 32.3 | 3.128 | 2.571 | 3 | 9165 / 4657 | 48 / 24 |  |
| conv2d_k2_s1_none_rf1_resource_dataflow | 224 | 1.6 | 3.046 | 2.719 | 9 | 37069 / 23107 | 0 / 0 |  |
| conv2d_k2_s1_ring_rf1_resource_dataflow | 226 | 1.6 | 3.046 | 2.944 | 5 | 11425 / 8379 | 0 / 0 |  |
| conv2d_k2_s1_none_rf4_resource_dataflow | 597 | 4.1 | 3.046 | 3.557 | 3 | 12125 / 8449 | 278 / 290 |  |
| conv2d_k2_s1_ring_rf4_resource_dataflow | 596 | 4.1 | 3.128 | 2.607 | 4 | 10271 / 6916 | 48 / 24 |  |
| conv2d_k1_s2_none_rf1_latency_fused | 194 | 1.2 | 3.455 | 2.577 | 3 | 13284 / 7576 | 0 / 0 |  |
| conv2d_k1_s2_none_rf4_latency_fused | 745 | 4.8 | 3.164 | 3.499 | 0 | 7816 / 3042 | 56 / 98 |  |
| conv2d_k1_s2_none_rf1_latency_dataflow | 196 | 1.3 | 3.046 | 2.665 | 8 | 13871 / 8519 | 0 / 0 |  |
| conv2d_k1_s2_none_rf4_latency_dataflow | 200 | 1.3 | 3.046 | 3.499 | 0 | 7898 / 3797 | 56 / 98 |  |
| conv2d_k1_s2_none_rf1_resource_fused | 199 | 1.3 | 2.919 | 2.724 | 2 | 13397 / 7303 | 0 / 0 |  |
| conv2d_k1_s2_none_rf4_resource_fused | 3155 | 20.2 | 2.887 | 2.889 | 4 | 5542 / 4029 | 56 / 56 |  |
| conv2d_k1_s2_none_rf1_resource_dataflow | 196 | 1.3 | 3.046 | 2.665 | 8 | 13871 / 8519 | 0 / 0 |  |
| conv2d_k1_s2_none_rf4_resource_dataflow | 200 | 1.3 | 3.046 | 3.499 | 0 | 7898 / 3797 | 56 / 98 |  |
| conv2d_k1_s1_diag_rf4_latency_fused | 744 | 5.2 | 3.164 | 2.683 | 3 | 4312 / 2076 | 32 / 32 |  |
| conv2d_k1_s1_diag_rf4_latency_dataflow | 593 | 4.1 | 3.046 | 2.488 | 4 | 4337 / 2760 | 32 / 32 |  |
| conv2d_k1_s1_diag_rf4_resource_fused | 2945 | 20.5 | 2.913 | 2.4 | 8 | 4196 / 3002 | 32 / 32 |  |
| conv2d_k1_s1_diag_rf4_resource_dataflow | 593 | 4.1 | 3.046 | 2.488 | 4 | 4337 / 2760 | 32 / 32 |  |
| conv2d_k1_s1_sym_rf4_latency_fused | 744 | 5.2 | 3.164 | 2.685 | 3 | 4376 / 2173 | 32 / 32 |  |
| conv2d_k1_s1_sym_rf4_latency_dataflow | 593 | 4.1 | 3.046 | 2.488 | 4 | 4337 / 2760 | 32 / 32 |  |
| conv2d_k1_s1_sym_rf4_resource_fused | 2945 | 20.5 | 2.913 | 2.4 | 8 | 4196 / 3002 | 32 / 32 |  |
| conv2d_k1_s1_sym_rf4_resource_dataflow | 593 | 4.1 | 3.046 | 2.488 | 4 | 4337 / 2760 | 32 / 32 |  |
| conv2d_k1_s1_none_rf3_latency_fused | 564 | 3.9 | 3.164 | 3.499 | 0 | 10685 / 3685 | 80 / 143 |  |
| conv2d_k1_s1_none_rf3_latency_dataflow | 451 | 3.1 | 3.046 | 3.499 | 0 | 10739 / 4391 | 80 / 143 |  |
| conv2d_k1_s1_none_rf3_resource_fused | 3309 | 23.0 | 2.887 | 2.404 | 9 | 6603 / 4441 | 80 / 80 |  |
| conv2d_k1_s1_none_rf3_resource_dataflow | 451 | 3.1 | 3.046 | 3.499 | 0 | 10739 / 4391 | 80 / 143 |  |
| maxpool2d_k1_s1_none_rf1_latency_fused | 191 | 1.3 | 3.046 | 2.667 | 3 | 3272 / 1317 | 0 / 0 |  |
| maxpool2d_k1_s2_none_rf1_latency_fused | 191 | 1.3 | 3.046 | 2.663 | 3 | 3822 / 2273 | 0 / 0 |  |
| avgpool2d_k1_s1_none_rf1_latency_fused |  |  |  |  |  |  /  |  /  | ERROR NotImplementedError: AvgPool2d layers ['c'] have no hls4ml export path yet. |
| avgpool2d_k1_s2_none_rf1_latency_fused |  |  |  |  |  |  /  |  /  | ERROR NotImplementedError: AvgPool2d layers ['c'] have no hls4ml export path yet. |
| conv3d_k1_s1_none_rf1_latency_fused | 387 | 1.0 | 4.532 | 2.789 | 4 | 39672 / 22039 | 0 / 0 |  |
| conv3d_k1_s1_none_rf4_latency_fused | 1461 | 3.8 | 4.532 | 3.739 | 3 | 14486 / 7470 | 308 / 328 |  |
| conv3d_k1_s1_none_rf1_latency_dataflow | 387 | 1.0 | 4.532 | 2.725 | 9 | 42658 / 26539 | 0 / 0 |  |
| conv3d_k1_s1_none_rf4_latency_dataflow | 1044 | 2.7 | 4.532 | 3.313 | 2 | 14888 / 10052 | 308 / 328 |  |
| conv3d_k1_s1_none_rf1_resource_fused | 387 | 1.0 | 4.532 | 2.563 | 4 | 39653 / 20707 | 0 / 0 |  |
| conv3d_k1_s1_none_rf4_resource_fused | 7567 | 19.7 | 4.532 | 2.903 | 2 | 14688 / 9366 | 251 / 251 |  |
| conv3d_k1_s1_none_rf1_resource_dataflow | 387 | 1.0 | 4.532 | 2.725 | 9 | 42658 / 26539 | 0 / 0 |  |
| conv3d_k1_s1_none_rf4_resource_dataflow | 1044 | 2.7 | 4.532 | 3.313 | 2 | 14888 / 10052 | 308 / 328 |  |
| maxpool3d_k1_s1_none_rf1_latency_fused | 461 | 1.2 | 4.532 | 2.717 | 3 | 4231 / 2278 | 0 / 0 |  |

2 case(s) flagged.
