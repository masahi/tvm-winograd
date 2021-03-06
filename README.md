Results below are obtained before AutoTVM was introduced.

When AutoTVM was merged, the TVM Direct convolution schedule was replaced with a completely new one.

So the "TVM Direct" results below do not represent current performance of TVM. It should be much better now.

<br/>

All numbers in msec.

TVM Direct convolution cannot handle specific inputs shape. These are denoted as N/A.

## R9 Nano

| (batch,CI,size,CO) | TVM Winograd (This code) | TVM Direct | MIOpen Winograd |
|------------- |:-------------:|:-------------:|:-------------:|
| (1, 128, 122, 128) | 1.174 | N/A | 0.433
| (1, 128, 128, 128) | 1.139 | 1.404 | 0.437
| (1, 64, 56, 64) | 0.136 | 0.349 | 0.044
| (1, 64, 64, 32) | 0.173 | 0.743 | 0.044
| (1, 64, 224, 64) | 1.233 | 1.865 | 0.373
| (1, 64, 112, 128) | 0.650 | 1.126 | 0.206
| (1, 512, 28, 512) | 0.747 | N/A | 0.422
| (1, 128, 28, 128) | 0.179 | 0.218 | 0.066
| (1, 256, 14, 256) | 0.316 | 0.463 | 0.109
| (8, 128, 122, 128) | 9.288 | N/A | 4.204
| (16, 64, 56, 64) | 1.397 | N/A | 0.377
| (32, 64, 64, 32) | 5.732 | N/A | 0.473
| (64, 128, 32, 128) | 5.980 | N/A | 2.243

<br/>

## GTX1070 Ti

| (batch,CI,size,CO) | TVM Winograd (This code) | TVM Direct | TVM Winograd NVPTX (This code) | TVM Direct NVPTX | cuDNN Winograd |
|------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| (1, 128, 122, 128) | 1.668 | 2.272 | 1.099 | N/A | 0.391
| (1, 128, 128, 128) | 0.947 | 1.740 | 0.950 | 1.797 | 0.395
| (1, 64, 56, 64) | 0.180 | 0.165 | 0.116 | 0.162 | 0.038
| (1, 64, 64, 32) | 0.114 | 0.480 | 0.078 | 0.481 | 0.025
| (1, 64, 224, 64) | 1.252 | 1.817 | 1.238 | 1.722 | 0.362
| (1, 64, 112, 128) | 0.759 | 1.236 | 0.556 | 1.216 | 0.187
| (1, 512, 28, 512) | 1.103 | N/A | 0.654 | N/A | 0.603
| (1, 128, 28, 128) | 0.124 | 0.120 | 0.074 | 0.117 | 0.043
| (1, 256, 14, 256) | 0.182 | 0.487 | 0.118 | 0.332 | 0.098
| (8, 128, 122, 128) | 11.175 | N/A | 7.644 | N/A | 3.293
| (16, 64, 56, 64) | 1.783 | N/A | 1.330 | N/A | 0.391
| (32, 64, 64, 32) | 2.849 | N/A | 2.300 | N/A | 0.531
| (64, 128, 32, 128) | 3.707 | N/A | 3.651 | N/A | 1.589
