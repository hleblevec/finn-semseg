# Cityscapes Semantic Segmentation on FPGA with FINN

This repository contains source files to build and deploy on Alveo U250 board a quantized ResNet18-Unet semantic segmentation model trained on Cityscapes dataset using the FINN framework from https://github.com/Xilinx/finn.  

Requirements:

* Vitis 2022.2
* XRT 2.13
* Docker
* Python >= 3.8
* Pynq >= 3.0.1


A pre-synthesized XCLBIN file is available in `/deploy/driver/bitfile/` and can be used along with the pynq-based driver to deploy and evaluate the model.

To re-generate binaries, do:

```SHELL
#clone FINN repository at correct commit
get-finn.sh
cd finn/
#launch the build
./run-docker.sh build_custom ../scripts
```

Newly generated bitfile should be in `/outputs/bitfile/`.
