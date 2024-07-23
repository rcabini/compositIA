# CompositIA

CompositIA is a fully automated system designed to calculate body composition from thoraco-abdominal CT scans.

![Pipeline](pipeline.png)

## Overview

**CompositIA** consists of three main components:

1. **MultiResUNet**: Identifies CT slices intersecting the L1 and L3 vertebrae.
2. **UNet for L1 Segmentation**: Segments the L1 vertebra from the CT slice at the L1 spinal level into spongiosa tissue (spun) and cortical tissue (cort).
3. **UNet for L3 Segmentation**: Segments the CT slice at the L3 spinal level into visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), and skeletal muscle area (SMA).

### Model Implementations

- **MultiResUNet** is based on the implementation by Ibtehaz and Sohel Rahman described in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608019302503?via%3Dihub).
- **UNet** models are based on the implementation by Ronneberger et al. detailed in [this work](https://arxiv.org/pdf/1505.04597.pdf).

All models are developed using **TensorFlow 2**.

[//]: # (L3 U-net weights are available at this [link](https://drive.google.com/file/d/1wUEumfrXRGFBlY6pT9z1NB1Eg9_Ni2UT/view?usp=share_link).)

### Citation

Please cite the following [paper](https://arxiv.org/) when using CompositIA:

    bla bla

## Installation Instructions

### Creating a Virtual Environment

First, create a virtual environment with Anaconda:

```bash
conda create --name comp --file requirements.txt
conda activate comp
```