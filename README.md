# compositIA

Fully automatic system to calculate body composition from toraco-abdominal CT scans.

compositIA consists of three blocks:

* CNN to predict the position of L1 and L3.
* U-net to segment the L1 vertebra from the CT slice at the L1 spinal level. The L1 segmentation is composed of two different regions: spungiosa tissue (spun) and cortical tissue (cort).
* U-net to segment the CT slice at the L3 spinal level in the following regions: visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), skeletal muscle area (SMA).

**U-Net** are based on the implementation proposed by Ronneberger et al. `<https://arxiv.org/pdf/1505.04597.pdf>` developed with **Tensorflow 2**. 

L3 U-net weights are available at this [link](https://drive.google.com/file/d/1wUEumfrXRGFBlY6pT9z1NB1Eg9_Ni2UT/view?usp=share_link).

## Python environment

```
conda create --name tf --file requirements.txt
```
