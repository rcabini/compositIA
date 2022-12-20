# compositIA

Fully automatic system to calculate body composition from CT scans. 

compositIA consists of three blocks:

* CNN to predict the position of L1 and L3.
* U-net to segment L1.
* U-net to segment the CT slice at the L3 spinal level in the following regions: visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), skeletal muscle area (SMA).
