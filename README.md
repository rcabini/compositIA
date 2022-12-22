# compositIA

Fully automatic system to calculate body composition from toraco-abdominal CT scans.

compositIA consists of three blocks:

* CNN to predict the position of L1 and L3.
* U-net to segment the L1 vertebra from the CT slice at the L1 spinal level.
* U-net to segment the CT slice at the L3 spinal level in the following regions: visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), skeletal muscle area (SMA).

L3 U-net weights are available at this (link)[https://drive.google.com/file/d/1s577ebXqz8-1Ymnr6tDrDGXHvPKAuCp4/view?usp=share_link].
