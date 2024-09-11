# CompositIA

CompositIA is a fully automated tool designed to calculate body composition from thoraco-abdominal computed tomography (CT) scans.

![](pipeline.png)

**CompositIA** consists of three blocks:

* **`MultiResUNet`** to predict CT slices intersecting the first lumbar vertebra (L1) and third lumbar vertebra (L3).
* **`UNetL1`** to segment the L1 vertebra from the CT slice at the L1 spinal level. The L1 segmentation is composed of two different regions: trabecular bone tissue and cortical tissue.\
**`UNetL3`** to segment the CT slice at the L3 spinal level in the following regions: visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), skeletal muscle area (SMA).
* Quantification of body composition indices.

**`MultiResUNet`** is based on the implementation proposed by Ibtehaz, and Sohel Rahman described in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608019302503?via%3Dihub). **`UNet`** is based on the implementation proposed by Ronneberger et al. detailed in the [work](https://arxiv.org/pdf/1505.04597.pdf). 
All the models are developed using **Tensorflow 2**. 

A web application with a user-friendly interface is available [here](http://www.dp-lab.io/compositia), allowing users to upload CT scans, run the analysis, and view results directly in the browser. A standalone Windows version is available for download from this link [link](http://www.dp-lab.io/compositia).

[//]:<Please cite the following [paper](https://arxiv.org/) when using CompositIA:>
  
## Installation instructions

First, clone the CompositIA repository:
```bash
git clone https://github.com/rcabini/compositIA.git
```
Create a virtual environment with Anaconda:
```bash
conda env create -f environment.yml
conda activate environment
```

## Train CompositIA

### Data preparation
CompositIA supports data in the form of 3D NIfTI images (files ending in `.nii.gz`). For DICOM to NIfTI conversion, we recommend using ITK-Snap. Organize the NIfTI files of both images and segmentations as follows: 

    Dataset/Images/
    ├── 001
    |   ├── data.nii.gz
    ├── 002
    |   ├── data.nii.gz
    ├── 003
    |   ├── data.nii.gz
    ├── ...
    
    Dataset/Segmentations/
    ├── 001
    |   ├── segmentation.nii.gz
    ├── 002
    |   ├── segmentation.nii.gz
    ├── 003
    |   ├── segmentation.nii.gz
    ├── ...

CompositIA was trained and tested by using a k-fold cross-validation strategy. To create the train-test splits, run:

    cd utils
    python k-fold_generator.py --data_folder path_to_Dataset/ --output_folder path_to_Dataset/ --k 5

This command generates the `k-fold-test.txt` file, which lists the filenames of the test set for each k-fold. Replace `path_to_Dataset` with the actual path to the Dataset folder. The default number of folds (`k`) is 5.

To create dataset necessary to train the CompositIA tool you sholud create three different dataset for the three models:

* To create dataset to train the L1/L3 localization model, please run:
```bash
  cd slicer/
  python data_generator.py --data_folder path_to_Dataset/ --output_folder path_to_Dataset/slicer/ --ktxt path_to_Dataset/k-fold-test.txt
```
* To create dataset to train the L1 segmentation model, please run:
```bash
  cd L1scripts/
  python data_generator.py --data_folder path_to_Dataset/ --output_folder path_to_Dataset/L1/
```
* To create dataset to train the L3 segmentation model, please run:
```bash
  cd L3scripts/
  python data_generator.py --data_folder path_to_Dataset/ --output_folder path_to_Dataset/L3/
```
where `path_to_Dataset` should be replaced with the path to the Dataset folder.

### Training on k-folds
To to train the CompositIA tool you sholud train the three different models:

* L1/L3 localization model training, run:
```bash
  cd slicer/
  python main_slicer_CV.py --data_folder path_to_Dataset/slicer/ --weights_folder ./weights_slicer/ --ktxt path_to_Dataset/k-fold-test.txt
```
* L1 segmentation training, run:
```bash
  cd L1scripts/
  python main_L1_CV.py --data_folder path_to_Dataset/L1/ --weights_folder ./weights_L1/ --ktxt path_to_Dataset/k-fold-test.txt
```
* L3 segmentation training, run:
```bash
  cd L3scripts/
  python main_L3_CV.py --data_folder path_to_Dataset/L3/ --weights_folder ./weights_L3/ --ktxt path_to_Dataset/k-fold-test.txt
```

### Testing on k-folds
To to test the CompositIA tool on the k-folds you sholud test the three different models:

* L1/L3 localization model testing, run:
```bash
  cd slicer/
  python run_slicer_CV.py --data_folder path_to_Dataset/slicer/ --weights_folder ./weights_slicer/ --output_folder ./results_slicer/ --ktxt path_to_Dataset/k-fold-test.txt --nifti_folder ./path_to_Dataset/
```
* L1 segmentation testing, run:
```bash
  cd L1scripts/
  python run_L1_CV.py --data_folder path_to_Dataset/L1/ --weights_folder ./weights_L1/ --output_folder ./results_L1/ --ktxt path_to_Dataset/k-fold-test.txt
```
* L3 segmentation testing, run:
```bash
  cd L3scripts/
  python run_L3_CV.py --data_folder path_to_Dataset/L3/ --weights_folder ./weights_L3/ --output_folder ./results_L3/ --ktxt path_to_Dataset/k-fold-test.txt
```

## Run CompositIA on a new dataset

To run the complete CompositIA tool on a new thoraco-abdominal CT scan, you should run:

    python CompositIA.py --input_path path_to_input/data.nii.gz 
                         --output_folder path_to_output/
                         --weights_slicer path_to_weights_slicer/weights.hdf5
                         --weights_L1 path_to_weights_L1/weights.hdf5
                         --weights_L3 path_to_weights_L3/weights.hdf5

where `path_to_input` is the path to the input NIfTI CT and `path_to_output` is the path to the output directory where all the results will be saved. The output directory includes: 

* `scores.json`: contains the body composition indices computed by CompositIA;
* `planes.png`: highlights the predicted position of L1 and L3 planes;
* `L1slice.png` and `L3slice.png`: extracted slices of the CT scan;
* `L1segmentation.png` and `L3segmentation.png`: segmentations predicted by CompositIA. 

Replace the paths with the appropriate paths to the weight files. Weights of all the pretrained models are available [here](http://www.dp-lab.io/compositia/proc_2024/weights). 

To run the complete CompositIA tool by using custom weights replace the paths with the appropriate paths to your custom weight files.

## License
CompositIA is licensed under the EUPL-1.2 license. See the file LICENSE for more details.
