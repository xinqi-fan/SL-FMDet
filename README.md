# A Deep Learning based Light-weight Face Mask Detector to Fight Against COVID-19

by Xinqi Fan, Mingjie Jiang, and Hong Yan

## Introduction
This repository is for our paper A Deep Learning based Light-weight Face Mask Detector with Residual Context Attention and Gaussian Heatmap to Fight Against COVID-19. 

Coronavirus disease 2019 has seriously affected the world. One major protective measure for individuals is to wear masks in public areas. In this paper, we propose a single-shot light-weight face mask detector (SL-FMDet) to meet the low computational requirements for embedded systems, as well as achieve high performance. To cope with the low feature extraction capability caused by the light-weight model, we propose two novel methods to enhance the model's feature extraction process. First, to extract rich context information and focus on crucial face masks related regions, we propose a novel residual context attention module. Second, in order to learn more discriminated features for faces with and without masks, we introduce a novel auxiliary task as synthesized Gaussian heatmap regression.

![](https://github.com/xinqi-fan/Face-Mask-Detection/blob/main/figure/pipeline.png)
Figure. Pipeline of SL-FMDet

## Usage
### Requirement
Python 3.6

PyTorch 1.6

Pandas 1.2.3


### Download
Clone the repository:
```
git clone https://github.com/xinqi-fan/Face-Mask-Detection.git
cd Face-Mask-Detection
```

### Prepare data

* Please download AIZOO dataset, or Moxa3K dataset.
* Modify the path to your dataset in the following codes, run them to generate the corresponding heatmaps, and place them at the same folder as where train and test data placed.

```
python tools/heatmap_gaussian_aizoo.py OR tools/heatmap_gaussian_moxa.py
```

### Train the model

* Download [ImageNet pretrained weights](https://drive.google.com/file/d/1BODjD9TtoXtGrna5dc-63GbpmMySW416/view?usp=sharing), and place it in the weight folder.
* Run the following code (More settings if not available at input arguments can be found in data/config.py.).

```
python train_valid_mask_heatmap.py --dataset_choice AIZOO/Moxa3K --dataset_root PATH_TO_DATASET
```

### Test the model

* Set the weights from trained model and run the following code.

```
python test_mask_heatmap.py --dataset_choice AIZOO/Moxa3K --trained_model PATH_TO_WEIGHT
```
### Evaluate the model

* Modified the path inside evaluation_mAP.py file and run

```
python evaluation_mAP.py
```

## Result

![](https://github.com/xinqi-fan/Face-Mask-Detection/blob/main/figure/result.png)

Figure. Result demonstration


## Citation
To be updated

## Comment
We welcome any pull request for improving the code.

## Contact
Please contact "xinqi.fan@my.cityu.edu.hk".
