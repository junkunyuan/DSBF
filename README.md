# DSBF

## Introduction
This repository contains the implementation code for paper:

<FONT size=10>**Domain-Specific Bias Filtering for Single Labeled Domain Generalization**</FONT>

Junkun Yuan, Xu Ma, Defang Chen, Kun Kuang, Fei Wu, Lanfen Lin

*arXiv preprint, 2021*

[[arXiv](https://arxiv.org/abs/2110.00726)]

## Brief Abstract for the Paper
<p align="center">
    <img src="framework.png" width="900"> <br>
</p>

Domain generalization (DG) utilizes multiple labeled
source datasets to train a generalizable model for unseen target
domains. However, due to expensive annotation costs, the requirements
of labeling all the source data are hard to be met in real-world applications. 

We investigate a Single Labeled Domain Generalization (SLDG) task with only one source domain being labeled, which is more practical and challenging than the Conventional Domain Generalization (CDG). A major obstacle in the SLDG task is the discriminability-generalization bias: discriminative information in the labeled source dataset may contain domain-specific bias, constraining the generalization of the trained model. 

To tackle this challenging task, we propose Domain-Specific Bias Filtering (DSBF), which initializes a discriminative model with the labeled source data and filters out its domain-specific bias with the unlabeled source data for generalization improvement. We divide the filtering process into (1) feature extractor debiasing using k-means clustering-based semantic feature re-extraction and (2) classifier calibrating using attention-guided semantic feature projection.

## Requirements
You may need to build suitable Python environment by installing the following packages (Anaconda is recommended).
* python 3.6
* pytorch 1.7.1 (with cuda 11.0 and cudnn 8.0)
* torchvision 0.8.2
* tensorboardx 2.1
* numpy 1.19

Device:
* GPU with VRAM >5GB (strictly).
* Memory >8GB.

## Data Preparation
We list the adopted datasets in the following.

| Datasets | Download link|
| :-: | :- |
| PACS [1]</a> | https://dali-dl.github.io/project_iccv2017.html |
| Office-Home [2] | https://www.hemanthdv.org/officeHomeDataset.html | 
|Office-Caltech-Home [2, 3] | https://people.eecs.berkeley.edu/~jhoffman/domainadapt & https://www.hemanthdv.org/officeHomeDataset.html|

Please note:
- Office-Caltech-Home dataset is constructed by choosing the common classes from Office-Caltech [3] and Office-Home [2] datasets, please see our paper for more details.
- Our dataset split follows previous works like RSC ([Code](https://github.com/DeLightCMU/RSC)) [4].
- Although these datasets are open-sourced, you may need to have permission to use the datasets under the datasets' license. 
- If you're a dataset owner and do not want your dataset to be included here, please get in touch with us via a GitHub issue. Thanks!

## Usage
1. Prepare the datasets. 
2. Update the .txt files under folder "DSBF/dataset/pthList/" with your dataset path.
3. Run the code with command: 
```
nohup sh run.sh > run.txt 2>&1 &
```
4. Check results in DSBF/**dataset**-**task**-**target-data**.txt.

## Updates
- [12/27/2021] We uploaded a new arXiv version. See [new arXiv version](https://arxiv.org/abs/2110.00726).


## Citation
If you find our code or idea useful for your research, please cite our work.
```bib
@article{yuan2021domain,
  title={Domain-Specific Bias Filtering for Single Labeled Domain Generalization},
  author={Yuan, Junkun and Ma, Xu and Chen, Defang and Kuang, Kun and Wu, Fei and Lin, Lanfen},
  journal={arXiv preprint arXiv:2110.00726},
  year={2021}
}
```

## Contact
If you have any questions, feel free to contact us through email (yuanjk@zju.edu.cn or maxu@zju.edu.cn) or GitHub issues. Thanks!

## References
[1] Li, Da, et al. "Deeper, broader and artier domain generalization." Proceedings of the IEEE international conference on computer vision. 2017.

[2] Venkateswara, Hemanth, et al. "Deep hashing network for unsupervised domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[3] Saenko, Kate, et al. "Adapting visual category models to new domains." European conference on computer vision. Springer, Berlin, Heidelberg, 2010.

[4] Huang, Zeyi, et al. "Self-challenging improves cross-domain generalization." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer International Publishing, 2020.