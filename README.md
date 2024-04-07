# MedViT: A Robust Vision Transformer for Generalized Medical Image Classification

This repo is the official implementation of ["MedViT: A Robust Vision Transformer for Generalized Medical Image Classification"](https://arxiv.org/abs/2302.09462).

## Train & Test --- Prepare data
- (beginner friendly) We will soon provide new code with a custom dataset and pre-trained weights on Imagenet.🍉 
- (New version) We have updated the code ["Instructions.ipynb"](https://github.com/Omid-Nejati/MedViT/blob/main/Instructions.ipynb), incorporating the installation requirements and adding a section on adversarial robustness. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omid-Nejati/MedViT/blob/main/Instructions.ipynb)
- (Previous version) Please go to ["Colab_MedViT.ipynb"](https://github.com/Omid-Nejati/MedViT/blob/main/Colab_MedViT.ipynb) for complete detail on dataset preparation and Train/Test procedure. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omid-Nejati/MedViT/blob/main/Colab_MedViT.ipynb)

## Introduction
Convolutional Neural Networks (CNNs) have advanced existing medical systems for automatic disease diagnosis. However, there are still concerns about the reliability of deep medical diagnosis systems against the potential threats of adversarial attacks since inaccurate diagnosis could lead to disastrous consequences in the safety realm. In this study, we propose a highly robust yet efficient CNN-Transformer hybrid model which is equipped with the locality of CNNs as well as the global connectivity of vision Transformers. To mitigate the high quadratic complexity of the self-attention mechanism while jointly attending to information in various representation subspaces, we construct our attention mechanism by means of an efficient convolution operation. Moreover, to alleviate the fragility of our Transformer model against adversarial attacks, we attempt to learn smoother decision boundaries. To this end, we augment the shape information of an image in the high-level feature space by permuting the feature mean and variance within mini-batches. With less computational complexity, our proposed hybrid model demonstrates its high robustness and generalization ability compared to the state-of-the-art studies on a large-scale collection of standardized MedMNIST-2D datasets. 
<div style="text-align: center">
<img src="images/result.png" title="MedViT-S" height="60%" width="60%">
</div>
Figure 1. Comparison between MedViTs and the baseline ResNets, in terms of average ACC-Parameters and average AUC-Parametrs trade-off over all 2D datasets.</center>


## Overview

<div style="text-align: center">
<img src="images/structure.png" title="MedViT-S" height="75%" width="75%">
</div>
Figure 2. The overall hierarchical architecture of MedViT.</center>


## Visualization

Visual inspection of MedViT-T and ResNet-18 using Grad-CAM on MedMNIST-2D datasets. The green rectangles is
used to show a specific part of the image that contains information relevant to the diagnosis or analysis of a medical condition,
where the superiority of our proposed method can be clearly seen.
![MedViT-V](images/visualize.png)
<center>Figure 3. The heat maps of the output feature from ResNet and MedViT.</center>

## Citation
If you find this project useful in your research, please consider cite:
```
@article{manzari2023medvit,
  title={MedViT: A robust vision transformer for generalized medical image classification},
  author={Manzari, Omid Nejati and Ahmadabadi, Hamid and Kashiani, Hossein and Shokouhi, Shahriar B and Ayatollahi, Ahmad},
  journal={Computers in Biology and Medicine},
  volume={157},
  pages={106791},
  year={2023},
  publisher={Elsevier}
}
```

## Acknowledgement
We heavily borrow the code from [RVT](https://github.com/vtddggg/Robust-Vision-Transformer) and [LocalViT](https://github.com/ofsoundof/LocalViT).
