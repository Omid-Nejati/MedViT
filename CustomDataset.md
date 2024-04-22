# Usage

First, clone the repository:
```
git clone https://github.com/Omid-Nejati/MedViT
```
Then, install required packages.

```
pip install -r requirements.txt
```
## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

#### Training

To train MedViT-small on ImageNet  using 8 gpus for 300 epochs, run:

```shell
cd CustomDataset/
bash train.sh 8 --model MedViT_small  --batch-size 30 --lr 5e-4 --warmup-epochs 20 --weight-decay 0.1 --data-path your_data_path
```
Finetune MedViT-small with 384x384 input size for 30 epochs, run:
```shell
cd CustomDataset/
bash train.sh 8 --model MedViT_small --batch-size 30 --lr 5e-6 --warmup-epochs 0 --weight-decay 1e-8 --epochs 30 --sched step --decay-epochs 60 --input-size 384 --resume ../checkpoints/MedViT_small_im1k.pth --finetune --data-path your_data_path 

```

#### Evaluation 

To evaluate the performance of MedViT-small on ImageNet using 8 gpus, run:
```shell
cd CustomDataset/
bash train.sh 8 --model MedViT_small --batch-size 30 --lr 5e-4 --warmup-epochs 20 --weight-decay 0.1 --data-path your_data_path --resume ../checkpoints/MedViT_small_im1k.pth --eval
```
