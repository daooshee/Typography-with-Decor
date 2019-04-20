# Typography-with-Decor
<img src="https://github.com/daooshee/Typography-with-Decor/blob/master/example.jpg" width="800" alt="Example"/>

**Typography with Decor: Intelligent Text Style Transfer** <br>
[Wenjing Wang](https://daooshee.github.io/website/), [Jiaying Liu](http://icst.pku.edu.cn/struct/people/liujiaying.html), [Shuai Yang](http://www.icst.pku.edu.cn/struct/people/yangs/index.html), and [Zongming Guo](http://www.icst.pku.edu.cn/vip/people-guozm.html) <br>
CVPR, 2019

## Setup

### Prerequisites

* Python 3 (Python 2 also works, but needs small revision.)

* Pytorch

* sklearn

* matplotlib

* opencv-python

### Getting Started

* Clone this repo:
```
git clone https://github.com/daooshee/Typography-with-Decor.git
cd Typography-with-Decor
```

* Download pre-trained models from [Google Drive](https://drive.google.com/open?id=1Y0ujOSF1KUepPZ7cIYGmJg04iPMbiywJ).

   Save `netG.pth` to `pre-trained/netG.pth`
   
   Save `netD.pth` to `pre-trained/netD.pth`
   
   Save `netSeg.pth` to `Segmentation/netSeg.pth`
   
* Download datasets from [Google Drive](https://drive.google.com/open?id=1eNYlxpGY7bU8nd36rlu12gkRTGSjxtKu).

   Save and unzip `Decoration.zip` to `Data/Decoration/`. These are 4k icons collected from www.shareicon.net, used as decors.
   
   Save and unzip `TextEffects.zip` to `Data/TextEffects/`. It contains 60 different kinds of text effects with 52 English letters of 19 fonts, totally 59k images.
   
   These two parts are required by the one-shot fine-tuning process.

### Quick Testing
* Some examples (teasor and Fig. 14 in the paper) are provided in `Input/`. For decor segmentation + one-shot fine-tuning + basal text effect transfer + decor recomposition, simply run: <br>
```
python runtest.py
```

* Results can be found in `result/`. 

* Variables such as gpu id, batch size, training iteration for one-shot fine-tuning can be setted in `runtest.py`.

### Testing Custom Images
* For custom images, save target glyph image in `Input/content/`, save input style/glyph image pairs in `Input/style/`. The input glyph image should be named with `_glyph.png`.

* For pre-processing the glyph image, using the following commands with Matlab: 
```
x = imread('input.png'); 

I = im2bw(x(:,:,1));
I2 = bwdist(I, 'euclidean');
I3 = bwdist(~I, 'euclidean');
x(:,:,3) = min(255, I2);
x(:,:,2) = min(255, I3);

imwrite(x,'result.png');
```
* Write the names of your custom images in `runtest.py`.

* Run `python runtest.py`.

### Training Basal Text Effect Transfer

* If you are interested in training from the start, for training basal text effect transfer, first download datasets from [Google Drive](https://drive.google.com/open?id=1eNYlxpGY7bU8nd36rlu12gkRTGSjxtKu). 

   Save and unzip `Decoration.zip` to `Other/BasalTextEffectTransfer/Data/Decoration`. 
   
   Save and unzip `TextEffects.zip` to `Other/BasalTextEffectTransfer/Data/TextEffects`.
   
* `cd Other/BasalTextEffectTransfer/`.

* Run `python train.py --gpu 0 `. More parameters can be found by `python train.py --help`.

### Training Decor Segmentation

* For training decor segmentation, first download datasets from [Google Drive](https://drive.google.com/open?id=1eNYlxpGY7bU8nd36rlu12gkRTGSjxtKu).

   Save and unzip `Decoration.zip` to `Other/DecorSegmentation/Data/Decoration`.
   
   Save and unzip `TextEffects.zip` to `Other/DecorSegmentation/Data/TextEffects`.
   
   Save and unzip `Texture.zip` to `Other/DecorSegmentation/Data/Texture`.
   
   Save and unzip `WildDataMask.zip` to `Other/DecorSegmentation/Data/WildDataMask`. 
   

* `cd Other/DecorSegmentation/`.

* First train with with L1 loss for 30 epochs with a mini-batch size of 90:
```
python train_domain_adaptation.py --gpu 0 --batchSize 90 --niter 17000 --use_decay
```

* Then train with L1 loss and perceptual loss for 50 epochs with a mini-batch size of 15:
```
python train_domain_adaptation.py --gpu 0 --batchSize 15 --niter 170000 --use_decay --use_perceptual
```  

* Finally train with full loss for 5000 iterations with a mini-batch size of 10.
```
python train_domain_adaptation.py --gpu 0 --batchSize 10 --niter 5000 --use_decay --use_perceptual --domain_adaptation
```  

## Citation

If you use this code for your research, please cite our paper:
```
@InProceedings{typography2019,
author = {Wang, Wenjing and Liu, Jiaying and Yang, Shuai and Guo, Zongming},
title = {Typography with Decor: Intelligent Text Style Transfer},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
