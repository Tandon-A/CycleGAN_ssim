# Similarity Functions 

Generative Deep Learning Models such as the Generative Adversarial Networks (GAN) are used for various image manipulation problems such as scene editing (removing an object from an image) , image generation, style transfer, producing an image as the end result. 

To improve the quality of these generated images it is important to use an objective function (loss function) which is better suited to human perceptual judgements. In this post, I would present a brief overview of different loss functions used for this task. 

## The Notorious L2 Loss 

According to error sensitivity theory, a distorted image is seen as a sum of the undistorted image and an error signal. 

| Undistorted Image | Error Signal | Distorted Image |
|:-----------------:|:------------:|:---------------:|
<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ggcolab.png" width = "300" alt="Undistorted Image"> |  <img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/gg_noise.png" width = "300" alt="Error Signal">  | <img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/gg_dist.png" width = "300" alt="Distorted Image"> |

###### Figure 2: Distorted Image = Undistorted Image + Error Signal 

Loss in quality of an image is thus assumed to be related to the visibility of the error signal. L2 loss tries to quantify this error signal by taking the mean of squared difference between intensities (pixel values) of the distorted and the undistorted image. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;L(X,Y)&space;=&space;||X&space;-&space;Y||_{2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;L(X,Y)&space;=&space;||X&space;-&space;Y||_{2}" title="L(X,Y) = ||X - Y||_{2}" /></a>

###### Formula 1: L2 loss 

L2 loss has been the de facto standard for the industry for quite a long time now and this is mainly due to the following reasons - 
* It is simple and inexpensive to compute. 
* Additive in nature; error due to independent distortions can be added together. 
* It is convex, symmetric and differentiable, making it a good metric in the context of optimization.
* Provides the Maximum Likelihood estimate when noise is assumed to be independent and identically distributed following a    Gaussian distribution. 

Due to its simple structure, researchers have used l2 loss in all types of problems, from regression to image processing.  
L2 loss assumes that pixels or signal points are independent of each other whereas images are highly structured - ordering of the pixels carry important information about the contents of an image. For a given error signal, L2 loss remains the same irrespective of the correlation between the original signal and the error signal even though this correlation can have strong impact on perceptual similarity. 

Failure of these assumptions makes L2 loss an unsuitable candidate to improve the quality of generated images. 

## SSIM Loss 

Loss in quality of an image is thus not only related to the visibility of the error signal. Contrary to the L2 loss, the structural similarity (SSIM) index provides a measure of the similarity by comparing two images based on luminance similarity, contrast similarity and structural similarity information. 

### Formulation:

Luminance of an image signal is estimated by mean intensity. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\mu&space;_{x}&space;=&space;\frac{1}{N}\sum_{i=1}^{N}x_{i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\mu&space;_{x}&space;=&space;\frac{1}{N}\sum_{i=1}^{N}x_{i}" title="\mu _{x} = \frac{1}{N}\sum_{i=1}^{N}x_{i}" /></a>

###### Formula 2: Mean Intensity [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)
Luminance of two images is then compared by

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;l(x,y)&space;=&space;\frac{2\mu&space;_{x}\mu&space;_{y}&space;&plus;&space;C_{1}}{\mu&space;_{x}^{2}&space;&plus;&space;\mu&space;_{y}^{2}&space;&plus;&space;C_{1}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;l(x,y)&space;=&space;\frac{2\mu&space;_{x}\mu&space;_{y}&space;&plus;&space;C_{1}}{\mu&space;_{x}^{2}&space;&plus;&space;\mu&space;_{y}^{2}&space;&plus;&space;C_{1}}" title="l(x,y) = \frac{2\mu _{x}\mu _{y} + C_{1}}{\mu _{x}^{2} + \mu _{y}^{2} + C_{1}}" /></a>
###### Formula 3: Luminance Similarity [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)

Contrast is determined by difference of the luminance between the object and other objects in the field of view. This is done by calculating the standard deviation of the image signal. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\sigma&space;_{x}&space;=&space;(\frac{1}{N-1}&space;\sum_{i=1}^{N}(x_{i}&space;-&space;\mu&space;_{x})^{2})^{\frac{1}{2}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\sigma&space;_{x}&space;=&space;(\frac{1}{N-1}&space;\sum_{i=1}^{N}(x_{i}&space;-&space;\mu&space;_{x})^{2})^{\frac{1}{2}}" title="\sigma _{x} = (\frac{1}{N-1} \sum_{i=1}^{N}(x_{i} - \mu _{x})^{2})^{\frac{1}{2}}" /></a>

###### Formula 4: Contrast of image signal [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)

Contrast similarity is then found out by, 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;c(x,y)&space;=&space;\frac{&space;2\sigma&space;_{x}\sigma&space;_{y}&space;&plus;&space;C_{2}&space;}{&space;\sigma&space;_{x}^{2}&space;&plus;&space;\sigma&space;_{y}^{2}&space;&plus;&space;C_{2}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;c(x,y)&space;=&space;\frac{&space;2\sigma&space;_{x}\sigma&space;_{y}&space;&plus;&space;C_{2}&space;}{&space;\sigma&space;_{x}^{2}&space;&plus;&space;\sigma&space;_{y}^{2}&space;&plus;&space;C_{2}}" title="c(x,y) = \frac{ 2\sigma _{x}\sigma _{y} + C_{2} }{ \sigma _{x}^{2} + \sigma _{y}^{2} + C_{2}}" /></a>

###### Formula 5: Contrast Similarity [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)


Structural information is represented by strong inter-dependencies between spatially close pixels. Normalizing the incoming signals by first subtracting mean intensities and then dividing by respective standard deviation projects the image signals as unit vectors on hyperplanes defined by, 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\sum_{i=1}^{N}x&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\sum_{i=1}^{N}x&space;=&space;0" title="\sum_{i=1}^{N}x = 0" /></a>
###### Formula 6: Hyperplane [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)

These unit vectors are associated with the structural information.

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;s(x,y)&space;=&space;\frac{\sigma&space;_{xy}&space;&plus;&space;C_{3}}{&space;\sigma&space;_{x}\sigma&space;_{y}&space;&plus;&space;C_{3}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;s(x,y)&space;=&space;\frac{\sigma&space;_{xy}&space;&plus;&space;C_{3}}{&space;\sigma&space;_{x}\sigma&space;_{y}&space;&plus;&space;C_{3}}" title="s(x,y) = \frac{\sigma _{xy} + C_{3}}{ \sigma _{x}\sigma _{y} + C_{3}}" /></a> ,

where 

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma&space;_{xy}&space;=&space;\frac{1}{N-1}\sum_{i=1}^{N}(x_{i}&space;-&space;\mu&space;_{x})(y_{i}&space;-&space;\mu&space;_{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma&space;_{xy}&space;=&space;\frac{1}{N-1}\sum_{i=1}^{N}(x_{i}&space;-&space;\mu&space;_{x})(y_{i}&space;-&space;\mu&space;_{y})" title="\sigma _{xy} = \frac{1}{N-1}\sum_{i=1}^{N}(x_{i} - \mu _{x})(y_{i} - \mu _{y})" /></a>, gives the corelation between the two windows x and y. 
###### Formula 7:  Structural similarity [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)

SSIM Index is computed by taking into account the luminance, contrast and structural similarity. The constants C1, C2 and C3 are used to resolve cases where in denominator is tending to zero. 

<a href="https://www.codecogs.com/eqnedit.php?latex=SSIM(x,y)&space;=&space;l(x,y)&space;*&space;c(x,y)&space;*&space;s(x,y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SSIM(x,y)&space;=&space;l(x,y)&space;*&space;c(x,y)&space;*&space;s(x,y)" title="SSIM(x,y) = l(x,y) * c(x,y) * s(x,y)" /></a>

###### Formual 8: SSIM [[1]](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)

SSIM index is calculated as a local measure rather than a global measure in order to incorporate the fact that the human visual system (HVS) can perceive only a local area at high resolution at a time. In the above formulas x and y are windows on the full images X and Y (test/predicted image and reference image) 

In comparison to L2 loss, SSIM index is a better image quality measure as it is better suited to the HVS. The following figure shows that SSIM index varies for different distortions while L2 Loss remains constant, showing superiority of SSIM index over L2 loss. 

<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/comp.png" width = "400" alt="Undistorted Image">

###### Figure 3: Comparison of SSIM Index with L2 Loss as similarity measure [[2]](https://ieeexplore.ieee.org/document/4775883)

To take into account the scale at which local structure of an image is analyzed, researchers came up with a multi scale version of SSIM index, MS-SSIM. This is calculated by computing the SSIM index at various scales and then taking a weighted average of the computed values.

SSIM loss given by, 1 - SSIM Index, is used as the objective function for DL models. 

While SSIM loss may seem more suitable as compared to L2 loss, it was designed for grayscale images and sometimes fails in estimating quality of color images. Training DL models with SSIM loss can lead to shift of colors. 

To overcome this issue of SSIM loss, neural nets can be trained with a combination of different losses. 

<a href="https://www.codecogs.com/eqnedit.php?latex=L^{mix}&space;=&space;\alpha&space;L^{SSIM}&space;&plus;&space;(1-&space;\alpha)G_{\sigma&space;_{G}^{M}}L^{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L^{mix}&space;=&space;\alpha&space;L^{MS-SSIM}&space;&plus;&space;(1-&space;\alpha)G_{\sigma&space;_{G}^{M}}L^{1}" title="L^{mix} = \alpha L^{SSIM} + (1- \alpha)G_{\sigma _{G}^{M}}L^{1}" /></a>

###### Formula 9:  Combined Loss Formula [[3]](https://arxiv.org/pdf/1511.08861.pdf)

Here <a href="https://www.codecogs.com/eqnedit.php?latex=G_{\sigma&space;_{G}^{M}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G_{\sigma&space;_{G}^{M}}" title="G_{\sigma _{G}^{M}}" /></a> is a Guassian Window and <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is a small number (0.84) to weight the different loss functions involved. Guassian window is used to make L1 loss consistent with the MS-SSIM loss, in which the error at pixel q is propogated based on its contribution to MS-SSIM of the central pixel. 


|Input Image |SSIM Image|SSIM + L1 |SSIM + L2|
|:----------:|:--------:|:--------:|:-------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/org5.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re5_ssim.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re5_ssiml1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re5_ssiml2.png) |
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/org7.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re7_ssim.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re7_ssiml1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re7_ssiml2.png) |
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/org8.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re8_ssim.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re8_ssiml1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_comp/re8_ssiml2.png) |

###### Figure 4: SSIM Color Change (Reconstructions of Input Image produced by CycleGAN model trained on Monet-Photo Database). A variation is observed in the sky color in the SSIM reconstruction of the second input image. 

## Deep Features as Perceptual Metric 

SSIM loss has been well adopted in the industry but it has its own limitations. SSIM loss cannot incorporate large geometric distortions. 

<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_fail.png" width = "500" alt="Undistorted Image">

###### Figure 5: SSIM Index Failure [[4]](https://arxiv.org/abs/1801.03924)
Finding a linear function which fits the human similarity measure is an onerous task due to the complex context-dependent nature of human judgement. 

In light of this problem, researchers in the community have used distance between features of images passed through DL models (VGG, Alex) as a similarity measure between images. Distance between the features is generally calculated as l2 distance or cosine distance. 

<a href="https://www.codecogs.com/eqnedit.php?latex=L_{P}(x,y)&space;=&space;||\phi&space;^{l_{p}}(x)&space;-&space;\phi&space;^{l_{p}}(y)||_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{P}(x,y)&space;=&space;||\phi&space;^{l_{p}}(x)&space;-&space;\phi&space;^{l_{p}}(y)||_{1}" title="L_{P}(x,y) = ||\phi ^{l_{p}}(x) - \phi ^{l_{p}}(y)||_{1}" /></a> 

where <a href="https://www.codecogs.com/eqnedit.php?latex=\phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /></a> is the pretrained DL model (VGG or Alex) and <a href="https://www.codecogs.com/eqnedit.php?latex=l_{p}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{p}" title="l_{p}" /></a> represents the network layer. 
###### Formula 10: Perceptual Loss [[5]](https://arxiv.org/abs/1803.02077)


<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/Perceptual.png" width = "350" alt="Perceptual Metric">

###### Figure 6: Perceptual Loss

Another loss metric is the recently proposed contextual loss which also measures distances between features computed using DL models. While calculating this metric, features of the reference image and the predicted/test image  are treated as two separate collections. For every feature in the reference image feature collection, a similar feature is found amongst the test image feature collection. This is done by calculating the cosine distance between the two feature vectors and then converting that to a similaity value by exponentiating it. Maximum value of this metric is then taken as the similarity value of the two images. Authors of this [research paper](https://arxiv.org/abs/1803.02077) show promising results on using the contextual loss parameter. 


## Comparison of different loss functions

I trained CycleGAN [[6]](https://arxiv.org/abs/1703.10593) model on Monet-Photo database with different loss functions used for calculating the cycle consistency loss. Some sample comparisons are provided below. The project is available [here](https://github.com/Tandon-A/CycleGAN_ssim). 


<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/img_proj.png" width="400" alt="Project Wroking">

###### Figure 7: Project Details


#### Photo to Monet Paintings

|Input Image |L1 Image |SSIM Image |SSIM + L1 |SSIM + L2(a) |SSIM + L2(b) |SSIM + L1 + L2(b)|
|:----------:|:-------:|:---------:|:--------:|:-----------:|:-----------:|:---------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/org/orgB2.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/p2m/ex1.png) |
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/org/orgB5.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/p2m/ex2.png) |


#### Monet to Photo Paintings

|Input Image |L1 Image |SSIM Image |SSIM + L1 |SSIM + L2(a) |SSIM + L2(b) |SSIM + L1 + L2(b)|
|:----------:|:-------:|:---------:|:--------:|:-----------:|:-----------:|:---------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/org/orgA2.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/m2p/ex1.png) |
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/org/orgA9.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/m2p/ex2.png) |



P.S. - I am implementing perceptual loss using deep features metric in this project. Stay tuned for that. 


## References 


1. Wang, Zhou, et al. "Image quality assessment: from error visibility to structural similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
2. Wang, Zhou, and Alan C. Bovik. "Mean squared error: Love it or leave it? A new look at signal fidelity measures." IEEE signal processing magazine 26.1 (2009): 98-117.
3. Zhao, Hang, et al. "Loss functions for image restoration with neural networks." IEEE Transactions on Computational Imaging 3.1 (2017): 47-57.
4. Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." arXiv preprint (2018).
5. Mechrez, Roey, Itamar Talmi, and Lihi Zelnik-Manor. "The contextual loss for image transformation with non-aligned data." arXiv preprint arXiv:1803.02077 (2018).
6. Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." arXiv preprint (2017).
