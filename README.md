# CycleGAN_ssim

This project is an extension of the project [Image Editing using GAN](https://github.com/Tandon-A/Image-Editing-using-GAN). 

Cycle Consistent Generative Adversarial Networks (CycleGAN) as described in the [paper](https://arxiv.org/abs/1703.10593) have been implemented with [SSIM loss](https://arxiv.org/abs/1511.08861) to produce images of better visual quality. The code is developed in tensorflow.


<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/CycleGAN_working.png" width="600" alt="CycleGAN model">

## Prerequisites

* Python 3.3+
* Tensorflow 
* pillow (PIL)
* matplotlib 
* (Optional) [Monet-Photo Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)


## Results 
Trained CycleGAN model on Monet-Photo Database.

### Comparison with L1 Loss

#### Photo to Monet Paintings

| Input Image | Output Image - L1 | Ouput Image - SSIM | Input Image | Output Image - L1 | Ouput Image - SSIM |
|:-----------:|:-----------------:|:------------------:|:-----------:|:-----------------:|:------------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/org/orgB2.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/l1/monetB2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/ssim/monetB2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/org/orgB5.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/l1/monetB5.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/ssim/monetB6.png) |




#### Monet Paintings to Photo 

| Input Image | Output Image - L1 | Ouput Image - SSIM | Input Image | Output Image - L1 | Ouput Image - SSIM |
|:-----------:|:-----------------:|:------------------:|:-----------:|:-----------------:|:------------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/org/orgA2.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/l1/realA2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/ssim/realA2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/org/orgA9.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/l1/realA9.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/ssim/realA9.png) |




### Generated Samples

#### Photo to Monet Paintings

| Input Image | Output Image | Input Image | Output Image | Input Image | Output Image |
|:-----------:|:------------:|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/orgB22.jpg)  |  ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/monetB22.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/orgB25.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/monetB25.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/orgB27.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/monetB27.png) 


#### Monet Paintings to Photo 

| Input Image | Output Image | Input Image | Output Image | Input Image | Output Image |
|:-----------:|:------------:|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/orgA7.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/realA7.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/orgA15.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/realA15.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/orgA27.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/realA27.png) 



## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/LICENSE) file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
