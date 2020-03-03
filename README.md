# Papers
|         Name         | Published |          Method            |          Comments          |
| :------------------: | :-------: | :------------------------: | :------------------------: |
| Blind Image Super-Resolution with Spatially Variant Degradations | 2019/SIGA | Use a generator network to synthesize the HR given a LR and the blur kernel, a kernel discriminator to analyze the error in generated HR, an optimization procedure to recover the kernel and the HR by minimizing the error| Estimate spatially varying kernels to handle images with composited content |
| Blind Super-Resolution With Iterative Kernel Correction | 2019/ICCV | Proposed an Iterative Kernel Correction (IKC) framework consists of a SR network given a LR and the blur kernel, a kernel predictor given a LR, and a corrector to estimate kernel error given the HR result and the current kernel | Use an iterative method to correct the kernel gradually and achieve blur kernel estimation in blind SR problem |
| Learning a Single Convolutional Super-Resolution Network for Multiple Degradations | 2018/CVPR | Propose a general framework with dimensionality stretching strategy that enables a SR network to take blur kernel and noise level as input | Consider the kernel as input, but do not estimate the kernel |
| “Zero-Shot” Super-Resolution using Deep Internal Learning | 2018/CVPR | Train a small image-specific CNN at test time on examples extracted solely from the input image | Unsupervised SR methods, flexible dataset for training, image-specific kernel is estimated from the LR test image using other method, can handle different imaging conditions |
| Meta-Transfer Learning for Zero-Shot Super-Resolution | 2020/CVPR | A three stages training scheme: a large-scale training with bicubic degradation data, a meta-transfer learning with diverse blur kernels data, and a self-supervision meta-test phase | Generate result with a few gradient descent update |
# Network Strategies
## Input the kernel
- Learn a mapping that map the vectorized kernel to a low dimensional representation and assemble it at each pixel or region to obtain the kernel maps
- Reduce the dimensionality of the kernel space by principal component analysis (PCA) and stretch the result into kernel maps

- Predict a residual image that is then added to a bicubicly upsampled image to produce the output
- spatial feature transform (SFT) layer: use the kernel maps to predict affine transformation for the input feature maps by a scaling and shifting operation
- sub-pixel convolution layer: Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.

## Learning
Perceptual Losses for Real-Time Style Transfer and Super-Resolution
- feature loss for preserving image content and overall spatial structure, but not color, texture
- style loss for preserving stylistic features but not spatial structure

Decouple Learning for Parameterized Image Operators
- use a weight learning network to adaptively predict the weights of the base network

Dynamic Convolution: Attention over Convolution Kernels
- dynamic convolution: aggregates multiple convolution kernels dynamically based upon the attentions
