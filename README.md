# SISR
|         Name         | Published |      Method        |          Comments          |
| :------------------: | :-------: | :----------------: | :------------------------: |
| Zoom to Learn, Learn to Zoom | 2019/CVPR | introduce a new dataset, SR-RAW, for super-resolution from raw data, with optical ground truth, propose a novel contextual bilateral loss for training | SR-RAW dataset from real sensor, contextual bilateral loss |
| Towards Real Scene Super-Resolution with Raw Images | 2019/CVPR | propose a method to generate realistic training data by simulating the imaging process, develop a dual network architecture for training | generate realistic training data, dual network |
| Second-order Attention Network for Single Image Super-Resolution | 2019/CVPR | propose SAN with second-order channel attention (SOCA) module that adaptively rescale the channel-wise features and present a non-locally enhanced residual group (NLRG) structure | second-order channel attention module, non-locally enhanced residual group structure | 


# Blind SR
|         Name         | Published |      Method        |          Comments          |
| :------------------: | :-------: | :----------------: | :------------------------: |
| Learning a Single Convolutional Super-Resolution Network for Multiple Degradations | 2018/CVPR | Propose a general framework with dimensionality stretching strategy that enables a SR network to take blur kernel and noise level as input | Consider the kernel as input, but do not estimate the kernel |
| Blind Image Super-Resolution with Spatially Variant Degradations | 2019/SIGA | Use a generator network to synthesize the HR given a LR and the blur kernel, a kernel discriminator to analyze the error in generated HR, an optimization procedure to recover the kernel and the HR by minimizing the error| Estimate spatially varying kernels to handle images with composited content |
| Blind Super-Resolution With Iterative Kernel Correction | 2019/CVPR | Proposed an Iterative Kernel Correction (IKC) framework consists of a SR network given a LR and the blur kernel, a kernel predictor given a LR, and a corrector to estimate kernel error given the HR result and the current kernel | Use an iterative method to correct the kernel gradually and achieve blur kernel estimation in blind SR problem |
| Kernel Modeling Super-Resolution on Real Low-Resolution Images | 2019/ICCV | Generate a GAN-augmented blur kernel pool by extracting real blur kernels with a kernel estimation algorithm, then construct a paired LR-HR training dataset based on the kernel pool | Blur kernel augmentation by GAN |
| “Zero-Shot” Super-Resolution using Deep Internal Learning | 2018/CVPR | Train a small image-specific CNN at test time on examples extracted solely from the input image | Unsupervised SR methods, image-specific kernel is estimated for training dataset, can handle different imaging conditions |
| Meta-Transfer Learning for Zero-Shot Super-Resolution | 2020/CVPR | A three stages training scheme: a large-scale training with bicubic degradation data, a meta-transfer learning with diverse blur kernels data, and a self-supervision meta-test phase | Generate result with a few gradient descent update |

# Exemplar Guided SR
|         Name         | Published |      Method        |          Comments          |
| :------------------: | :-------: | :----------------: | :------------------------: |
| Learning Warped Guidance for Blind Face Restoration | 2018/ECCV | Predict flow to warp the guided image, and then take LR image and warped guidance as input to produce the result. Use landmark loss and total variation regularization for training | Warp the guidance, use landmark loss and tv loss |
| Exemplar Guided Face Image Super-Resolution without Facial Landmarks | 2019/CVPRW | Warp the guided image to align its contents by a subnetwork, train the network in an adversarial generative manner with identity loss | Warp the guidance, use adversarial loss and identity loss |


# Network Strategies
- Learn a mapping that map the vectorized kernel to a low dimensional representation and assemble it at each pixel or region to obtain the kernel maps
- Reduce the dimensionality of the kernel space by principal component analysis (PCA) and stretch the result into kernel maps

- Predict a residual image that is then added to a bicubicly upsampled image to produce the output
- spatial feature transform (SFT) layer: use the kernel maps to predict affine transformation for the input feature maps by a scaling and shifting operation

Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network
- sub-pixel convolution layer

Perceptual Losses for Real-Time Style Transfer and Super-Resolution
- feature loss for preserving image content and overall spatial structure, but not color, texture
- style loss for preserving stylistic features but not spatial structure

Decouple Learning for Parameterized Image Operators
- use a weight learning network to adaptively predict the weights of the base network

Dynamic Convolution: Attention over Convolution Kernels
- dynamic convolution: aggregates multiple convolution kernels dynamically based upon the attentions

Deep Network Interpolation for Continuous Imagery Effect Transition

CFSNet: Toward a Controllable Feature Space for Image Restoration
