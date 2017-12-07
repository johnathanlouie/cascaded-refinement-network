This is an implementation of the convolutional neural network described in
["Photographic Image Synthesis with Cascaded Refinement Networks"](http://cqf.io/papers/Photographic_Image_Synthesis_ICCV2017.pdf)
by Qifeng Chen and Vladlen Koltun. There are some differences between 
[their implementation](https://github.com/CQFIO/PhotographicImageSynthesis) 
and this one. You may find more information at their [website](http://cqf.io/ImageSynthesis/).

# Required Python Libraries
* Tensorflow
* Keras
* OpenCV
* Pillow
* Numpy
* h5py
* Python 3

# Dataset
Please download the dataset from Cityscape. We used gtFine_trainvaltest (labels) and leftImg8bit_trainvaltest (data).

Link: https://www.cityscapes-dataset.com/downloads/

# Quick Start
1. Clone this repository.
2. Download the dataset from Cityscape.
3. Prepare a save file to begin training by using the `prepvgg` and then `prepcrn` subcommands.
4. Then train by using the `train` subcommand.
5. To synthesize images, use the `generate` subcommand after training.
6. Run `python3 crn.py --help` for more information.

# Warning
Running this neural network requires a substantial amount of memory. Training
the network in 256p requires at least 40 GB for a batch size of 1. Training in
1024p requires at least 120 GB for a batch size of 5.

256p is enabled. To use the code for 512p and 1024p, uncomment the extra modules.

# Differences
* Uses batch normalization instead of layer normalization.
* Uses an earlier version of their loss function.
* Uses max pooling instead of bilinear subsampling.

# Reference
Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks. In ICCV 2017.
