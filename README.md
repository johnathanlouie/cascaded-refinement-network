# Readme

This is an implementation of the convolutional neural network decribed in the paper "Photographic Image Synthesis with Cascaded Refinement Networks."

# Required Python Libraries
Tensorflow or Theano
Keras
OpenCV
Pillow
Numpy
h5py

# Dataset

Please download the Dataset from Cityscape for sementic layout example.

Link: https://www.cityscapes-dataset.com/downloads/

# Quick Start
1. Clone this repository.
2. Download the dataset from Cityscape.
3. Prepare a save file to store the model and weights to begin training by using the prepvgg and then prepcrn subcommands.
4. Then train by using the train subcommand.
5. To synthesize images, use the generate subcommand after training.
3. Run "python3 crn.py --help" for more information.

## Warning
Running this neural network requires a substantial amount of memory. Training the network in 256p requires at least 40 GB for a batch size of 1. Training in 1024p requires at least 120 GB for a batch size of 5.

## Reference

Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks. In ICCV 2017.
