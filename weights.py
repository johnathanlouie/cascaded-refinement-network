import keras
import numpy

def getWeights():
   vgg = keras.applications.vgg19.VGG19(False, "imagenet")

   a=[
      "block1_conv1",
      "block1_conv2",
      "block2_conv1",
      "block2_conv2",
      "block3_conv1",
      "block3_conv2",
      "block3_conv3",
      "block3_conv4",
      "block4_conv1",
      "block4_conv2",
      "block4_conv3",
      "block4_conv4",
      "block5_conv1",
      "block5_conv2",
      "block5_conv3",
      "block5_conv4"
      ]

   for i in a:
       numpy.save(i, vgg.get_layer(i).get_weights())