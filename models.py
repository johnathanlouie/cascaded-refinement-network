from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, concatenate, Input, Lambda
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.layers.advanced_activations import LeakyReLU


def normalize_crn_output(x):
    """The output of the final convolutional layer is a very dark image because its input has been normalized. This function will help it scale better."""
    return (x + 1.0) / 2.0 * 255.0


def append_layer(x, fnum, conv, bn):
    """Each module has a 3x3 convolutional layer with LReLU for activation followed by batch normalization."""
    x = Conv2D(filters=fnum, kernel_size=3, padding="same", name=conv)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(name=bn)(x)
    return x


def append_module(prev, sem, mnum, fnum):
    """Append a module to the end of another module. I used this function to build the Cascaded Refinement Network module by module."""
    # Module 0 takes the semantic layout downsampled to 4x8 as input.
    if mnum == 0:
        x = sem

    # The rest of the modules upsamples the output of the previous module and concatenates it with a downsampled semantic layout.
    else:
        x = UpSampling2D(size=2)(prev)
        x = concatenate([x, sem])

    # Naming convention for the layers.
    conv1 = "conv%d_1" % mnum
    bn1 = "bn%d_1" % mnum
    conv2 = "conv%d_2" % mnum
    bn2 = "bn%d_2" % mnum

    # Two convolutional layers.
    x = append_layer(x, fnum, conv1, bn1)
    x = append_layer(x, fnum, conv2, bn2)
    return x


def crn1024():
    """Build the Cascaded Refinement Network."""
    minput = Input(shape=(1024, 2048, 1), name="crn_input")

    # Downsample the semantic layout in advance.
    dn8 = minput
    dn7 = MaxPooling2D(pool_size=2, strides=2, name="pool7")(dn8)
    dn6 = MaxPooling2D(pool_size=2, strides=2, name="pool6")(dn7)
    dn5 = MaxPooling2D(pool_size=2, strides=2, name="pool5")(dn6)
    dn4 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(dn5)
    dn3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(dn4)
    dn2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(dn3)
    dn1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(dn2)
    dn0 = MaxPooling2D(pool_size=2, strides=2, name="pool0")(dn1)

    # Each module doubles the resolution and has progressively fewer filters.
    x = append_module(None, dn0, 0, 1024)
    x = append_module(x, dn1, 1, 1024)
    x = append_module(x, dn2, 2, 1024)
    x = append_module(x, dn3, 3, 1024)
    x = append_module(x, dn4, 4, 1024)
    x = append_module(x, dn5, 5, 512)
    x = append_module(x, dn6, 6, 512)
    x = append_module(x, dn7, 7, 128)
    x = append_module(x, dn8, 8, 32)

    # The last module is followed by a convolutional layer with 3 filters for the 3 color channels.
    x = Conv2D(filters=3, kernel_size=1, activation=None)(x)

    # The output is very dark, so scale the intensity.
    moutput = Lambda(normalize_crn_output, name="crn_output")(x)
    model = Model(inputs=minput, outputs=moutput)
    return model


def crn512():
    """Build the Cascaded Refinement Network."""
    minput = Input(shape=(512, 1024, 1), name="crn_input")

    # Downsample the semantic layout in advance.
    dn7 = minput
    dn6 = MaxPooling2D(pool_size=2, strides=2, name="pool6")(dn7)
    dn5 = MaxPooling2D(pool_size=2, strides=2, name="pool5")(dn6)
    dn4 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(dn5)
    dn3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(dn4)
    dn2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(dn3)
    dn1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(dn2)
    dn0 = MaxPooling2D(pool_size=2, strides=2, name="pool0")(dn1)

    # Each module doubles the resolution and has progressively fewer filters.
    x = append_module(None, dn0, 0, 1024)
    x = append_module(x, dn1, 1, 1024)
    x = append_module(x, dn2, 2, 1024)
    x = append_module(x, dn3, 3, 1024)
    x = append_module(x, dn4, 4, 1024)
    x = append_module(x, dn5, 5, 512)
    x = append_module(x, dn6, 6, 512)
    x = append_module(x, dn7, 7, 128)

    # The last module is followed by a convolutional layer with 3 filters for the 3 color channels.
    x = Conv2D(filters=3, kernel_size=1, activation=None)(x)

    # The output is very dark, so scale the intensity.
    moutput = Lambda(normalize_crn_output, name="crn_output")(x)
    model = Model(inputs=minput, outputs=moutput)
    return model


def crn256():
    """Build the Cascaded Refinement Network."""
    minput = Input(shape=(256, 512, 1), name="crn_input")

    # Downsample the semantic layout in advance.
    dn6 = minput
    dn5 = MaxPooling2D(pool_size=2, strides=2, name="pool5")(dn6)
    dn4 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(dn5)
    dn3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(dn4)
    dn2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(dn3)
    dn1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(dn2)
    dn0 = MaxPooling2D(pool_size=2, strides=2, name="pool0")(dn1)

    # Each module doubles the resolution and has progressively fewer filters.
    x = append_module(None, dn0, 0, 1024)
    x = append_module(x, dn1, 1, 1024)
    x = append_module(x, dn2, 2, 1024)
    x = append_module(x, dn3, 3, 1024)
    x = append_module(x, dn4, 4, 1024)
    x = append_module(x, dn5, 5, 512)
    x = append_module(x, dn6, 6, 512)

    # The last module is followed by a convolutional layer with 3 filters for the 3 color channels.
    x = Conv2D(filters=3, kernel_size=1, activation=None)(x)

    # The output is very dark, so scale the intensity.
    moutput = Lambda(normalize_crn_output, name="crn_output")(x)
    model = Model(inputs=minput, outputs=moutput)
    return model


def vgg19(height, width):
    """Build a pretrained VGG19 for perceptual loss."""
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=(height, width, 3))

    # Only the input and the second convolutional layer from each block is used in the loss function.
    vgg2 = Model(inputs=vgg.input, outputs=[
            vgg.input,
            vgg.get_layer("block1_conv2").output,
            vgg.get_layer("block2_conv2").output,
            vgg.get_layer("block3_conv2").output,
            vgg.get_layer("block4_conv2").output,
            vgg.get_layer("block5_conv2").output
            ])

    # Ensure VGG19 cannot be trained.
    vgg2.trainable = False
    for l in vgg2.layers:
        l.trainable = False
    return vgg2


def combine_crn_vgg19(crn, vgg):
    """Append VGG19 to the end of CRN for perceptual loss."""
    output = vgg(crn.output)
    model = Model(inputs=crn.input, outputs=output)
    # Add the optimizer with the learning rate, the 6 loss functions for the 6 outputs from VGG19, and hyperweights.
    model.compile(
            optimizer=Adam(lr=0.0001),
            loss=[
                    "mean_absolute_error",
                    "mean_absolute_error",
                    "mean_absolute_error",
                    "mean_absolute_error",
                    "mean_absolute_error",
                    "mean_absolute_error"
            ],
            loss_weights=[1.0, 1/2.6, 1/4.8, 1/3.7, 1/5.6, 10.0/1.5]
    )
    return model


def extract_crn(model):
    """Extract CRN from the CRN+VGG19 save file."""
    return Model(inputs=model.input, outputs=model.get_layer("crn_output").output)