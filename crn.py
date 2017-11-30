import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model, load_model
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, concatenate, Input
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.layers.advanced_activations import LeakyReLU
import cv2
import numpy as n

def create_crn():
    minput = Input(shape=(1024, 2048, 1))
    L8 = minput
    L7 = MaxPooling2D(pool_size=2, strides=2, name="mod7_pool")(L8)
    L6 = MaxPooling2D(pool_size=2, strides=2, name="mod6_pool")(L7)
    L5 = MaxPooling2D(pool_size=2, strides=2, name="mod5_pool")(L6)
    L4 = MaxPooling2D(pool_size=2, strides=2, name="mod4_pool")(L5)
    L3 = MaxPooling2D(pool_size=2, strides=2, name="mod3_pool")(L4)
    L2 = MaxPooling2D(pool_size=2, strides=2, name="mod2_pool")(L3)
    L1 = MaxPooling2D(pool_size=2, strides=2, name="mod1_pool")(L2)
    L0 = MaxPooling2D(pool_size=2, strides=2, name="mod0_pool")(L1)
    # module 0
    x = L0
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod0_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod0_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 1
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L1])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod1_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod1_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 2
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L2])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod2_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod2_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 3
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L3])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod3_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod3_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 4
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L4])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod4_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod4_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 5
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L5])
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod5_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod5_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 6
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L6])
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod6_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod6_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 7
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L7])
    x = Conv2D(filters=128, kernel_size=3, padding="same", name="mod7_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, padding="same", name="mod7_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # module 8
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L8])
    x = Conv2D(filters=32, kernel_size=3, padding="same", name="mod8_conv1")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same", name="mod8_conv2")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    moutput = Conv2D(filters=3, kernel_size=1, activation=None)(x)
    model = Model(inputs=minput, outputs=moutput)
    return model

def create_vgg():
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(1024, 2048, 3))
    vgg2 = Model(inputs=vgg.input, outputs=[
            vgg.input,
            vgg.get_layer('block1_conv2').output,
            vgg.get_layer('block2_conv2').output,
            vgg.get_layer('block3_conv2').output,
            vgg.get_layer('block4_conv2').output,
            vgg.get_layer('block5_conv2').output
            ])
    vgg2.trainable = False;
    for l in vgg2.layers:
        l.trainable = False
    return vgg2

def create_training_model(crn, vgg):
    output = vgg(crn.output)
    model = Model(inputs=crn.input, outputs=output)
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

def load_data(loc):
    inputNames = sorted(os.listdir(loc))
    ndata = len(inputNames)
    ishape = (ndata, 1024, 2048, 1)
    data = n.ndarray(shape=ishape, dtype=n.uint8)
    for i in range(ndata):
        data[i] = cv2.imread(loc + '/' + inputNames[i], 0).reshape((1024, 2048, 1))
    return data

def load_labels(loc):
    outputNames = sorted(os.listdir(loc))
    ndata = len(outputNames)
    oshape = (ndata, 1024, 2048, 3)
    labels = n.ndarray(shape=oshape, dtype=n.uint8)
    for i in range(ndata):
        labels[i] = cv2.imread(loc + '/' + outputNames[i], 1).reshape((1024, 2048, 3))
    return labels

def main():
    parser = argparse.ArgumentParser(description="Cascaded Refinement Networks for photorealistic image synthesis.")
    parser.add_argument("-l", "--load", help="Load the model from this file.")
    parser.add_argument("-s", "--save", help="Save the model with architecture, weights, training configuration, and optimization state to this file.")
    subparsers = parser.add_subparsers(dest="subparser")
    subparser1 = subparsers.add_parser('train', help='Train the model using semantic layouts as input and ground truth images as output.')
    subparser1.add_argument("layouts", help="Directory in which the semantic layouts are stored.")
    subparser1.add_argument("images", help="Directory in which the ground truth images are stored.")
    subparser1.add_argument("-v", "--vgg", help="Load VGG19 from this file.")
    subparser1.add_argument("-c", "--create", help="Save VGG19 to this file.")
    subparser1.add_argument("-b", "--batch", help="Number of samples per gradient update.", type=int, default=None)
    subparser1.add_argument("-e", "--epoch", help="Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.", type=int, default=1)
    subparser2 = subparsers.add_parser('generate', help='Synthesize images using semantic layouts as input.')
    subparser2.add_argument("layouts", help="Directory in which the semantic layouts are stored.")
    subparser2.add_argument("outputs", help="Directory to which the synthesized images are written.")
    args = parser.parse_args()

    crn = None
    if args.load == None:
        crn = create_crn()
    else:
        crn = load_model(args.load)

    if args.subparser == "train":
        vgg = None
        if args.vgg == None:
            vgg = create_vgg()
        else:
            vgg = load_model(args.vgg)
        if args.create != None:
            vgg.save(args.create)
        data = load_data(args.layouts)
        raw_labels = load_labels(args.images)
        labels = vgg.predict(raw_labels)
        training_model = create_training_model(crn, vgg)
        training_model.fit(x=data, y=labels, batch_size=args.batch, epochs=args.epoch)
    elif args.subparser == "generate":
        data = load_data(args.layouts)
        result = crn.predict(data)
        for i in range(data.shape[0]):
            cv2.imwrite(args.outputs + '/' + str(i) + '.png', result[i])
    else:
        print("No commands. Try --help.")
    if args.save != None:
        crn.save(args.save)
    return

main()