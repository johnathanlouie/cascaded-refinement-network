import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import Model, load_model
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, concatenate, Input
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.layers.advanced_activations import LeakyReLU
import cv2
import numpy

def create_crn():
    minput = Input(shape=(1024, 2048, 1), name="crn_input")
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
    x = LeakyReLU(alpha=0.2, name="mod0_lrelu1")(x)
    x = BatchNormalization(name="mod0_norm1")(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod0_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod0_lrelu2")(x)
    x = BatchNormalization(name="mod0_norm2")(x)
    # module 1
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L1])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod1_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod1_lrelu1")(x)
    x = BatchNormalization(name="mod1_norm1")(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod1_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod1_lrelu2")(x)
    x = BatchNormalization(name="mod1_norm2")(x)
    # module 2
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L2])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod2_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod2_lrelu1")(x)
    x = BatchNormalization(name="mod2_norm1")(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod2_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod2_lrelu2")(x)
    x = BatchNormalization(name="mod2_norm2")(x)
    # module 3
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L3])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod3_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod3_lrelu1")(x)
    x = BatchNormalization(name="mod3_norm1")(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod3_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod3_lrelu2")(x)
    x = BatchNormalization(name="mod3_norm2")(x)
    # module 4
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L4])
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod4_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod4_lrelu1")(x)
    x = BatchNormalization(name="mod4_norm1")(x)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", name="mod4_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod4_lrelu2")(x)
    x = BatchNormalization(name="mod4_norm2")(x)
    # module 5
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L5])
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod5_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod5_lrelu1")(x)
    x = BatchNormalization(name="mod5_norm1")(x)
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod5_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod5_lrelu2")(x)
    x = BatchNormalization(name="mod5_norm2")(x)
    # module 6
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L6])
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod6_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod6_lrelu1")(x)
    x = BatchNormalization(name="mod6_norm1")(x)
    x = Conv2D(filters=512, kernel_size=3, padding="same", name="mod6_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod6_lrelu2")(x)
    x = BatchNormalization(name="mod6_norm2")(x)
    # module 7
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L7])
    x = Conv2D(filters=128, kernel_size=3, padding="same", name="mod7_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod7_lrelu1")(x)
    x = BatchNormalization(name="mod7_norm1")(x)
    x = Conv2D(filters=128, kernel_size=3, padding="same", name="mod7_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod7_lrelu2")(x)
    x = BatchNormalization(name="mod7_norm2")(x)
    # module 8
    x = UpSampling2D(size=2)(x)
    x = concatenate([x, L8])
    x = Conv2D(filters=32, kernel_size=3, padding="same", name="mod8_conv1")(x)
    x = LeakyReLU(alpha=0.2, name="mod8_lrelu1")(x)
    x = BatchNormalization(name="mod8_norm1")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same", name="mod8_conv2")(x)
    x = LeakyReLU(alpha=0.2, name="mod8_lrelu2")(x)
    x = BatchNormalization(name="mod8_norm2")(x)
    moutput = Conv2D(filters=3, kernel_size=1, activation=None, name="crn_output")(x)
    model = Model(inputs=minput, outputs=moutput)
    return model

def create_vgg():
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=(1024, 2048, 3))
    vgg2 = Model(inputs=vgg.input, outputs=[
            vgg.input,
            vgg.get_layer("block1_conv2").output,
            vgg.get_layer("block2_conv2").output,
            vgg.get_layer("block3_conv2").output,
            vgg.get_layer("block4_conv2").output,
            vgg.get_layer("block5_conv2").output
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

def create_testing_model(model):
    return Model(inputs=model.input, outputs=model.get_layer("crn_output").output)

def size_data(loc):
    inputNames = sorted(os.listdir(loc))
    return len(inputNames)

def load_data(loc, start, end):
    inputNames = sorted(os.listdir(loc))[start:end]
    ndata = len(inputNames)
    ishape = (ndata, 1024, 2048, 1)
    data = numpy.ndarray(shape=ishape, dtype=numpy.uint8)
    for i in range(ndata):
        data[i] = cv2.imread(loc + "/" + inputNames[i], cv2.IMREAD_GRAYSCALE).reshape((1024, 2048, 1))
    return data

def load_labels(loc, start, end):
    outputNames = sorted(os.listdir(loc))[start:end]
    ndata = len(outputNames)
    oshape = (ndata, 1024, 2048, 3)
    labels = numpy.ndarray(shape=oshape, dtype=numpy.uint8)
    for i in range(ndata):
        labels[i] = cv2.imread(loc + "/" + outputNames[i], cv2.IMREAD_COLOR).reshape((1024, 2048, 3))
    return labels

def proc_args():
    parser = argparse.ArgumentParser(description="Cascaded Refinement Networks for photorealistic image synthesis.")
    subparsers = parser.add_subparsers(dest="subparser")
    subparser1 = subparsers.add_parser("train", help="Train the model using semantic layouts as input and ground truth images as output.")
    subparser1.add_argument("load", help="Load the model from this file.")
    subparser1.add_argument("save", help="Save the model with architecture, weights, training configuration, and optimization state to this file.")
    subparser1.add_argument("vgg", help="Load VGG19 from this file.")
    subparser1.add_argument("semantic", help="Directory in which the semantic layouts are stored.")
    subparser1.add_argument("truth", help="Directory in which the ground truth images are stored.")
    subparser1.add_argument("-b", "--batchsize", help="Number of samples per gradient update.", type=int, default=5)
    subparser1.add_argument("-e", "--epochs", help="Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.", type=int, default=1)
    subparser2 = subparsers.add_parser("generate", help="Synthesize images using semantic layouts as input.")
    subparser2.add_argument("load", help="Load the model from this file.")
    subparser2.add_argument("semantic", help="Directory in which the semantic layouts are stored.")
    subparser2.add_argument("synthesized", help="Directory to which the synthesized images are written.")
    subparser3 = subparsers.add_parser("prepcrn", help="Prepare CRN for use.")
    subparser3.add_argument("save", help="Save CRN to this file.")
    subparser3.add_argument("vgg", help="Load VGG19 from this file.")
    subparser4 = subparsers.add_parser("prepvgg", help="Prepare VGG19 for use.")
    subparser4.add_argument("save", help="Save VGG19 to this file.")
    args = parser.parse_args()
    return args

def main():
    args = proc_args()
    if args.subparser == "train":
        batch_file = "batch"
        epoch_file = "epoch"
        b = 0
        if os.path.isfile(batch_file):
            file1  = open(batch_file, "r")
            b = int(file1.read())
            file1.close()
        e = 0
        if os.path.isfile(epoch_file):
            file2  = open(epoch_file, "r")
            e = int(file2.read())
            file2.close()
        training_model = load_model(args.load)
        vgg = load_model(args.vgg)
        data_size = size_data(args.semantic)
        while args.epochs > e:
            while b < data_size:
                b2 = b + args.batchsize
                data = load_data(args.semantic, b, b2)
                raw_labels = load_labels(args.truth, b, b2)
                labels = vgg.predict(raw_labels)
                training_model.train_on_batch(x=data, y=labels)
                training_model.save(args.save)
                file1  = open(batch_file, "w")
                file1.write(str(b2))
                file1.flush()
                file1.close()
                b = b2
            e += 1
            file2  = open(epoch_file, "w")
            file2.write(str(e))
            file2.flush()
            file2.close()
    elif args.subparser == "generate":
        testing_model = create_testing_model(load_model(args.load))
        result = testing_model.predict(load_data(args.semantic))
        for i in range(data.shape[0]):
            cv2.imwrite(args.outputs + "/" + str(i) + ".png", result[i])
    elif args.subparser == "prepcrn":
        create_training_model(create_crn(), load_model(args.vgg)).save(args.save)
    elif args.subparser == "prepvgg":
        create_vgg().save(args.save)
    else:
        print("No commands. Try --help.")
    return

main()