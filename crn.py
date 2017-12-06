import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import Model, load_model
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, concatenate, Input, Lambda
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.layers.advanced_activations import LeakyReLU
import cv2
import numpy
import signal

def normalize_crn_output(x):
    return (x + 1.0) / 2.0 * 255.0

def append_crn_module(prev, sem, mnum, fnum):
    if mnum > 0:
        x = UpSampling2D(size=2)(prev)
        x = concatenate([x, sem])
    else:
        x = sem
    conv1 = "conv%d_1" % mnum
    bn1 = "bn%d_1" % mnum
    conv2 = "conv%d_2" % mnum
    bn2 = "bn%d_2" % mnum
    x = Conv2D(filters=fnum, kernel_size=3, padding="same", name=conv1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(name=bn1)(x)
    x = Conv2D(filters=fnum, kernel_size=3, padding="same", name=conv2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(name=bn2)(x)
    return x

def create_crn(h, w):
    minput = Input(shape=(h, w, 1), name="crn_input")
#    L8 = minput
#    L7 = MaxPooling2D(pool_size=2, strides=2, name="pool7")(L8)
    L6 = minput #MaxPooling2D(pool_size=2, strides=2, name="pool6")(L7)
    L5 = MaxPooling2D(pool_size=2, strides=2, name="pool5")(L6)
    L4 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(L5)
    L3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(L4)
    L2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(L3)
    L1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(L2)
    L0 = MaxPooling2D(pool_size=2, strides=2, name="pool0")(L1)
    # Module 0
    x = append_crn_module(None, L0, 0, 1024)
    # Module 1
    x = append_crn_module(x, L1, 1, 1024)
    # Module 2
    x = append_crn_module(x, L2, 2, 1024)
    # Module 3
    x = append_crn_module(x, L3, 3, 1024)
    # Module 4
    x = append_crn_module(x, L4, 4, 1024)
    # Module 5
    x = append_crn_module(x, L5, 5, 512)
    # Module 6
    x = append_crn_module(x, L6, 6, 512)
    # Module 7
#    x = append_crn_module(x, L7, 7, 128)
    # Module 8
#    x = append_crn_module(x, L8, 8, 32)
    x = Conv2D(filters=3, kernel_size=1, activation=None)(x)
    moutput = Lambda(normalize_crn_output, name="crn_output")(x)
    model = Model(inputs=minput, outputs=moutput)
    return model

def create_vgg(h, w):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=(h, w, 3))
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

def load_data(loc, start, end, h, w, c):
    mode = cv2.IMREAD_GRAYSCALE if c == 1 else cv2.IMREAD_COLOR
    filenames = sorted(os.listdir(loc))[start:end]
    ndata = len(filenames)
    shape = (ndata, h, w, c)
    data = numpy.ndarray(shape=shape, dtype=numpy.uint8)
    for i in range(ndata):
        data[i] = cv2.imread(loc + "/" + filenames[i], mode).reshape((h, w, c))
    return data

def proc_args():
    parser = argparse.ArgumentParser(description="Cascaded Refinement Networks for photorealistic image synthesis.")
    subparsers = parser.add_subparsers(dest="subparser")
    subparser1 = subparsers.add_parser("train", help="Train the model using semantic layouts as input and ground truth images as output.")
    subparser1.add_argument("load", help="Load the model from this file.")
    subparser1.add_argument("save", help="Save the model with architecture, weights, training configuration, and optimization state to this file.")
    subparser1.add_argument("vgg", help="Load VGG19 from this file.")
    subparser1.add_argument("semantic", help="Directory in which the semantic layouts are stored.")
    subparser1.add_argument("truth", help="Directory in which the preprocessed ground truth images are stored.")
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
    subparser5 = subparsers.add_parser("preptruth", help="Preprocess the ground truth images into Numpy files.")
    subparser5.add_argument("source", help="The directory containing the original ground truth images.")
    subparser5.add_argument("destination", help="The directory to which the preprocessed images are stored.")
    args = parser.parse_args()
    return args

def read_temp_file(url):
    count = (0, 0)
    if os.path.isfile(url):
        file = open(url, "r")
        count = tuple(map(int, file.read().split()))
        file.close()
    return count

def write_temp_file(url, epoch, batch):
    file = open(url, "w")
    file.write("%d %d" % (epoch, batch))
    file.close()
    return

class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        return

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        return

def main():
    args = proc_args()
    height = 256
    width = 512
    if args.subparser == "train":
        tempfile = "tmp0"
        e, b = read_temp_file(tempfile)
        training_model = load_model(args.load, custom_objects={'normalize_crn_output': normalize_crn_output})
        vgg = load_model(args.vgg)
        data_size = size_data(args.semantic)
        while e < args.epochs:
            while b < data_size:
                if GracefulKiller().kill_now:
                    training_model.save(args.save)
                    write_temp_file(tempfile, e, b)
                    return
                data = load_data(args.semantic, b, b + args.batchsize, height, width, 1)
                raw_labels = load_data(args.truth, b, b + args.batchsize, height, width, 3)
                labels = vgg.predict(raw_labels)
                training_model.train_on_batch(x=data, y=labels)
                b += args.batchsize
            b = 0
            e += 1
        training_model.save(args.save)
        if os.path.isfile(tempfile):
            os.remove(tempfile)
    elif args.subparser == "generate":
        testing_model = create_testing_model(load_model(args.load, custom_objects={'normalize_crn_output': normalize_crn_output}))
        for i in range(size_data(args.semantic)):
            if GracefulKiller().kill_now:
                return
            data = load_data(args.semantic, i, i + 1, height, width, 1)
            result = testing_model.predict(data)
            filename = "%s/%d.png" % (args.synthesized, i)
            cv2.imwrite(filename, result[0])
    elif args.subparser == "prepcrn":
        create_training_model(create_crn(height, width), load_model(args.vgg)).save(args.save)
    elif args.subparser == "prepvgg":
        create_vgg(height, width).save(args.save)
    elif args.subparser == "preptruth":
        print("Deprecated.")
    return

main()