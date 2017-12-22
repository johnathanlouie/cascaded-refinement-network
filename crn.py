import os
import argparse
import time
import numpy
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import load_model
import models


def size_data(directory):
    """Returns the number of samples in data and label directories."""
    return len(os.listdir(directory))


def load_images(directory, start, batch_size, height, width, channels):
    """Load the data or labels as Numpy arrays. Channels last."""
    mode = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    end = start + batch_size
    filenames = sorted(os.listdir(directory))[start:end]
    ndata = len(filenames)
    shape = (ndata, height, width, channels)
    data = numpy.ndarray(shape=shape, dtype=numpy.uint8)
    for i in range(ndata):
        data[i] = cv2.imread(directory + "/" + filenames[i], mode).reshape((height, width, channels))
    return data


def load_data(directory, start, batch_size):
    return load_images(directory, start, batch_size, 256, 512, 1)


def load_labels(directory, start, batch_size):
    return load_images(directory, start, batch_size, 256, 512, 3)


def proc_args():
    """Parse program arguments."""
    parser = argparse.ArgumentParser(description="Cascaded Refinement Networks for photorealistic image synthesis.")
    sub1 = parser.add_subparsers(dest="subparser1")
    sub1.required = True
    sub1_1 = sub1.add_parser("train", help="Train the model using semantic layouts as input and ground truth images as output.")
    sub1_1.add_argument("load", help="Load the model from this file.")
    sub1_1.add_argument("save", help="Save the model with architecture, weights, training configuration, and optimization state to this file.")
    sub1_1.add_argument("vgg", help="Load VGG19 from this file.")
    sub1_1.add_argument("semantic", help="Directory in which the semantic layouts are stored.")
    sub1_1.add_argument("truth", help="Directory in which the preprocessed ground truth images are stored.")
    sub1_1.add_argument("-b", "--batchsize", help="Number of samples per gradient update.", type=int, default=5)
    sub1_1.add_argument("-e", "--epochs", help="Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.", type=int, default=1)
    sub1_2 = sub1.add_parser("generate", help="Synthesize images using semantic layouts as input.")
    sub1_2.add_argument("load", help="Load the model from this file.")
    sub1_2.add_argument("semantic", help="Directory in which the semantic layouts are stored.")
    sub1_2.add_argument("synthesized", help="Directory to which the synthesized images are written.")
    sub1_3 = sub1.add_parser("prepare", help="Prepare save files for the program to work with.")
    sub2 = sub1_3.add_subparsers(dest="subparser2")
    sub2.required = True
    sub2_1 = sub2.add_parser("crn", help="Prepare CRN for use.")
    sub2_1.add_argument("save", help="Save CRN to this file.")
    sub2_1.add_argument("vgg", help="Load VGG19 from this file.")
    sub2_2 = sub2.add_parser("vgg", help="Prepare VGG19 for use.")
    sub2_2.add_argument("save", help="Save VGG19 to this file.")
    args = parser.parse_args()
    return args


def read_temp_file(url):
    """This temporary file specifies from which sample to start processing if the runtime was cut short or start from the beginning if it is missing."""
    count = (0, 0)
    if os.path.isfile(url):
        file = open(url, "r")
        count = tuple(map(int, file.read().split()))
        file.close()
    return count


def write_temp_file(url, epoch, batch):
    """If the program does not complete the specified number of epochs then it saves the epochs and current sample ID. The next run will start from where it left off."""
    file = open(url, "w")
    file.write("%d %d\n" % (epoch, batch))
    file.close()
    return


def main():
    # Keep track of elapsed time for Slurm.
    time_begin = time.time()
    # Handle arguments.
    args = proc_args()
    if args.subparser1 == "train":
        # Name of the file storing the epoch and sample number if the program ends before completion due to Slurm's time limit.
        tempfile = "tmp0"
        # Get the epoch and sample number or start from the beginning.
        e, b = read_temp_file(tempfile)
        # Load a saved CRN+VGG19 file to work on.
        training_model = load_model(args.load, custom_objects={'normalize_crn_output': models.normalize_crn_output})
        # Load the pretrained VGG19 to generate the labels.
        vgg = load_model(args.vgg)
        # Get the number of samples in the dataset, so it knows when the epoch will end.
        data_size = size_data(args.semantic)
        # Count the number of epochs so far.
        while e < args.epochs:
            # Count the number of samples processed so far.
            while b < data_size:
                batch_begin = time.time()
                # Slurm allocated 48 minutes. Graceful exit at 40 minutes for an 8 minute buffer. Each training batch has a long running time.
                if (time.time() - time_begin) >= (60 * 40):
                    os.remove(args.save)
                    os.rename(args.load, args.save)
                    # Save weights, architecture, training configuration, and optimization state.
                    training_model.save(args.load)
                    # Save where it left off.
                    write_temp_file(tempfile, e, b)
                    # Exit.
                    return
                # Load a batch of semantic layouts.
                data = load_data(args.semantic, b, args.batchsize)
                # Load a batch of ground truth images.
                raw_labels = load_images(args.truth, b, args.batchsize)
                # Use VGG19 to produce the labels from ground truth images.
                labels = vgg.predict(raw_labels)
                # Train a batch.
                training_model.train_on_batch(x=data, y=labels)
                print("batch time: %d" % (time.time() - batch_begin))
                # Increment the sample counter by the batch size.
                b += args.batchsize
            # When the current epoch is finished, set the sample counter to zero.
            b = 0
            # Increment the epoch counter.
            e += 1
        # When all the epochs finished running, save the model.
        training_model.save(args.save)
        # Remove the epoch/sample counter file since it completed the requested number of epochs.
#        if os.path.isfile(tempfile):
#            os.remove(tempfile)
    elif args.subparser1 == "generate":
        # Load the CRN+VGG19 architecture and extract CRN from it.
        custom_object = {'normalize_crn_output': models.normalize_crn_output}
        crn_vgg = load_model(args.load, custom_objects=custom_object)
        testing_model = models.extract_crn(crn_vgg)
        # Synthesize and save all images.
        for i in range(size_data(args.semantic)):
            # Stop if Slurm time limit.
            if (time.time() - time_begin) >= (60 * 45):
                return
            data = load_data(args.semantic, i, 1)
            result = testing_model.predict(data)
            filename = "%s/%05d.png" % (args.synthesized, i)
            cv2.imwrite(filename, result[0])
    elif args.subparser1 == "prepare":
        if args.subparser2 == "crn":
            # This program needs an initial saved model to work with. Save an untrained CRN+VGG19 model.
            models.combine_crn_vgg19(models.crn256(), load_model(args.vgg)).save(args.save)
        elif args.subparser2 == "vgg":
            # The prepcrn subcommand requires a pretrained VGG19 save file to work with.
            models.vgg19(256, 512).save(args.save)
    return


main()