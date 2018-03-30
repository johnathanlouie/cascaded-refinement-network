import cv2
import os


def resize(source, destination, channel_type, interpolation, height):
    width = height * 2
    resolution = (width, height)
    for i in sorted(os.listdir(source)):
        img = cv2.imread("%s/%s" % (source, i), channel_type)
        img2 = cv2.resize(img, resolution, interpolation=interpolation)
        cv2.imwrite("%s/%s" % (destination, i), img2)
    return


def resize_data(source, destination, height):
    resize(source, destination, cv2.IMREAD_GRAYSCALE, cv2.INTER_NEAREST, height)
    return
    

def resize_label(source, destination, height):
    resize(source, destination, cv2.IMREAD_COLOR, cv2.INTER_CUBIC, height)
    return