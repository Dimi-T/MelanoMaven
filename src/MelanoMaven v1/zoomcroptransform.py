import cv2 as cv
from wand.image import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import numpy as np

tf.random.set_seed(17)                              # set seed for tensorflow for consistency in results
np.random.seed(17)
random.seed(17)

def get_count():
    try: 
        return len(os.listdir('../../output/'))              # get the number of images in the output folder
    except FileNotFoundError:
        return 0

def zoom(img, scale):                               # funtion to zoom in to the center of the image to ensure the lesion 
                                                    # of interest is the main focus of the image
    height, width, _ = [ scale * val for val in img.shape ]
    cx, cy = width/2, height/2

    img = cv.resize( img, (0, 0), fx=scale, fy=scale)
    img = img[ int(round(cy - height/scale * 0.5)) : int(round(cy + height/scale * 0.5)),
               int(round(cx - width/scale * 0.5)) : int(round(cx + width/scale * 0.5)),
               : ]
    
    return img

def transform(source_img):                        # function to transform the image to jpeg format

    save = "input/" + source_img.split('.')[0] + '.jpeg'
    img = Image(filename=source_img)
    img.format = 'jpeg'
    img.save(filename=save)
    os.remove(source_img)

def decode(img):                             # function to decode image and resize it to 299x299, as required by InceptionV3 model

    ext = img.split('.')[-1]
    img_path = tf.io.read_file(img)

    match ext:
        case 'jpg':
            img_proc = tf.image.decode_jpeg(img_path, channels=3)
        case 'jpeg':
            img_proc = tf.image.decode_jpeg(img_path, channels=3)
        case 'png':
            img_proc = tf.image.decode_png(img_path, channels=3)
        case _:
            print("Error: Image format not supported")
    img_proc = tf.image.convert_image_dtype(img_proc, tf.float32)
    return tf.image.resize(img_proc, [299, 299])

def resize(img, scale):                     # function to resize the image to the desired scale

    name = img
    img = cv.imread(img)
    img = cv.imwrite(name, zoom(img, scale))


def get_images(scale=1.5, modify="False"):                # function to get images from the input folder, detect the image format
                                                          # and convert the image to jpeg format if not supported by tensorflow

    imgs = os.listdir('../../input/')
    for i in range(len(imgs)):
        imgs[i] = '../../input/' + imgs[i]

    imgs_renamed = []
    COUNT = get_count() + 1

    """ 4.5 is the default scale factor for picture taken at 25cm distance
        from the lens with a Samsung A71 camera and a zoom factor of 1x """
    for img in imgs:
        os.rename(img, f"../../input/test_{COUNT}.png")
        img = f"../../input/test_{COUNT}.png"
        COUNT += 1

        unsupported_type = ['heic', 'heics', 'HEIC', 'HEICS', 'JPEG', 'JPG', 'PNG']
        if img.split('.')[-1] in unsupported_type:
            transform(img)
            img = img.split('.')[-1] + '.jpeg'
            print(img)
        if modify == "True":
            resize(img, scale)
        decode(img)
        imgs_renamed.append(img)
    return imgs_renamed