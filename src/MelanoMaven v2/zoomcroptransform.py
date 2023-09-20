import cv2 as cv
from wand.image import Image
import os


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

    save = '../..' + source_img.split('.')[-2] + '.jpeg'
    img = Image(filename=source_img)
    img.format = 'jpeg'
    img.save(filename=save)
    os.remove(source_img)


def resize(img, scale):                     # function to resize the image to the desired scale

    name = img
    img = cv.imread(img)
    img = cv.imwrite(name, zoom(img, scale))


def prepare(img_path, scale=4.5):                # function to get images from the input folder, detect the image format


    """ 4.5 is the default scale factor for picture taken at 25cm distance
        from the lens with a Samsung A71 camera and a zoom factor of 1x """

    unsupported_type = ['heic', 'heics', 'HEIC', 'HEICS', 'png', 'PNG', 'jpg', 'JPG']

    if img_path.split('.')[-1] in unsupported_type:
        transform(img_path)
        img_path = '../..' + img_path.split('.')[-2] + '.jpeg'

    resize(img_path, scale)
    return img_path