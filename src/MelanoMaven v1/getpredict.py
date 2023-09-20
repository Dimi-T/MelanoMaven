import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub
os.environ['PLT_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from zoomcroptransform import get_images
import sys
                                                                            # remove previous output files

custom_objects = {"KerasLayer": hub.KerasLayer}                                                 # load model
model = tf.keras.models.load_model("../../models/melanoma_model.h5", custom_objects=custom_objects)                                                                             # load weights
model.load_weights("../../models/melanoma_model_0.353.h5")

def predict(model, input, threshold=0.20):                                                      # function to evaluate the probability of the 
                                                                                                # input skin lesion being malignant
    img = tf.keras.preprocessing.image.load_img(input, target_size=(299, 299))                  # load the image as a tensor
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)                                                                # add a batch dimension
    img = tf.keras.applications.inception_v3.preprocess_input(img)                              # preprocess the image according to the InceptionV3 model
    img = tf.image.convert_image_dtype(img, tf.float32)                                         # convert the image to float32

    pred = model.predict(img)
    score = pred.squeeze()                                                                      # get the probability of the image being malignant

    if score >= threshold:                                                                      # map the probability to Benign or Malignant outcomes
        result = "Malignant"
        score = 100 * score
    else:
        result = "Benign"
        score = 100 * (1 - score)


    plt.imshow(img[0])                                                                          # save the image with the prediction
    plt.axis('off')
    plt.title(f"{score:.3f}% {result}")
    plt.savefig("../../output/"+ input.split("/")[-1].split(".")[0] + "_prediction.png")
    

    return result
try:
    scale_flag = float(sys.argv[1])
    modify_flag = sys.argv[2]
except:
    scale_flag = 1.0
    modify_flag = "False"

input_imgs = get_images(scale=scale_flag, modify=modify_flag)                                                          # test the model with all images from the input folder
print(input_imgs)
for img in input_imgs:
    predict(model, img)
