import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import numpy as np

tf.random.set_seed(17)                  # set seed for tensorflow for consistency in results
np.random.seed(17)
random.seed(17)

df_train = pd.read_csv("../../csv/train.csv")     # read csv files as pandas dataframe
df_valid = pd.read_csv("../../csv/valid.csv")

training_samples = len(df_train)
validation_samples = len(df_valid)

train_ds = tf.data.Dataset.from_tensor_slices((df_train["fpath"], df_train["type"]))    # creata tensorflow datasets from pandas dataframe
valid_ds = tf.data.Dataset.from_tensor_slices((df_valid["fpath"], df_valid["type"]))

def decode_img(img):                                                             # function to decode image and resize it to 299x299,                                
                                                                                 # as required by InceptionV3 model
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return tf.image.resize(img, [299, 299])

def get_path(file_path, label):                                                 # function to read image from file path and decode it

    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, label


train_ds = train_ds.map(get_path)                                            # map the function to the datasets
valid_ds = valid_ds.map(get_path)   

def prepare(ds, cache = True, batch_size = 64, shuffle_buffer_size = 1000, batch = True, prefetch = True, repeat = True):
                                                                                # function to prepare the dataset for training
    if cache:                                                                   # cache the dataset for faster training                       
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size = shuffle_buffer_size)                        # shuffle the dataset for better training

    if repeat == True:                                                        # repeat the dataset to train for multiple epochs
        ds = ds.repeat()

    if batch:                                                                 # batch the dataset for faster training                        
        ds = ds.batch(batch_size)

    if prefetch:                                                            
        ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return ds

train_ds = prepare(train_ds, cache = "../../cache/train_cache",batch_size = 64)
valid_ds = prepare(valid_ds, cache = "../../cache/valid_cache",batch_size = 64)
