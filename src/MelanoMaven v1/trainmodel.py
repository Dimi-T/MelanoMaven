import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import random
import numpy as np
import tensorflow_hub as hub

from preparedata import train_ds, valid_ds, training_samples, validation_samples    # import prepared datasets

tf.random.set_seed(17)
np.random.seed(17)
random.seed(17)


model_source = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"    # use InceptionV3 model from tensorflow hub

model = tf.keras.Sequential([hub.KerasLayer(model_source, trainable = False, output_shape = [2048]),  
                            tf.keras.layers.Dense(1, activation = "sigmoid")])    # use sigmoid activation function for binary classification
model.build([None, 299, 299, 3])    
model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])  

model_name = "../../models/MelanoMaven v1/melanoma_model"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name + "_{val_loss:.3f}.h5", save_best_only=True, verbose=1)
                                                            # save the model with the lowest validation loss

epochs = 100
batch = 64

model.fit(train_ds, epochs = epochs, steps_per_epoch = training_samples // batch,
                    validation_data = valid_ds, validation_steps = validation_samples // batch,
                    callbacks = [model_checkpoint], verbose = 1)
                                                            # train the model
model.save(model_name + ".h5")