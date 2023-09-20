import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_hub as hub
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import glob
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.metrics import sensitivity_score, specificity_score

from preparedata import prepare, get_path


tf.random.set_seed(17)
np.random.seed(17)
random.seed(17)

custom_objects = {"KerasLayer": hub.KerasLayer}                                                 # load the model using custom objects 
model = tf.keras.models.load_model("../../models/melanoma_model.h5", custom_objects=custom_objects)   # as KerasLayer is not a native tensorflow object

weights = glob.glob("../../models/melanoma_model_*.h5")                                               # load the model with the lowest validation loss
weights = [float('0.' + weight.split('_')[2].split('.')[1]) for weight in weights]
weights.sort()
weight = weights[0]

model.load_weights(f"../../models/melanoma_model_{weight}.h5")

df_test = pd.read_csv("../../csv/test.csv")                                                        # load the test dataset as a pandas dataframe
testing_samples = len(df_test)

test_ds = tf.data.Dataset.from_tensor_slices((df_test["fpath"], df_test["type"]))          # create a tensorflow dataset from the pandas dataframe
test_ds = test_ds.map(get_path)
test_ds = prepare(test_ds, cache = "../../cache/test_cache",batch = False, repeat = False, prefetch = False) # prepare the dataset for testing only

img_test = np.zeros((testing_samples, 299, 299, 3))                                     # create numpy arrays to store the images and labels
label_test = np.zeros((testing_samples))

for i, (img, label) in enumerate(test_ds.take(testing_samples)):                    # store the images and labels in the numpy arrays

    img_test[i] = img
    label_test[i] = label.numpy()


loss, acc = model.evaluate(img_test, label_test, verbose=0)                         # evaluate the model on the test dataset
print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'--'*25}\nAccuracy and loss before thresholding:")
print(f"Loss: {round(loss, 4)}, Accuracy: {round(acc, 4)}\n{'--'*25}\n{'--'*25}\n")

def predict_by_threshold(threshold=0.23):                                      # function to predict the labels using a threshold

    label_pred = model.predict(img_test)
    prediction = np.empty((testing_samples))

    for i in range(testing_samples):                                        # map the predicted probabilities to 0 or 1 using the threshold
        if label_pred[i][0] >= threshold:
            prediction[i] = 1
        else:
            prediction[i] = 0
    return prediction

label_pred = predict_by_threshold(threshold=0.35)
accuracy = accuracy_score(label_test, label_pred)                        # calculate the accuracy using the predicted labels

print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'--'*25}\nAccuracy after thresholding:")
print(f"Accuracy: {round(acc, 4)}\n{'--'*25}\n{'--'*25}\n")

sensitivity = sensitivity_score(label_test, label_pred)                 # calculate the sensitivity and specificity using the predicted labels
specificity = specificity_score(label_test, label_pred)

print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'--'*25}\nSensitivity and specificity:")
print(f"Sensitivity: {round(sensitivity, 4)}, Specificity: {round(specificity, 4)}\n{'--'*25}\n{'--'*25}")

def get_confusion_matrix(label_test, label_pred):                     # function to generate the confusion matrix

    cm = confusion_matrix(label_test, label_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'--'*25}\nConfusion Matrix:")
    print(cm)

    plt.subplots(figsize=(10, 10))
    sb.heatmap(cm, annot=True, fmt=".4f",
               xticklabels=[f"pred_{i}" for i in ["benign", "malignant"]],
                yticklabels=[f"true_{i}" for i in ["benign", "malignant"]])
    
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("True Label")
    try:
        os.remove("../../output/melanomaven_v1_confusion_matrix.png")            # remove the old confusion matrix
    except:
        pass
    plt.savefig("../../output/melanomaven_v1_confusion_matrix.png")              # save the new confusion matrix

get_confusion_matrix(label_test, label_pred)                                   # generate the confusion matrix
print(f"{'*'*40}\n{'*'*40}")