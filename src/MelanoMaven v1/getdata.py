import os
import zipfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import get_file


def get_data():                                                 # function to download and save the datasets from the provided source

    training = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip"
    validation = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip"
    testing = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip"

    for cur, link in enumerate([training, validation, testing]):
        cur_file = f"cur{cur}.zip"
        name = os.path.join(os.getcwd(), cur_file)
        data_f = get_file(origin=link, fname=name)
        with zipfile.ZipFile(data_f, "r") as f:
            f.extractall("../../data")
        os.remove(cur_file)
