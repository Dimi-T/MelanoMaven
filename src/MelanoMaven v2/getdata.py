import requests
import zipfile
import os


training = "https://huggingface.co/datasets/DimiT/MelanoMaven_ISIC_2017-2020/resolve/main/train.zip"
validation = "https://huggingface.co/datasets/DimiT/MelanoMaven_ISIC_2017-2020/resolve/main/valid.zip"
testing = "https://huggingface.co/datasets/DimiT/MelanoMaven_ISIC_2017-2020/resolve/main/test.zip"

try:
  os.mkdir("../../Licenta_cod/datav2")                                    # create the necessary directories
  os.mkdir("../../Licenta_cod/datav2/train")
  os.mkdir("../../Licenta_cod/datav2/valid")
  os.mkdir("../../Licenta_cod/datav2/test")
except:
  pass

def download_data(url, destination):                                     # function to download and save the datasets from the provided source
  
  response = requests.get(url)
  with open(destination, "wb") as file:
    file.write(response.content)

  with zipfile.ZipFile(destination, "r") as zip_ref:
    zip_ref.extractall(destination.replace(destination.split('/')[-1], ''))

  os.remove(destination)


def get_data():                                                   
  download_data(url=training, destination = "../../Licenta_cod/datav2/train.zip")
  download_data(url=validation, destination = "../../Licenta_cod/datav2/valid.zip")
  download_data(url=testing , destination = "../../Licenta_cod/datav2/test.zip")