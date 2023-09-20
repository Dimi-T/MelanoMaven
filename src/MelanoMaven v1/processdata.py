import os
import shutil
import glob
import pandas as pd


def create_csv(dir, types_dict):                            # function to create csv files from the raw dataset

    dir_name = os.path.basename(dir)
    types = list(types_dict)
    df = pd.DataFrame([],[],columns=["fpath", "type"])
    index = 0
    for type in types:
        for fpath in glob.glob(os.path.join(dir, type, "*")):
            df.loc[index] = [fpath, types_dict[type]]
            index += 1

    outf = f"../../csv/{dir_name}.csv"
    df.to_csv(outf)

def get_csv():                                              # function to create csv files for datasets and map them to the correct labels

    if os.path.isdir("../csv") == True:                     # remove the csv directory if it already exists
        shutil.rmtree("../csv")
    
    os.mkdir("../../csv")
        
    create_csv("../../data/train", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
    create_csv("../../data/valid", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
    create_csv("../../data/test", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})

get_csv() 