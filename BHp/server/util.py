import json
import pickle
import numpy as np


__locations=None
__data_columns=None
__model=None

def load_saved_artifacts():
    print("loading the data")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r',encoding="utf-8") as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:]
        print("Loaded")

    global  __model
    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        print("Entry to pickel")
        __model=pickle.load(f)
    print("Loading the saved artifacts")

def get_location_name():
    load_saved_artifacts()
    print(__locations)
    return __locations

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)


if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_name())
    print(get_estimated_price('1st phase jp nagar',1000,3,3))