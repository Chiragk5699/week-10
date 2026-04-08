import numpy as np
import pandas as pd
import pickle


# Load both models
with open("model_1.pickle", "rb") as f:
    model_1 = pickle.load(f)

with open("model_2.pickle", "rb") as f:
    model_2 = pickle.load(f)

# Same roast_category function from train.py
def roast_category(roast):
    roast_map = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return roast_map.get(roast, None)

def predict_rating(df_X):
    predictions = []

    for _, row in df_X.iterrows():
        roast_num = roast_category(row["roast"])

        if roast_num is None:
            # Unknown roast → use model_1 (100g_USD only)
            x = np.array([[row["100g_USD"]]])
            pred = model_1.predict(x)[0]
        else:
            # Valid roast → use model_2 (both features)
            x = np.array([[row["100g_USD"], roast_num]])
            pred = model_2.predict(x)[0]

        predictions.append(pred)

    return np.array(predictions)
