import pandas as pd
import pickle
from train import extract_features

# read the data
data = pd.read_csv("test.csv", header=None)
data_features = extract_features(data)
# load decision tree model
filename = 'SVM_model.pkl'
with open(filename, 'rb') as file:
    svc = pickle.load(file)

# predict the result and save it in Result.csv in one column
result = svc.predict(data_features)
result_S = pd.Series(result, name=None)
result_S.to_csv("Result.csv", index=False, header=None)
