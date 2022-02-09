import pandas as pd
import numpy as np
import pickle5 as pickle

def cleanFeatures(input):
    id=input["enrollee_id"]
    ds=input
    ds.drop(ds.columns[[1,2,4,5,8,9,11,12,13]], axis=1, inplace=True)

    ds["city_development_index"] = ds["city_development_index"].astype('double')
    ds['city_development_index'].fillna(value=ds['city_development_index'].mean(), inplace=True)
    ds['company_size'].fillna(value='Unknown', inplace=True)
    ds['enrolled_university'].fillna(value='no_enrollment', inplace=True)
    ds["education_level"].fillna(value=np.random.choice(["Graduate","Masters","High School", "Phd","Primary School"],p=[0.61,0.23,0.1,0.03,0.03]),inplace=True)

    encoded_columns = pd.get_dummies(ds['company_size'])
    ds = ds.join(encoded_columns).drop('company_size', axis=1)
    ds = ds.drop("Unknown", axis=1)

    encoded_columns = pd.get_dummies(ds['enrolled_university'])
    ds = ds.join(encoded_columns).drop('enrolled_university', axis=1)
    ds = ds.drop("no_enrollment", axis=1)

    encoded_columns = pd.get_dummies(ds['education_level'])
    ds = ds.join(encoded_columns).drop('education_level', axis=1)
    ds = ds.drop("Graduate", axis=1)

    return id, ds

# input=pd.read_csv("../resources/aug_test.csv")
# id, X=cleanFeatures(input)
# model=pickle.load(open("random_forest.sav", 'rb'))
# pred=model.predict(X)
# pred=pd.DataFrame(pred,columns=["Prediction"])
# rez=pd.concat([id,pred],axis=1)
# rez.to_csv("aug_test_results.csv")

def dajPrediction(id):
    ds=pd.read_csv("../resources/aug_test_results.csv")
    x=ds[ds["enrollee_id"]==id]
    print(x)

dajPrediction(217)



