import numpy as np
import pandas as pd
import tensorflow as tf



ds=pd.read_csv("aug_train.csv")
# provjera duplikata u setu
duplicate=ds["enrollee_id"].duplicated()
for i in duplicate:
    if i:
        print(i)
#0 duplikata



