import numpy as np
import pandas as pd
import tensorflow as tf



ds=pd.read_csv("aug_train.csv")
# provjera duplikata u setu
# duplicate=ds["enrollee_id"].duplicated()
# for i in duplicate:
#     if i:
#         print(i)
#0 duplikata

#provjera broja redova sa nepotpunim vrijednostima
# dim0=ds.shape
# ds.dropna(inplace = True)
# print(dim0)
# print(ds.shape)
# print((dim0[0]-ds.shape[0])/dim0[0])
#oko 50% redova imaju bar jednu nedostajucu vrijednost, prevelik procent da ih sve izbacimo

empty=[]
for i in ds:
    empty.append([ds[i].isnull().sum(),i])
empty.sort(reverse=True)
print(empty)
#najvise nedostajucih podataka je: company_type, company_size, gender, major_discipline
#zatim education level, last new job i enrolled university, ostala polja-zanemarljiv broj NaN




