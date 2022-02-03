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

# empty=[]
# for i in ds:
#     empty.append([ds[i].isnull().sum(),i])
# empty.sort(reverse=True)
# for i in range(len(empty)):
#     empty[i][0]/=ds.shape[0]
# print(empty)
#najvise nedostajucih podataka je: company_type 32%, company_size 30%, gender 23%, major_discipline 14%
#zatim education level 2%, last new job 2% i enrolled university 2%, ostala polja-zanemarljiv broj NaN
#na prvi pogled logicki je zakljuciti da parametri: last new job, enrolled university,company_size igraju vecu ulogu


#fali nezanemarljiv procent podataka o rodu
# print(ds['gender'].value_counts())
# rod=ds[["gender","target"]]
# rod=rod.replace(to_replace='Male', value=2)
# rod=rod.replace(to_replace='Female', value=3)
# rod=rod.replace(to_replace='Other', value=4)
# rod=rod.fillna(5)
# print(rod.corr())
#vidimo da je korelacija 0.07, sto nam govori da rod nema velik efekat na rezultat
#popunjavamo kolonu rod sa istom raspodjelom kao i trenutna-90.2% musko, 8.4% zensko, 1.3% other
#popunjavamo na ovaj nacin jer odlucivanjem za samo jedno opciju npr. svi elementi musko izaziva +23% disbalansa
dim=ds.shape
for i in range(dim[0]):
    if pd.isna(ds.at[i,'gender']):
        ds.at[i,'gender']=np.random.choice(['Male','Female','Other'],p=[0.9,0.08,0.02])



