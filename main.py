import numpy as np
import pandas as pd
import tensorflow as tf



ds=pd.read_csv("aug_train.csv")

# PROVJERA DUPLIKATA
# duplicate=ds["enrollee_id"].duplicated()
# for i in duplicate:
#     if i:
#         print(i)
#0 duplikata

#PROVJERA BROJA REDOVA SA NEPOTPUNIM VRIJEDNOSTIMA
# dim0=ds.shape
# ds.dropna(inplace = True)
# print(dim0)
# print(ds.shape)
# print((dim0[0]-ds.shape[0])/dim0[0])
#oko 50% redova imaju bar jednu nedostajucu vrijednost, prevelik procent da ih sve izbacimo


#histogram broja nedostajucih elemenata u redovima
# x=np.zeros([13,1])
# dim=ds.shape
# for i in range(dim[0]):
#     x[ds.iloc[i, ].isnull().sum(),0]+=1
# print(x)

#izbacujemo sve osobe koji imaju 5 ili vise nepopunjenih informacija- gubimo oko 250 redova sto je prihvatljivo
ds=ds.dropna(axis=0,thresh=5)

# empty=[]
# for i in ds:
#     empty.append([ds[i].isnull().sum(),i])
# empty.sort(reverse=True)
# for i in range(len(empty)):
#     empty[i][0]/=ds.shape[0]
# print(empty)
#najvise nedostajucih podataka je: company_type 31%, company_size 30%, gender 22%, major_discipline 13%
# last new job 1% education level 1% i enrolled university 1.4%,experience 0.2%
#na prvi pogled logicki je zakljuciti da parametri: last new job, enrolled university,company_size igraju vecu ulogu


#PODACI O KATEGORIJI GENDER
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

for i in range(ds.shape[0]):
    if pd.isna(ds.at[ds.index[i],'gender']):
        ds.at[ds.index[i],'gender']=np.random.choice(['Male','Female','Other'],p=[0.9,0.08,0.02])

#najvise nedostajucih podataka je: company_type 31%, company_size 30%, major_discipline 13%
# last new job 1% education level 1% i enrolled university 1.4%,experience 0.2%
#uklanjamo sve elemente koji nemaju last new job, education level, experience ili enrolled university (gubimo oko 4% seta)

ds=ds.dropna(axis=0,subset=['experience','enrolled_university','last_new_job','education_level'])

print(ds['company_type'].value_counts())
