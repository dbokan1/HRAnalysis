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

#print(ds['major_discipline'].value_counts())
# rod=ds.loc[:,["company_type","target","company_size","major_discipline"]]
# rod['company_type']=rod['company_type'].astype('category').cat.codes
# rod['company_size']=rod['company_size'].astype('category').cat.codes
# rod['major_discipline']=rod['major_discipline'].astype('category').cat.codes
# rod['target']=rod['target'].astype('category').cat.codes
# print(rod.corr())
#iz korelacije preostalih podataka vidimo da su velicina i tip firme usko povezane
#s obzirom da su bitne vrijednosti, najbolje je uvesti "Unknown" kako ne bi ugrozili integritet podataka

ds['company_type'].fillna(value='Unknown',inplace=True)
ds['company_size'].fillna(value='Unknown',inplace=True)
#jos jedna opcija je umetanje najcesce opcije, da se dalje testirati

#print(ds['major_discipline'].value_counts())
#major_discipline su raspoređeni 77% STEM, ostali jednocifreni i fali 15% materijala- popunicemo tim omjerima
ds['major_discipline'].fillna(value=np.random.choice(["STEM","Humanities","Other","Business Degree","Arts","No Major"],p=[0.88,0.041,0.023,0.02,0.02,0.016]),inplace=True)
#print(ds['major_discipline'].value_counts())

#training_hours numericki tip
# ds["training_hours"]=ds["training_hours"].astype('int')
# print("Mean: ",ds["training_hours"].mean())                   #srednja vrijednost 65.34
# print("Variance: ",ds["training_hours"].var())                #varijansa 3607
# print("Mode: ",ds["training_hours"].mode())                   #najvjerovatnija vrijednost 28
# print("Std:",ds["training_hours"].std())
ds=ds[(ds['training_hours'] < 250)]
#mozemo ukloniti outliere koji ne pripadaju u 3eps zoni oko mean vrijednosti-vrijednosti od 240 i iznad (uklanjamo cca 600 osoba)
ds["city_development_index"]=ds["city_development_index"].astype('double')
up=ds["city_development_index"].mean()+3*ds["city_development_index"].std()
down=ds["city_development_index"].mean()-3*ds["city_development_index"].std()
ds=ds[(ds['city_development_index']-up<0)]
ds=ds[(ds['city_development_index']-down>0)]

ds.to_csv("train_clean.csv")




