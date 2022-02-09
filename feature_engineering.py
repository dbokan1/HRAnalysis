#analiza kolona iz eda:
    #city_development_index: 0.32 najjaca korelacija, dvije kategorije jako zastupljene (5k i 2.5k), trebalo bi balansirati i zadrzati
    #enrolled_university: 0.15 dobra korelacija, relativno balansiran (13k:4k:1k), eventualno neke manje elemente duplirati
    #education_level: 0.08 relativno slaba korelacija (ujedno najvise koreliran sa target), blaze disbalansiran (11:4:2:1), nije ni sa cim posebno korelirana
    #company_size: dobra korelacija (0.19 druga najveca), balansirana kolona

#city: 0.04 korelacija, jako koreliran sa city_development_index,relativno nebalansiran, previse kategorija
#gender: slaba korelacija, disbalansiran, trebalo bi ukloniti
#relevent_experience: 0.13 dobra korelacija, relativno balansirana (13k:6k- dvije kategorije), jako koreliran sa enroled university, major discipline i company size
#major_discipline: 0.013 relativno slaba korelacija, tesko disbalansirana, trebalo bi ukloniti
#experience: 0.005 slaba korelacija (najveca sa last new job 0.06), balansirana kolona
#company_type: dobra korelacija (0.12 cetvrta najveca), disbalansirana kolona, koreliran sa nekoliko drugih vrijednosti
#last_new_job: 0.01 slaba korelacija, relativno balansirana (jedna kolona je oko 40%, ostale uporedive), jaka korelacija sa relevent_experience, company_size i type
#training_hours: 0.02 relativno slaba korelacija, balansiran set, slabo koreliran sa svime


#biramo elemente koji imaju najvisu korelaciju sa target, a uklanjamo elemente koji imaju malu korelaciju sa target i veliku korelaciju sa korisnim elementima
#zadrzavamo: city_development_index, company_size, enrolled_university, education_level
#izbacujemo: city, gender, relevent_experience, major_discipline, experience, company_type, last_new_job, training_hours


#kategoricki one hot koding:
# encoded_columns = pd.get_dummies(data['column'])
# data = data.join(encoded_columns).drop('column', axis=1)


import pandas as pd

ds=pd.read_csv("train_clean.csv")
ds.drop(ds.columns[[0,1,2,4,5,8,9,11,12,13]], axis=1,inplace=True)

#KODIRANJE PODATAKA

ds["city_development_index"]=ds["city_development_index"].astype('double')


#company_size ima 8 kategorija i jedna unknown-optimalno za sortirani dummy encoding, gdje cemo unknown izostaviti
encoded_columns = pd.get_dummies(ds['company_size'])
ds = ds.join(encoded_columns).drop('company_size', axis=1)
ds=ds.drop("Unknown",axis=1)

#one-hot enkodiranje enrolled university
encoded_columns = pd.get_dummies(ds['enrolled_university'])
ds = ds.join(encoded_columns).drop('enrolled_university', axis=1)
ds=ds.drop("no_enrollment",axis=1)

#mozemo uduplati primary school i phd kako bi imale relevantniju reprezentaciju
phd=ds[ds["education_level"]=="Phd"]
primary=ds[ds["education_level"]=="Primary School"]
ds=pd.concat([ds,primary,phd])
encoded_columns = pd.get_dummies(ds['education_level'])
ds = ds.join(encoded_columns).drop('education_level', axis=1)
ds=ds.drop("Graduate",axis=1)

#ds.to_csv("features_train.csv")
