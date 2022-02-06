import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
ds=pd.read_csv("train_clean.csv")
relevant=ds.loc[:,"city":"target"]
for i in relevant:
    relevant[i]=relevant[i].astype('category').cat.codes

corr=relevant.corr()
print(corr["target"])
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()
#sa korelacione heatmape uocavamo: nema kolone koja posebno utice na target
#najveci efekat ima company_size, company_type, experience
#development index ima minimalan negativni efekat-bitan
#gender major discipline experience imaju korelaciju blizu 0
#city je slabo koreliran i sa cim, isto tako training hours, experience i gender
#najvece korelacije sa target:city_development_index 0.32, company_size 0.19,enrolled_university 0.15, relevent_experience 0.13,
#company_type 0.12, education level 0.08, city 0.04, training hours 0.02, last_new_job 0.016, major_discipline 0.013
#najneznacajnije-experience,gender,major_discipline,last_new_job,city


#city i city_index su usko korelirane
#city_index i target su usko krelirane
#gender nije ni sa cim usko koreliran
#relevent_experience je sa enroled university, company size i company_type usko koreliran
#enroled_university- sa company_size i type


