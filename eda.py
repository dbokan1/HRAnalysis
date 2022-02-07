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

corr=relevant.corr().abs()

print(corr[["city","education_level"]])
print(corr["city"].sum())
print(corr["education_level"].sum())

# sns.heatmap(corr,
#         xticklabels=corr.columns,
#         yticklabels=corr.columns)
# plt.show()

#sa korelacione heatmape uocavamo:
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
# relevant=ds.loc[:,"city":"target"]
# for i in relevant:
#     relevant[i]=relevant[i].astype('category')
#
# #BOX PLOTOVI
# # plt.figure(figsize = (10, 6))
# # ax = sns.boxplot(x='company_size', y='company_type', data=relevant)
# # plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
# # plt.xticks(rotation=45)
# # plt.show()
#
# #
# sns.countplot(x=relevant.columns[0], alpha=0.7, data=relevant)
# plt.show()
# #vidimo da oko 50% stanovnika dolazi iz dva grada, raspodjela je relativno nebalansirana, visoki peekovi u odnosu na ostatak, nekoliko zastupljenijih gradova
# #
# sns.countplot(x=relevant.columns[1], alpha=0.7, data=relevant)
# plt.show()
# #slicna situacija i sa city_development_index, ima jedna ili dvije jako zastupljene kategorije i velik broj manjih
#
# sns.countplot(x=relevant.columns[2], alpha=0.7, data=relevant)
# plt.show()
# # raspodjela rodova je također nebalansirana, 90% osoba su muskog roda, oko 9% zenskog, 1% other
#
# sns.countplot(x=relevant.columns[3], alpha=0.7, data=relevant)
# plt.show()
# #kategorija relevent_experience je omjera 13000:6000 i postoje samo dvije kategorije
# #
# sns.countplot(x=relevant.columns[4], alpha=0.7, data=relevant)
# plt.show()
# #enrolled university je također nebalansirana- no enrollment 13000, full time course 4000, part time 1000
#
# sns.countplot(x=relevant.columns[5], alpha=0.7, data=relevant)
# plt.show()
# # education level blaze disbalansiran, graduate 11000, masters 4000, high school 2000,phd i primari skoro 1000
#
# sns.countplot(x=relevant.columns[6], alpha=0.7, data=relevant)
# plt.show()
# #major_discipline tesko disbalansiran, 16000/18000 je STEM, ostale kategorije ispod 500
# #
# sns.countplot(x=relevant.columns[7], alpha=0.7, data=relevant)
# plt.show()
# #experience prva kategorija koja je relativno balansirana, iako kategorija >20 je veća od ostalih, prestavlja samo 3000 elemenata, velik broj podjednako zastupljenih kategorija
#
# sns.countplot(x=relevant.columns[8], alpha=0.7, data=relevant, order=["<10","10/49","50-99","100-500","500-999","1000-4999","5000-9999","10000+","Unknown"])
# plt.show()
# #company_size podaci su balansirani, najvise ima u opsegu 50-99(oko 3000)
#
# sns.countplot(x=relevant.columns[9], alpha=0.7, data=relevant)
# plt.show()
# #company_type kolona je disbalansirana, Pvt Ltd ima oko 10k elemenata, ostali znatno manje
#
# sns.countplot(x=relevant.columns[10], alpha=0.7, data=relevant)
# plt.show()
# #last_new_job jedna kategorija je dominantno zastupljena, ostale su balansirane
#
# relevant["training_hours"]=relevant["training_hours"].astype('int')
# sns.countplot(x=relevant.columns[11], alpha=0.7, data=relevant)
# plt.show()
# #distribucija ove kategorije podsjeca na rayleighovu distribuciju
# #posto je numericki tip mozemo naci srednju vrijednost i varijansu
# # print("Mean: ",relevant["training_hours"].mean())                   #srednja vrijednost 65.34
# # print("Variance: ",relevant["training_hours"].var())                #varijansa 3607
# # print("Mode: ",relevant["training_hours"].mode())                   #najvjerovatnija vrijednost 28
# # print("Std:",relevant["training_hours"].std())                       #standardna devijacija 60
#
#
#
# sns.countplot(x=relevant.columns[12], alpha=0.7, data=relevant)
# plt.show()
#vidimo da 14000 osoba ima target 0, dok 4000 target 1

#primjecujemo da ima nebalansiranih kolona, kao sto su major discipline, company_type, gender
#problem disbalansiranih kolona je sto ce model pri treniranju stvoriti bias ka najzastupljenijim vrijednostima i kompromizirati integritet ucenja
#jos jedan problem je koristenje accuracy metrike, za disbalansirane setove daje pogresne podatke i trebali bi koristiti neke druge
# s obzirom da gender nije usko koreliran ni sa cim, prijedlog je tu kolonu izbaciti
# major discipline je samo vezan za grad i univerzitet osobe, prijedlog je tu kolonu izbaciti
#company_type je bitan za target ali je također disbalansiran- moramo ili ukloniti neke elemente ili duplirati manje zastupljene
