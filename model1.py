from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle5 as pickle

ds=pd.read_csv("features_train.csv")
Y=ds["target"]
X=ds.drop("target",axis=1)
smote = SMOTE()
X, Y = smote.fit_resample(X, Y)
X_train,X_test,y_train,y_test = train_test_split(X, Y , test_size=0.25, random_state=0)

# model = LogisticRegression()
# model.fit(X_train, y_train,)
#
# predictions=model.predict(X_test)
# cm=confusion_matrix(y_test, predictions)
# print(metrics.classification_report(y_test, predictions))
# TN,FP,FN,TP=confusion_matrix(y_test, predictions).ravel()
# print('True Positive(TP)= ', TP)
# print('False Positive(FP)=', FP)
# print('True Negative(TN)= ', TN)
# print('False Negative(FN)=', FN)
# accuracy =  (TP+TN) /(TP+FP+TN+FN)
# print('Preciznost logisticke regresije= {:0.3f}'.format(accuracy))
#Zakljucujemo da LogisticRegression nije dobar za ovaj problem

#Radi efikasnije pretrage dobrog modela uvodimo više njih i uporedo testiramo
models = {}

from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier(n_neighbors=5)


accuracy_test, accuracy_train, precision_test, precision_train, recall_test, recall_train, roc_auc_test, roc_auc_train = {}, {}, {}, {}, {}, {}, {}, {}
for key in models.keys():
    models[key].fit(X_train, y_train)
    predictions = models[key].predict(X_test)
    pred_train=models[key].predict(X_train)

    accuracy_test[key] = accuracy_score(predictions, y_test)
    accuracy_train[key] = accuracy_score(pred_train, y_train)

    precision_test[key] = precision_score(predictions, y_test)
    precision_train[key] = precision_score(pred_train, y_train)

    recall_test[key] = recall_score(predictions, y_test)
    recall_train[key] = recall_score(pred_train, y_train)

    roc_auc_test[key] = roc_auc_score(predictions, y_test)
    roc_auc_train[key] = roc_auc_score(pred_train, y_train)

# filename='random_forest.sav'
# pickle.dump(models['Random Forest'], open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy Test', 'Precision Test', 'Recall Test', 'ROC-AUC Test','Accuracy Train','Precision Train', 'Recall Train', 'ROC-AUC Train'])
df_model['Accuracy Test'] = accuracy_test.values()
df_model['Accuracy Train'] = accuracy_train.values()
df_model['Precision Test'] = precision_test.values()
df_model['Precision Train'] = precision_train.values()
df_model['Recall Test'] = recall_test.values()
df_model['Recall Train'] = recall_train.values()
df_model['ROC-AUC Test'] = roc_auc_test.values()
df_model['ROC-AUC Train'] = roc_auc_train.values()
df_model.sort_values(by='Precision Test', ascending=False)
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(df_model)

ax = df_model.plot.bar(rot=45)
ax.legend(ncol=len(models.keys()), bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 14})
plt.tight_layout()
plt.show()

#Accuracy- standardna metrika, govori koliko tacno nas sistem klasifikuje elemente
#Precision-odnos tacno predviđenih pozitivnih vrijednosti u odnosu na sve pozitivne vrijednosti
#Recall-odnos tacno predviđenih pozitivnih vrijednosti u odnosu na sve observacije
#AUC-ROC- govori koliko model razlikuje klase, kombinacija precision i recall

#znacajnije nam je gledati Precision i Recall metrike nego Accuracy, zbog postojanja Accuracy paradoksa za binarnu klasifikaciju

#                               Accuracy Test  Precision Test  Recall Test  ROC-AUC Test  Accuracy Train  Precision Train  Recall Train  ROC-AUC Train
# Support Vector Machines       0.572722        0.955844     0.547782      0.666296        0.563937         0.957367      0.533543       0.671112
# Decision Trees                0.822722        0.825455     0.828251      0.822597        0.999822         0.999642      1.000000       0.999824
# Random Forest                 0.838039        0.828831     0.851387      0.838061        0.999778         0.999821      0.999731       0.999778
# Naive Bayes                   0.768247        0.846494     0.739338      0.774346        0.768182         0.850246      0.727879       0.775754
# Logistic Regression           0.762520        0.769610     0.767815      0.762368        0.762943         0.773668      0.754410       0.763089
# K-Nearest Neighbor            0.660762        0.715065     0.655008      0.661543        0.775952         0.830542      0.746198       0.779389

#primjecujemo da je Random Forest overall najbolje fitan, posebno u roc aoc metrici, nakon njega Decision Trees
#SVM ima najbolji precision
#Decision Trees i Random Forest su overfitani na trening setu
