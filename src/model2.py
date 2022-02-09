import pandas
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,KFold, StratifiedKFold, LeaveOneOut, train_test_split, GridSearchCV
import pickle5 as pickle

ds=pandas.read_csv("../resources/features_train.csv")
Y=ds["target"]
X=ds.drop("target",axis=1)
smote = SMOTE()
X, Y = smote.fit_resample(X, Y)
X_train,X_test,y_train,y_test = train_test_split(X, Y , test_size=0.25, random_state=0)


def metrics(Y_test,Y_pred):
    print("Accuracy :",accuracy_score(Y_pred,Y_test))
    print("Precision Score : ",precision_score(Y_pred,Y_test))
    print("Recall Score : ",recall_score(Y_pred,Y_test))
    print("ROC AUC Score : ",roc_auc_score(Y_pred,Y_test))
    print("Classification Report\n\n",classification_report(Y_pred,Y_test))
    print(confusion_matrix(Y_pred,Y_test))

def train_GS(model,params):
    gsc = GridSearchCV(model, params, cv=3, verbose=2) #postavljamo cross validation na 5
    gsc.fit(X_train,y_train)
    gsc_best = gsc.best_estimator_
    print("Best Parameters : ", gsc.best_estimator_)
    y_pred=gsc_best.predict(X_test)
    metrics(y_pred,y_test)
    filename = '../models/random_forest_tuned.sav'
    # pickle.dump(gsc_best, open(filename, 'wb'))

#train_GS(LogisticRegression(random_state=123,solver='liblinear', penalty='l2', max_iter=5000),dict(C=np.logspace(1, 4, 10))) #sve oko 0.81
#train_GS(KNeighborsClassifier(),{'n_neighbors': np.arange(1, 5)})
#train_GS(RandomForestClassifier(random_state=0),{'n_estimators': [10,50,100, 200,500],'max_features': ['auto', 'sqrt', 'log2'],'max_depth' : [4,5,6,7,8],'criterion' :['gini', 'entropy'] }) #sve oko 0.83

# Best Parameters :  RandomForestClassifier(max_depth=8, n_estimators=500, random_state=0)
# Accuracy : 0.824453915823122
# Precision Score :  0.8129016312407316
# Recall Score :  0.8542857142857143
# ROC AUC Score :  0.8236710146059516
# Classification Report
#
#                precision    recall  f1-score   support
#
#          0.0       0.84      0.79      0.81      3658
#          1.0       0.81      0.85      0.83      3850
#
#     accuracy                           0.82      7508
#    macro avg       0.83      0.82      0.82      7508
# weighted avg       0.83      0.82      0.82      7508


#Biramo stratifiedKFold cross validation jer dobro radi za nebalansirane podatke
stratifiedkf=KFold(n_splits=10)
score=cross_val_score(RandomForestClassifier(),X,Y,cv=stratifiedkf,scoring='recall')
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))

#n=10, accuracy
# Cross Validation Scores are [0.24475524 0.73459873 0.75124875 0.65867466 0.71395271 0.72261072
#  0.63503164 0.95304695 0.94771895 0.95437895]
# Average Cross Validation score :0.7316017316017317
#n=10, recall
# Cross Validation Scores are [1.         0.75414365 0.71449704 0.68758717 0.67906336 0.7080292
#  0.96668549 0.96503497 0.96070596 0.95737596]
# Average Cross Validation score :0.8393122786091013



#Ideja: dizajnirati neuralnu mrezu, zasad nije pokazala bolje rezultate od Random Forest
#
# def create_baseline():
#     model = Sequential()
#     model.add(Dense(64, input_shape=(16,), activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Recall()])
#     return model
#
#
# model=create_baseline()
# history=model.fit(X_train,y_train,epochs=100,verbose=1,batch_size=16,shuffle=True)

# print(history.history.keys())
# plt.plot(history.history['recall'])
# plt.title('model recall')
# plt.ylabel('recall')
# plt.xlabel('epoch')
# plt.show()

# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

# predictions=model.predict(X_test).astype("bool")
# print("Accuracy:", accuracy_score(predictions, y_test))
# print("Precision:",precision_score(predictions, y_test))
# print("Recall:",recall_score(predictions, y_test))

#model.save("nn_prec_1")
# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, Y, cv=kfold)