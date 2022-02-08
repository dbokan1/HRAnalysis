import pandas
import keras
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ds=pandas.read_csv("features_train.csv")
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

def train_model(model,params):
    gsc = GridSearchCV(model, params, cv=5)
    gsc.fit(X_train,y_train)
    gsc_best = gsc.best_estimator_
    print("Best Parameters : ", gsc.best_estimator_)
    y_pred=gsc_best.predict(X_test)
    metrics(y_pred,y_test)

train_model(LogisticRegression(random_state=123,solver='liblinear', penalty='l2', max_iter=5000),dict(C=np.logspace(1, 4, 10))) #sve oko 0.81
#train_model(KNeighborsClassifier(),{'n_neighbors': np.arange(1, 5)})
#train_model(RandomForestClassifier(random_state=0),{'n_estimators': [50, 100, 200]}) #sve oko 0.83


#
#
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
# # summarize history for accuracy
# plt.plot(history.history['recall'])
# plt.title('model recall')
# plt.ylabel('recall')
# plt.xlabel('epoch')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
#
# plt.show()
# predictions=model.predict(X_test).astype("bool")
# print("Accuracy:", accuracy_score(predictions, y_test))
# print("Precision:",precision_score(predictions, y_test))
# print("Recall:",recall_score(predictions, y_test))

#model.save("nn_prec_1")
# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))