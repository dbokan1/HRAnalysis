from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd

ds=pd.read_csv("features_train.csv")
y=ds["target"]
x=ds.drop("target",axis=1)

X_train,X_test,y_train,y_test = train_test_split(x, y , test_size=0.25, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions=model.predict(X_test)
cm=confusion_matrix(y_test, predictions)

TN,FP,FN,TP=confusion_matrix(y_test, predictions).ravel()

print('True Positive(TP)= ', TP)
print('False Positive(FP)=', FP)
print('True Negative(TN)= ', TN)
print('False Negative(FN)=', FN)

accuracy =  (TP+TN) /(TP+FP+TN+FN)
print('Preciznost logisticke regresije= {:0.3f}'.format(accuracy))
