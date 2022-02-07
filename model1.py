from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

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

models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    # Fit the classifier model
    models[key].fit(X_train, y_train)

    # Prediction
    predictions = models[key].predict(X_test)

    # Calculate Accuracy, Precision and Recall Metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()


ax = df_model.plot.bar(rot=45)
ax.legend(ncol=len(models.keys()), bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 14})
plt.tight_layout()
plt.show()