import pandas as pd

df = pd.read_csv('processed_titanic.csv')

print(df.head())
print(df.dtypes)
df.drop(columns=["Name"], inplace=True)

from sklearn.model_selection import train_test_split
Xr = df.drop(columns=['Age'])
yr = df['Age']

Xcl = df.drop(columns=['Survived', 'PassengerId'])
ycl = df['Survived']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.4, random_state=42)
Xcl_train, Xcl_test, ycl_train, ycl_test = train_test_split(Xcl, ycl, test_size=0.4, random_state=42)


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_modelcl = LinearRegression()
linear_model.fit(Xr_train, yr_train)
linear_modelcl.fit(Xcl_train, ycl_train)
yr_pred_test = linear_model.predict(Xr_test)
ycl_predr_test = linear_modelcl.predict(Xcl_test)

from sklearn.metrics import mean_squared_error
print("\nMSE:", mean_squared_error(yr_test, yr_pred_test))

from sklearn.metrics import root_mean_squared_error
print("RMSE:", root_mean_squared_error(yr_test, yr_pred_test))

from sklearn.metrics import mean_absolute_error
print("MAE:", mean_absolute_error(yr_test, yr_pred_test))

from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression(max_iter=5000)
logreg_model.fit(Xcl_train, ycl_train)
ycl_pred_test = logreg_model.predict(Xcl_test)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
print("\nAccuracy:", accuracy_score(ycl_test, ycl_pred_test))
print("Precision:", precision_score(ycl_test, ycl_pred_test))
print("Recall:", recall_score(ycl_test, ycl_pred_test))
print("F1 Score:", f1_score(ycl_test, ycl_pred_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ycl_test, ycl_pred_test)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

ycl_predr_test_binary = [1 if x >= 0.5 else 0 for x in ycl_predr_test]
cma = confusion_matrix(ycl_test, ycl_predr_test_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cma, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

yr_pred_test_binary = [1 if x >= 0.5 else 0 for x in yr_pred_test]
yrefefwf = [1 if x >= 0.5 else 0 for x in yr_test]
cmx = confusion_matrix(yrefefwf, yr_pred_test_binary)
plt.figure(figsize=(4, 4))
sns.heatmap(cmx, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("\nPrecision:", precision_score(yrefefwf, yr_pred_test_binary))
print("Recall:", recall_score(yrefefwf, yr_pred_test_binary))

