from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import numpy as np
import csv

features = []
for row in csv.reader(open('/home/vidhikatkoria/VR/GA2/CIFAR_VLAD copy/CIFAR_VLAD_SIFT_FEATURES.csv', 'r'), delimiter=','):
     features.append([float(i) for i in row])
train_label=features[len(features)-1]
train_features=features[:len(features)-1]

z = list(zip(train_features, train_label))
TRAIN, TEST = train_test_split(z, test_size=0.1, random_state=72)
train_features, train_label = zip(*TRAIN)
test_features, test_label = zip(*TEST)


test_features=np.array(test_features, dtype=np.float32)
train_features=np.array(train_features, dtype=np.float32)
test_label=np.array(test_label, dtype=np.float32)
train_label=np.array(train_label, dtype=np.float32)

print("\nK nearest Neighbours")
knn=KNeighborsClassifier()
knn.fit(train_features,train_label)
knn_predict=knn.predict(test_features)
actual_output=list(test_label)
print("\nAccuracy Score ", metrics.accuracy_score(actual_output, knn_predict))
print('Confusion Matrix : ',confusion_matrix(actual_output,knn_predict))
# print('Report : ',classification_report(actual_output,knn_predict))

print("\nlogistic regression")
log_reg=LogisticRegression()
log_reg.fit(train_features,train_label)
log_predict=log_reg.predict(test_features)
actual_output=list(test_label)
print("\nAccuracy Score ",metrics.accuracy_score(actual_output, log_predict))
print('Confusion Matrix : ',confusion_matrix(actual_output,log_predict))
# print('Report : ',classification_report(actual_output,log_predict))

print("\nSVM classification")
svmachine=SVC(kernel='poly',degree=100000)
svmachine.fit(train_features,train_label)
svmachine_predict=svmachine.predict(test_features)
actual_output=list(test_label)
print("\nAccuracy Score ",metrics.accuracy_score(actual_output, svmachine_predict))
print('Confusion Matrix : ',confusion_matrix(actual_output,svmachine_predict))
# print('Report : ',classification_report(actual_output,svmachine_predict))

print("\nada boost")
ada=GradientBoostingClassifier()
ada.fit(train_features,train_label)
ada_predict=ada.predict(test_features)
actual_output=list(test_label)
print("\nAccuracy Score ", metrics.accuracy_score(actual_output, ada_predict))
print('Confusion Matrix : ',confusion_matrix(actual_output,ada_predict))
# print('Report : ',classification_report(actual_output,knn_predict))
