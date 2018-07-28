from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd,numpy as np

data = pd.read_csv("/root/Documents/titanic dataset/train.csv",sep=',')
data=data.drop(['Name','SibSp','Pclass','Fare','Ticket','Cabin','Parch','Embarked','PassengerId'],1)
#print(data)
data=data.fillna(0)
feature=np.array(data.iloc[:,2:])
label=np.ravel(data.Survived)


#USING LOGISTIC REGRESSION
lr=LogisticRegression(penalty='l2')
lr.fit(feature,label)

xtrain,xtest,ytrain,ytest=train_test_split(feature,label,test_size=0.5)
predict=lr.predict(xtest)
print('THE ACCURACY  IN LOGISTIC IS',accuracy_score(ytest,predict))
#USING DECISION TREE
clf=DecisionTreeClassifier(max_depth=7)
clf.fit(feature,label)

xtrain1,xtest1,ytrain1,ytest1=train_test_split(feature,label,test_size=0.5)
predict=clf.predict(xtest1)
print('THE ACCURACY  IN DECISION TREE IS',accuracy_score(ytest1,predict))

#USING K NEIGHBOUR
knn=DecisionTreeClassifier()
knn.fit(feature,label)

xtrain2,xtest2,ytrain2,ytest2=train_test_split(feature,label,test_size=0.5)
predict=knn.predict(xtest2)
print('THE ACCURACY  IN KNEIGHBOR IS',accuracy_score(ytest1,predict))

