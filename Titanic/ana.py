'''
@author：KongWeiKun
@file: ana.py
@time: 18-1-6 下午10:06
@contact: 836242657@qq.com
'''
'''
利用xgboost
'''
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.grid_search import  GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb

#读取数据
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
# print(train.info(),test.info())

def clean_data(titanic):
    titanic["Age"] = titanic['Age'].fillna(titanic['Age'].median())
    titanic['child'] = titanic['Age'].apply(lambda x:1 if x < 15 else 0)

    titanic['sex'] = titanic['Sex'].apply(lambda x:1 if x == "male" else 0)

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3

    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1

    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0

    titanic["name"] = titanic["Name"].apply(getName)

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic


# 对数据进行清洗
train_data = clean_data(train)
test_data = clean_data(test)

features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]
X_train = train[features].as_matrix()
Y_train = train['Survived']
# print(X_train)
#训练模型
model = xgb.XGBClassifier(max_depth=2,n_estimators=300,learning_rate=0.1,silent=True,objective='binary:logistic').fit(X_train,Y_train)
#测试特征
X_test = test[features].as_matrix()
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
print('写入文件')
submission.to_csv("./data/xgboost_submission.csv",index=False)
