'''
@author：KongWeiKun
@file: analysis.py
@time: 18-1-9 下午4:07
@contact: 836242657@qq.com
'''
'''
简单的练手
'''
import pandas as pd
import numpy as np

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
print('--------------------------加载数据-------------------------------')
# print(train.info(),test.info())
print('--------------------------清洗数据-------------------------------')
#age 缺省值比较多 采用Age的平均值进行填充
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
#sex 属性改变 由男 女变为1 0
train['Sex'] = train['Sex'].apply(lambda x:1 if x=='male' else 0)
test['Sex'] = test['Sex'].apply(lambda x:1 if x=='male' else 0)

#特征选择
feature = ['Age','Sex']
from sklearn import tree
print('--------------------------建模-------------------------------')
dt = tree.DecisionTreeClassifier()
dt = dt.fit(train[feature],train['Survived'])
print('--------------------------预测-------------------------------')
predict_data = dt.predict(test[feature])
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predict_data
})
print('--------------------------写入文件-------------------------------')
submission.to_csv('./data/submission_decision_tree.csv',index=False)#不添加索引