'''
@author：KongWeiKun
@file: analysis.py
@time: 17-12-5 下午3:33
@contact: 836242657@qq.com
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('----------------------------------------------------------------------------------------------------------------------------')
df_train=pd.read_csv('./resourses/train.csv')
# print(df_train.head(2))
print(df_train.isnull().sum())#查看数据的缺省值

#把时间数据分割成年 月 日新特征
# df_train['year']=pd.DatetimeIndex(df_train['datetime']).year
# # print(df_train['year'][:10])
# df_train['month']=pd.DatetimeIndex(df_train['datetime']).month
# # print(df_train['month'][:2])
# df_train['day']=pd.DatetimeIndex(df_train['datetime']).day
# df_train['hour']=pd.DatetimeIndex(df_train['datetime']).hour
# # print(df_train['hour'][:4])
# df_train=df_train.drop(['datetime'],axis=1)#去除原特征值
print(df_train.head(2))

##单变量EDA 探索之 分组均值曲线
# fig,axs = plt.subplots(2,2,sharey=True)
# df_train.groupby('weather').mean().plot(y='count',marker='o',ax=axs[0,0])
# df_train.groupby('humidity').mean().plot(y='count',marker='*',figsize=(24,16),ax=axs[0,1])
# df_train.groupby('temp').mean().plot(y='count',marker='o',ax=axs[1,0])
# df_train.groupby('windspeed').mean().plot(y='count',marker='.',ax=axs[1,1])
# plt.savefig('single.jpg')
# plt.show()

# df_train.groupby('hour').mean().plot(y='count',marker='o')
# plt.title('mean count per hour')
# plt.savefig('hour.jpg')
# plt.show()

#分组散点图
# fig,axs = plt.subplots(2,3,sharey=True)
# df_train.plot(x='temp',y='count',kind='scatter',figsize=(24,16),ax=axs[0,0],color='magenta')
# df_train.plot(x='day',y='count',kind='scatter',ax=axs[0,1],color='cyan')
# df_train.plot(x='humidity',y='count',kind='scatter',ax=axs[0,2],color='bisque')
# df_train.plot(x='windspeed',y='count',kind='scatter',ax=axs[1,0],color = 'coral')
# df_train.plot(x='month',y='count',kind='scatter',ax=axs[1,1],color='darkblue')
# df_train.plot(x='hour',y='count',kind = 'scatter',ax=axs[1,2],color='deeppink')
# plt.savefig('every_group_scatter.jpg')
# plt.show()

#两两相关性分析
# column=df_train.columns
# corr=df_train[column].corr()
# plt.figure()
# plt.matshow(corr)
# plt.colorbar()
# plt.savefig('relative_of_both.jpg')
# plt.show()

df = df_train
name = df_train.drop(['count','casual','registered'],axis=1).columns
target = df_train['count'].values
feature = df_train.drop(['count','casual','registered'],axis=1).values
print(feature)
# from sklearn import preprocessing
# feature_scaled = preprocessing.scale(feature)
#
# from sklearn import cross_validation
# from sklearn import linear_model
# from sklearn import svm
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.learning_curve import learning_curve
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import explained_variance_score
# from sklearn.grid_search import GridSearchCV
#
# cv = cross_validation.ShuffleSplit(len(feature_scaled),n_iter=5,test_size=0.2,random_state=0)
# print('岭回归结果：')
# for train,test in cv:
#     reg2 = linear_model.Ridge().fit(feature_scaled[train],target[train])
#     print('train score:{0:.3f},test score:{1:.3f}\n'.format(reg2.score(feature_scaled[train],target[train]),reg2.score(feature_scaled[test],target[test])))
#
# print('svm结果：')
# for train,test in cv:
#     reg3 = svm.SVR().fit(feature_scaled[train],target[train])
#     print('train score:{0:.3f},test score:{1:.3f}\n'.format(reg3.score(feature_scaled[train],target[train]),reg3.score(feature_scaled[test],target[test])))
#
# print('随机森林回归结果：')
# for train,test in cv:
#     reg4 = RandomForestRegressor(n_estimators=100).fit(feature_scaled[train],target[train])
#     print('train score:{0:.3f},test score:{1:.3f}\n'.format(reg4.score(feature_scaled[train],target[train]),reg4.score(feature_scaled[test],target[test])))


#通过随机森特征选择，去掉特征对count贡献最小的三个特征‘workingday’,'holiday','day'
# feature = df_train.drop(['count','casual','registered','holiday','workingday','day'],axis=1).values
# from sklearn import preprocessing
# feature_scaled = preprocessing.scale(feature)
#
# #再次进行算法实现
# print('随机森林回归结果：')
# for train,test in cv:
#     reg4 = RandomForestRegressor(n_estimators=100).fit(feature_scaled[train],target[train])
#     print('train score:{0:.3f},test score:{1:.3f}\n'.format(reg4.score(feature_scaled[train],target[train]),reg4.score(feature_scaled[test],target[test])))

# # 随机森林优化
# target = df_train['count'].values
# feature = df_train.drop(['count', 'casual', 'registered'], axis=1).values
# feature_scaled = preprocessing.scale(feature)
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#     feature_scaled, target, test_size=0.2, random_state=0)
#
# tuned_parameters = [{'n_estimators': [10, 100, 500],
#                      'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}]
# scores = ['r2']
#
# for score in scores:
#
#     print(score)
#     clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
#     clf.fit(X_train, y_train)
#
#     print(clf.best_estimator_)
#     print("得分分别是:")
#     # grid_scores_的返回值:
#     #    * a dict of parameter settings
#     #    * the mean score over the cross-validation folds
#     #    * the list of scores for each fold
#     for params, mean_score, scores in clf.grid_scores_:
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean_score, scores.std() / 2, params))


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt
#
#
# title = "Learning Curves (Random Forest, n_estimators = 100)"
# cv = cross_validation.ShuffleSplit(feature_scaled.shape[0], n_iter=10, test_size=0.2, random_state=0)
# estimator = RandomForestRegressor(n_estimators=100, max_depth=10)
# plot_learning_curve(estimator, title, feature_scaled, target, (0.0, 1.01), cv=cv, n_jobs=1)
#
# plt.show()