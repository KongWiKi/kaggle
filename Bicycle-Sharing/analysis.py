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
# print(df_train.isnull().sum())#查看数据的缺省值

#把时间数据分割成年 月 日新特征
df_train['year']=pd.DatetimeIndex(df_train['datetime']).year
# print(df_train['year'][:10])
df_train['month']=pd.DatetimeIndex(df_train['datetime']).month
# print(df_train['month'][:2])
df_train['day']=pd.DatetimeIndex(df_train['datetime']).day
df_train['hour']=pd.DatetimeIndex(df_train['datetime']).hour
# print(df_train['hour'][:4])
df_train=df_train.drop(['datetime'],axis=1)#去除原特征值
# print(df_train.head(2))

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