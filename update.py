'''
@author：KongWeiKun
@file: update.py
@time: 18-1-9 下午6:51
@contact: 836242657@qq.com
'''
import pip
from subprocess import call

for dist in pip.get_installed_distributions():
    print('待更新: %s'%dist)
    # call("pip install --upgrade"+' '+dist.project_name,shell=True)
    # print('更新完毕%s'%dist)
