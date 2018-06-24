'''
@author：KongWeiKun
@file: test.py
@time: 18-1-12 下午12:44
@contact: 836242657@qq.com
'''
import tensorflow as tf
import numpy as np

hello = tf.constant('Hello, TensorFlow')
sess = tf.Session()
print(sess.run(hello))
