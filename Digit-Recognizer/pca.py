"""
@time : 2019/12/18下午12:37
@Author: kongwiki
@File: pca.py
@Email: kongwiki@163.com
"""
import numpy as np


def pca(X, k):
	"""
	PCA 实现
	:param X: 特征数据集
	:param k:  降到的目标维度
	:return: 降维后的数据
	"""
	n_samples, n_features = X.shape
	# 每个特征的均值
	mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
	# 标准化
	norm_X = X - mean
	# scatter matrix
	scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
	# 计算特征根和特征向量
	eig_val, eig_vec = np.linalg.eig(scatter_matrix)
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
	# 倒序排列
	eig_pairs.sort(reverse=True)
	# 选取前k个特征值
	feature = np.array([ele[1] for ele in eig_pairs[:k]])
	# 返回新的特征矩阵
	data = np.dot(norm_X, np.transpose(feature))
	return data
