import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('D:\python_exercise\python_ML\pca\data.txt')
#print(data)
# 按列求平均，即各个特征的平均
meanVal = np.mean(data, axis=0)  # 按列求均值
newdata = (data - meanVal) / np.std(data, axis=0)
# print(newdata)
def pca_by_feature(newdata, need_accumulative_contribution_rate=0.75):
    """协方差矩阵/相关矩阵求解主成分及其因子载荷量和贡献率（打印到控制台）
    :param data: 数据
    :param need_accumulative_contribution_rate: 需要达到的累计方差贡献率
    :return: 重构后的数据
    """
    # np.cov用来求协方差矩阵，参数rowvar=0说明数据一行代表一个样本
    covMat = np.cov(newdata, rowvar=0)
    print('协方差矩阵：', covMat)
    # 求解相关矩阵的特征值和特征向量
    features_value, features_vector = np.linalg.eig(covMat)
    print('特征值：',features_value)
    print('特征矩阵：',features_vector)#n_features*n_features矩阵，每一列是是一个特征向量
    # 依据特征值大小排序特征值
    value=sorted(features_value,reverse=True)#不影响原数组，用sort()方法会改变原数组
    print(value)
    #X[:, 1][:, np.newaxis] 索引多维数组的某一列时，返回的仍然是列的结构
    #np.vstack():在竖直方向上堆叠 np.hstack():在水平方向上平铺
    # 计算所需的主成分数量
    total_features_value = sum(value)  # 特征值总和
    need_accumulative_contribution_rate *= total_features_value
    n_principal_component = 0  # 所需的主成分数量
    accumulative_contribution_rate = 0
    while accumulative_contribution_rate < need_accumulative_contribution_rate:
        accumulative_contribution_rate += value[n_principal_component]
        n_principal_component += 1
    #求出所有主成分对应的特征向量
    Indice = np.argsort(features_value)#将原数组从小到大排序好并返回排好的数据在原数组中的位置
    ValIndice = Indice[-1:-(n_principal_component + 1):-1]#求出最大几个特征值的索引
    vector = features_vector[:, ValIndice]
    print(vector)
    # 输出单位特征向量和主成分的方差贡献率
    print("单位特征向量和主成分的方差贡献率")
    for i in range(n_principal_component):
        print("主成分:", i,
              "方差贡献率:", value[i] / total_features_value,
              "特征向量:", vector[:, i])
    #降维后的数据
    lowdata=newdata.dot(vector)
    #重构数据
    redata=lowdata.dot(vector.T)+meanVal
    return redata

    # # 计算各个主成分的因子载荷量：factor_loadings[i][j]表示第i个主成分对第j个变量的相关关系，即因子载荷量
    # factor_loadings = np.vstack(
    #     [[np.sqrt(features_value[i]) * features_vector[j][i] / np.sqrt(R[j][j]) for j in range(n_features)]
    #      for i in range(n_principal_component)]
    # )
    # #np.vstack 在竖直方向上堆叠
    # # 输出主成分的因子载荷量和贡献率
    # print("主成分的因子载荷量和贡献率")
    # for i in range(n_principal_component):
    #     print("主成分:", i, "因子载荷量:", factor_loadings[i])
    # print("所有主成分对变量的贡献率:", [np.sum(factor_loadings[:, j] ** 2) for j in range(n_features)])
redata=pca_by_feature(data)
plt.scatter(data[:,0],data[:,1])
plt.scatter(redata[:,0],redata[:,1],c='r')
#plt.show()
#SVD分解
def pca_by_svd(X, k):
    """数据矩阵奇异值分解进行的主成分分析算法
    :param X: 样本矩阵X
    :param k: 主成分个数k
    :return:
    """
    n_samples = X.shape[1]#求样本个数即列数
    # 构造新的n×m矩阵
    T = X.T / np.sqrt(n_samples - 1)
    # 对矩阵T进行截断奇异值分解
    U, S, V = np.linalg.svd(T)
    U= U[:,:k]
    # 求k×n的样本主成分矩阵
    return np.dot(X,U)
print(pca_by_svd(data, 1))





