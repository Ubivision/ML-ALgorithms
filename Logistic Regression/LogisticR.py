import pandas as pd
import numpy as np

'''
除了代价函数和梯度下降部分不一样以外，其余部分和多元线性回归一致
'''
'读取数据'
# 路径自行修改
train = pd.read_csv('data/train.csv')


'数据预处理'
# 显示数据集基本信息（列数、列名、非空行个数、数据类型）
#print(train.info())

# 用年龄的中位数填充缺失的年龄
train['Age'] = train['Age'].fillna(train['Age'].median())
# fillna()--Pandas函数，填充空数据，参数为想要填充的数值
# median()--numpy函数，求中位数

# 统计登船地点中出现最多的那一个，用此登船地点填充缺失的登船地点
emb = train['Embarked'].value_counts()
# value_counts()--统计指定列每个元素出现的次数
fillstr = emb.idxmax(axis=1)
# idxmax()--返回在给定dataframe上第一次出现最大值的索引，axis=0时，返回每列最大值索引，
# axis=1时，返回每行最大值索引
train['Embarked'] = train['Embarked'].fillna(fillstr)

# 为方便后续计算，将本来是用字符表示的性别和登船地点修改为用数字表示
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train.loc[train['Embarked'] == 'C', 'Embarked'] = 0
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 1
train.loc[train['Embarked'] == 'S', 'Embarked'] = 2
# loc()中第一个参数表示要筛选的行的条件，第二个参数表示要筛选的列的条件


'特征归一化（特征缩放）（选择哪些特征作为属性具有主观性，可自行根据分析结果选择）'
x = train[["Pclass", "Age", "Parch", "Fare", "Sex", "Embarked"]]
y = train['Survived']

y_train = y.values.reshape(-1, 1)
# train集的y矩阵
# 本来survived中一列的数据，经过values后会返回成为一行的数据，所以需要reshape
# reshape(-1, 1):将矩阵重组为x行一列的形式，其中x通过元素个数/1来计算

avg = np.zeros(x.shape[1])
std = np.zeros(x.shape[1])
# 初始化一个特征个数个的全0数组，用于存放特征的平均值和标准差

avg = np.mean(x)
std = np.std(x)
# mean和std都默认按列求均值和标准差，这块我也没搞懂，不知道为什么就默认按列求了

x_norm = (x - avg) / std
# 特征缩放公式


'初始化矩阵'
x_norm = x_norm.values
# 将x_norm的dataframe数据结构转换为numpy的array，方便计算
x_norm = np.insert(x_norm, 0, np.ones(x_norm.shape[0]), axis=1)
x_norm = np.array(x_norm, dtype=np.float64)
# 为数据矩阵的最左边一列添加1，方便之后的矩阵运算

theta = np.zeros(x_norm.shape[1]).reshape(-1, 1)
# 初始化参数矩阵

'代价函数'
def cost(x, y, theta):
    m = x.shape[0]
    z = x.dot(theta)
    h = 1 / (np.exp(-z) + 1)
    cost = -1/m*np.sum((y*np.log(h)+(1-y)*np.log(1-h)))
    return cost

'梯度下降'
def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    cost_history = np.zeros(num_iters)
    z = x.dot(theta)
    h = 1 / (np.exp(-z) + 1)
    for i in range(num_iters):
        theta = theta - alpha/m*np.dot(x.T, (h-y))
        cost_history[i] = cost(x, y, theta)
    return theta, cost_history

'规定学习率，迭代次数，并开始训练'
alpha = 0.01
iters = 800
[theta, cost_history] = gradientDescent(x_norm, y_train, theta, alpha, iters)
print(cost_history)


'保存训练结果，因为得到的预测结果是介于0-1之间的数，故通过阈值法将结果转换为只有0和1'
result = x_norm.dot(theta)
for i in range(len(result)):
    if result[i] >= 0.5:
        result[i] = 1
    else:
        result[i] = 0


'计算训练集正确率'
num_correct = 0
for i in range(len(result)):
    if result[i] == y[i]:
        num_correct += 1
accuracy_train = num_correct / len(result)
print('在训练集上的正确率为：'+str(accuracy_train))



'''
测试集（处理过程和训练集差不多，只是测试集中没有Survived列，需要上传结果到kaggle后才能知道准确率）
'''
'读取数据'
test = pd.read_csv('data/test.csv')
# print(test.info())


'数据预处理'
test['Age'] = test['Age'].fillna(test['Age'].median())

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test.loc[test['Embarked'] == 'C', 'Embarked'] = 0
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 1
test.loc[test['Embarked'] == 'S', 'Embarked'] = 2


'特征归一化'
x = test[["Pclass", "Age", "Parch", "Fare", "Sex", "Embarked"]]
avg = np.zeros(x.shape[1])
std = np.zeros(x.shape[1])

avg = np.mean(x)
std = np.std(x)

x = (x - avg) / std

x_test = x.values
x_test = np.insert(x_test, 0, np.ones(x.shape[0]), axis=1)


'保存预测结果，因为得到的预测结果是介于0-1之间的数，故通过阈值法将结果转换为只有0和1'
result = x_test.dot(theta)
for i in range(len(result)):
    if result[i] >= 0.5:
        result[i] = 1
    else:
        result[i] = 0


'导出kaggle要求格式的预测文件'
passengerId = test['PassengerId'].values.reshape(-1, 1)
list = np.hstack((passengerId, result))
column = ['PassengerId', 'Survived']
resultFile = pd.DataFrame(columns=column, data=list)
resultFile.to_csv('data/LogesticRSubmission.csv', index=None)
