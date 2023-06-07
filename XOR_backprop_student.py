import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import math


#### 实验1：利用PyTorch的自动求导机制完成反向传播，并验证与DIY的结果是否一致
# Hint：所有的参数和运算都用torch中的数据形式和函数进行定义，数据也需要打包为tensor形式

#定义模型，并初始化参数
#MLP模型：              
#       输入层  n_x个神经元#
#       隐藏层  n_h个神经元#
#       输出层  n_y个神经元#
def initialize_parameters(n_x, n_h, n_y, W_correct):
    
    # 模型参数用tensor表示，使用标准正态分布分配参数#
    # 输入层到隐藏层的权重和偏置#
    # W1 = torch.randn(n_h, n_x).double() * W_corect
    # b1 = torch.zeros(n_h, 1).double()
    # #隐藏层到输出层的权重和偏置#
    # W2 = torch.randn(n_y, n_h).double() * W_corect
    # b2 = torch.zeros(n_y, 1).double()
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * W_correct
    #偏置#
    b1 = np.zeros((n_h, 1))
    #隐藏层-输出层#
    #权重#
    W2 = np.random.randn(n_y, n_h) * W_correct
    #偏置#
    b2 = np.zeros((n_y, 1))
    
    W1, b1 = torch.tensor(W1), torch.tensor(b1)
    W2, b2 = torch.tensor(W2), torch.tensor(b2)
    #请求梯度#
    W1.requires_grad_()
    b1.requires_grad_()
    W2.requires_grad_()
    b2.requires_grad_()

    # 将参数传递给下一个变量#
    parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return parameters


#前向传播
def forward_prop(X,parameters):
    #首先提取相关的参数#
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 利用torch中的tensor运算，依次计算每层的z和a
    # 例如：
    # z1 = torch.add(torch.matmul(W, X), b1)
    # a1 = 1/(1+torch.exp(-z1))
    #之后计算向前传播的过程#
    #每一层分为两步：线性求和和非线性变换#

    #输入层到隐藏层的线性求和：使用矩阵线性变换#
    z1 = torch.matmul(W1, X) + b1
    #输入层到隐藏层的非线性变换：使用sigmoid激活函数#
    a1 = torch.sigmoid(z1)

    #隐藏层到输出层的线性求和#
    z2 = torch.matmul(W2, a1) + b2
    #隐藏层到输出层的非线性变换#
    a2 = torch.sigmoid(z2)

    y_pred = a2
    
    return y_pred


#损失函数计算, 加入Method选择节点
def calculate_cost(y_pred, Y, Method):

    #通过对比预测值和真实值来计算损失#
    # 定义损失函数，均方根或交叉熵
    #使用均方根#
    if Method == 'RMSE':
        RMSE_lose = torch.nn.MSELoss()
        cost = RMSE_lose(y_pred, Y)
    elif Method == 'BCE':
        BCE_lose = torch.nn.BCELoss()
        cost = BCE_lose(y_pred, Y)

    return cost


# 利用训练完的模型进行预测
def predict(X, parameters):
    #采用向前传播函数#
    y_predict = forward_prop(X, parameters)

    return y_predict #> 0.5#

# 主程序
if __name__ == '__main__':
    # # 准备训练数据和标签
    # np.random.seed(2)
    # #样本值#
    # N = 200  
    # x = np.random.uniform(0, 1, N)
    # y = np.random.uniform(0, 1, N)
    # # 对于每个类别，我们生成了N个具有噪声的样本。#
    # # 第一象限：1，1：0 #
    # X1 = np.column_stack((x, y)) 
    # Y1 = np.zeros(N)  
    # # 第三象限：0，0：0 #
    # X2 = np.column_stack((-x, -y)) 
    # Y2 = np.zeros(N)  
    # # 第二象限：1，0：0 #
    # X3 = np.column_stack((-x, y))  
    # Y3 = np.ones(N) 
    # # 第四象限：0，1：0 #
    # X4 = np.column_stack((x, -y))
    # Y4 = np.ones(N)  

    # # 合并样本 #
    # X = np.concatenate((X1, X2, X3, X4), axis=0)
    # Y = np.concatenate((Y1, Y2, Y3, Y4), axis=0)

    # # 转换为torch.tensor #
    # X, Y = torch.tensor(X), torch.tensor(Y)


    # # 打乱数据 #
    # indices = torch.randperm(X.shape[0])
    # X, Y = X[indices], Y[indices]
    # # # 将数据分成两个类别 #
    # X0 = X[Y.squeeze() == 0]
    # X1 = X[Y.squeeze() == 1]
    # X, Y = X.T, Y.unsqueeze(1).T
    # # # 绘制类别0的散点图
    # plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 0')

    # # 绘制类别1的散点图
    # plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1')

    # plt.legend()  # 显示图例
    # plt.show()  # 显示图像
    #4个样本调试#
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float64)
    Y = np.array([[0, 1, 1, 0]], dtype=np.float64)
    
    X, Y = torch.tensor(X), torch.tensor(Y)

    # 超参数设置

    num_of_iters = 10000      # 迭代次数
    learning_rate = 0.1     # 学习率

    # 模型定义与初始化 # n_x, n_y: 输入/输出的神经元个数，中间是自己定义的MLP神经元个数
    
    # init_params = initialize_parameters(n_x, #自行补充#, n_y) 
    #输入层2个神经元，输出层1个神经元，隐藏层配置5个#
    init_params = initialize_parameters(2, 10, 1, 1)
    #提供W1接口，验证DIV是否成功#
    # W1 = init_params["W1"]

    # 训练模型
    #cost_iter存储每一次更新权重后的损失函数#
    cost_iter = [] # 保存损失函数的变化
    for i in range(0, num_of_iters): # 每次迭代

        # 正向传播
        y_pred=forward_prop(X, init_params)
        
        # 计算损失函数
        cost=calculate_cost(y_pred, Y, 'RMSE')

        # 梯度下降并更新参数
        # 自动求导
        cost.backward()

        # 查看自动求导之后各参数的梯度，例如：print("dW1:", W1.grad.numpy())，检验是否与DIY的结果一致
        #查看输入层到隐藏层的权重梯度#
        # print("dw1:",W1.grad.numpy())

        with torch.no_grad():
            # 关闭梯度计算，实现参数更新#
            for params in init_params.values():
                params -= learning_rate * params.grad
        #梯度清零：
        #   每次调用.backward()后，梯度会被累积（而不是被替换）到.grad属性中去。
        #   如果我们不清零梯度，那么就会把新的梯度值和上一步的梯度值混在一起，这会导致计算错误。# 

        for params in init_params.values():
            params.grad.zero_()
    
        cost_iter.append(cost.item())
    
    # 可设置迭代停止条件
    #比如若选取95%的置信度，那么可以设置cost<5%停止#

    # 观察损失函数下降情况
    #每100次打点#
        if(i%100==0):
            print('cost after iteration#{:d}:{:f}'.format(i,cost))
    
    # 得到最终的参数值，例如：W1_star = W1.detach()
    # print(init_params)

   
    # 测试数据

    X_test = torch.tensor([[0,0,1,1], [0,1,0,1]], dtype=torch.float64)
    y_predict = predict(X_test, init_params)
    print(y_predict)
    # print(init_params["W1"])

    # # # 绘制损失函数随迭代次数变化的曲线图，cost_iter, num_of_iters
    # plt.plot(range(num_of_iters), cost_iter)
    # plt.title('RMSE of Datasets') #此处可以更改名字#
    # plt.xlabel('Num')
    # plt.ylabel('Cost')
    # plt.show()
    