import numpy as np
import matplotlib.pyplot as plt
import torch

#### 实验1： 手动完成MLP的反向传播进行训练


#定义模型，并初始化参数（利用numpy数组进行定义）
#模型：
#   输入层 2
#   隐藏层 10
#   输出层 1
def initialize_parameters(n_x, n_h, n_y, W_correct):
    #输入层-隐藏层#
    np.random.seed(2)
    #权重#
    W1 = np.random.randn(n_h, n_x) * W_correct
    #偏置#
    b1 = np.zeros((n_h, 1))
    #隐藏层-输出层#
    #权重#
    W2 = np.random.randn(n_y, n_h) * W_correct
    #偏置#
    b2 = np.zeros((n_y, 1))
    # 将参数打包为dictionary形式
    # 例如： parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return parameters

#前向传播
def forward_prop(X,parameters):
    # 从字典dictionary里读取出所有的参数
    # 例如 W1=parameters["W1"]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]    
    # 利用矩阵运算，依次计算每层的z和a
    #使用sigmod函数#
    Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    y_pred = A2
    cache={"A1":A1,"A2":A2} # 需要保留到BackProp计算梯度的缓存
    return y_pred, cache

#计算损失函数
def calculate_cost(y_pred,Y):
    
    # 定义损失函数，均方根或交叉熵
    cost = np.sum((y_pred - Y)**2 / y_pred.shape[0])
    # 注意：返回的cost是标量
    return cost
#sigmode函数梯度#
def dsigmoid(x):
    return x * (1 - x)

def backward_prop(X, Y,parameters, learning_rate):
    # Get parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    y_pred, cache = forward_prop(X, parameters)
    A1 = cache["A1"]
    A2 = cache["A2"]

    m = X.shape[1]  
    dZ2 = (1/m) * 2 * (A2 - Y) * dsigmoid(A2)  
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * dsigmoid(A1)  
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    new_parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

    return new_parameters

# 利用训练完的模型进行预测
def predict(X, parameters):

    y_predict, cache = forward_prop(X, parameters)

    return y_predict




# 主程序
if __name__ == '__main__':
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float64)
    Y = np.array([[0, 1, 1, 0]], dtype=np.float64)


    # 超参数设置

    num_of_iters = 10000     # 迭代次数
    learning_rate =  0.1   # 学习率

    # 模型定义与初始化 # n_x, n_y: 输入/输出的神经元个数，中间是自己定义的MLP神经元个数
    
    init_params = initialize_parameters(2, 10, 1, 1) 
    # 训练模型

    parameters = init_params
    cost_iter = [] # 保存损失函数的变化
    for i in range(0, num_of_iters): # 每次迭代

        # 正向传播
        y_pred, cache=forward_prop(X, parameters)

        # 计算损失函数
        cost=calculate_cost(y_pred, Y)

        # 梯度下降并更新参数
        parameters=backward_prop(X, Y,parameters, learning_rate)
        
        cost_iter.append(cost)
        
        # 可设置迭代停止条件



        # 观察损失函数下降情况
        if(i%100==0):
            print('cost after iteration#{:d}:{:f}'.format(i,cost))
    

   
    # 测试数据

    X_test = torch.tensor([[0,0,1,1], [0,1,0,1]], dtype=torch.float64)
    y_predict = predict(X_test, parameters)
    print(y_predict)

    # print('for example({:d},{:d}) is {:d}'.format(X_test[0][0], X_test[1][0], y_predict))
    

    # 绘制损失函数随迭代次数变化的曲线图，cost_iter, num_of_iters
    plt.plot(range(num_of_iters), cost_iter)
    plt.title('RMSE of Datasets') #此处可以更改名字#
    plt.xlabel('Num')
    plt.ylabel('Cost')
    plt.show()
    
