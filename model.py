import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 定义一个线性神经网络模型
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        # 定义第一个全连接层
        self.linear1 = nn.Linear(input_size, hidden_size1)
        # 定义第二个全连接层
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        # 定义第三个全连接层
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # 使用ReLU激活函数处理第一个全连接层的输出
        x = F.relu(self.linear1(x))
        # 使用ReLU激活函数处理第二个全连接层的输出
        x = F.relu(self.linear2(x))
        # 第三个全连接层的输出
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        # 定义模型保存的文件夹路径
        model_folder_path = './model'
        # 如果文件夹不存在，则创建它
        if not os.path.exists(model_folder_path):
           os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # 保存模型的状态字典
        torch.save(self.state_dict(), file_name)

# 定义一个Q学习训练器
class QTrainer:
    def __init__(self, model, lr, gamma):
        # 学习率
        self.lr = lr
        # 折扣因子
        self.gamma = gamma
        # 神经网络模型
        self.model = model
        # 优化器，使用Adam优化器
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # 损失函数，使用均方误差损失函数
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 将输入转换为张量
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # 如果输入是一维的，扩展为二维
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. 预测当前状态的Q值
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 如果游戏没有结束，更新Q值
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new

        # 2. 计算损失
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # 反向传播
        loss.backward() 

        # 更新参数
        self.optimizer.step()