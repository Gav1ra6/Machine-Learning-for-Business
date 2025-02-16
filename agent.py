import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer


MAX_MEMORY = 10000
BATCH_SIZE = 1000
LR = 0.001

# 定义智能体类
class Agent:
    def __init__(self):
        # 游戏次数
        self.n_games = 0
        # 探索率
        self.epsilon = 0                                               
        # 折扣因子
        self.gamma = 0.9                                                 
        # 记忆队列，用于存储经验
        self.memory = deque(maxlen=MAX_MEMORY)                           
        # 神经网络模型
        self.model = Linear_QNet(11, 256, 256, 3)
        # 训练器
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # 获取蛇头的位置
        head = game.snake[0]
        # 定义蛇头周围的四个点
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # 判断蛇的移动方向
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # 定义状态列表
        state = [
            # 前方是否有危险
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # 右侧是否有危险
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # 左侧是否有危险
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # 移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 食物的位置
            game.food.x < game.head.x,    # 食物在左边
            game.food.x > game.head.x,    # 食物在右边
            game.food.y < game.head.y,    # 食物在上边
            game.food.y > game.head.y     # 食物在下边
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        # 将经验存储到记忆队列中
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # 从记忆队列中随机采样一个小批量
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        # 解压缩小批量数据
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # 训练一步
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # 训练一步
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # 随机移动，平衡探索和利用
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # 随机选择一个动作
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # 根据模型预测选择动作
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    # 存储每局游戏的得分
    plot_scores = []
    # 存储每局游戏的平均得分
    plot_mean_scores = []
    # 总得分
    total_score = 0
    # 最高得分记录
    record = 0
    # 创建智能体
    agent = Agent()
    # 创建游戏
    game = SnakeGameAI()
    while True:
        # 获取旧状态
        state_old = agent.get_state(game)

        # 获取动作
        final_move = agent.get_action(state_old)

        # 执行动作并获取新状态
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 训练短期记忆
        agent.train_short_memory(state_old, final_move, reward, state_new,done)

        # 存储经验
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 训练长期记忆，绘制结果
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                # 如果得分超过记录，保存模型
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

        

if __name__ == '__main__':
    train()