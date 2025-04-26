from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
import numpy as np
from collections import deque

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

# 导入FeUdal模型
from scripts.FeUdal.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic

class MyAgent(BaseAgent):
    def __init__(self, player_side: str, ac_name: str, target_name: str = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param ac_name: 控制的單位名稱（例如 B 船）
        :param target_name: 目標單位名稱（例如 A 船），可選
        :param log_level: 日誌級別，預設為INFO，可設置為logging.DEBUG啟用詳細日誌
        """
        super().__init__(player_side)
        self.ac_name = ac_name
        self.target_name = target_name  # 用於追蹤特定目標（例如 A 船）
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(f"MyAgent_{ac_name}")
        self.logger.setLevel(logging.INFO)
        
        # FeUdal网络参数
        class Args:
            def __init__(self):
                self.manager_hidden_dim = 64
                self.worker_hidden_dim = 64
                self.state_dim_d = 6
                self.embedding_dim_k = 16
                self.n_actions = 4  # 上下左右

        self.args = Args()
        self.input_size = 6  # [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        
        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # 初始化FeUdal网络
        self.manager = Feudal_ManagerAgent(self.input_size, self.args).to(self.device)
        self.worker = Feudal_WorkerAgent(self.input_size, self.args).to(self.device)
        self.critic = FeUdalCritic(self.input_size, self.args).to(self.device)
        
        self.manager_optimizer = torch.optim.Adam(self.manager.parameters(), lr=1e-3)
        self.worker_optimizer = torch.optim.Adam(self.worker.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)  # 使用manager的critic部分
        
        # 修改記憶存儲方式，使用列表存儲完整的episode
        self.episode_memory = []  # 存儲當前episode的經驗
        self.completed_episodes = []  # 存儲已完成的episodes
        self.max_episodes = 10  # 最多保存的episode數量
        
        self.epsilon = 1
        self.gamma = 0.99
        self.batch_size = 256
        self.update_freq = 100
        self.total_steps = 0
        self.done_condition = 0.005
        self.train_interval = 10

        # 初始化隐藏状态
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()


        self.single_past_goals = []
        self.batch_past_goals = []

        self.best_distance = 1000000
        self.total_reward = 0

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None
        self.prev_action = None
        self.prev_goal = None
        self.prev_critic_value = None

        # 初始化訓練統計記錄器（如果有的話）
        self.stats_logger = Logger()
        
        # 添加遊戲結束標記
        self.episode_step = 0
        self.max_episode_steps = 200
        self.episode_done = False
        self.episode_reward = 0

    def get_unit_info_from_observation(self, features: FeaturesFromSteam, unit_name: str) -> Unit:
        """
        從觀察中獲取指定單位的資訊。
        """
        units = features.units
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None
    
    def get_contact_info_from_observation(self, features: FeaturesFromSteam, contact_name: str) -> dict:
        """
        從觀察中獲取指定接觸點（敵方單位）的資訊。
        """
        contacts = features.contacts
        for contact in contacts:
            if contact['Name'] == contact_name:
                return contact
        return None
    def get_distance(self, state):
        """计算智能体与目标之间的距离，支持NumPy数组和PyTorch张量"""
        if isinstance(state, torch.Tensor):
            # 使用PyTorch操作
            return torch.sqrt((state[0] - state[4])**2 + (state[1] - state[5])**2)
        else:
            # 使用NumPy操作
            return np.sqrt((state[0] - state[4])**2 + (state[1] - state[5])**2)

    def debug_print(self, message):
        """只在調試模式下打印信息"""
        if self.debug_mode:
            print(message)
            
    def action(self, features: FeaturesFromSteam, VALID_FUNCTIONS: AvailableFunctions) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            self.logger.warning(f"找不到單位: {self.ac_name}")
            return action  # 如果找不到單位，返回空動作
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        current_state = self.get_state(features)
        
        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.get_reward(self.prev_state, current_state)
            done = self.get_distance(current_state) < self.done_condition
            self.total_reward += reward
            self.episode_reward += reward
            
            # 將經驗添加到當前episode的記憶中
            self.episode_memory.append((self.prev_state, current_state, self.prev_action, reward, done, self.prev_goal, self.prev_critic_value))
            
            # 檢查遊戲是否結束
            if done or self.episode_step > self.max_episode_steps:
                self.episode_done = True
                self.logger.info(f"遊戲結束! 總獎勵: {self.episode_reward:.4f}")
                
                # 將完成的episode添加到已完成episodes列表中
                if len(self.episode_memory) > 0:
                    self.completed_episodes.append(self.episode_memory)
                    # 限制已完成的episodes數量
                    if len(self.completed_episodes) > self.max_episodes:
                        self.completed_episodes.pop(0)
                
                # 在遊戲結束時進行訓練
                self.train()
                # 重置遊戲狀態
                return self.reset()
            
        self.logger.debug("訓練完成")
        
        # 選擇動作
        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Manager生成目标
            _, goal, self.manager_hidden = self.manager(state_tensor, self.manager_hidden)
            
            # Worker根据目标选择动作
            q_values, self.worker_hidden = self.worker(
                state_tensor, 
                self.worker_hidden,
                goal
            )
            # Critic估計狀態值
            critic_value = self.critic(state_tensor)
        if random.random() < self.epsilon:
            action = random.randint(0, self.args.n_actions - 1)
            self.logger.debug(f"隨機選擇動作: {action}")
        else:
            action = q_values.argmax().item()
            self.logger.debug(f"根據Q值選擇動作: {action}")
        
        # 執行動作
        action_cmd = self.apply_action(action, ac)
        self.logger.debug(f"應用動作命令: {action_cmd}")
        
        # 更新前一步資料
        self.prev_state = current_state
        self.prev_action = action
        self.prev_goal = goal
        self.prev_critic_value = critic_value
        self.total_steps += 1
        self.episode_step += 1

        # 更新 epsilon
        self.epsilon = max(0.1, self.epsilon * 0.999)
        self.logger.debug(f"更新epsilon: {self.epsilon:.4f}")
        
        return action_cmd
    
    def normalize_state(self, state):
        min_lon, max_lon = 121.0, 122.5  # 根據你的地圖範圍調整
        min_lat, max_lat = 23.5, 24.5
        min_heading, max_heading = 0.0, 360.0
        min_speed, max_speed = 0.0, 30.0
        norm_state = np.zeros_like(state)
        norm_state[0] = (state[0] - min_lon) / (max_lon - min_lon)  # B_lon
        norm_state[1] = (state[1] - min_lat) / (max_lat - min_lat)  # B_lat
        norm_state[2] = state[2] / max_heading  # B_heading
        norm_state[3] = state[3] / max_speed  # B_speed   
        norm_state[4] = (state[4] - min_lon) / (max_lon - min_lon)  # A_lon
        norm_state[5] = (state[5] - min_lat) / (max_lat - min_lat)  # A_lat
        return norm_state   

    def get_state(self, features: FeaturesFromSteam) -> np.ndarray:
        """
        獲取當前狀態向量，包含自身單位和目標的資訊。
        :return: numpy 陣列，例如 [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        """
        # 獲取自身單位（B 船）的資訊
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            # 如果找不到單位，返回預設狀態
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        # 獲取目標單位（A 船）的資訊
        if self.target_name:
            target = self.get_contact_info_from_observation(features, self.target_name) or \
                    self.get_unit_info_from_observation(features, self.target_name)
            if target:
                # 根據 target 的型別提取經緯度
                if isinstance(target, dict):  # 來自 features.contacts
                    target_lon = float(target.get('Lon', 0.0))
                    target_lat = float(target.get('Lat', 0.0))
                else:  # 來自 features.units (Unit 物件)
                    target_lon = float(target.Lon)
                    target_lat = float(target.Lat)
            else:
                target_lon, target_lat = 0.0, 0.0  # 目標未找到時的預設值
        else:   
            target_lon, target_lat = 0.0, 0.0  # 未指定目標時的預設值
        
        raw_state = np.array([float(ac.Lon), float(ac.Lat), float(ac.CH), float(ac.CS), target_lon, target_lat])
        normalized_state = self.normalize_state(raw_state)
        
        # 將 NumPy 數組轉換為 PyTorch 張量並返回
        # 返回狀態向量：[自身經度, 自身緯度, 自身航向, 自身航速, 目標經度, 目標緯度]
        return normalized_state
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def get_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        distance = self.get_distance(state)
        next_distance = self.get_distance(next_state)
         # print(f"Distance: {distance:.4f} -> {next_distance:.4f}")
        #reward = -100 * next_distance
        # if (distance - next_distance) > 0:
        #     reward = 2*(1-next_distance)  # 放大距離變化
        # else:
        #     reward = 10*(-next_distance)  # 放大距離變化
        
        self.logger.debug(f"距離變化: {distance:.4f} -> {next_distance:.4f}")
            
        reward = 500 * (distance-next_distance)
        # print(f"Reward: {reward}")
        if next_distance + 0.01 < self.best_distance:
            self.best_distance = next_distance   #如果當前距離比最佳距離近0.5公里 換他當最佳距離 然後給獎勵
            reward += 3
            self.logger.debug(f"新的最佳距離: {self.best_distance:.4f}")
        if next_distance < self.done_condition:
            reward += 20
        # if next_distance > 0.25:
        #     reward -= 100
        # print(f"FinalReward: {reward:.4f}")
            self.logger.debug("達到目標條件!")
        reward -= 0.1
        self.logger.debug(f"獎勵: {reward:.4f}")
            
        return reward

    def apply_action(self, action: int, ac: Unit) -> str:
        """將動作轉換為 CMO 命令"""
        step_size = 0.005
        lat, lon = float(ac.Lat), float(ac.Lon)
        if action == 0:  # 上
            heading = 0
        elif action == 1:  # 下
            heading = 180
        elif action == 2:  # 左
            heading = 270
        elif action == 3:  # 右
            heading = 90
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=self.ac_name,
            heading=heading,
            speed=30
        )

    def train(self):
        """訓練FeUdal網路"""
        # 檢查是否有足夠的episodes進行訓練
        if len(self.completed_episodes) < 1:
            self.logger.warning("沒有足夠的episodes進行訓練")
            return
            
        # 從已完成的episodes中隨機選擇一個episode
        episode = random.choice(self.completed_episodes)
        
        batch = episode
            
        # 解包批次數據
        states, next_states, actions, rewards, dones, goals, critic_values = zip(*batch)
        
        # 轉換為張量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 修复：直接将张量堆叠而不使用numpy
        goals_tensor = torch.stack([g.detach() for g in goals]).to(self.device)
        critic_values_tensor = torch.stack([cv.detach() for cv in critic_values]).to(self.device)
        
        # 獲取實際的批次大小
        actual_batch_size = states.size(0)

        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()
        #----------------------------- Calculate manager loss -----------------------------------
        
        # # 为训练创建新的隐藏状态，批次大小为实际批次大小
        # train_manager_hidden = (
        #     self.manager_hidden[0].expand(actual_batch_size, -1),
        #     self.manager_hidden[1].expand(actual_batch_size, -1)
        # )
        # train_worker_hidden = (
        #     self.worker_hidden[0].expand(actual_batch_size, -1),
        #     self.worker_hidden[1].expand(actual_batch_size, -1)
        # )
        current_goals = []
        for t in range(actual_batch_size):
            _, goal, self.manager_hidden = self.manager(states[t].unsqueeze(0), self.manager_hidden)
            current_goals.append(goal)
        current_goals = torch.stack(current_goals).squeeze(1)
        
        current_values = self.critic(states)

        state_diff = next_states - states
        returns = compute_returns(rewards, self.gamma, current_values, dones)
        advantages = returns - current_values
        # 標準化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        temp_state_diff = F.normalize(state_diff, dim=-1, p=2)
        temp_goals = F.normalize(current_goals, dim=-1, p=2)
        cos_sim = F.cosine_similarity(temp_state_diff, temp_goals, dim=-1)

        # Manager更新
        manager_loss = -(advantages.detach() * cos_sim).mean()
        
        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        self.manager_optimizer.step()

        #----------------------------- Calculate worker loss -----------------------------------
        self.worker_hidden = self.worker.init_hidden()
        current_q = []
        target_q = []

        for t in range(actual_batch_size):
            q, self.worker_hidden = self.worker(states[t].unsqueeze(0), self.worker_hidden, current_goals[t].detach())
            current_q.append(q.squeeze(0).gather(0, actions[t]))
        current_q = torch.stack(current_q)
        
        
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()
        with torch.no_grad():
            for t in range(actual_batch_size):
                _, next_goal, self.manager_hidden = self.manager(next_states[t].unsqueeze(0), self.manager_hidden)
                next_q, self.worker_hidden = self.worker(next_states[t].unsqueeze(0), self.worker_hidden, next_goal)
                next_q = next_q.max(1)[0]
                target = rewards + (1 - dones) * self.gamma * next_q
                target_q.append(target.squeeze(0))
        target_q = torch.stack(target_q)
        worker_loss = nn.MSELoss()(current_q, target_q)
        
        self.worker_optimizer.zero_grad()
        worker_loss.backward()
        self.worker_optimizer.step()

        #----------------------------- Calculate critic loss -----------------------------------
        critic_loss = 0.5 * (advantages ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 記錄訓練統計數據  
        distance = self.get_distance(states[0])
        
        # 每10步記錄一次
        # if self.total_steps % 10 == 0:
        total_loss = manager_loss.item() + worker_loss.item()
        self.logger.info(f"步數: {self.total_steps}, 距離: {distance:.4f}, 損失: {total_loss:.4f}, 總獎勵: {self.total_reward:.4f}")
        self.stats_logger.log_stat("distance", distance.item(), self.total_steps)
        self.stats_logger.log_stat("manager_loss", manager_loss.item(), self.total_steps)
        self.stats_logger.log_stat("worker_loss", worker_loss.item(), self.total_steps)
        self.stats_logger.log_stat("return", self.total_reward, self.total_steps)

        # if self.total_steps % self.update_freq == 0:
        #     self.manager_hidden = self.manager.init_hidden()
        #     self.worker_hidden = self.worker.init_hidden()
        #     self.logger.debug(f"更新FeUdal網路，步數: {self.total_steps}")
    
    def reset(self):
        """重置遊戲狀態，準備開始新的episode"""
        self.best_distance = 1000000
        self.prev_state = None
        self.prev_action = None
        self.prev_goal = None
        self.prev_critic_value = None
        self.episode_step = 0
        self.episode_done = False
        self.episode_reward = 0
        self.manager_hidden = self.manager.init_hidden()
        self.worker_hidden = self.worker.init_hidden()
        # 清空當前episode的記憶
        self.episode_memory = []
        self.logger.info("重置遊戲狀態，準備開始新的episode")
        
        # 重置單位位置（如果需要）
        return set_unit_position(
            side=self.player_side,
            unit_name=self.ac_name,
            latitude=24.04,
            longitude=122.18
        )


def compute_returns(rewards, gamma, values, terminated): 
    """
    rewards: 奖励张量 [batch_size]
    gamma: 折扣因子
    values: 价值估计 [batch_size]
    terminated: 终止标志 [batch_size]
    返回: returns张量 [batch_size]
    """
    # 在您的代码中，这些都是一维张量，没有序列长度维度
    batch_size = rewards.shape[0]
    
    # 预分配returns张量
    returns = torch.zeros_like(values)
    
    # 计算每个样本的回报
    for b in range(batch_size):
        # 如果终止，则回报就是奖励
        if terminated[b]:
            returns[b] = rewards[b]
        else:
            # 否则，回报是奖励加上折扣的价值估计
            returns[b] = rewards[b] + gamma * values[b]
            
    return returns