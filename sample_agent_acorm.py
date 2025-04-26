from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import FeaturesFromSteam, Unit
import numpy as np
from collections import deque

import random

import torch
import torch.nn as nn

# from scripts.MyDQN.logger import Logger
from logger import Logger
import logging

# import acorm
from algorithm.acorm import ACORM_Agent

from geopy.distance import geodesic


class MyAgent(BaseAgent):
    def __init__(self, player_side: str, ac_name: str, target_name: str, args=None, replay_buffer=None):
        """s
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
        

        # # DQN 參數
        # self.input_size = 6  # [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        # self.hidden_size = 64
        # self.num_actions = 4  # 上下左右
        
        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # self.model = DQN(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        # self.target_model = DQN(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        # self.target_model.load_state_dict(self.model.state_dict())
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # ACORM
        self.args = args
        self.agent_n = ACORM_Agent(args)
        self.last_onehot_a_n = np.zeros((args.N, args.action_dim)) # (1, 8)
        # self.replay_buffer = replay_buffer 

        # 修改記憶存儲方式，使用列表存儲完整的episode
        self.episode_memory = []  # 存儲當前episode的經驗
        self.completed_episodes = []  # 存儲已完成的episodes
        self.max_episodes = 10  # 最多保存的episode數量


        self.memory = deque(maxlen=10000)
        self.epsilon = 1
        # self.gamma = 0.99
        # self.batch_size = 128
        self.update_freq = 100
        self.steps = 0 # (total)
        self.game_step = 0  # (episode)
        self.done_condition = 0.005
        self.train_interval = 10

        self.single_past_goals = []
        self.batch_past_goals = []

        self.best_distance = 1000000
        self.total_reward = 0

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None
        self.prev_action = None
        self.prev_features = None #?

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
        unit_s = []
        # print("features.units----------------------",units)
        for unit in units:
            if unit.Name in unit_name:
                unit_s.append(unit)
        return unit_s
        # return None
    
    def get_contact_info_from_observation(self, features: FeaturesFromSteam, contact_name: str) -> dict:
        """
        從觀察中獲取指定接觸點（敵方單位）的資訊。
        """
        contacts = features.contacts
        # print("1111111111111111111",contacts)
        contact_s = []
        for contact in contacts:
            if contact['Name'] in contact_name:
                contact_s.append(contact)
        return contact_s
        # return None
    def get_distance(self, state: np.ndarray) -> float:
        """
        計算兩船之間的距離
        """
        return np.sqrt((state[0] - state[4])**2 + (state[1] - state[5])**2)

    def debug_print(self, message):
        """只在調試模式下打印信息"""
        if self.debug_mode:
            print(message)
            
    def action(self, step_id,features: FeaturesFromSteam, VALID_FUNCTIONS: AvailableFunctions) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        print("total_step:", self.steps)
        self.logger.debug("開始執行動作")
        action = ""
        print("features=====================",features)
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        # print("ac------------------------",ac)
        if ac is None:
            self.logger.warning(f"找不到單位: {self.ac_name}")
            return action  # 如果找不到單位，返回空動作
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        current_state = self.get_state(features) # current_state [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        
        obs_n = torch.tensor([current_state], dtype=torch.float32).to("cpu")
        avail_a_n = np.ones((self.args.N, self.args.action_dim))

        
        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.get_reward(self.prev_state, current_state)
            done = self.get_distance(current_state) < self.done_condition
            self.total_reward += reward
            self.episode_reward += reward
            
            # 將經驗添加到當前episode的記憶中(model所需資訊)
            self.episode_memory.append((self.prev_state, current_state, self.prev_action, reward, done, ))
            
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

        # 更新 epsilon
        self.epsilon = max((self.args.epsilon - self.args.epsilon_decay * self.steps),0.02)
        print("epsilon===============",self.epsilon)

        # 選擇動作
        role_embedding = self.agent_n.get_role_embedding(obs_n, self.last_onehot_a_n)
        a_n = self.agent_n.choose_action(obs_n, self.last_onehot_a_n, role_embedding, avail_a_n, self.epsilon)
        
        print("a_n======================",a_n)
        self.last_onehot_a_n = np.eye(self.args.action_dim)[a_n]
        print("last_onehot_a_n================",self.last_onehot_a_n)
        
        # 輸出動作字串(lua)
        action_cmd = self.apply_action(a_n, ac)
        self.logger.debug(f"應用動作命令: {action_cmd}")
        self.logger.info(f"應用動作命令: {action_cmd}")

        # 更新前一步資料
        self.prev_state = current_state
        self.prev_action = a_n
        self.prev_features = features
        self.steps += 1 # (total)
        self.game_step += 1 # (episode)

        return action_cmd
    
    
    def latlon_to_xy(lat, lon, origin_lat=24, origin_lon=119):
        """
        將經緯度轉為以 (origin_lat, origin_lon) 為原點的 x, y 座標，範圍 [-1, 1]。
        場景範圍：Lat [23, 25], Lon [118, 120]。
        """
        x_km = geodesic((origin_lat, origin_lon), (origin_lat, lon)).km
        y_km = geodesic((origin_lat, origin_lon), (lat, origin_lon)).km
        width_km = geodesic((origin_lat, origin_lon-1), (origin_lat, origin_lon+1)).km
        height_km = geodesic((origin_lat-1, origin_lon), (origin_lat+1, origin_lon)).km
        if lon < origin_lon:
            x_km *= -1
        if lat < origin_lat:
            y_km *= -1
        x_scaled = (x_km / width_km) * 2
        y_scaled = (y_km / height_km) * 2
        return round(x_scaled, 2), round(y_scaled, 2)
    
    def get_state(self, features: FeaturesFromSteam) -> np.ndarray:
        """
        獲取當前狀態向量，包含 3 個己方單位和 3 個敵方單位的資訊。
        狀態向量結構：
        - 己方（3 個）：[x, y, heading, speed, fuel, weapon1, weapon2, weapon3] × 3
        - 敵方（3 個）：[x, y, heading, speed, fuel] × 3
        總維度：24 (己方) + 15 (敵方) = 39
        """
        max_speed = 30.0  # 最大速度
        state = []

        # 1. 處理己方單位（agent_n）

        for i in range(len(self.ac_name)):
            if i < len(self.ac_name):
                unit_name = self.ac_name[i]
                unit = self.get_unit_info_from_observation(features, unit_name)
                if unit:
                    # 提取經緯度並轉為 x, y
                    lon = float(unit.Lon) # unit.get('Lon', 0.0)
                    lat = float(unit.Lat) # unit.get('Lat', 0.0)
                    x, y = self.latlon_to_xy(lat, lon)
                    
                    # 提取航向、速度、燃料
                    heading = float(unit.CH) / 360.0  # [0, 360] -> [0, 1] unit.get('CH', 0.0)
                    speed = float(unit.CS) / max_speed  # [0, 30] -> [0, 1] unit.get('CS', 0.0)
                    # CurrentFuel=8500.0, MaxFuel=8500.0 unit.get('Fuel', {}).get('CQ', 0.0) unit.get('Fuel', {}).get('MQ', 1.0)
                    fuel = float(unit.CurrentFuel) / float(unit.MaxFuel)  # [0, MQ] -> [0, 1] 
                    
                    # # 提取武器數量（假設 3 種武器）
                    # weapons = [0.0, 0.0, 0.0]  # 默認值
                    # if unit_name in self.max_weapons:
                    #     unit_weapons = unit.get('Weapons', [])
                    #     for j in range(min(3, len(unit_weapons))):
                    #         cl = float(unit_weapons[j].get('CL', 0.0))
                    #         ml = self.max_weapons[unit_name][j]
                    #         weapons[j] = cl / ml if ml > 0 else 0.0
                    
                    # # 添加到狀態 暫時不使用武器資訊
                    # state.extend([x, y, heading, speed, fuel] + weapons)
                    state.extend([x, y, heading, speed, fuel])
                else:
                    # 單位不存在，填充 0
                    # state.extend([0.0] * 8)
                    state.extend([0.0] * 5)
            else:
                # 預留單位，填充 0
                # state.extend([0.0] * 8)
                state.extend([0.0] * 5)

        # 2. 處理敵方單位（agent_n）
        for i in range(len(self.target_name)):
            if i < len(self.target_name):
                unit_name = self.target_name[i]
                unit = self.get_unit_info_from_observation(features, unit_name)
                if unit:
                    # 提取經緯度並轉為 x, y
                    lon = float(unit.Lon) # unit.get('Lon', 0.0)
                    lat = float(unit.Lat) # unit.get('Lat', 0.0)
                    x, y = self.latlon_to_xy(lat, lon)
                    
                    # 提取航向、速度、燃料
                    heading = float(unit.CH) / 360.0  # [0, 360] -> [0, 1] unit.get('CH', 0.0)
                    speed = float(unit.CS) / max_speed  # [0, 30] -> [0, 1] unit.get('CS', 0.0)
                    # CurrentFuel=8500.0, MaxFuel=8500.0 unit.get('Fuel', {}).get('CQ', 0.0) unit.get('Fuel', {}).get('MQ', 1.0)
                    fuel = float(unit.CurrentFuel) / float(unit.MaxFuel)  # [0, MQ] -> [0, 1] 
                    
                    # 敵方無武器信息
                    state.extend([x, y, heading, speed, fuel])
                else:
                    # 單位不存在或未探測到，填充 0
                    state.extend([0.0] * 5)
            else:
                # 預留單位，填充 0
                state.extend([0.0] * 5)
        print("state=========================",np.array(state, dtype=np.float32))

        return np.array(state, dtype=np.float32)

    
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
        # if action == 0:  # 上
        #     heading = 0
        # elif action == 1:  # 下
        #     heading = 180
        # elif action == 2:  # 左
        #     heading = 270
        # elif action == 3:  # 右
        #     heading = 90
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=self.ac_name,
            heading=action[0] * 45,
            speed=30
        )
    
    def train(self,state):
        """訓練 ACORM """
        # 檢查是否有足夠的episodes進行訓練
        if len(self.completed_episodes) < 1:
            self.logger.warning("沒有足夠的episodes進行訓練")
            return
        
        # 從已完成的episodes中隨機選擇一個episode
        episode = random.choice(self.completed_episodes)
        
        batch = episode
            
        # 解包批次數據
        states, next_states, actions, rewards, dones, goals, critic_values = zip(*batch)
        
        loss, _, _ = self.agent_n.train(self.replay_buffer)
        print("----------",state.shape)
        # print("self.args.epsilon==========",self.args.epsilon)

        # 記錄訓練統計數據  
        distance = self.get_distance(state[0])
        #每10步記錄一次
        if self.steps % 10 == 0:
            self.logger.info(f"步數: {self.steps}, 距離: {distance:.4f}, 損失: {loss.item():.4f}, 總獎勵: {self.total_reward:.4f}")
            self.stats_logger.log_stat("distance", distance.item(), self.steps)
            self.stats_logger.log_stat("loss", loss.item(), self.steps)
            self.stats_logger.log_stat("return", self.total_reward, self.steps)

        # self.steps += 1
        if self.steps % self.update_freq == 0:
            # self.target_model.load_state_dict(self.model.state_dict())
            self.logger.debug(f"更新目標網路，步數: {self.steps}")

    # def train(self, state, action, reward, next_state, done):
    #     """訓練 DQN 模型"""
    #     self.memory.append((state, action, reward, next_state, done))
    #     if len(self.memory) >= self.batch_size:
    #         batch = random.sample(self.memory, self.batch_size)
    #         states, actions, rewards, next_states, dones = zip(*batch)
            
    #         states = torch.tensor(states, dtype=torch.float32).to(self.device)
    #         actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
    #         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    #         next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
    #         dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
            
    #         current_q = self.model(states).gather(1, actions)
    #         next_q = self.target_model(next_states).max(1)[0]
    #         target_q = rewards + (1 - dones) * self.gamma * next_q
            
    #         loss = nn.MSELoss()(current_q.squeeze(), target_q)
    #         self.logger.debug(f"損失: {loss.item():.4f}")
            
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
            
    #         # 記錄訓練統計數據  
    #         distance = self.get_distance(state)
            
    #         # 每10步記錄一次
    #         if self.steps % 10 == 0:
    #             self.logger.info(f"步數: {self.steps}, 距離: {distance:.4f}, 損失: {loss.item():.4f}, 總獎勵: {self.total_reward:.4f}")
    #             self.stats_logger.log_stat("distance", distance, self.steps)
    #             self.stats_logger.log_stat("loss", loss.item(), self.steps)
    #             self.stats_logger.log_stat("return", self.total_reward, self.steps)

    #         # self.steps += 1
    #         if self.steps % self.update_freq == 0:
    #             self.target_model.load_state_dict(self.model.state_dict())
    #             self.logger.debug(f"更新目標網路，步數: {self.steps}")
    
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
