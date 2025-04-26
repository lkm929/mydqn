import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
from geopy.distance import geodesic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定 log 檔案路徑
file_path = 'D:\\pycmo-main\\pycmo\\scripts\\demo_m\\log.txt'

# 超參數
BATCH_SIZE = 64
LR = 5e-4
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
GAMMA = 0.99
MEMORY_CAPACITY = 20000
TARGET_UPDATE_FREQ = 1000

# 24 個動作空間
actions = [
    [0, 5], [45, 5], [90, 5], [135, 5], [180, 5], [225, 5], [270, 5], [315, 5],
    [0, 0], [45, 0], [90, 0], [135, 0], [180, 0], [225, 0], [270, 0], [315, 0],
    [0, -5], [45, -5], [90, -5], [135, -5], [180, -5], [225, -5], [270, -5], [315, -5]
]

def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=len(actions)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SimpleEnv:
    def __init__(self, start, target, heading, speed):
        self.init_pos = start
        self.target = target
        self.head, self.spee = heading, speed
        self.reset()

    def reset(self):
        self.lon, self.lat = self.init_pos
        self.heading, self.speed = self.head, self.spee
        self.I = haversine(*self.init_pos, *self.target)
        self.current_distance = self.I
        self.prev_distance = self.I
        self.best_distance = self.I
        return [(self.lon - 120) / 10, (self.lat - 25) / 10, self.heading / 360, self.speed / 40]

    def step(self, action_idx):
        heading_change, acc = actions[action_idx]
        self.heading = (self.heading + heading_change) % 360
        MAX_SPEED = 40
        self.speed = min(MAX_SPEED, max(0, self.speed + acc))
        speed_in_kmh = self.speed * 1.852
        distance = speed_in_kmh / 3600.0
        done = False
        bearing = self.heading
        new_position = geodesic(kilometers=distance).destination((self.lat, self.lon), bearing)
        self.lat, self.lon = new_position.latitude, new_position.longitude

        D = haversine(self.lon, self.lat, *self.target)
        self.current_distance = D

        #reward的計算
        reward  = (self.prev_distance - self.current_distance) * 20   #先前距離 - 當前距離 (有變近就會是正的)
        
        if self.current_distance + 0.5 < self.best_distance:
            self.best_distance = self.current_distance   #如果當前距離比最佳距離近0.5公里 換他當最佳距離 然後給獎勵
            reward += 3

        if self.current_distance < 0.1:
            done = True

        self.prev_distance = self.current_distance

        state = [(self.lon - 120) / 10, (self.lat - 25) / 10, self.heading / 360, self.speed / 40]
        print(f"距離: {D:.4f} km, 方位變化: {heading_change}°, 加速度: {acc}, "
              f"航向: {self.heading}°, 速度(節): {self.speed}")
        return state, reward, done

# 創建主網絡和目標網絡
model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=MEMORY_CAPACITY)

def epsilon_by_frame(frame_idx):
    return FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * math.exp(-1.0 * frame_idx / 50000) #到50000t episilon衰減至0.01
# 初始化 log 檔案
with open(file_path, "w", encoding="utf-8") as f:
    f.write("t,distance,reward\n")



env = SimpleEnv(start=(125.91, 31.533), target=(128.27, 33.77), heading=0, speed=0 )
state = env.reset()
done = False
t = 0
episode_reward = 0

while not done:
    epsilon = epsilon_by_frame(t)
    if random.random() < epsilon:
        action_idx = random.randrange(len(actions))
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=device).float().unsqueeze(0)
            action_idx = model(state_tensor).argmax().item()

    next_state, reward, done = env.step(action_idx)
    memory.append((state, action_idx, reward, next_state, done))

    if len(memory) >= BATCH_SIZE:
        batch = random.sample(memory, BATCH_SIZE)
        states, actions_batch, rewards_batch, next_states, dones_batch = zip(*batch)
        states_v = torch.tensor(states, device=device).float()
        actions_v = torch.tensor(actions_batch, device=device).unsqueeze(1)
        rewards_v = torch.tensor(rewards_batch, device=device).unsqueeze(1)
        next_states_v = torch.tensor(next_states, device=device).float()
        dones_v = torch.tensor(dones_batch, device=device).unsqueeze(1)

        q_values = model(states_v).gather(1, actions_v)
        next_q_values = target_model(next_states_v).max(1)[0].detach().unsqueeze(1)
        expected_q = rewards_v + GAMMA * next_q_values * (~dones_v)
        loss = loss_fn(q_values, expected_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if t % TARGET_UPDATE_FREQ == 0:
        target_model.load_state_dict(model.state_dict())
    
    state = next_state
    episode_reward += reward
    t += 1
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{t},{env.current_distance:.4f},{episode_reward:.4f}\n")


print(f"Episode Reward: {episode_reward:.2f}, Steps: {t}")