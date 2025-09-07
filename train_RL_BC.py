import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn
from collections import deque
from src.BC_only import BC_Dataset,EarlyStopping
from src.Environment_Agent import Environment, StudentModel, Critic
from src.RL_BC import Train_Update
import torch.nn.functional as F


# Khởi tạo thông số
L = 3000 #pre-training steps
T = 10000 #training steps
M = 500 #data collection steps
N = 64 #batch size
L2_WEIGHT = 1e-4
TAU_SOFT_UPDATE = 0.001
GAMMA = 0.99
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC= 0.001
EXPERT_BUFFER_SIZE = 500000
AGENT_BUFFER_SIZE = 1000000
EARLY_STOPPING_MAINTAIN = 5
DROP_OUT = 0.5
LOW_BOUNDER_ENVIRONMENT = -0.2
UP_BOUNDER_ENVIRONMENT = 0.2
PRE_TRAIN_DATA_RATE = 0.1 #Full dataset: 500,000 sample
TRAINING_DATA_EXPERT_AGENT_RATE = 0.25 # In RL period: 0.25 from expert, 0.75 from agent buffer
# Đường link dữ liệu trên Huggingface
link = "hf://datasets/NathanGavenski/CartPole-v1/teacher.jsonl"
#Xác định cấu hình phần cứng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Khởi tạo môi trường
ENV_NAME = 'CartPole-v1'
REN_MODE = "rgb_array" #Human or rgb_array
env = Environment(env_name=ENV_NAME,render_mode=REN_MODE,low_bounder=LOW_BOUNDER_ENVIRONMENT,up_bounder=UP_BOUNDER_ENVIRONMENT)
env_test = Environment(env_name=ENV_NAME,render_mode=REN_MODE,low_bounder=LOW_BOUNDER_ENVIRONMENT,up_bounder=UP_BOUNDER_ENVIRONMENT)

# Trích xuất tập dữ liệu chuyên gia
expert_data = BC_Dataset(file_path=link,sample_rate=PRE_TRAIN_DATA_RATE) #[state,action,reward,next_state]
for exp in expert_data.list:
    exp[3] = env.next_state(state=exp[0],action=exp[1])


# Khởi tạo actor và critic
actor = StudentModel(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=DROP_OUT)
target_actor = StudentModel(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=DROP_OUT)
target_actor.load_state_dict(actor.state_dict())
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)

critic = Critic(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=DROP_OUT)
target_critic = Critic(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=DROP_OUT)
target_critic.load_state_dict(critic.state_dict())
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)

# Khởi tạo bộ nhớ đệm actor, critic
memory_actor = deque(list(expert_data.list),maxlen=EXPERT_BUFFER_SIZE)
memory_critic = deque(maxlen=AGENT_BUFFER_SIZE)

train = Train_Update(actor_model=actor,actor_optimizer=optimizer_actor,actor_target=target_actor,
                     critic_model=critic,critic_optimizer=optimizer_critic,critic_target=target_critic,
                     batch=N,expert_rate=TRAINING_DATA_EXPERT_AGENT_RATE,bc_weight=1,actor_weight=1,critic_weight=1,L2_weight=L2_WEIGHT,
                     Tau_soft_update=TAU_SOFT_UPDATE,GAMMA=GAMMA)
early_stopper = EarlyStopping(patience=EARLY_STOPPING_MAINTAIN,mode="maintain")
avg_reward =[]
timestep = 0
best_pretrain = 0
# Pretrain
for i in range(L):
    train(expert_relay=memory_actor,agent_relay=memory_critic,pretrain=True)
    timestep += 1
    if i % 10 ==0:
        _,agv,_,_ = env_test.simulate_agent(actor,num_episodes=100)
        avg_reward.append([timestep, agv, 0])
        if agv > best_pretrain:
            best_pretrain = agv
            best_weights_actor = actor.state_dict()
            best_weights_critic = critic.state_dict()
        if i % 100 == 0:
            print(f"Pretrain: Timestep {i}, Avg_reward: {agv}")

Out_training = False
actor.load_state_dict(best_weights_actor)
critic.load_state_dict(best_weights_critic)
for i in range(T):
    state = env.reset()
    episode_reward = 0
    for j in range(M):
        with torch.no_grad():
            actor.eval()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = actor(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            memory_critic.append([state,action,reward,next_state])
            episode_reward += reward
            state = next_state
            timestep += 1
            if terminated or truncated:
                state = env.reset()
                break
        train(expert_relay=memory_actor, agent_relay=memory_critic, pretrain=False)
    _, agv, _, _ = env_test.simulate_agent(actor, num_episodes=1000)
    avg_reward.append([timestep, agv, episode_reward])
    print(f"Train: Timestep {timestep}, Avg_reward: {agv}")
    if early_stopper(val_loss=agv, model=actor, maintaince_score=500):
        Out_training = True
        print(f"Mô hình đã đạt hiệu suất yêu cầu ở Timestep {timestep},episode {L+i}")
        break
    if Out_training:
        break

# Lưu các thông số theo dõi
df  = pd.DataFrame(data = avg_reward,columns = ["Timestep", "Average Reward","Episode Reward"])
episodes = df["Timestep"].tolist()
avg_rewards = df["Average Reward"].tolist()
eps_rewards = df["Episode Reward"].tolist()

# Lưu model
torch.save(early_stopper.best_weights, f'RL_BC_only_{ENV_NAME}.pth')
print("Mô hình đã được lưu thành công!")

#Lưu tham số theo dõi
df.to_csv(f'RL_BC_training_{ENV_NAME}.csv', index=False)

# Vẽ biểu đồ
plt.plot(episodes, avg_rewards, label='Average Reward', color='blue')
plt.title('Learning Curve: Average Reward over Timestep')
plt.xlabel('Timestep')
plt.ylabel('Average Reward')
plt.grid(True)
plt.legend()
plt.savefig("RL_BC Average Reward")
plt.show(block=False)
plt.close()

plt.plot(episodes, eps_rewards, label='Episode Reward', color='blue')
plt.title('Learning Curve: Episode Reward')
plt.xlabel('Episodes')
plt.ylabel('Total Episode Reward')
plt.grid(True)
plt.legend()
plt.savefig("RL_BC Episode Reward")
plt.show(block=False)
plt.close()

env.close()
