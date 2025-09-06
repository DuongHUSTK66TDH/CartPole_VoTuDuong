import gymnasium as gym
import torch
import pandas as pd
import numpy as np
from Environment_Agent import StudentModel,Environment
from pkg_resources import resource_stream, resource_exists

#Environment init
FILE_MODEL = ["RL_BC_only_CartPole-v1.pth","Good BC_only//Ver 1//BC_only_CartPole-v1.pth","RL_only result//LR=0.001//RL_CartPole-v1.pth"]
MODEL_NAME = ["RL_BC","BC_only","RL_only"]
device = "cuda" if torch.cuda.is_available() else "cpu"
env_test = Environment(env_name="CartPole-v1", render_mode="rgb_array", low_bounder=-0.2, up_bounder=0.2)
result =[]
for i in range(len(FILE_MODEL)):
    model_test = StudentModel(no_of_obs=env_test.state_size,no_of_action=env_test.action_size,drop_out=0.5)
    model_test_state_dic = torch.load(FILE_MODEL[i])
    model_test.load_state_dict(model_test_state_dic)
    model_test.eval()
    _,avg_res,goal,fail = env_test.simulate_agent(model=model_test,num_episodes=100000)
    result.append([MODEL_NAME[i],avg_res,goal,fail])
    print(avg_res,goal,fail)
    #env_test.video_simulation(agent_model=model_test,model_name="RL+BC",video_path="vids//RL_BC_Simulation.mp4")

df  = pd.DataFrame(data = result,columns = ["Model_name", "Average Reward", "No_fail","Fail_rate"])
df.to_csv(f"Test_competition.csv")

env_test.close()