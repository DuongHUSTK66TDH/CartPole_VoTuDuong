import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width


def plot_model_comparison(models, avg_rewards, fail_rates):
    """
    Vẽ biểu đồ cột kép để so sánh phần thưởng trung bình và tỷ lệ thất bại của các mô hình.

    Args:
        models (list): Danh sách tên các mô hình, ví dụ: ['Model A', 'Model B', 'Model C'].
        avg_rewards (list): Danh sách phần thưởng trung bình của các mô hình.
        fail_rates (list): Danh sách tỷ lệ thất bại của các mô hình (giá trị từ 0 đến 1).
    """
    x = np.arange(len(models))  # Vị trí các cột
    width = 0.35  # Chiều rộng của các cột

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Trục Y bên trái: Phần thưởng trung bình
    ax1.set_xlabel('Mô hình', fontsize=12)
    ax1.set_ylabel('Phần thưởng trung bình', color='tab:blue', fontsize=12)
    reward_bars = ax1.bar(x - width / 2, avg_rewards, width, label='Phần thưởng trung bình', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Trục Y bên phải: Tỷ lệ thất bại
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tỷ lệ thành công', color='tab:red', fontsize=12)
    fail_bars = ax2.bar(x + width / 2, fail_rates, width, label='Tỷ lệ thành công', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1)  # Đảm bảo trục tỷ lệ từ 0 đến 1

    # Thêm nhãn trên các cột
    def autolabel(bars, ax, fmt=''):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 điểm offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(reward_bars, ax1, fmt='{:.1f}')
    autolabel(fail_bars, ax2, fmt='{:.2f}')

    # Đặt nhãn cho trục X
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    # Tiêu đề và legend
    plt.title('So sánh Hiệu suất của các Mô hình', fontsize=16)
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("So sánh Hiệu suất của các Mô hình")
    plt.show()


# Đường dẫn đến tệp của bạn
RL_FILE_PATH = "RL_only result\\LR=0.001\\RL_Training_reward_CartPole-v1.txt"
RL_BC_FILE_PATH = "RL_BC_training_CartPole-v1.csv"
# Đọc tệp tin và tạo DataFrame
LR_df = pd.read_csv(RL_FILE_PATH)
LR_BC_df = pd.read_csv(RL_BC_FILE_PATH)

plot_model_comparison(models=["RL_BC","BC_only","RL_only"],avg_rewards=[500,500,499.99916],fail_rates=[0,0,99.976])
#Sử dụng khi dùng RL:
# Trục X: các episode
# RL_timestep= LR_df["Time Step"].tolist()
# # Trục Y: các giá trị phần thưởng
# RL_avg_reward = LR_df["Average Reward"].tolist()
# RL_episode_reward = LR_df["Episode Reward"].tolist()
# #RL_episodes = LR_df["Epoch"].tolist()
#
# #Sử dụng khi dùng RL_BC:
# RL_BC_timestep = LR_BC_df["Timestep"].tolist()
# RL_BC_reward = LR_BC_df["Average Reward"].tolist()
#
# # Thêm nhãn và tiêu đề để biểu đồ rõ ràng hơn
# plt.plot(RL_timestep,RL_avg_reward , linestyle='-',label="RL only",linewidth=1,color="red")
# plt.plot(RL_BC_timestep,RL_BC_reward  , linestyle='-',label="RL-BC",linewidth=1,color="blue")
# plt.title("Episode Reward")
# plt.ylabel("Episode Reward")
# plt.xlabel("Timestep")
# plt.legend(loc='lower right')
# plt.savefig("Compare RL+BC")
# plt.show()
# plt.close()

# # Sử dụng khi dùng BC:
# # Trục X: các epoch
# epochs = df["Epoch"].tolist()
# # Trục Y: các tiêu chí theo dõi
# train_loss = df["Train_loss"].tolist()
# train_acc = df["Train_acc"].tolist()
# val_loss = df["Val_loss"].tolist()
# val_acc = df["Val_acc"].tolist()
# avg_reward = df["Val_reward"].tolist()
#
# plt.plot(epochs, train_acc, color="blue", label="train")
# plt.plot(epochs, val_acc, color="orange", label="valid")
# plt.title("Training curve - Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.savefig("BC_only Training Curve - Accuracy")
# plt.show()
# plt.close()