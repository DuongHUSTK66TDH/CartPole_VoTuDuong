import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F

class MyCombinedLoss(nn.Module):
    def __init__(self, bc_weight, actor_weight, critic_weight, L2_weight):
        super().__init__()
        self.BC_weight = bc_weight
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.L2_weight = L2_weight

    def forward(self, predict_action_actor, current_q_value, R1, batch_action,
                q_value_for_predicted_actions, loss_L2_critic, loss_L2_actor):
        # 1. Behavior Cloning Loss
        bc_loss = 0.5 * F.mse_loss(predict_action_actor, batch_action)
        # 2. Q-learning Loss
        q_loss = 0.5 * F.mse_loss(q_value_for_predicted_actions, R1)
        # 3. Actor Loss
        actor_loss = -torch.mean(q_value_for_predicted_actions)
        # 4. Total Loss
        combined_loss = (self.BC_weight * bc_loss +
                         self.critic_weight * q_loss +
                         self.actor_weight * actor_loss +
                         self.L2_weight * (loss_L2_actor + loss_L2_critic))
        return combined_loss


class Train_Update:
    def __init__(self, actor_model, actor_optimizer, actor_target,
                 critic_model, critic_optimizer, critic_target,
                 batch, expert_rate, bc_weight, actor_weight, critic_weight, L2_weight,
                 Tau_soft_update,GAMMA,device="gpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch = batch
        self.expert_nums = int(expert_rate * batch)
        self.agent_nums = batch - self.expert_nums
        self.combine_loss = MyCombinedLoss(bc_weight, actor_weight, critic_weight, L2_weight)
        self.Tau = Tau_soft_update
        self.GAMMA = GAMMA
        self.device =device
    def soft_update(self, target_net, main_net):
        """Cập nhật trọng số của target network bằng soft update."""
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(self.Tau * main_param.data + (1.0 - self.Tau) * target_param.data)

    def __call__(self, expert_relay, agent_relay, pretrain=True):
        if pretrain or len(agent_relay) < self.agent_nums:
            transitions = random.sample(expert_relay, self.batch)
        else:
            transitions = random.sample(expert_relay, self.expert_nums)
            transitions.extend(random.sample(agent_relay, self.agent_nums))

        # Chuyển đổi sang Tensor
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device)
        # Sử dụng dtype torch.float32 cho action nếu không gian hành động liên tục
        batch_action = F.one_hot(torch.tensor(batch_action, dtype=torch.int64).to(self.device), num_classes=2).float()

        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(self.device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(self.device)
        #batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(-1).to(device)

        # Tính toán Q-value cho hành động được dự đoán bởi Actor
        predicted_actions_actor = self.actor_model(batch_state)
        q_value_for_predicted_actions = self.critic_model(batch_state, predicted_actions_actor)
        # Tính toán Q-value hiện tại từ batch dữ liệu
        current_q_value = self.critic_model(batch_state, batch_action)
        # Tính toán Target Q-value (R1)
        with torch.no_grad():
            next_actions = self.actor_target(batch_next_state)
            next_q_values = self.critic_target(batch_next_state, next_actions)
            R1 = batch_reward + self.GAMMA * next_q_values

        # Tính toán L2 Regularization Loss
        loss_L2_actor = sum(p.pow(2.0).sum() for p in self.actor_model.parameters())
        loss_L2_critic = sum(p.pow(2.0).sum() for p in self.critic_model.parameters())

        # Tính toán total loss
        total_loss = self.combine_loss(predict_action_actor=predicted_actions_actor,
                                       current_q_value=current_q_value,R1=R1,
            batch_action=batch_action,
            q_value_for_predicted_actions=q_value_for_predicted_actions,
            loss_L2_critic=loss_L2_critic,
            loss_L2_actor=loss_L2_actor)

        # Zero_grad cho cả hai optimizer trước khi gọi backward
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Thực hiện backward cho hàm loss tổng hợp
        total_loss.backward()

        # Cập nhật trọng số của cả hai mạng
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Cập nhật soft update cho các mạng mục tiêu
        self.soft_update(self.actor_target, self.actor_model)
        self.soft_update(self.critic_target, self.critic_model)
        return total_loss
