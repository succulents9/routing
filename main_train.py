import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from DQN.environment import GraphEnv
from env.LEO_network import SatelliteNetwork
from DQN.model import QNetwork, ReplayBuffer
from DQN.QMIX import QMIXNetwork
from utils.plot import plot_metrics

batch_size = 32
gamma = 0.99
qmix_episodes = 200
update_target_freq = 20
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995
learning_rate = 1e-3
save_dir = "../saved_models"
os.makedirs(save_dir, exist_ok=True)

def extract_global_state(states):
    region_features = []
    for s in states:
        s_tensor = torch.FloatTensor(s)
        pooled = s_tensor.mean(dim=0)
        region_features.append(pooled)
    global_state = torch.cat(region_features, dim=0)
    return global_state

class SimpleQMIXNetwork(nn.Module):
    def __init__(self, state_dim, num_agents, hidden_dim):
        super(SimpleQMIXNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        self.agent_layer = nn.Sequential(
            nn.Linear(num_agents * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state, local_q_values):
        batch_size = global_state.size(0)
        state_features = self.state_layer(global_state)
        state_features = state_features.view(batch_size, self.num_agents, -1)
        weighted_qs = local_q_values.unsqueeze(-1) * state_features
        combined = weighted_qs.view(batch_size, -1)
        q_total = self.agent_layer(combined)
        return q_total.squeeze(-1)

if __name__ == "__main__":
    node_dim = 1
    edge_dim = 2
    hidden_dim = 64
    action_dim = 8
    sa_files_dir = "C:/Users/49753/Desktop/GNN+DRL_0424V2/sat_data_1616.txt"
    sat_network = SatelliteNetwork(sa_files_dir)
    region_num = sat_network.grid_cols * sat_network.grid_rows
    per_region_feature_dim = node_dim
    state_dim = region_num * per_region_feature_dim
    envs = []
    for region_id in range(1, region_num + 1):
        env = GraphEnv(sat_network, region_id)
        env.initialize_environment()
        envs.append(env)
    local_q_networks = []
    target_local_q_networks = []
    for region_id in range(1, region_num + 1):
        q_net = QNetwork(node_dim, edge_dim, hidden_dim, action_dim)
        q_net.load_state_dict(torch.load(f"./models/region_{region_id}_model_16.pth"))
        q_net.eval()
        local_q_networks.append(q_net)
        target_net = QNetwork(node_dim, edge_dim, hidden_dim, action_dim)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()
        target_local_q_networks.append(target_net)
    qmix_net = SimpleQMIXNetwork(state_dim=state_dim, num_agents=region_num, hidden_dim=hidden_dim)
    target_qmix_net = SimpleQMIXNetwork(state_dim=state_dim, num_agents=region_num, hidden_dim=hidden_dim)
    target_qmix_net.load_state_dict(qmix_net.state_dict())
    optimizers = [optim.Adam(q.parameters(), lr=learning_rate) for q in local_q_networks]
    qmix_optimizer = optim.Adam(qmix_net.parameters(), lr=learning_rate)
    buffers = [ReplayBuffer(capacity=5000) for _ in range(region_num)]
    epsilon = epsilon_start
    for episode in tqdm(range(qmix_episodes)):
        states = [env.reset(None) for env in envs]
        done_flags = [False] * region_num
        while not all(done_flags):
            actions = []
            for idx in range(region_num):
                if done_flags[idx]:
                    actions.append(0)
                    continue
                data = states[idx]
                q_values = local_q_networks[idx](data.x.unsqueeze(0), data.edge_index, data.edge_attr, torch.zeros(data.num_nodes, dtype=torch.long)).detach()
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, action_dim)
                else:
                    action = q_values.argmax().item()
                actions.append(action)
            next_states = []
            rewards = []
            dones = []
            infos = []
            for idx, env in enumerate(envs):
                if done_flags[idx]:
                    next_states.append(states[idx])
                    rewards.append(0)
                    dones.append(True)
                    infos.append({})
                    continue
                next_state, reward, done, info = env.step(actions[idx])
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                buffers[idx].add(states[idx], actions[idx], reward, next_state, done)
            states = next_states
            done_flags = dones
        if all([len(buffer) >= batch_size for buffer in buffers]):
            batch_data = [buffer.sample(batch_size) for buffer in buffers]
            local_q_evals = []
            local_q_targets = []
            for idx in range(region_num):
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_data[idx]
                x_batch = state_batch.x
                edge_index_batch = state_batch.edge_index
                edge_attr_batch = state_batch.edge_attr
                batch_batch = torch.zeros(x_batch.size(0), dtype=torch.long)
                next_x_batch = next_state_batch.x
                next_edge_index_batch = next_state_batch.edge_index
                next_edge_attr_batch = next_state_batch.edge_attr
                next_batch_batch = torch.zeros(next_x_batch.size(0), dtype=torch.long)
                q_eval = local_q_networks[idx](x_batch, edge_index_batch, edge_attr_batch, batch_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                next_q = target_local_q_networks[idx](next_x_batch, next_edge_index_batch, next_edge_attr_batch, next_batch_batch).max(1)[0]
                q_target = reward_batch + gamma * (1 - done_batch.float()) * next_q
                local_q_evals.append(q_eval)
                local_q_targets.append(q_target)
            q_eval_stack = torch.stack(local_q_evals, dim=1)
            q_target_stack = torch.stack(local_q_targets, dim=1)
            # 这里你应当传入全局状态，示例先用extract_global_state(states)
            global_state = extract_global_state([s.x.numpy() for s in states]).unsqueeze(0)
            q_total_eval = qmix_net(global_state, q_eval_stack)
            q_total_target = target_qmix_net(global_state, q_target_stack)
            qmix_loss = nn.MSELoss()(q_total_eval, q_total_target.detach())
            qmix_optimizer.zero_grad()
            qmix_loss.backward()
            qmix_optimizer.step()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % update_target_freq == 0:
            for idx in range(region_num):
                target_local_q_networks[idx].load_state_dict(local_q_networks[idx].state_dict())
            target_qmix_net.load_state_dict(qmix_net.state_dict())
        if episode % 10 == 0:
            avg_reward = np.mean([sum(b.reward_buffer) for b in buffers])
            print(f"[Episode {episode}] QMIX Loss: {qmix_loss.item():.4f}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")
    torch.save(qmix_net.state_dict(), os.path.join(save_dir, "qmix_final_model.pth"))
