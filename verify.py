import numpy as np
import os
import torch
import traceback
import random
import matplotlib.pyplot as plt
from scipy.constants import c
from DQN.environment import GraphEnv
from DQN.model import DQNAgent, ReplayBuffer, QNetwork
from env.LEO_network import SatelliteNetwork
from main_eval import evaluate_route, evaluate_full_task
from utils.plot import plot_result, save_simulation_result

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def generate_tasks(task_baseline, region_envs):
    random.seed(40)
    demand_ratios = [0, 0, 2, 0, 1, 0, 2, 1, 0, 2, 2, 1, 2, 1, 2, 0]
    task_to_num_tasks = {2: 300, 3: 200, 4: 200, 5: 80, 6: 75, 7: 50, 8: 50, 9: 50, 10: 50}

    all_regions = list(range(1, 17))
    region_total_flows = [r * task_baseline for r in demand_ratios]
    num_tasks = task_to_num_tasks[task_baseline]

    total_flow = sum(region_total_flows)
    region_task_counts = [max(1, int(num_tasks * (f / total_flow))) if r > 0 else 0
                          for r, f in zip(demand_ratios, region_total_flows)]

    diff = num_tasks - sum(region_task_counts)
    for i in range(abs(diff)):
        idx = i % 12
        if demand_ratios[idx] > 0:
            region_task_counts[idx] += 1 if diff > 0 else -1

    tasks = []
    for region_idx, task_count in enumerate(region_task_counts):
        if task_count == 0:
            continue

        src_region = region_idx + 1
        if src_region not in region_envs:
            print(f"跳过区域 {src_region}，因为没有对应的环境")
            continue

        src_env = region_envs[src_region]
        valid_src_nodes = list(src_env.node_id_to_index.keys())

        if not valid_src_nodes:
            print(f"跳过区域 {src_region}，因为没有有效的节点")
            continue

        avg_flow = region_total_flows[region_idx] / task_count
        for i in range(task_count):
            source_node = random.choice(valid_src_nodes)
            p = {2: 0.5, 3: 0.53, 4: 0.58, 5: 0.63, 6: 0.7, 7: 0.7, 8: 0.75, 9:0.8 , 10: 0.85}
            if random.random() < p[task_baseline]:
                dst_region_candidates = [r for r in all_regions if
                                         r != src_region and demand_ratios[r - 1] > 0 and r in region_envs]
                if not dst_region_candidates:
                    dst_region = src_region
                else:
                    dst_region = random.choice(dst_region_candidates)
            else:
                dst_region = src_region

            if dst_region not in region_envs:
                print(f"跳过目标区域 {dst_region}，因为没有对应的环境")
                continue

            dst_env = region_envs[dst_region]
            valid_dst_nodes = list(dst_env.node_id_to_index.keys())

            if not valid_dst_nodes:
                print(f"跳过目标区域 {dst_region}，因为没有有效的节点")
                continue

            if dst_region == src_region:
                valid_dst_nodes = [n for n in valid_dst_nodes if n != source_node]
                if not valid_dst_nodes:
                    print(f"区域 {dst_region} 中没有不同于源节点的目标节点")
                    continue

            destination_node = random.choice(valid_dst_nodes)

            task = {
                'src_region': src_region,
                'dst_region': dst_region,
                'source_node': source_node,
                'destination_node': destination_node,
                'flow_demand': avg_flow,
                'baseline': task_baseline
            }
            tasks.append(task)

    print(f"生成了 {len(tasks)} 个有效任务")
    return tasks


def load_region_models(region_ids, node_dim, edge_dim, hidden_dim, action_dim,
                       model_path_template="C:/Users/49753/Desktop/GNN+DRL_0424V2/saved_models/region_{}_model_16.pth"):
    agents_dict = {}
    for region_id in region_ids:
        model_path = model_path_template.format(region_id)

        if not os.path.exists(model_path):
            print(f"区域 {region_id} 的模型不存在，使用默认模型")
            model_path = "./saved_models/region_1_model_16_1.pth"

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        actual_action_dim = state_dict['fc3.weight'].size(0)
        print(f"区域 {region_id} 模型的实际action_dim为: {actual_action_dim}")

        agent = DQNAgent(node_dim, edge_dim, hidden_dim, actual_action_dim)
        agent.q_network.load_state_dict(state_dict)
        agent.q_network.eval()
        agents_dict[region_id] = agent

    return agents_dict


if __name__ == "__main__":
    import os
    import traceback
    import torch

    region_num = 16

    node_dim = 1
    edge_dim = 2
    hidden_dim = 64
    action_dim = 8
    region_delay = 0
    inter_delay = 0

    sat_network = SatelliteNetwork("./starlink.txt")
    sat_network.run_simulation(0)
    for region_id in range(1, 17):
        sat_network.build_gateway_table(0, region_id)

    region_envs = {}
    for region_id in range(1, 17):
        print(f"\n区域{region_id} ")
        env = GraphEnv(sat_network, region_id)
        env.initialize_environment()
        region_envs[region_id] = env

    region_ids = list(range(1, 17))
    agents_dict = load_region_models(region_ids, node_dim, edge_dim, hidden_dim, action_dim)

    for total_flow in range(2, 11):
        print(f"\n========== 正在模拟 total_flow = {total_flow} Gbps ==========")

        delay_list = []
        loss_list = []
        link_u_list = []
        delivery_rate_list = []

        tasks = generate_tasks(total_flow, region_envs)
        total_tasks = len(tasks)
        print(f"开始处理 {total_tasks} 个任务...")

        for task_idx, task in enumerate(tasks):
            try:
                delay, delivery_rate, loss, link_u = evaluate_full_task(agents_dict, region_envs, sat_network,
                                                                         task)
                delay_list.append(delay)
                loss_list.append(loss)
                link_u_list.append(link_u)
                delivery_rate_list.append(delivery_rate)

                if (task_idx + 1) % 100 == 0:
                    progress = (task_idx + 1) / total_tasks * 100
                    print(f"已完成: {progress:.1f}% ({task_idx+1}/{total_tasks})")

            except Exception as e:
                print(f"任务失败: {e}")
                print(f"失败任务详情: {task}")
                traceback.print_exc()
                continue

        print(f"所有任务处理完成! 成功完成")

        avg_delay = sum(delay_list) / len(delay_list) if delay_list else 0
        avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
        avg_utilization = sum(link_u_list) / (len(link_u_list) * 15) if link_u_list else 0
        avg_delivery = sum(delivery_rate_list) / len(delivery_rate_list) if delivery_rate_list else 0

        save_simulation_result(
            total_flow,
            delay_list, loss_list, link_u_list,
            delivery_rate_list,
            save_dir=f"new_result"
        )
