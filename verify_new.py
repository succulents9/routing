import numpy as np
import os
import torch
import traceback
import random
import matplotlib.pyplot as plt
from DQN.environment import GraphEnv
from DQN.model import DQNAgent
from env.LEO_network import SatelliteNetwork
from main_eval import evaluate_full_task
from utils.plot import save_simulation_results

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def generate_tasks_by_total_flow(total_flow_mbps, region_envs):
    random.seed(40)
    demand_ratios = [0, 0, 2, 0, 1, 0, 2, 1, 0, 2, 2, 1, 2, 1, 2, 0]
    all_regions = list(range(1, 17))

    total_flow_gbps = total_flow_mbps / 1000.0

    task_to_num_tasks = {80: 30,120:80 ,160: 80,200: 80, 240: 80,280: 80,320: 80, 350:80,}
    cross_prob_map = {80: 0.35,120: 0.65,160: 0.6,200: 0.65,240: 0.7,280: 0.75,320: 0.8,350:0.8,}

    num_tasks = task_to_num_tasks.get(total_flow_mbps, 300)
    cross_prob = cross_prob_map.get(total_flow_mbps, 0.6)

    total_ratio = sum(demand_ratios)
    region_total_flows = [r / total_ratio * total_flow_gbps for r in demand_ratios]
    region_task_counts = [int(num_tasks * (f / total_flow_gbps)) if r > 0 else 0
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
            continue

        src_env = region_envs[src_region]
        valid_src_nodes = list(src_env.node_id_to_index.keys())
        if not valid_src_nodes:
            continue

        avg_flow = region_total_flows[region_idx] / task_count

        for _ in range(task_count):
            source_node = random.choice(valid_src_nodes)

            if random.random() < cross_prob:
                dst_region_candidates = [r for r in all_regions if r != src_region and demand_ratios[r - 1] > 0 and r in region_envs]
                dst_region = random.choice(dst_region_candidates) if dst_region_candidates else src_region
            else:
                dst_region = src_region

            if dst_region not in region_envs:
                continue

            dst_env = region_envs[dst_region]
            valid_dst_nodes = list(dst_env.node_id_to_index.keys())
            if not valid_dst_nodes:
                continue

            if dst_region == src_region:
                valid_dst_nodes = [n for n in valid_dst_nodes if n != source_node]
                if not valid_dst_nodes:
                    continue

            destination_node = random.choice(valid_dst_nodes)

            task = {
                'src_region': src_region,
                'dst_region': dst_region,
                'source_node': source_node,
                'destination_node': destination_node,
                'flow_demand': avg_flow,
                'total_flow': total_flow_gbps
            }
            tasks.append(task)

    return tasks

def load_region_models(region_ids, node_dim, edge_dim, hidden_dim, action_dim,
                    model_path_template="./saved_models/region_{}_model_16.pth"):
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
    region_num = 16
    node_dim = 1
    edge_dim = 2
    hidden_dim = 64
    action_dim = 8

    sat_network = SatelliteNetwork("./starlink.txt")
    sat_network.run_simulation(0)
    for region_id in range(1, 17):
        sat_network.build_gateway_table(0, region_id)

    region_envs = {}
    for region_id in range(1, 17):
        print(f"\n初始化区域 {region_id}")
        env = GraphEnv(sat_network, region_id)
        env.initialize_environment()
        region_envs[region_id] = env

    agents_dict = load_region_models(list(range(1, 17)), node_dim, edge_dim, hidden_dim, action_dim)

    total_flows_mbps = [80, 120, 160, 200, 240, 280, 320,350]

    for total_flow_mbps in total_flows_mbps:
        total_flow_gbps = total_flow_mbps
        print(f"\n========== 正在模拟 total_flow = {total_flow_gbps:.3f} Mbps ==========")

        delay_list = []
        loss_list = []
        link_u_list = []
        delivery_rate_list = []
        throughput_list = []

        tasks = generate_tasks_by_total_flow(total_flow_gbps, region_envs)
        total_tasks = len(tasks)
        print(f"开始处理任务...")

        for task_idx, task in enumerate(tasks):
            try:
                delay, delivery_rate, loss, link_u, throughput = evaluate_full_task(
                    agents_dict, region_envs, sat_network, task
                )
                delay_list.append(delay)
                loss_list.append(loss)
                link_u_list.append(link_u)
                delivery_rate_list.append(delivery_rate)
                throughput_list.append(throughput)

                if (task_idx + 1) % 100 == 0:
                    progress = (task_idx + 1) / total_tasks * 100
                    

            except Exception as e:
                print(f"任务失败: {e}")
                print(f"失败任务详情: {task}")
                traceback.print_exc()
                continue

        print(f"所有任务处理完成!")

        avg_delay = sum(delay_list) / len(delay_list) if delay_list else 0
        avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
        avg_utilization = sum(link_u_list) / (len(link_u_list) * 15) if link_u_list else 0
        avg_delivery = sum(delivery_rate_list) / len(delivery_rate_list) if delivery_rate_list else 0
        avg_throughput = sum(throughput_list) / len(throughput_list) if throughput_list else 0

        save_simulation_results(
            total_flow_gbps,
            delay_list, loss_list, link_u_list,
            delivery_rate_list, throughput_list,
            save_dir="new_result"
        )