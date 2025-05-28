import random
import networkx as nx
import os
from env.LEO_network import SatelliteNetwork
from DQN.environment import GraphEnv
from DQN.model import DQNAgent
import torch
import traceback

def test_region_reachability(region_id, sat_network, agent, num_tests=10):
    print(f"\n===== 测试区域 {region_id} 的路径可达性 =====")
    region_satellites = sat_network.region_satellites.get(region_id, [])
    if not region_satellites:
        print(f"区域 {region_id} 中没有卫星")
        return 0, []

    env = GraphEnv(sat_network, region_id)
    env.initialize_environment()
    print(f"区域 {region_id} 包含 {len(region_satellites)} 个卫星")

    success_count = 0
    failed_tasks = []

    for test in range(num_tests):
        if len(region_satellites) < 2:
            print(f"区域 {region_id} 中卫星数量不足，无法进行测试")
            return 0, []

        source = random.choice(region_satellites)
        while True:
            destination = random.choice(region_satellites)
            if destination != source:
                break

        task = {
            'src_region': region_id,
            'dst_region': region_id,
            'source_node': env.node_id_to_index[source],
            'destination_node': env.node_id_to_index[destination],
            'flow_demand': 0.1,
            'baseline': 1.0
        }

        try:
            state = env.reset_task(task)
            done = False
            max_steps = 50
            step_count = 0
            while not done and step_count < max_steps:
                action = agent.select_action(state, epsilon=0)
                next_state, reward, done, info = env.step(action)
                state = next_state
                step_count += 1

            if done and info.get('success', False):
                print(f"测试 {test+1}: 从 {source} 到 {destination} 可达，步数: {step_count}")
                success_count += 1
            else:
                print(f"测试 {test+1}: 从 {source} 到 {destination} 不可达，步数: {step_count}")
                failed_tasks.append(task)
        except Exception as e:
            print(f"测试 {test+1}: 从 {source} 到 {destination} 出错: {e}")
            traceback.print_exc()
            failed_tasks.append(task)

    if num_tests > 0:
        success_rate = (success_count / num_tests) * 100
        print(f"区域 {region_id} 的可达率: {success_rate:.2f}%")
        return success_rate, failed_tasks
    else:
        print("未进行测试")
        return 0, []

def main():
    print("正在初始化卫星网络...")
    sat_network = SatelliteNetwork("e:\\GNN+DRL\\sat_data_1616.txt")
    sat_network.run_simulation(0)

    node_dim = 1
    edge_dim = 2
    hidden_dim = 64
    action_dim = 8

    agents = {}
    for region_id in range(1, 17):
        model_path = f"e:\\GNN+DRL\\saved_models\\region_{region_id}_model_16_1.pth"
        if not os.path.exists(model_path):
            print(f"区域 {region_id} 的模型不存在，使用区域1的模型")
            model_path = "e:\\GNN+DRL\\saved_models\\region_1_model_1681.pth"
        agent = DQNAgent(node_dim, edge_dim, hidden_dim, action_dim)
        try:
            agent.q_network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            agent.q_network.eval()
            agents[region_id] = agent
            print(f"成功加载区域 {region_id} 的模型: {model_path}")
        except Exception as e:
            print(f"加载区域 {region_id} 的模型失败: {e}")
            model_path = "e:\\GNN+DRL\\saved_models\\region_1_model_1681.pth"
            agent.q_network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            agent.q_network.eval()
            agents[region_id] = agent
            print(f"使用区域1的模型作为备选")

    results = {}
    problem_tasks = {}

    for region_id in range(1, 17):
        agent = agents[region_id]
        success_rate, failed_tasks = test_region_reachability(region_id, sat_network, agent, num_tests=20)
        results[region_id] = success_rate
        problem_tasks[region_id] = failed_tasks

    print("\n===== 区域路径可达性测试结果汇总 =====")
    for region_id, rate in results.items():
        status = "✓ 完全可达" if rate == 100 else "✗ 部分不可达" if rate > 0 else "✗ 完全不可达"
        print(f"区域 {region_id}: {rate:.2f}% - {status}")

    problem_regions = [region_id for region_id, rate in results.items() if rate < 100]
    if problem_regions:
        print(f"\n需要注意的区域: {problem_regions}")
        print("\n问题任务详情:")
        for region_id in problem_regions:
            print(f"区域 {region_id} 的问题任务:")
            for i, task in enumerate(problem_tasks[region_id]):
                print(f"  任务 {i+1}: {task}")
    else:
        print("\n所有区域都是完全可达的")

if __name__ == "__main__":
    main()
