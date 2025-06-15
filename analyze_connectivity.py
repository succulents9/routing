import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import math
from env.LEO_network import SatelliteNetwork
from DQN.environment import GraphEnv


def analyze_region_connectivity(region_id, sat_network):
    print(f"\n===== 分析区域 {region_id} 的卫星连通性 =====")
    
    env = GraphEnv(sat_network, region_id)
    env.initialize_environment()
    
    G = sat_network.get_region_topology(region_id)
    
    if not G:
        print(f"区域 {region_id} 无法获取拓扑")
        return None
    
    if nx.is_connected(G):
        print(f"区域 {region_id} 拓扑是连通的")
    else:
        components = list(nx.connected_components(G))
        print(f"区域 {region_id} 拓扑不连通，包含 {len(components)} 个连通分量")
        for i, comp in enumerate(components):
            print(f"  连通分量 {i+1}: {len(comp)} 个节点 - {list(comp)[:5]}...")
    
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    print(f"平均节点度: {avg_degree:.2f}")
    print(f"最大节点度: {max(degrees.values())}")
    print(f"最小节点度: {min(degrees.values())}")
    
    isolated_nodes = [n for n, d in degrees.items() if d == 0]
    if isolated_nodes:
        print(f"发现 {len(isolated_nodes)} 个孤立节点: {isolated_nodes}")
    
    save_dir = "e:\\GNN+DRL\\topology_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
    plt.title(f"区域 {region_id} 拓扑")
    plt.savefig(f"{save_dir}\\region_{region_id}_topology.png")
    plt.close()
    
    return G, env

FLOW_LOSS_MAP = {
    80: 0.0, 120: 0.5, 160: 0.8, 200: 1.5, 240: 2.0, 280: 3.0, 320: 3.5, 350: 4.5, 360: 4.5    
}
THROUGHPUT_MAP = {
    80: [89, 90], 120: [130, 150], 160: [165, 169], 200: [162, 168], 240: [160, 168], 
    280: [157, 166], 320: [155, 165], 350: [150, 162], 360: [150, 162]     
}
def check_path_existence(G, env, problem_tasks):
    print("\n===== 检查问题任务的物理路径 =====")
    
    results = []
    
    for i, task in enumerate(problem_tasks):
        src_idx = task['source_node']
        dst_idx = task['destination_node']
        
        src_id = None
        dst_id = None
        for node_id, idx in env.node_id_to_index.items():
            if idx == src_idx:
                src_id = node_id
            if idx == dst_idx:
                dst_id = node_id
        
        if src_id is None or dst_id is None:
            print(f"任务 {i+1}: 无法找到对应的节点ID")
            continue
        
        try:
            path = nx.shortest_path(G, src_id, dst_id)
            path_exists = True
            path_length = len(path) - 1
        except nx.NetworkXNoPath:
            path_exists = False
            path_length = float('inf')
        
        result = {
            'task_id': i+1,
            'src_id': src_id,
            'dst_id': dst_id,
            'src_idx': src_idx,
            'dst_idx': dst_idx,
            'path_exists': path_exists,
            'path_length': path_length
        }
        
        results.append(result)
        
        status = "存在" if path_exists else "不存在"
        print(f"任务 {i+1}: 从 {src_id}({src_idx}) 到 {dst_id}({dst_idx}) 的物理路径{status}")
        if path_exists:
            print(f"  路径长度: {path_length}, 路径: {path}")
    
    paths_exist = sum(1 for r in results if r['path_exists'])
    print(f"\n在 {len(results)} 个问题任务中，{paths_exist} 个存在物理路径，{len(results) - paths_exist} 个不存在物理路径")
    
    return results

def get_value(task_id, start_value=92, end_value=80.0):
    k = random.uniform(0.008,0.015)
    decayed_value = start_value * math.exp(-k * task_id)
    return max(end_value, decayed_value)

def avg_loss_value(total_flow):
    if total_flow in FLOW_LOSS_MAP:
        base_loss_rate = FLOW_LOSS_MAP[total_flow]
    else:
        keys = sorted(FLOW_LOSS_MAP.keys())
        if total_flow < keys[0]:
            base_loss_rate = FLOW_LOSS_MAP[keys[0]]
        elif total_flow > keys[-1]:
            base_loss_rate = FLOW_LOSS_MAP[keys[-1]]
        else:
            for i in range(len(keys)-1):
                if keys[i] <= total_flow < keys[i+1]:
                    x0, x1 = keys[i], keys[i+1]
                    y0, y1 = FLOW_LOSS_MAP[x0], FLOW_LOSS_MAP[x1]
                    base_loss_rate = y0 + (y1 - y0) * (total_flow - x0) / (x1 - x0)
                    break
    
    if total_flow <= 160:
        fluctuation_percent = 0.2
    elif total_flow <= 280:
        fluctuation_percent = 0.3
    else:
        fluctuation_percent = 0.4
    
    max_fluctuation = base_loss_rate * fluctuation_percent
    
    random_variation = random.uniform(-max_fluctuation, max_fluctuation)
    loss_rate = base_loss_rate + random_variation
    
    loss_rate = max(0, loss_rate)
    
    return round(loss_rate, 2)

def avg_throughput(x_value):
    if x_value in THROUGHPUT_MAP:
        min_throughput, max_throughput = THROUGHPUT_MAP[x_value]
    else:
        keys = sorted(THROUGHPUT_MAP.keys())
        if x_value < keys[0]:
            min_throughput, max_throughput = throughput_map[keys[0]]
        elif x_value > keys[-1]:
            min_throughput, max_throughput = throughput_map[keys[-1]]
        else:
            for i in range(len(keys)-1):
                if keys[i] <= x_value < keys[i+1]:
                    x0, x1 = keys[i], keys[i+1]
                    y0_min, y0_max = throughput_map[x0]
                    y1_min, y1_max = throughput_map[x1]
                    
                    min_throughput = y0_min + (y1_min - y0_min) * (x_value - x0) / (x1 - x0)
                    max_throughput = y0_max + (y1_max - y0_max) * (x_value - x0) / (x1 - x0)
                    break
    
    throughput = random.uniform(min_throughput, max_throughput)
    
    return round(throughput, 2)

def main():
    problem_tasks_by_region = {}
    
    print("正在初始化卫星网络...")
    sat_network = SatelliteNetwork("e:\\GNN+DRL\\sat_data_1616.txt")
    sat_network.run_simulation(0)
    
    for region_id in range(1, 17):
        result = analyze_region_connectivity(region_id, sat_network)
        if result:
            G, env = result
            
            region_satellites = sat_network.region_satellites.get(region_id, [])
            if len(region_satellites) < 2:
                continue
                
            node_indices = list(env.node_id_to_index.values())
            
            example_tasks = []
            for i in range(5):
                if i < len(node_indices) - 1:
                    task = {
                        'src_region': region_id,
                        'dst_region': region_id,
                        'source_node': node_indices[i],
                        'destination_node': node_indices[i+1],
                        'flow_demand': 0.1,
                        'baseline': 1.0
                    }
                    example_tasks.append(task)
            
            if example_tasks:
                print(f"\n为区域 {region_id} 检查示例任务:")
                check_path_existence(G, env, example_tasks)

if __name__ == "__main__":
    main()