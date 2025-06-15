import random
from env.LEO_network import SatelliteNetwork
from DQN.QMIX import RegionRouter
from scipy.constants import c


def evaluate_route(agent, env, task):
    if task['source_node'] == task['destination_node']:
        raise ValueError(f"源节点和目标节点不能相同: {task['source_node']}")
        
    state = env.reset_task(task)
    done = False

    total_loss = 0.0
    total_delay = 0.0
    total_util = 0.0
    delivery_packet = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

        total_delay += info.get("delay", 0.0)
        total_loss += info.get("loss", 0.0)
        total_util += info.get("utilization", 0.0)
        delivery_packet += 1

    success = env.remaining_demand <= 0

    metrics = {
        "success": success,
        "delivery_packet": delivery_packet,
        "delay": round(total_delay, 2),
        "loss": round(total_loss, 2),
        "util": round(total_util, 2),
    }
    return metrics


def evaluate_full_task(agents_dict, region_envs, sat_network, task):
    region_router = RegionRouter(sat_network.inter_region_gateways)
    src = task['source_node']
    dst = task['destination_node']
    src_region = task['src_region']
    dst_region = task['dst_region']

    total_delay = 0.0
    total_packets = 0
    total_loss = 0.0
    total_util = 0.0
    packet_num = task['flow_demand'] * 1e9 / (20 * 8e3)

    src_env = region_envs[src_region]
    if src not in src_env.node_id_to_index:
        raise ValueError(f"源节点 {src} 不在区域 {src_region} 的环境中")

    if src_region == dst_region:
        env = region_envs[src_region]
        agent = agents_dict[src_region]

        if dst not in env.node_id_to_index:
            raise ValueError(f"目标节点 {dst} 不在区域 {src_region} 的环境中")

        local_task = task.copy()
        metrics = evaluate_route(agent, env, local_task)
        total_delay += metrics['delay']
        total_packets += metrics['delivery_packet']
        total_loss += metrics['loss']
        total_util += metrics['util']

        throughput = (total_packets * 160000) / (total_delay / 1000) if total_delay > 0 else 0
        return total_delay, total_packets / packet_num, total_loss / packet_num, total_util, throughput

    while src_region != dst_region:
        gateways = region_router.get_next_hop_gateway(src_region, dst_region)
        if not gateways:
            raise ValueError(f"找不到从 {src_region} 到 {dst_region} 的网关对")
        g_src, g_dst = gateways

        env = region_envs[src_region]
        if g_src not in env.node_id_to_index:
            raise ValueError(f"网关节点 {g_src} 不在区域 {src_region} 的环境中")

        if src != g_src:
            sub_task = {
                'src_region': src_region,
                'dst_region': src_region,
                'source_node': src,
                'destination_node': g_src,
                'flow_demand': task['flow_demand'],
            }

            agent = agents_dict[src_region]
            metrics = evaluate_route(agent, env, sub_task)
            total_delay += metrics['delay']
            total_packets += metrics['delivery_packet']
            total_loss += metrics['loss']
            total_util += metrics['util']

        p1 = sat_network.satellite_dict[g_src].get_position_at_time(0)
        p2 = sat_network.satellite_dict[g_dst].get_position_at_time(0)
        region_delay = SatelliteNetwork.calculate_distance(p1, p2) / c
        total_delay += region_delay * 1e3

        src = g_dst
        src_region = sat_network.satellite_dict.get(src).region

    if src != dst:
        dst_env = region_envs[dst_region]
        if dst not in dst_env.node_id_to_index:
            raise ValueError(f"目标节点 {dst} 不在区域 {dst_region} 的环境中")

        sub_task = {
            'src_region': dst_region,
            'dst_region': dst_region,
            'source_node': src,
            'destination_node': dst,
            'flow_demand': task['flow_demand'],
        }
        env = region_envs[dst_region]
        agent = agents_dict[dst_region]

        metrics = evaluate_route(agent, env, sub_task)

        total_delay += metrics['delay']
        total_packets += metrics['delivery_packet']
        total_loss += metrics['loss']
        total_util += metrics['util']

    throughput = (total_packets * 160000) / (total_delay / 1000) if total_delay > 0 else 0

    return total_delay, total_packets / packet_num, total_loss / packet_num, total_util, throughput