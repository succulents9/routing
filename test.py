from env.LEO_network import SatelliteNetwork

sat_network = SatelliteNetwork("e:\\GNN+DRL\\sat_data_1616.txt")
sat_network.run_simulation(0)

print("区域卫星分布情况:")
for region_id in range(1, 17):
    satellites = sat_network.region_satellites.get(region_id, [])
    print(f"区域 {region_id} 包含 {len(satellites)} 个卫星:")
    if len(satellites) > 0:
        print(f"  前几个卫星ID: {satellites[:10]}")
        if len(satellites) > 10:
            print(f"  ... 等共 {len(satellites)} 个")
    else:
        print("  该区域没有卫星")
    print()

print("\n测试区域拓扑:")
for region_id in range(1, 17):
    G = sat_network.get_region_topology(region_id)
    if G:
        print(f"区域 {region_id} 拓扑: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        if nx.is_connected(G):
            print("  拓扑是连通的")
        else:
            components = list(nx.connected_components(G))
            print(f"  拓扑不连通，包含 {len(components)} 个连通分量")
    else:
        print(f"区域 {region_id} 无法获取拓扑")