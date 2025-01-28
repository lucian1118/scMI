import torch
import random
import numpy as np
from collections import Counter
from torch_geometric.data import HeteroData


def neighbors(data):
    all_neighbors = {node_type: {} for node_type in data.node_types}
    all_neighbors_type = {node_type: {} for node_type in data.node_types}
            
    for node_type in data.node_types:
        for node in range(len(data[node_type].x)):
            all_neighbors[node_type][node] = []
            all_neighbors_type[node_type][node] = []   
            for edge_type in data.edge_types:
                if edge_type[0] == node_type:
                    edge_index = data[edge_type].edge_index
                    neighbors = edge_index[1][edge_index[0] == node].tolist()

                    if edge_type[1] in ["cc", "rev_cc"]:
                        neighbors = neighbors * 150
                    all_neighbors[node_type][node].extend(neighbors)
                    all_neighbors_type[node_type][node].extend([edge_type[2]] * len(neighbors))
    
    return all_neighbors, all_neighbors_type


def subgraph_random_walk_with_restart(data, all_neighbors, all_neighbors_type, start_node, start_node_type, top_k, iter, walk_length, restart_prob):
    
    sampled_nodes = {node_type: [] for node_type in data.node_types}
    
    for _ in range(iter):
        current_node = start_node
        current_type = start_node_type
        sampled_nodes[current_type].append(current_node)

        for _ in range(walk_length):

            if len(all_neighbors[current_type][current_node]) == 0:
                break

            if random.random() < restart_prob:
                current_node = start_node
                current_type = start_node_type
            else:
                random_index = random.randint(0, len(all_neighbors[current_type][current_node]) - 1)
                current_node, current_type = all_neighbors[current_type][current_node][random_index], all_neighbors_type[current_type][current_node][random_index]
            
            sampled_nodes[current_type].append(current_node)
    
    for node_type in data.node_types:
        sampled_nodes[node_type] = [item for item, _ in Counter(sampled_nodes[node_type]).most_common(top_k)]
        
    subgraph = data.subgraph(sampled_nodes)
    
    adjusted_graph = HeteroData()
    adjusted_graph[start_node_type].x = subgraph[start_node_type].x
    for node_type in subgraph.node_types:
        adjusted_graph[node_type].x = subgraph[node_type].x
        if adjusted_graph[node_type].x.size(0) < top_k:
            adjusted_graph[node_type].x = torch.cat([adjusted_graph[node_type].x, torch.zeros(top_k - adjusted_graph[node_type].x.size(0), adjusted_graph[node_type].x.size(1))])
    for edge_type in subgraph.edge_types:
        adjusted_graph[edge_type].edge_index = subgraph[edge_type].edge_index
    
    for node_type in data.node_types:
        adjusted_graph[node_type].n_id = sampled_nodes[node_type]

    return adjusted_graph


def generate_subgraphs(data, types, top_k=5, iter=20, walk_length=10, restart_prob=0.1):
    
    all_neighbors, all_neighbors_type = neighbors(data)
    
    subgraphs = []
        
    for node_type in types:
        for node in range(len(data[node_type].x)):
            subgraph = subgraph_random_walk_with_restart(
                data, all_neighbors, all_neighbors_type, start_node=node, start_node_type=node_type,
                top_k=top_k, iter=iter, walk_length=walk_length, restart_prob=restart_prob
            )
            subgraphs.append((node_type, node, subgraph))
    
    return subgraphs