import json
import networkx as nx
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# 读取JSON文件
with open('1015.json', 'r') as file:
    data = json.load(file)

# 创建一个无向图
G = nx.Graph()

# 解析节点和边
for node in data["nodes"]:
    node_id = node["id"]
    x, y = node["coordinate"]["x"], node["coordinate"]["y"]

    # 添加节点到图中
    G.add_node(node_id, pos=(x, y))

    # 添加边到图中
    for edge in node["edges"]:
        destination = edge["destination"]
        weight = edge["weight"]
        G.add_edge(node_id, destination, weight=weight)

# 获取并排序节点列表
nodes = sorted(G.nodes, key=int)
print("Sorted Nodes:", nodes)

# 创建节点到索引的映射，从1开始
node_to_index = {node: idx + 1 for idx, node in enumerate(nodes)}
print("Node to Index Mapping:", node_to_index)

# 创建一个新的图，使用排序后的节点和更新后的边
sorted_G = nx.Graph()

# 添加排序后的节点到新图中
for node in nodes:
    sorted_G.add_node(node_to_index[node], pos=G.nodes[node]['pos'])

# 添加边到新图中，使用更新后的节点索引
for u, v, data in G.edges(data=True):
    sorted_G.add_edge(node_to_index[u], node_to_index[v], weight=data['weight'])

# 提取边索引并转换为数值索引
edges = [(node_to_index[u], node_to_index[v]) for u, v in G.edges]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
# print("Edge Index Tensor:")
# print(edge_index)

# 获取邻接矩阵，使用排序后的节点列表
adj_matrix = nx.to_numpy_array(sorted_G, nodelist=sorted(sorted_G.nodes), weight='weight')

# # 输出邻接矩阵
# print("Adjacency Matrix:")
# print(adj_matrix)

# 计算所有边的距离
distances = []
for u, v in sorted_G.edges():
    x1, y1 = sorted_G.nodes[u]['pos']
    x2, y2 = sorted_G.nodes[v]['pos']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distances.append(distance)

# 找到最大距离用于归一化
max_distance = max(distances)

# 更新权重为归一化后的距离
for u, v in sorted_G.edges():
    x1, y1 = sorted_G.nodes[u]['pos']
    x2, y2 = sorted_G.nodes[v]['pos']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    normalized_weight = (distance / max_distance) * 100  # 归一化
    sorted_G[u][v]['weight'] = normalized_weight
#
# # 输出每条边的权重
# for u, v, data in sorted_G.edges(data=True):
#     print(f"Edge from {u} to {v} has weight {data['weight']}")