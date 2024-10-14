import json
import networkx as nx
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# 读取JSON文件
with open('../data/simpledata1.json', 'r') as file:
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

# # 输出每条边的权重
# for u, v, data in G.edges(data=True):
#     print(f"Edge from {u} to {v} has weight {data['weight']}")
# # 输出直连边的条数
# num_edges = G.number_of_edges()
# print(f"Number of direct edges: {num_edges}") #443

# # 获取节点列表
# nodes_unsorted = list(G.nodes)
# print("Unsorted Nodes:", nodes_unsorted)
#
# # 排序节点列表
# nodes_sorted = sorted(nodes_unsorted, key=int)
# print("Sorted Nodes:", nodes_sorted)

# 获取并排序节点列表
nodes = sorted(G.nodes, key=int)

# 获取邻接矩阵，使用排序后的节点列表
adj_matrix = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

# np.savetxt("data/adjacency_matrix1.txt", adj_matrix, fmt='%.2f')

# 输出节点和邻接矩阵
# print("Nodes:", nodes)
# print("Adjacency Matrix:")
# print(adj_matrix)

# 计算所有边的距离
distances = []
for u, v in G.edges():
    x1, y1 = G.nodes[u]['pos']
    x2, y2 = G.nodes[v]['pos']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distances.append(distance)

# 找到最大距离用于归一化
max_distance = max(distances)

# 更新权重为归一化后的距离
for u, v in G.edges():
    x1, y1 = G.nodes[u]['pos']
    x2, y2 = G.nodes[v]['pos']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    normalized_weight = (distance / max_distance) * 100  # 归一化
    G[u][v]['weight'] = normalized_weight

# 输出更新后的邻接矩阵
np.set_printoptions(threshold=np.inf, precision=2)
adj_matrix = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

# 保存邻接矩阵到文件
# np.savetxt("data/adjacency_matrix.txt", adj_matrix, fmt='%.2f')


# 创建节点到索引的映射，从1开始
node_to_index = {node: idx + 1 for idx, node in enumerate(nodes)}

# 提取边索引并转换为数值索引
edges = [(node_to_index[u], node_to_index[v]) for u, v in G.edges]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 保存 edge_index 到文件
with open("data/edge_index.txt", "w") as f:
    for row in edge_index:
        f.write(" ".join(map(str, row.tolist())) + "\n")

print("Edge Index saved to 'edge_index.txt'")


# 验证边索引是否准确
def are_directly_connected(G, node1, node2):
    # 使用实际的节点ID
    return G.has_edge(node1, node2)


node1 = '328'
node2 = '249'

connected_nodes = list(G.neighbors(node1))

# 输出直接连接的节点
print(f"Nodes directly connected to node {node1}: {connected_nodes}")

if are_directly_connected(G, node1, node2):
    print(f"Node {node1} and Node {node2} are directly connected.")
else:
    print(f"Node {node1} and Node {node2} are not directly connected.")

# # 可视化图
# pos = nx.get_node_attributes(G, 'pos')
# plt.figure(figsize=(10, 8))  # 调整图的大小
# nx.draw(G, pos, with_labels=True, node_size=300, node_color="lightblue", font_size=10)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
# plt.show()
