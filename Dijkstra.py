import heapq
from datainput import G

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes}

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, attributes in graph[current_node].items():
            weight = attributes['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    paths = {node: [] for node in graph.nodes}
    for node in graph.nodes:
        if distances[node] < float('inf'):
            current = node
            while current is not None:
                paths[node].insert(0, current)
                current = predecessors[current]

    formatted_distances = {node: f"{dist:.2f}" for node, dist in distances.items()}

    return formatted_distances, paths

# 遍历所有节点作为起始节点
all_results = {}
for start_node in G.nodes:
    shortest_paths, paths = dijkstra(G, start_node)
    all_results[start_node] = (shortest_paths, paths)

# 保存结果到文件
with open("data/all_shortest_paths.txt", "w") as f:
    for start_node, (distances, paths) in all_results.items():
        f.write(f"From Node {start_node}:\n")
        for node in distances:
            f.write(f"  To Node {node}: Distance = {distances[node]}, Path = {paths[node]}\n")
        f.write("\n")

print("All shortest paths saved to 'all_shortest_paths.txt'")

# # 使用示例
# start_node = '1'  # 替换为你的起始节点
# shortest_paths, paths = dijkstra(G, start_node)
#
# # 保存结果到文件
# with open("shortest_paths.txt", "w") as f:
#     for node in shortest_paths:
#         f.write(f"Node {node}: Distance = {shortest_paths[node]}, Path = {paths[node]}\n")
#
# print(f"Shortest paths from {start_node} saved to 'shortest_paths.txt'")