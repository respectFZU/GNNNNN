import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import networkx as nx
from smallbatch.data import G, edge_index
import torch.nn.functional as F
import random
from tqdm import tqdm

# 增强节点特征
input_dim = 32
hidden_dim = 128
output_dim = G.number_of_nodes() + 1
node_features = torch.randn((G.number_of_nodes() + 1, input_dim))


# 初始化模型
class PathPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PathPredictor, self).__init__()
        self.conv1 = SAGEConv(input_dim + 2, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, current_node, target_node):
        start_feature = torch.zeros((x.size(0), 1), device=x.device)
        target_feature = torch.zeros((x.size(0), 1), device=x.device)
        start_feature[current_node] = 1.0
        target_feature[target_node] = 1.0
        x = torch.cat([x, start_feature, target_feature], dim=1)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.fc(x)

        probabilities = F.softmax(x, dim=1)

        masked_probabilities = torch.zeros_like(probabilities)
        for node in range(probabilities.size(0)):
            node_str = str(node)
            if node_str in G:
                neighbors = list(G.neighbors(node_str))
                neighbors = [int(n) for n in neighbors]
                mask = torch.zeros_like(probabilities[node])
                mask[neighbors] = 1.0
                masked_probabilities[node] = probabilities[node] * mask
                if masked_probabilities[node].sum() > 0:
                    masked_probabilities[node] /= masked_probabilities[node].sum()

        return masked_probabilities


model = PathPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 确保 edge_index 的形状正确
print(edge_index.shape)

# 训练模型
num_epochs = 1
node_pairs = [(i, j) for i in range(1, 27) for j in range(1, 27) if i != j]
random.shuffle(node_pairs)

# # 暂存错误路径：出现预测点和实际点不同，暂存起点终点当前点三元组，计算损失
# error_paths = []

for epoch in range(num_epochs):
    # 暂存错误路径：出现预测点和实际点不同，暂存起点终点当前点三元组，计算损失
    error_paths = []
    # 训练路径节点
    for start_node, target_node in tqdm(node_pairs, desc=f"Epoch {epoch+1}"):
        try:
            shortest_path = nx.dijkstra_path(G, source=str(start_node), target=str(target_node))
            shortest_path = [int(node) for node in shortest_path]

            path_data = Data(x=node_features, edge_index=edge_index)

            model.train()
            current_node = start_node
            valid_path = True

            while current_node != target_node:
                optimizer.zero_grad()

                # 获取当前节点的特征
                output = model(path_data.x, path_data.edge_index, current_node, target_node)
                next_node = output[current_node].argmax().item()

                # 打印输出概率
                print(f"Current node: {current_node}, Target node: {target_node}")
                row = output[current_node]
                non_zero_indices = row > 0
                non_zero_probs = row[non_zero_indices]
                non_zero_indices = non_zero_indices.nonzero(as_tuple=True)[0]
                if non_zero_probs.numel() > 0:
                    print(f"Node {current_node}:")
                    for idx, prob in zip(non_zero_indices, non_zero_probs):
                        print(f"  To node {idx.item()}: {prob.item():.2f}")

                # 检查预测是否正确
                correct_node = shortest_path[shortest_path.index(current_node) + 1]
                if next_node == correct_node:
                    # 更新当前节点
                    current_node = next_node
                else:
                    valid_path = False
                    error_paths.append((start_node, target_node, current_node))

                    # 计算加权损失
                    weight = torch.ones(output.size(1))
                    weight[correct_node] = 5.0  # 对正确节点施加更高权重
                    step_loss = F.cross_entropy(output[current_node].unsqueeze(0),
                                                torch.tensor([correct_node], dtype=torch.long),
                                                weight=weight)

                    # 反向传播和优化
                    step_loss.backward()
                    optimizer.step()
                    break

            if valid_path:
                print(f"Epoch {epoch + 1}, Start {start_node}, Target {target_node}, Path completed")

        except nx.NetworkXNoPath:
            print(f"No path from {start_node} to {target_node}")

    # 训练错误路径
    for start_node, target_node, current_node in tqdm(error_paths, desc="Re-training"):
        try:
            shortest_path = nx.dijkstra_path(G, source=str(current_node), target=str(target_node))
            shortest_path = [int(node) for node in shortest_path]

            path_data = Data(x=node_features, edge_index=edge_index)

            model.train()
            valid_path = True

            while current_node != target_node:
                optimizer.zero_grad()

                # 获取当前节点的特征
                output = model(path_data.x, path_data.edge_index, current_node, target_node)
                next_node = output[current_node].argmax().item()

                # 打印输出概率
                print(f"Current node: {current_node}, Target node: {target_node}")
                row = output[current_node]
                non_zero_indices = row > 0
                non_zero_probs = row[non_zero_indices]
                non_zero_indices = non_zero_indices.nonzero(as_tuple=True)[0]
                if non_zero_probs.numel() > 0:
                    print(f"Node {current_node}:")
                    for idx, prob in zip(non_zero_indices, non_zero_probs):
                        print(f"  To node {idx.item()}: {prob.item():.2f}")

                # 检查预测是否正确
                correct_node = shortest_path[shortest_path.index(current_node) + 1]
                if next_node == correct_node:
                    # 更新当前节点
                    current_node = next_node
                else:
                    valid_path = False

                    # 计算加权损失
                    weight = torch.ones(output.size(1))
                    weight[correct_node] = 5.0  # 对正确节点施加更高权重
                    step_loss = F.cross_entropy(output[current_node].unsqueeze(0),
                                                torch.tensor([correct_node], dtype=torch.long),
                                                weight=weight)

                    # 反向传播和优化
                    step_loss.backward()
                    optimizer.step()
                    break

            if valid_path:
                print(f"Re-training Start {start_node}, Target {target_node}, Path completed")

        except nx.NetworkXNoPath:
            print(f"No path from {start_node} to {target_node}")

# 保存模型
torch.save(model.state_dict(), 'gcn_model.pth')

# 加载模型
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()

# 测试模型并计算准确率
correct_count = 0
total_count = 0
shortest_path_count = 0
non_shortest_path_count = 0
invalid_path_count = 0
incomplete_path_count = 0

for start_node in tqdm(range(1, G.number_of_nodes() + 1), desc="Testing"):
    for target_node in range(1, G.number_of_nodes() + 1):
        if start_node == target_node:
            continue

        try:
            correct_path = nx.dijkstra_path(G, source=str(start_node), target=str(target_node))
            correct_path = [int(node) for node in correct_path]

            with torch.no_grad():
                predicted_path = [start_node]
                current_node = start_node
                valid_path = True
                visited = set()

                while current_node != target_node:
                    output = model(node_features, edge_index, current_node, target_node)
                    next_node = output[current_node].argmax().item()

                    # 打印输出概率
                    print(f"Current node: {current_node}, Start node: {start_node}, Target node: {target_node}")
                    row = output[current_node]
                    non_zero_indices = row > 0
                    non_zero_probs = row[non_zero_indices]
                    non_zero_indices = non_zero_indices.nonzero(as_tuple=True)[0]
                    if non_zero_probs.numel() > 0:
                        print(f"Node {current_node}:")
                        for idx, prob in zip(non_zero_indices, non_zero_probs):
                            print(f"  To node {idx.item()}: {prob.item():.2f}")

                    if current_node in correct_path and next_node == correct_path[correct_path.index(current_node) + 1]:
                        # 正确预测，继续
                        current_node = next_node
                    else:
                        # 检查是否为合法边
                        if G.has_edge(str(current_node), str(next_node)):
                            if next_node in visited:
                                print("检测到循环，路径预测失败")
                                valid_path = False
                                break
                            predicted_path.append(next_node)
                            visited.add(next_node)
                            current_node = next_node
                            if len(predicted_path) - len(correct_path) > 3:
                                # print("路径过长，寻路失败")
                                # valid_path = False
                                break
                        else:
                            print("路径预测失败")
                            valid_path = False
                            break

                    if len(predicted_path) > len(correct_path) + 3:
                        print("路径过长，寻路失败")
                        valid_path = False
                        break

                if current_node != target_node:
                    incomplete_path_count += 1

            if valid_path and current_node == target_node:
                correct_count += 1
                if predicted_path == correct_path:
                    shortest_path_count += 1
                    print(f"起点 {start_node} 到终点 {target_node} 最短路径预测成功")
                else:
                    non_shortest_path_count += 1
                    print(f"起点 {start_node} 到终点 {target_node} 非最短路径预测成功")

            total_count += 1

            print(f"Target node: {target_node}")
            print(f"Correct path: {correct_path}")
            print(f"Predicted path: {predicted_path}")

        except nx.NetworkXNoPath:
            print(f"No path from {start_node} to {target_node}")

accuracy = correct_count / total_count
print(f"Accuracy: {accuracy:.2f}")
print(f"Number of correct paths: {correct_count}")
print(f"Number of shortest paths: {shortest_path_count}")
print(f"Number of non-shortest paths: {non_shortest_path_count}")
print(f"Number of invalid paths: {invalid_path_count}")
print(f"Number of incomplete paths: {incomplete_path_count}")