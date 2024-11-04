import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import networkx as nx
from ModelRecord.data import G, edge_index
import random
from tqdm import tqdm
import json
from ModelRecord.embeding import node_features

class PathPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PathPredictor, self).__init__()
        self.conv1 = SAGEConv(input_dim * 3, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, current_node, target_node, hidden=None):
        start_feature = x[current_node].unsqueeze(0).repeat(x.size(0), 1)
        target_feature = x[target_node].unsqueeze(0).repeat(x.size(0), 1)

        # 第一层卷积
        x = torch.cat([x, start_feature, target_feature], dim=1)
        x = self.conv1(x, edge_index).relu()

        # 第二层卷积
        target_node_feature = x[target_node].unsqueeze(0).repeat(x.size(0), 1)
        x = torch.cat([x, target_node_feature], dim=1)
        x = self.conv2(x, edge_index).relu()

        # 使用 LSTM 处理隐藏状态
        x = x.unsqueeze(0)  # 添加 batch 维度
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)  # 移除 batch 维度

        # 第三层卷积
        x = self.conv3(x, edge_index).relu()

        # 全连接层
        x = self.fc(x)
        probabilities = F.softmax(x, dim=1)

        masked_probabilities = torch.zeros_like(probabilities)
        for node in range(probabilities.size(0)):
            if node in G:
                neighbors = list(G.neighbors(node))
                neighbors = [int(n) for n in neighbors if int(n) < x.size(0)]
                mask = torch.zeros_like(probabilities[node])
                mask[neighbors] = 1.0
                masked_probabilities[node] = probabilities[node] * mask
                if masked_probabilities[node].sum() > 0:
                    masked_probabilities[node] /= masked_probabilities[node].sum()

        return masked_probabilities, hidden

# 初始化模型
input_dim = node_features.size(1)
hidden_dim = 128
output_dim = G.number_of_nodes()
model = PathPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# 确保 edge_index 的形状正确
print(edge_index.shape)

# 训练模型
num_epochs = 100
node_pairs = [(i, j) for i in range(G.number_of_nodes()) for j in range(G.number_of_nodes()) if i != j]
random.shuffle(node_pairs)

# 保存每个 epoch 的准确率指标
accuracy_metrics = []

for epoch in range(num_epochs):
    error_paths = []
    for start_node, target_node in tqdm(node_pairs, desc=f"Epoch {epoch + 1}"):
        try:
            # 使用整数索引
            shortest_path = nx.dijkstra_path(G, source=start_node, target=target_node)

            model.train()
            current_node = start_node
            valid_path = True
            hidden = None

            while current_node != target_node:
                optimizer.zero_grad()
                output, hidden = model(node_features, edge_index, current_node, target_node, hidden)
                next_node = output[current_node].argmax().item()

                correct_node = shortest_path[shortest_path.index(current_node) + 1]
                if next_node == correct_node:
                    current_node = next_node
                else:
                    valid_path = False
                    error_paths.append((start_node, target_node, current_node))

                    num_classes = output.size(1)

                    # 获取当前节点的输出
                    outputs = output[current_node].unsqueeze(0)

                    # 创建 one-hot 编码的标签
                    one_hot_labels = torch.zeros(num_classes)

                    # 确保 correct_node 在范围内
                    if correct_node < num_classes:
                        one_hot_labels[correct_node] = 1.0
                    else:
                        print(f"Error: Correct node {correct_node} is out of bounds.")
                        break

                    # 将 one-hot 标签转换为整数标签
                    integer_labels = torch.argmax(one_hot_labels).unsqueeze(0)

                    # 使用整数标签计算交叉熵损失
                    step_loss = F.cross_entropy(outputs, integer_labels)

                    # 反向传播和优化
                    step_loss.backward()
                    optimizer.step()
                    break

            if valid_path:
                print(f"Epoch {epoch + 1}, Start {start_node}, Target {target_node}, Path completed")

        except nx.NetworkXNoPath:
            print(f"No path from {start_node} to {target_node}")
        except nx.NodeNotFound as e:
            print(e)

    scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), 'GraphSage_small.pth')

    # 加载模型
    model.load_state_dict(torch.load('GraphSage_small.pth'))
    model.eval()

    # 测试模型并计算准确率
    correct_count = 0
    total_count = 0
    shortest_path_count = 0
    non_shortest_path_count = 0
    invalid_path_count = 0
    incomplete_path_count = 0

    results = []

    for start_node in tqdm(range(G.number_of_nodes()), desc="Testing"):
        for target_node in range(G.number_of_nodes()):
            if start_node == target_node:
                continue

            try:
                correct_path = nx.dijkstra_path(G, source=start_node, target=target_node)

                with torch.no_grad():
                    predicted_path = [start_node]
                    current_node = start_node
                    valid_path = True
                    visited = set()
                    prev_hidden = None

                    while current_node != target_node:
                        output, hidden_state = model(node_features, edge_index, current_node, target_node, prev_hidden)
                        prev_hidden = hidden_state  # 更新隐藏状态
                        next_node = output[current_node].argmax().item()

                        if current_node in correct_path and next_node == correct_path[
                            correct_path.index(current_node) + 1]:
                            current_node = next_node
                            predicted_path.append(current_node)
                        else:
                            if G.has_edge(current_node, next_node):
                                if next_node in visited:
                                    valid_path = False
                                    print(f"循环检测失败: {current_node} -> {next_node}")
                                    break
                                predicted_path.append(next_node)
                                visited.add(next_node)
                                current_node = next_node
                                if len(predicted_path) - len(correct_path) > 3:
                                    print(f"路径过长失败: {predicted_path}")
                                    break
                            else:
                                valid_path = False
                                invalid_path_count += 1
                                print(f"无效边失败: {current_node} -> {next_node}")
                                break

                    if current_node != target_node:
                        valid_path = False
                        incomplete_path_count += 1
                        print(f"路径不完整失败: {predicted_path}")

                if valid_path and current_node == target_node:
                    correct_count += 1
                    if predicted_path == correct_path:
                        shortest_path_count += 1
                        print(f"起点 {start_node} 到终点 {target_node} 最短路径预测成功")
                    else:
                        non_shortest_path_count += 1
                        print(f"起点 {start_node} 到终点 {target_node} 非最短路径预测成功")

                total_count += 1

                results.append({
                    "start_node": start_node,
                    "target_node": target_node,
                    "correct_path": correct_path,
                    "predicted_path": predicted_path,
                    "valid_path": valid_path
                })

                print(f"Target node: {target_node}")
                print(f"Correct path: {correct_path}")
                print(f"Predicted path: {predicted_path}")

            except nx.NetworkXNoPath:
                print(f"No path from {start_node} to {target_node}")
            except nx.NodeNotFound as e:
                print(e)

    accuracy = correct_count / total_count
    print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}")
    print(f"Number of correct paths: {correct_count}")
    print(f"Number of shortest paths: {shortest_path_count}")
    print(f"Number of non-shortest paths: {non_shortest_path_count}")
    print(f"Number of invalid paths: {invalid_path_count}")
    print(f"Number of incomplete paths: {incomplete_path_count}")

    # 保存当前 epoch 的准确率
    accuracy_metrics.append({
        "epoch": epoch + 1,
        "accuracy": accuracy,
        "correct_paths": correct_count,
        "shortest_paths": shortest_path_count,
        "non_shortest_paths": non_shortest_path_count,
        "invalid_paths": invalid_path_count,
        "incomplete_paths": incomplete_path_count
    })
    if shortest_path_count == 650:
        break


# 保存所有 epoch 的准确率指标
with open('accuracy_metrics.json', 'w') as f:
    json.dump(accuracy_metrics, f, indent=4)