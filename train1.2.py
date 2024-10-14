import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import networkx as nx
from datainput import G, edge_index
import torch.nn.functional as F

# 增强节点特征
input_dim = 32  # 增加输入特征维度
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
        self.conv4 = SAGEConv(hidden_dim, hidden_dim)  # 增加一层
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, start_node, target_node):
        start_feature = torch.zeros((x.size(0), 1), device=x.device)
        target_feature = torch.zeros((x.size(0), 1), device=x.device)
        start_feature[start_node] = 1.0
        target_feature[target_node] = 1.0
        x = torch.cat([x, start_feature, target_feature], dim=1)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()  # 新增层
        x = self.fc(x)
        return x

def custom_loss(output, shortest_path, G):
    path_loss = F.cross_entropy(output[shortest_path[:-1]], torch.tensor(shortest_path[1:], dtype=torch.long))
    neighbor_loss = 0.0
    for i in range(len(shortest_path) - 1):
        current_node = shortest_path[i]
        predicted_node = output[current_node].argmax().item()
        if str(predicted_node) not in G.neighbors(str(current_node)):
            neighbor_loss += 1.0
    total_loss = path_loss + 0.5 * neighbor_loss  # 调整惩罚项权重
    return total_loss

model = PathPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 调整学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 确保 edge_index 的形状正确
print(edge_index.shape)

# 训练模型
num_epochs = 5  # 增加训练轮数
start_node = 1

for epoch in range(num_epochs):
    for target_node in range(1, G.number_of_nodes() + 1):
        if start_node == target_node:
            continue
        try:
            shortest_path = nx.dijkstra_path(G, source=str(start_node), target=str(target_node))
            shortest_path = [int(node) for node in shortest_path]

            path_data = Data(x=node_features, edge_index=edge_index)

            model.train()
            optimizer.zero_grad()

            output = model(path_data.x, path_data.edge_index, start_node, target_node)

            loss = custom_loss(output, shortest_path, G)

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Start {start_node}, Target {target_node}, Loss: {loss.item()}")

        except nx.NetworkXNoPath:
            print(f"No path from {start_node} to {target_node}")

    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'gcn_model.pth')

# 加载模型
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()

# 测试模型并计算准确率
correct_count = 0
total_count = 0
invalid_path_count = 0

for target_node in range(1, G.number_of_nodes() + 1):
    if start_node == target_node:
        continue

    try:
        correct_path = nx.dijkstra_path(G, source=str(start_node), target=str(target_node))
        correct_path = [int(node) for node in correct_path]

        with torch.no_grad():
            output = model(node_features, edge_index, start_node, target_node)
            predicted_path = [start_node]
            current_node = start_node
            valid_path = True

            while current_node != target_node:
                next_node = output[current_node].argmax().item()

                if not G.has_edge(str(current_node), str(next_node)):
                    print(f"Invalid connection from {current_node} to {next_node}")
                    valid_path = False
                    invalid_path_count += 1
                    break

                predicted_path.append(next_node)
                current_node = next_node

                if len(predicted_path) > len(correct_path):
                    valid_path = False
                    break

            if valid_path and predicted_path[-1] != target_node:
                predicted_path.append(target_node)

        if valid_path and predicted_path == correct_path:
            correct_count += 1

        total_count += 1

        print(f"Target node: {target_node}")
        print(f"Correct path: {correct_path}")
        print(f"Predicted path: {predicted_path}")

    except nx.NetworkXNoPath:
        print(f"No path from {start_node} to {target_node}")

accuracy = correct_count / total_count
print(f"Accuracy: {accuracy:.2f}")
print(f"Number of invalid paths: {invalid_path_count}")