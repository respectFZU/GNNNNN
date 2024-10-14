import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from datainput import G, edge_index
from GCN import GCN
import random

# 假设我们有节点特征
input_dim = 16  # 输入特征维度
hidden_dim = 128  # 增加隐藏层维度
output_dim = G.number_of_nodes()  # 输出特征维度，表示节点数
node_features = torch.randn((G.number_of_nodes(), input_dim))  # 随机特征

# 初始化模型
class PathPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PathPredictor, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)  # 增加一层
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = self.gcn3(x, edge_index).relu()  # 增加一层
        x = self.fc(x)
        return x

model = PathPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 确保 edge_index 的形状正确
print(edge_index.shape)  # 应该输出 torch.Size([2, num_edges])

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    for start_node in range(1, G.number_of_nodes()):
        for target_node in range(1, G.number_of_nodes()):
            if start_node == target_node:
                continue
            try:
                shortest_path = nx.dijkstra_path(G, source=str(start_node), target=str(target_node))
                shortest_path = [int(node) for node in shortest_path]

                # 创建路径图
                path_data = Data(x=node_features, edge_index=edge_index)

                # 训练模型
                model.train()
                optimizer.zero_grad()

                # 前向传播
                output = model(path_data.x, path_data.edge_index)

                # 创建目标张量
                target = torch.tensor(shortest_path[1:], dtype=torch.long)  # 目标是路径上的节点序列

                # 计算损失
                loss = criterion(output[shortest_path[:-1]], target)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch+1}, Start {start_node}, Target {target_node}, Loss: {loss.item()}")

            except nx.NetworkXNoPath:
                print(f"No path from {start_node} to {target_node}")

# 训练完成后保存模型
torch.save(model.state_dict(), 'gcn_model.pth')

# 加载模型
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()

# 随机选择一个起点和终点
random_start_node = random.randint(1, G.number_of_nodes() - 1)
random_target_node = random.randint(1, G.number_of_nodes() - 1)

# 确保起点和终点不同
while random_start_node == random_target_node:
    random_target_node = random.randint(1, G.number_of_nodes() - 1)

# 计算正确路径
try:
    correct_path = nx.dijkstra_path(G, source=str(random_start_node), target=str(random_target_node))
    correct_path = [int(node) for node in correct_path]

    # 前向传播以获得预测
    with torch.no_grad():
        output = model(node_features, edge_index)
        predicted_path = [random_start_node]
        current_node = random_start_node

        while current_node != random_target_node:
            next_node = output[current_node].argmax().item()
            if next_node in predicted_path:  # 防止循环
                break
            predicted_path.append(next_node)
            current_node = next_node
            if len(predicted_path) > len(correct_path):  # 防止死循环
                break

    # 打印结果
    print(f"Random start node: {random_start_node}")
    print(f"Random target node: {random_target_node}")
    print(f"Correct path: {correct_path}")
    print(f"Predicted path: {predicted_path}")

except nx.NetworkXNoPath:
    print(f"No path from {random_start_node} to {random_target_node}")