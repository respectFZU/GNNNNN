import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from datainput import edge_index


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# # 示例：
# num_nodes = 339  # 节点数量
# input_dim = 16  # 输入特征维度
# hidden_dim = 32  # 隐藏层维度
# output_dim = 2  # 输出特征维度
# x = torch.randn((num_nodes, input_dim))  # 这里用随机数代替实际坐标
#
# # 检查 edge_index 中的最大索引
# max_index = edge_index.max().item()
# if max_index >= num_nodes:
#     raise ValueError(f"edge_index contains invalid index {max_index} for num_nodes {num_nodes}")
#
# # 初始化模型
# model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
#
# # 前向传播
# output = model(x, edge_index)
#
# # 输出结果
# print(output)
