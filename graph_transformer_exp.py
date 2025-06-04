import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

K_HOP = 2
# Node type mapping used in IMDB dataset (Movie, Actor)
type2idx = {"M": 0, "A": 1}


def load_imdb(path="ds/imdb/MA.txt"):
    """Load a simple IMDB bipartite graph."""
    g = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            src, dst = line.strip().split()
            g.add_edge(src, dst)
    return g


def type_encoder(node: str):
    res = [0.0] * len(type2idx)
    res[type2idx[node[0]]] = 1.0
    return res


def dist_encoder(src: str, dest: str, g: nx.Graph, k_hop: int, one_hot: bool = True):
    paths = list(nx.all_simple_paths(g, src, dest, cutoff=k_hop + 2))
    cnt = [k_hop + 1] * len(type2idx)
    for path in paths:
        res = [0] * len(type2idx)
        for i in range(1, len(path)):
            res[type2idx[path[i][0]]] += 1
        for k in range(len(type2idx)):
            cnt[k] = min(cnt[k], res[k])
    if one_hot:
        one_hot_list = [np.eye(k_hop + 2, dtype=np.float32)[cnt[i]] for i in range(len(type2idx))]
        return np.concatenate(one_hot_list)
    return np.array(cnt, dtype=np.float32)


def build_pyg_data(g: nx.Graph):
    node_map = {n: i for i, n in enumerate(g.nodes())}
    edge_index = []
    edge_attr = []
    for u, v in g.edges():
        edge_index.append([node_map[u], node_map[v]])
        edge_attr.append(dist_encoder(u, v, g, K_HOP))
        edge_index.append([node_map[v], node_map[u]])
        edge_attr.append(dist_encoder(v, u, g, K_HOP))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_features = torch.tensor([type_encoder(n) for n in g.nodes()], dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class GTModel(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.conv = TransformerConv(in_dim, hidden_dim, heads=2, edge_dim=edge_dim)
        self.lin = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, data: Data):
        h = self.conv(data.x, data.edge_index, data.edge_attr)
        h = torch.relu(h)
        return h


def main():
    g = load_imdb()
    data = build_pyg_data(g)
    model = GTModel(data.num_features, 32, data.edge_attr.size(-1))
    with torch.no_grad():
        out = model(data)
    print("Node embedding shape:", out.shape)


if __name__ == "__main__":
    main()
