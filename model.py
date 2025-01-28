import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv, Linear, norm
from torch_geometric.data import HeteroData, Batch
import torch_geometric.transforms as T

class HeteroGCN(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.metadata = data.metadata()
        
        self.trans = torch.nn.ModuleDict()
        for node_type in self.metadata[0]:
            self.trans[node_type] = Linear(data[node_type].x.size(-1), hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.atts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = RGCNConv(hidden_channels, hidden_channels, len(self.metadata[1]))
            self.convs.append(conv)

            att = RGATConv(hidden_channels, hidden_channels, len(self.metadata[1]) ** 2, attention_mechanism="within-relation")
            self.atts.append(att)
            
            nor = norm.BatchNorm(hidden_channels)
            self.norms.append(nor)
        
        self.fc = Linear(hidden_channels, out_channels)
    
    def gen_att_graph(self, subgraph):
        att_graph = HeteroData()
        for node_type in subgraph.node_types:
            att_graph[node_type].x = subgraph[node_type].x
        for node_type1 in self.metadata[0]:
            for node_type2 in self.metadata[0]:
                
                att_graph[node_type1, node_type1+node_type2, node_type2].edge_index = [[],[]]
                for node1 in range(len(att_graph[node_type1].x)):
                    for node2 in range(len(att_graph[node_type2].x)):
                        att_graph[node_type1, node_type1+node_type2, node_type2].edge_index[0].append(node1)
                        att_graph[node_type1, node_type1+node_type2, node_type2].edge_index[1].append(node2)
                        
                att_graph[node_type1, node_type1+node_type2, node_type2].edge_index = torch.tensor(att_graph[node_type1, node_type1+node_type2, node_type2].edge_index)
        
        return att_graph

    def forward(self, batch_size, heter_gcn, heter_att):
        for node_type in self.metadata[0]:
            heter_gcn[node_type].x = self.trans[node_type](heter_gcn[node_type].x)
            heter_att[node_type].x = self.trans[node_type](heter_att[node_type].x)
        homo_gcn, homo_att = heter_gcn.to_homogeneous(), heter_att.to_homogeneous()
        x, edge_index, edge_type = homo_gcn.x, homo_gcn.edge_index, homo_gcn.edge_type
        
        mask = torch.ones(homo_gcn.x.size()).view(batch_size, -1, homo_gcn.x.size()[-1]).to(x.device)
        mask[torch.rand(batch_size) < 0.5, 0, :] = 0
        mask = mask.view(homo_gcn.x.size())
        
        for conv, att, nor in zip(self.convs, self.atts, self.norms):
            x1 = conv(x * mask, edge_index, edge_type)
            x2 = att(x * mask, edge_index, edge_type)
            x = F.leaky_relu(nor(x1 + x2))
        x = self.fc(x + homo_gcn.x)
        x = x.view(batch_size, -1, x.size(-1))
        
        return x[:, 0, ], x[:, 1:, :]

import torch
import torch.nn.functional as F

class KMeansLoss(torch.nn.Module):
    def __init__(self, num_clusters, alpha=0.05):
        super(KMeansLoss, self).__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha

    def forward(self, embeddings):
        batch_size, embed_dim = embeddings.size()
        centers = torch.randn(self.num_clusters, embed_dim).to(embeddings.device)
        
        distances = torch.sqrt(((embeddings.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2))
        
        cluster_assignments = torch.argmin(distances, dim=1)
        
        cluster_loss = distances.gather(1, cluster_assignments.view(-1, 1)).squeeze()
        kmeans_loss = cluster_loss.mean()

        for i in range(self.num_clusters):
            cluster_points = embeddings[cluster_assignments == i]
            if cluster_points.size(0) > 0:
                centers[i] = cluster_points.mean(dim=0)
        
        return kmeans_loss * self.alpha
    
class ModelWithClustering(torch.nn.Module):
    def __init__(self, data, num_clusters, hidden_channels, out_channels, num_layers, alpha=0.05):
        super(ModelWithClustering, self).__init__()
        self.heterogcn = HeteroGCN(data, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
        self.kmeans_loss = KMeansLoss(num_clusters, alpha)

    def forward(self, batch_size, heter_gcn, heter_att):
        query, pos = self.heterogcn(batch_size, heter_gcn, heter_att)
        kmeans_loss = self.kmeans_loss(query)
        
        return query, pos, kmeans_loss

class GRNClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GRNClassifier, self).__init__()
        self.fc1 = Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.output = Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x