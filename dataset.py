import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import numpy as np
import random

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

def create_hetero_data(rna, atac, row_ind, col_ind):
    data = HeteroData()

    data["cellR"].x = torch.tensor(rna[:, rna.var["highly_variable"]].obsm["X_pca"])
    data["cellA"].x = torch.tensor(atac[:, atac.var["highly_variable"]].obsm["X_lsi"])
    data["gene"].x = torch.tensor(rna[:, rna.var["highly_variable"]].X.transpose())
    data["peak"].x = torch.tensor(atac[:, atac.var["highly_variable"]].X.toarray().transpose())

    data["cellR", "cc", "cellA"].edge_index = torch.tensor(np.array([row_ind, col_ind]))

    data["cellR", "cg", "gene"].edge_index = torch.tensor(np.array(np.where(rna[:, rna.var["highly_variable"]].X > 0)))

    data["cellA", "cp", "peak"].edge_index = torch.tensor(np.array(np.where((atac[:, atac.var["highly_variable"]].X > 0).toarray())))

    trans = ToUndirected()
    data = trans(data)

    return data


class SubgraphDataset(Dataset):
    def __init__(self, subgraphs, metadata):
        super(SubgraphDataset, self).__init__()
        self.ori_subgraphs = subgraphs
        self.metadata = metadata
        self.att_graph = [self.gen_att_graph(graph[2]) for graph in subgraphs]
    
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
    
    def __getitem__(self, index):
        type = self.ori_subgraphs[index][0]
        node = self.ori_subgraphs[index][1]
        ori_subgraph = self.ori_subgraphs[index][2]
        if ori_subgraph.node_items()[0][0] in ["cellR", "cellA"]:
            ori_feature = ori_subgraph.node_items()[0][1].x[0]
            mask = torch.ones(ori_subgraph["cellR"].x[0].size())
        else:
            ori_feature = torch.zeros(ori_subgraph["cellR"].x[0].size())
            mask = torch.zeros(ori_subgraph["cellR"].x[0].size())
        att_graph = self.att_graph[index]
        return type, node, ori_feature, mask, ori_subgraph, att_graph
    
    def __len__(self):
        return len(self.ori_subgraphs)
    
class GRNDataset(Dataset):
    def __init__(self, subgraphs, metadata, regulons, left, right):
        super(GRNDataset, self).__init__()
        self.ori_subgraphs = subgraphs
        self.metadata = metadata
        self.att_graph = [self.gen_att_graph(graph[2]) for graph in subgraphs]
        self.regulons = regulons
        self.left = left
        self.right = right
    
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
    
    def __getitem__(self, index):
        type = self.ori_subgraphs[index][0]
        node = self.ori_subgraphs[index][1]
        ori_subgraph = self.ori_subgraphs[index][2]
        if ori_subgraph.node_items()[0][0] in ["cellR", "cellA"]:
            ori_feature = ori_subgraph.node_items()[0][1].x[0]
            mask = torch.ones(ori_subgraph["cellR"].x[0].size())
        else:
            ori_feature = torch.zeros(ori_subgraph["cellR"].x[0].size())
            mask = torch.zeros(ori_subgraph["cellR"].x[0].size())
        att_graph = self.att_graph[index]
        if index % 2 == 0:
            geneA, geneB = self.regulons[(index // 2) % len(self.regulons)]
            label = torch.tensor([1])
        else:
            geneA, geneB = random.randint(self.left, self.right), random.randint(self.left, self.right)
            label = torch.tensor([0])
        
        geneA_ori = self.ori_subgraphs[geneA][2]
        geneA_att = self.att_graph[geneA]
        geneB_ori = self.ori_subgraphs[geneB][2]
        geneB_att = self.att_graph[geneB]
            
        return type, node, ori_feature, mask, ori_subgraph, att_graph, geneA_ori, geneA_att, geneB_ori, geneB_att, label
    
    def __len__(self):
        return len(self.ori_subgraphs)