import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from info_nce import InfoNCE
from model import ModelWithClustering
from subgraph import generate_subgraphs
from dataset import SubgraphDataset
from sklearn.cluster import KMeans

from utils import write_to_h5ad, save_cluster

def train_clustering(args, rna, atac, data):
    model = ModelWithClustering(data, args.num_clusters, hidden_channels=args.emb_dim, out_channels=args.emb_dim, num_layers=args.num_layers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    
    criterion1 = nn.MSELoss(reduction="mean")
    criterion2 = InfoNCE(temperature=args.tau, negative_mode="paired")
    
    print("Model Training...")
    
    for epoch in tqdm(range(args.num_epochs)):
        if epoch % 30 == 0:
            print("Sample subgraphs...")
            subgraphs = generate_subgraphs(data, ["cellR", "cellA"])
            dataset = SubgraphDataset(subgraphs, data.metadata())
            loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
        
        training_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            with torch.autograd.set_detect_anomaly(True):
                type, index, ori_fea, mask, ori_graph, att_graph = batch
                
                ori_graph, att_graph = ori_graph.to(device), att_graph.to(device)
                ori_fea, mask = ori_fea.to(device), mask.to(device)
                optimizer.zero_grad()
                
                query, pos, loss3 = model(len(type), ori_graph, att_graph)
                
                loss1 = criterion1(query * mask, ori_fea)
                query = query.repeat(1, pos.size()[1]).contiguous().view(-1, query.size()[-1])
                
                neg = pos.unsqueeze(1).tile(1, pos.size()[1], 1, 1)
                neg = pos[torch.randperm(pos.size()[0])].contiguous().view(query.size()[0], -1, query.size()[-1])
                pos = pos.contiguous().view(-1, query.size()[-1])
                
                loss2 = criterion2(query, pos, neg) * 0.1
                loss = loss1 + args.lamb * loss2 + loss3
                loss.backward()
                optimizer.step()
                
                training_loss += loss.item()
        
        print('[Epoch %d] loss: %.3f' % (epoch, training_loss / len(loader)))
    
    print("Writing embedding to h5ad files...")
    
    loader = DataLoader(dataset, shuffle=False,batch_size=128)
    
    nodes = []
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        type, index, _, _, ori_graph, att_graph = batch
        
        with torch.no_grad():
            ori_graph, att_graph = ori_graph.to(device), att_graph.to(device)
            query, pos, _ = model(len(type), ori_graph, att_graph)
            nodes.append(query)
    nodes = torch.cat(nodes).cpu().detach().numpy()
    cellR_cnt = len(rna)
    cellA_cnt = len(atac)
    rna.obsm["X_scmi"] = nodes[:cellR_cnt]
    atac.obsm["X_scmi"] = nodes[cellR_cnt:cellR_cnt + cellA_cnt]
    write_to_h5ad(rna, atac, nodes)
    torch.save(model.state_dict(), "data/scmi_clustering.pt")


def clustering(rna, atac, num_clusters):
    rna_emb = rna.obsm["X_scmi"]
    atac_emb = atac.obsm["X_scmi"]
    embedding = np.concatenate([rna_emb, atac_emb])
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(embedding)
    rna.obs["cluster"] = clusters[:len(rna)]
    atac.obs["cluster"] = clusters[len(rna):]
    save_cluster(rna, atac)
    print("Clusters have been successfully saved to 'data/*-scmi.h5ad', please find clusters in obs['cluster'] and visualization in 'data/cluster.png'.")
    