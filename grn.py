import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from info_nce import InfoNCE
from itertools import chain

from model import HeteroGCN, GRNClassifier
from subgraph import generate_subgraphs
from dataset import GRNDataset
from utils import write_gene_emb

def train_grn(args, rna, atac, data, regulons):
    model = HeteroGCN(data, hidden_channels=args.emb_dim, out_channels=args.emb_dim, num_layers=args.num_layers)
    cls = GRNClassifier(embedding_dim=args.emb_dim, hidden_dim=args.emb_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cls.to(device)
    
    params = chain(model.parameters(), cls.parameters())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    
    criterion1 = nn.MSELoss(reduction="mean")
    criterion2 = InfoNCE(temperature=args.tau, negative_mode="paired")
    criterion3 = nn.CrossEntropyLoss()
    
    print("Model Training...")
    
    for epoch in tqdm(range(args.num_epochs)):
        if epoch % 30 == 0:
            print("Sample subgraphs...")
            subgraphs = generate_subgraphs(data, ["cellR", "cellA", "gene"])
            dataset = GRNDataset(subgraphs, data.metadata(), regulons, len(rna) + len(atac), len(subgraphs) - 1)
            loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
        
        training_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            type, index, ori_fea, mask, ori_graph, att_graph, geneA_ori, geneA_att, geneB_ori, geneB_att, labels = batch
            
            ori_graph, att_graph = ori_graph.to(device), att_graph.to(device)
            ori_fea, mask = ori_fea.to(device), mask.to(device)
            optimizer.zero_grad()
            
            query, pos = model(len(type), ori_graph, att_graph)
            
            loss1 = criterion1(query * mask, ori_fea)
            query = query.repeat(1, pos.size()[1]).contiguous().view(-1, query.size()[-1])
            
            neg = pos.unsqueeze(1).tile(1, pos.size()[1], 1, 1)
            neg = pos[torch.randperm(pos.size()[0])].contiguous().view(query.size()[0], -1, query.size()[-1])
            pos = pos.contiguous().view(-1, query.size()[-1])
            loss2 = criterion2(query, pos, neg) * 0.1
            
            geneA_ori, geneA_att, geneB_ori, geneB_att, labels = geneA_ori.to(device), geneA_att.to(device), geneB_ori.to(device), geneB_att.to(device), labels.to(device)
            geneA, _ = model(len(type), geneA_ori, geneA_att)
            geneB, _ = model(len(type), geneB_ori, geneB_att)
            logits = cls(geneA, geneB)
            loss3 = criterion3(logits, labels.float()) * 0.1
            
            loss = loss1 + args.lamb * loss2 + loss3
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
        
        print('[Epoch %d] loss: %.3f' % (epoch, training_loss / len(loader)))
    
    print("Writing embedding to h5ad files...")
    
    loader = DataLoader(dataset, shuffle=False,batch_size=128)
    
    nodes = []
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        type, index, _, _, ori_graph, att_graph, _, _, _, _, _ = batch
        
        with torch.no_grad():
            ori_graph, att_graph = ori_graph.to(device), att_graph.to(device)
            query, pos = model(len(type), ori_graph, att_graph)
            nodes.append(query)
    nodes = torch.cat(nodes).cpu().detach().numpy()
    write_gene_emb(rna, atac, nodes)
    torch.save(model.state_dict(), "data/scmi_grn.pt")
    print("Gene embeddings have been successfully saved to 'data/rna-scmi.h5ad', please find gene embeddings in varm['X_scmi'].")