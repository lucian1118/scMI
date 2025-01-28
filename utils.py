import os
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

def write_to_h5ad(rna, atac, nodes):
    rna_emb = ad.read_h5ad("data/rna-match.h5ad").obsm["X_match"]
    atac_emb = ad.read_h5ad("data/atac-match.h5ad").obsm["X_match"]
    rna = ad.read_h5ad("data/rna-pp.h5ad")
    atac = ad.read_h5ad("data/atac-pp.h5ad")
    
    cellR_cnt = len(rna)
    cellA_cnt = len(atac)
    rna_emb += nodes[:cellR_cnt]
    atac_emb += nodes[cellR_cnt:cellR_cnt + cellA_cnt]
    
    rna.obsm["X_scmi"] = rna_emb
    atac.obsm["X_scmi"] = atac_emb
    
    rna.write("data/rna-scmi.h5ad", compression="gzip")
    atac.write("data/atac-scmi.h5ad", compression="gzip")


def save_cluster(rna, atac):
    combined = ad.concat([rna, atac])
    sc.pp.neighbors(combined, use_rep="X_scmi", metric="cosine")
    sc.tl.umap(combined)
    if "cell_type" in combined.obs.columns:
        sc.pl.umap(combined, color=["cell_type", "cluster", "domain"], wspace=0.65, show=False)
    else:
        sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65, show=False)
    plt.savefig("data/cluster.png")
    rna.write("data/rna-scmi.h5ad", compression="gzip")
    atac.write("data/atac-scmi.h5ad", compression="gzip")
    
def write_gene_emb(rna, atac, nodes):
    cellR_cnt = len(rna)
    cellA_cnt = len(atac)
    rna = rna[:, rna.var["highly_variable"]]
    rna.varm["X_scmi"] = nodes[cellR_cnt + cellA_cnt:]
    rna.write("data/rna-scmi.h5ad", compression="gzip")
    
class suppress_stdout_stderr(object):
    
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])