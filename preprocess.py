import anndata as ad
import scanpy as sc
import itertools
import pandas as pd
import seaborn as sns
import scglue
from matplotlib import rcParams
from itertools import chain
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from utils import suppress_stdout_stderr
import logging

def preprocess(args):
    print("Preprocessing Data...")
    
    rna = ad.read_h5ad(args.rna_h5ad)
    atac = ad.read_h5ad(args.atac_h5ad)
    
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=args.emb_dim, svd_solver="auto")
    
    scglue.data.lsi(atac, n_components=args.emb_dim, n_iter=15)
    
    rna.write("data/rna-pp.h5ad", compression="gzip")
    atac.write("data/atac-pp.h5ad", compression="gzip")
    
    return rna, atac
    
def pre_match(args, rna, atac):

    scglue.data.get_gene_annotation(
        rna, gtf=args.gtf_file,
        gtf_by="gene_name"
    )
        
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

    rna.var.dropna(axis=1, inplace=True)
    atac.var.dropna(axis=1, inplace=True)
    
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="X_pca"
    )
    
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep="X_lsi"
    )
    
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()
    
    scglue.models.fit_SCGLUE.logger.disabled = True
    scglue.models.nn.autodevice.logger.disabled = True
    scglue.models.PairedSCGLUEModel.logger.disabled = True
    scglue.models.scglue.PairedSCGLUETrainer .logger.disabled = True
    scglue.data.estimate_balancing_weight.logger.disabled = True
    scglue.graph.check_graph.logger.disabled = True
    scglue.models.plugins.LRScheduler.logger.disabled = True
    scglue.models.plugins.EarlyStopping.logger.disabled = True
    logging.getLogger().setLevel(logging.ERROR)
    with suppress_stdout_stderr():
        glue = scglue.models.fit_SCGLUE(
            {"rna": rna, "atac": atac}, guidance_hvf,
            model=scglue.models.PairedSCGLUEModel,
            init_kws={"latent_dim": args.emb_dim}
        )
    
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    
    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
    
    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
    
    return rna, atac

def match(args, rna, atac):
    
    rna_emb = rna.obsm["X_glue"]
    atac_emb = atac.obsm["X_glue"]
    
    if args.seurat_rna and args.seurat_atac:
        rna_seurat = np.load(args.seurat_rna)
        atac_seurat = np.load(args.seurat_atac)
        rna_emb = np.concatenate([rna_emb, rna_seurat], -1)
        atac_emb = np.concatenate([atac_emb, atac_seurat], -1)
        
    if args.bindsc_rna and args.bindsc_atac:
        rna_bindsc = np.load(args.bindsc_rna)
        atac_bindsc = np.load(args.bindsc_atac)
        rna_emb = np.concatenate([rna_emb, rna_bindsc], -1)
        atac_emb = np.concatenate([atac_emb, atac_bindsc], -1)
    
    if args.data_mode == "unmatch":
        dist_matrix = cdist(rna_emb, atac_emb, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
    else:
        row_ind, col_ind = list(range(len(rna))), list(range(len(atac)))
    
    rna.obsm["X_match"] = rna_emb
    atac.obsm["X_match"] = atac_emb
    
    rna.write("data/rna-match.h5ad", compression="gzip")
    atac.write("data/atac-match.h5ad", compression="gzip")
    
    return row_ind, col_ind, rna, atac


def pre_grn(args, rna, atac):
    grns = pd.read_csv(args.grn_file)
    grns = grns[(grns["confidence"] == "A") | (grns["confidence"] == "B") | (grns["confidence"] == "C")]
    g2i = {}
    for i, gene in enumerate(rna[:, rna.var["highly_variable"]].var.index):
        g2i[gene] = i
    cell_cnt = len(rna) + len(atac)
    regulons = []
    for source, target in zip(grns["source"], grns["target"]):
        if source in rna[:, rna.var["highly_variable"]].var.index and target in rna[:, rna.var["highly_variable"]].var.index:
            regulons.append((cell_cnt + g2i[source], cell_cnt + g2i[target]))
    return regulons
