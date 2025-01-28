import anndata as ad
from preprocess import preprocess, pre_match, match, pre_grn
from dataset import create_hetero_data
from embedding import train_embedding
from cell_clustering import train_clustering, clustering
from grn import train_grn
from args import parse_args

def main():
    args = parse_args()
    
    rna, atac = preprocess(args)
    rna, atac = pre_match(args, rna, atac)
    
    row_ind, col_ind, rna, atac = match(args, rna, atac)
    data = create_hetero_data(rna, atac, row_ind, col_ind)

    if args.mode == "embedding":
        train_embedding(args, rna, atac, data)
    elif args.mode == "cluster":
        train_clustering(args, rna, atac, data)
        rna = ad.read_h5ad("data/rna-scmi.h5ad")
        atac = ad.read_h5ad("data/atac-scmi.h5ad")
        clustering(rna, atac, args.num_clusters)
    elif args.mode == "grn":
        regulons = pre_grn(args, rna, atac)
        train_grn(args, rna, atac, data, regulons)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()