import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training and data processing")
    
    # Mode Choice
    parser.add_argument('--mode', type=str, default="embedding", help="Training mode")
    parser.add_argument('--num_clusters', type=int, help="Number of cluster size")

    # RNA, ATAC, GTF and GRN file paths and mode
    parser.add_argument('--rna_h5ad', type=str, default="data/10x-Multiome-Pbmc10k-RNA.h5ad", help="Path to RNA h5ad file")
    parser.add_argument('--atac_h5ad', type=str, default="data/10x-Multiome-Pbmc10k-ATAC.h5ad", help="Path to ATAC h5ad file")
    parser.add_argument('--gtf_file', type=str, default="data/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz", help="Path to GTF file")
    parser.add_argument('--grn_file', type=str, help="Path to GRN file")
    parser.add_argument('--data_mode', type=str, default="match", help="Dataset match or unmatch")
    
    # Optional embeddings
    parser.add_argument('--seurat_rna', type=str, default=None, help="Path to Seurat RNA embedding")
    parser.add_argument('--seurat_atac', type=str, default=None, help="Path to Seurat ATAC embedding")
    parser.add_argument('--bindsc_rna', type=str, default=None, help="Path to bindSC RNA embedding")
    parser.add_argument('--bindsc_atac', type=str, default=None, help="Path to bindSC ATAC embedding")

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--lamb', type=float, default=0.5, help="Lambda parameter")
    parser.add_argument('--m', type=int, default=20, help="Number of random walk iterations")
    parser.add_argument('--k', type=int, default=5, help="Top-k nodes for sampling")
    parser.add_argument('--tau', type=float, default=0.1, help="Tau parameter")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of layers in the model")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--emb_dim', type=int, default=256, help="Embedding size")
    
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)