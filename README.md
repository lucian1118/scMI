# scMI: Integrating scRNA-seq and scATAC-seq with inter-type attention heterogeneous graph neural networks

## Environment Setup
To set up the environment for running the code, ensure you have Python installed, and use the provided `requirements.txt` file to install the necessary dependencies:
```bash
# conda create -n scmi
# conda activate scmi

pip install -r requirements.txt
```
---

## Modes and Input Data and Modes

- `--mode`: The training mode. Options are:
  - `embedding`: To generate embeddings for RNA and ATAC cells.
  - `cluster`: To perform clustering on RNA and ATAC cells.
  - `grn`: To test gene regulatory networks.

- `--num_clusters`: The number of clusters. This parameter is required **only when** `--mode` is set to `cluster`.

The following input files are required for running the models:

### Data File Paths
- `--rna_h5ad`: Path to the RNA h5ad file.
- `--atac_h5ad`: Path to the ATAC h5ad file.
- `--gtf_file`: Path to the gene annotation file in GTF format.
- `--grn_file`: Path to the gene regulatory network (GRN) file, required **only when** `--mode` is set to `grn`.

### Dataset Mode
- `--data_mode`: Specify the dataset mode. Options are:
  - `match`: The RNA and ATAC data are matched (from the same cells).
  - `unmatch`: The RNA and ATAC data are unmatched.

---

## Command-Line Examples for Each Mode

### Embedding Mode (matched data)
```bash
python main.py –rna_h5ad <path_to_rna_h5ad> –atac_h5ad <path_to_atac_h5ad> –gtf_file <path_to_gtf_file> –mode embedding –data_mode match
```
### Clustering Mode (unmatched data)
```bash
python main.py –rna_h5ad <path_to_rna_h5ad> –atac_h5ad <path_to_atac_h5ad> –gtf_file <path_to_gtf_file> –mode cluster –data_mode unmatch –num_clusters 19
```
### GRN Mode (matched data)
```bash
python main.py –rna_h5ad <path_to_rna_h5ad> –atac_h5ad <path_to_atac_h5ad> –gtf_file <path_to_gtf_file> –grn_file <path_to_grn_file> –mode grn –data_mode match
```
Replace `<path_to_*>` with the paths to your respective data files.

---

## Other Parameters
- `--seurat_rna`: Path to the Seurat-generated RNA embedding file (optional for unmatched data).
- `--seurat_atac`: Path to the Seurat-generated ATAC embedding file (optional for unmatched data).
- `--bindsc_rna`: Path to the bindSC-generated RNA embedding file (optional for unmatched data).
- `--bindsc_atac`: Path to the bindSC-generated ATAC embedding file (optional for unmatched data).

### Training Hyperparameters
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate.
- `--lamb`: Lambda parameter.
- `--m`: Number of random walk iterations.
- `--k`: Number of top-k nodes for sampling.
- `--tau`: Tau parameter of InfoNCE loss.
- `--num_layers`: Number of layers in the model.
- `--epochs`: Total number of training epochs.
- `--emb_dim`: Size of the embedding dimension.

---

This repository provides the implementation of **scMI**, a method for integrating scRNA-seq and scATAC-seq data using inter-type attention heterogeneous graph neural networks. For more details, please refer to the paper or reach out with any questions!
