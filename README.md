# SPARK_Rec

A Knowledge Graph Enhanced Hybrid Geometric Recommendation System that integrates SVD decomposition, hyperbolic geometry, and knowledge graph embeddings for improved recommendation performance.

## Overview

SPARK_Rec is a hybrid recommendation system that combines SVD decomposition, hyperbolic geometry, and knowledge graph embeddings for improved recommendation accuracy.

## Key Features

- **Hybrid Geometric Spaces**: Euclidean and hyperbolic space fusion
- **Knowledge Graph Integration**: Entity and relation embeddings
- **Adaptive Fusion**: Popularity-aware mixing strategies
- **Graph Neural Networks**: Message passing on interaction and KG graphs
- **Contrastive Learning**: SVD and KG embedding alignment

## Requirements

```bash
Python >= 3.7
PyTorch >= 1.8.0
RecBole >= 1.1.1
numpy >= 1.19.0
scipy >= 1.6.0
pandas >= 1.2.0
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SPARK_Rec
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install RecBole** (if not already installed)
```bash
pip install recbole
```

## Data Preparation

### 1. Dataset Structure
Organize your dataset in the following format:
```
dataset/
├── [dataset_name]/
│   ├── [dataset_name].inter     # User-item interactions
│   ├── [dataset_name].kg        # Knowledge graph triples  
│   └── [dataset_name].link      # Item-entity alignments
```

### 2. SVD Preprocessing
Run SVD decomposition on the interaction matrix:
```bash
python preprocess_svd.py --dataset amazon-book --output_path ./preprocessed
```

### 3. Knowledge Graph Embeddings (Optional)
If you have pretrained KG embeddings:
```
kg_embeddings/
├── entity_embeddings.npy
└── relation_embeddings.npy
```

## Usage

### Basic Training
```bash
python run_kg_model.py \
    --dataset yelp2018 \
    --embedding_size 64 \
    --kg_embedding_size 64 \
    --epochs 200
```

### Advanced Configuration
```bash
python run_kg_model.py \
    --dataset amazon-book \
    --embedding_size 64 \
    --kg_embedding_size 64 \
    --hyperbolic_ratio 0.3 \
    --curvature 0.8 \
    --tucker_weight 0.5 \
    --use_tucker \
    --cl_weight 0.1 \
    --epochs 200 \
    --gpu_id 0
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--embedding_size` | Dimension of recommendation embeddings | 64 |
| `--kg_embedding_size` | Dimension of KG embeddings | 64 |
| `--hyperbolic_ratio` | Weight for hyperbolic space | 0.3 |
| `--curvature` | Curvature of hyperbolic space | 0.8 |
| `--tucker_weight` | Weight for TuckER scoring | 0.5 |
| `--use_tucker` | Enable TuckER framework | False |
| `--cl_weight` | Contrastive learning weight | 0.1 |
| `--kg_weight` | Knowledge graph loss weight | 0.01 |

## File Structure

```
SPARK_Rec/
├── KG_model_v2.py              # Main model implementation
├── run_kg_model.py             # Training and evaluation script
├── preprocess_svd.py           # SVD preprocessing utility
├── requirements.txt            # Python dependencies
├── dataset/                    # Dataset directory
├── preprocessed/               # SVD decomposition results
├── logs/                       # Training logs
└── README.md                   # This file
```

## Model Architecture

The model consists of several key components:

1. **SVD Embeddings**: User and item latent factors from matrix factorization
2. **Lorentz Manifold**: Hyperbolic space operations for hierarchical modeling
3. **Knowledge Graph Networks**: Entity and relation embedding learning
4. **Hybrid GNN**: Message passing in both Euclidean and hyperbolic spaces
5. **TuckER Scoring**: Tensor decomposition for KG completion
6. **Adaptive Fusion**: Popularity-aware mixing of different signals

## Evaluation

The model supports standard recommendation metrics:
- Recall@K
- NDCG@K  
- MRR
- Hit@K
- Precision@K

Results are automatically logged during training and saved to the `logs/` directory.

## Datasets

The model has been tested on the following datasets:
- Amazon-Book
- Alibaba-iFashion
- Yelp2018

Ensure your dataset follows the RecBole format specifications.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--embedding_size`
2. **SVD files not found**: Run the preprocessing script first
3. **Knowledge graph not loaded**: Check file formats and paths

### Debug Mode
```bash
python run_kg_model.py --batch_debug --dataset yelp2018
```

## License

This project is available for academic and research purposes.
