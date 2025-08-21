"""
Improved SVD preprocessing script - Using RecBole for data processing, ensuring train/test data separation
"""
import torch
import numpy as np
import scipy.sparse as sp
import os
import argparse
import time
import gc
from recbole.data import create_dataset, data_preparation
from recbole.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='RecBole compatible SVD preprocessing script')
    parser.add_argument('--dataset', type=str, default='amazon-book', help='Dataset name')
    parser.add_argument('--output_path', type=str, default='./preprocessed', help='SVD result output path')
    parser.add_argument('--alpha', type=float, default=1.0, help='Normalization smoothing coefficient')
    parser.add_argument('--q', type=int, default=400, help='SVD decomposition dimension, should be slightly larger than req_vec')
    parser.add_argument('--niter', type=int, default=30, help='SVD iteration count')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU for SVD computation')
    parser.add_argument('--config_file_list', type=str, default=None, nargs='+', help='Additional configuration files')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    return parser.parse_args()

def normalize_matrix(rate_matrix, alpha):
    """Normalize interaction matrix"""
    print("Normalizing interaction matrix...")
    
    # Get user and item degrees
    D_u = np.array(rate_matrix.sum(axis=1)).flatten() + alpha
    D_i = np.array(rate_matrix.sum(axis=0)).flatten() + alpha
    
    # Compute normalization coefficients
    D_u_sqrt_inv = 1.0 / np.sqrt(D_u)
    D_i_sqrt_inv = 1.0 / np.sqrt(D_i)
    
    # Create diagonal matrices
    D_u_sqrt_inv_diag = sp.diags(D_u_sqrt_inv)
    D_i_sqrt_inv_diag = sp.diags(D_i_sqrt_inv)
    
    # Normalize matrix
    norm_rate_matrix = D_u_sqrt_inv_diag @ rate_matrix @ D_i_sqrt_inv_diag
    
    return norm_rate_matrix

def compute_svd(rate_matrix, q, niter, use_gpu=False):
    """Compute SVD decomposition"""
    print(f"Computing SVD decomposition (q={q}, niter={niter})...")
    
    if use_gpu and torch.cuda.is_available():
        print("Using GPU for SVD computation")
        device = torch.device('cuda')
        
        # For large matrices, converting to dense matrix may cause memory issues
        # Try to use sparse format directly for SVD
        try:
            rate_tensor = torch.FloatTensor(rate_matrix.toarray()).to(device)
            start = time.time()
            U, S, V = torch.svd_lowrank(rate_tensor, q=q, niter=niter)
            end = time.time()
            
            U = U.cpu().numpy()
            S = S.cpu().numpy()
            V = V.cpu().numpy()
        except (MemoryError, RuntimeError) as e:
            print(f"GPU memory insufficient, switching to CPU: {e}")
            use_gpu = False
    
    if not use_gpu or not torch.cuda.is_available():
        print("Using CPU for SVD computation")
        # Use scipy's SVD
        start = time.time()
        U, S, V = sp.linalg.svds(rate_matrix, k=q)
        end = time.time()
        
        # Ensure singular values are in descending order
        idx = np.argsort(S)[::-1]
        U = U[:, idx]
        S = S[idx]
        V = V[idx, :].T
    
    print(f'SVD computation time: {end-start:.4f} seconds')
    print(f'Singular value range: {S.min():.6f} ~ {S.max():.6f}')
    
    return U, S, V

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== SVD preprocessing for {args.dataset} dataset ===")
    
    # Build configuration dictionary
    config_dict = {
        'dataset': args.dataset,
        'seed': args.seed,
        'reproducibility': True,
        'load_col': {'inter': ['user_id', 'item_id']},
        'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'group_by': 'user'},
    }
    
    # Load dataset using RecBole
    print("Loading dataset using RecBole...")
    config = Config(model='BPR', dataset=args.dataset, config_dict=config_dict, config_file_list=args.config_file_list)
    dataset = create_dataset(config)
    
    # Print dataset information
    print(f"Dataset information: {dataset}")
    print(f"Number of users: {dataset.user_num}")
    print(f"Number of items: {dataset.item_num}")
    print(f"Number of interactions: {dataset.inter_num}")
    
    # Dataset splitting - Use only training set for SVD computation
    print("Splitting dataset...")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Get training set interaction matrix (Important modification: Use only training set)
    print("Getting training set interaction matrix...")
    train_inter_matrix = train_data.dataset.inter_matrix(form='csr').astype(np.float32)
    print(f"Training set interaction matrix shape: {train_inter_matrix.shape}")
    
    # Save original data dimension information
    dimensions = {
        "n_users": dataset.user_num,
        "n_items": dataset.item_num,
        "dataset_name": args.dataset,
        "train_interactions": train_data.dataset.inter_num,
        "valid_interactions": valid_data.dataset.inter_num,
        "test_interactions": test_data.dataset.inter_num,
        "timestamp": time.time()
    }
    np.save(os.path.join(output_dir, 'svd_dims.npy'), dimensions)
    
    # Save interaction matrix
    sp.save_npz(os.path.join(output_dir, 'train_inter_matrix.npz'), train_inter_matrix)
    
    # Normalize interaction matrix
    norm_rate_matrix = normalize_matrix(train_inter_matrix, args.alpha)
    
    # Free memory
    del train_inter_matrix
    gc.collect()
    
    # Compute SVD
    U, S, V = compute_svd(norm_rate_matrix, args.q, args.niter, args.gpu)
    
    # Free memory
    del norm_rate_matrix
    gc.collect()
    
    # Validate SVD dimensions
    if U.shape[0] != dataset.user_num or V.shape[0] != dataset.item_num:
        print(f"Warning: SVD dimensions do not match dataset!")
        print(f"SVD user count: {U.shape[0]}, dataset user count: {dataset.user_num}")
        print(f"SVD item count: {V.shape[0]}, dataset item count: {dataset.item_num}")
    
    # Save SVD results
    np.save(os.path.join(output_dir, 'svd_u.npy'), U)
    np.save(os.path.join(output_dir, 'svd_v.npy'), V)
    np.save(os.path.join(output_dir, 'svd_value.npy'), S)
    
    # Create README file
    with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
        f.write(f"SVD Preprocessing Information\n")
        f.write(f"============================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of users: {dataset.user_num}\n")
        f.write(f"Number of items: {dataset.item_num}\n")
        f.write(f"Total interactions: {dataset.inter_num}\n")
        f.write(f"Training set interactions: {train_data.dataset.inter_num}\n")
        f.write(f"Validation set interactions: {valid_data.dataset.inter_num}\n")
        f.write(f"Test set interactions: {test_data.dataset.inter_num}\n")
        f.write(f"SVD dimension: {args.q}\n")
        f.write(f"SVD user matrix shape: {U.shape}\n")
        f.write(f"SVD item matrix shape: {V.shape}\n")
        f.write(f"Preprocessing method: SVD computed based on training set only, avoiding information leakage\n")
        f.write(f"Generation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"SVD preprocessing completed. Results saved to {output_dir}")
    print(f"SVD dimensions - U: {U.shape}, S: {S.shape}, V: {V.shape}")
    print("Important: This SVD decomposition is based on training set only, avoiding test set information leakage")
    print("Please ensure using the same RecBole configuration for model training to maintain consistent user/item IDs")

if __name__ == '__main__':
    main()