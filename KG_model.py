import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
import logging
from tqdm import tqdm

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.utils.enum_type import ModelType


class LorentzManifold(nn.Module):
    """Stable Lorentz space operation utility class"""
    
    def __init__(self, c=1.0, eps=1e-7, clip_value=100.0):
        """
        Initialize Lorentz space processing class
        
        Args:
            c: Curvature parameter (c=1/k, where k is Gaussian curvature)
            eps: Numerical stability parameter
            clip_value: Gradient clipping threshold
        """
        super(LorentzManifold, self).__init__()
        self.c = c  # Curvature
        self.eps = eps  # Numerical stability parameter
        self.clip_value = clip_value  # Clipping value
        
        # Time axis index
        self.time_idx = 0
    
    def minkowski_dot(self, x, y, keepdim=True):
        """
        Compute Minkowski inner product (inner product in hyperbolic space)
        
        Args:
            x, y: Batch point coordinates (..., dim)
            keepdim: Whether to keep output dimensions
        """
        # Minkowski inner product: -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
        res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        if keepdim:
            res = res.unsqueeze(-1)
        return res
    
    def minkowski_norm(self, x, keepdim=True):
        """Compute Minkowski norm (= sqrt(|<x,x>_M|))"""
        dot = self.minkowski_dot(x, x, keepdim=keepdim)
        return torch.sqrt(torch.abs(dot) + self.eps)
    
    def acosh(self, x):
        """Numerically stable inverse hyperbolic cosine"""
        # Ensure input is at least 1+eps
        x = torch.clamp(x, min=1.0 + self.eps)
        return torch.log(x + torch.sqrt(x.pow(2) - 1.0))
    
    def lorentz_to_poincare(self, x):
        """Convert from Lorentz model to Poincaré ball model"""
        # x_poincare = x_{1:} / (x_0 + 1)
        d = x.size(-1) - 1
        return x[..., 1:] / (x[..., 0:1] + 1.0)
    
    def poincare_to_lorentz(self, x):
        """Convert from Poincaré ball model to Lorentz model"""
        # Compute Euclidean norm
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        # Compute Lorentz coordinates
        t = 1.0 + x_norm_sq
        denom = 1.0 - x_norm_sq + self.eps
        # Concatenate time and space coordinates
        return torch.cat([t / denom, 2.0 * x / denom], dim=-1)
    
    def euclidean_to_lorentz(self, x, scale=0.5):
        """Convert Euclidean space vector to Lorentz model
        
        Args:
            x: Euclidean space vector
            scale: Scaling factor to control vector distribution range in hyperbolic space
        """
        # First scale the Euclidean vector
        norm = torch.norm(x, dim=-1, keepdim=True) + self.eps
        x_scaled = x * scale / norm
        
        # Ensure scaled vector norm is within safe range
        x_norm_sq = torch.sum(x_scaled * x_scaled, dim=-1, keepdim=True)
        
        # Compute Lorentz time component
        t = torch.sqrt(1.0 + x_norm_sq / self.c)
        
        # Concatenate time and space components
        return torch.cat([t, x_scaled], dim=-1)
    
    def project_to_lorentz(self, x):
        """Project to Lorentz manifold
        
        Project points to Lorentz manifold (hyperboloid) to satisfy -x_0^2 + x_1^2 + ... + x_n^2 = -1/c
        """
        # Get space components
        space_comp = x[..., 1:]
        
        # Compute sum of squares of space components
        space_norm_sq = torch.sum(space_comp * space_comp, dim=-1, keepdim=True)
        
        # Compute time component
        coeff = 1.0 / self.c
        time_comp = torch.sqrt(space_norm_sq + coeff)
        
        # Concatenate and return
        return torch.cat([time_comp, space_comp], dim=-1)
    
    def lorentz_distance(self, x, y):
        """Compute distance in Lorentz space"""
        # Ensure points are on manifold
        x = self.project_to_lorentz(x)
        y = self.project_to_lorentz(y)
        
        # Compute Minkowski inner product, smaller inner product means larger distance
        inner_prod = -self.minkowski_dot(x, y, keepdim=False)
        
        # Ensure inner product is at least 1
        inner_prod = torch.clamp(inner_prod, min=1.0 + self.eps)
        
        # Apply acosh and scale
        distance = self.acosh(inner_prod) / torch.sqrt(torch.tensor(self.c))
        
        # Clip maximum distance
        return torch.clamp(distance, max=self.clip_value)
    
    def exp_map_zero(self, x):
        """Exponential map at origin (from tangent space to manifold)"""
        # Extract norm of tangent space vector
        norm = torch.norm(x, dim=-1, keepdim=True) + self.eps
        
        # Compute coefficient
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        k = torch.cosh(sqrt_c * norm)
        res_t = k
        
        # Compute space components
        res_x = (torch.sinh(sqrt_c * norm) / (sqrt_c * norm)) * x
        
        # Concatenate result
        return torch.cat([res_t, res_x], dim=-1)
    
    def log_map_zero(self, x):
        """Logarithmic map at origin (from manifold to tangent space)"""
        # Ensure point is on manifold
        x = self.project_to_lorentz(x)
        
        # Time and space components
        t = x[..., 0:1]
        v = x[..., 1:]
        
        # Compute norm
        norm_v = torch.norm(v, dim=-1, keepdim=True) + self.eps
        
        # Compute scaling factor
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        factor = self.acosh(t) / (sqrt_c * norm_v)
        
        # Return tangent space vector
        return factor * v
    
    def mobius_addition(self, x, y):
        """Möbius addition operation (in Poincaré ball model)
        
        Note: Convert to Poincaré model, perform addition, then convert back to Lorentz model
        """
        # Convert to Poincaré model
        x_p = self.lorentz_to_poincare(x)
        y_p = self.lorentz_to_poincare(y)
        
        # Compute inner products and norms
        x_norm_sq = torch.sum(x_p * x_p, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y_p * y_p, dim=-1, keepdim=True)
        xy_inner = torch.sum(x_p * y_p, dim=-1, keepdim=True)
        
        # Numerator and denominator
        num = (1 + 2*xy_inner + y_norm_sq) * x_p + (1 - x_norm_sq) * y_p
        denom = 1 + 2*xy_inner + x_norm_sq * y_norm_sq
        
        # Möbius addition
        res_p = num / (denom + self.eps)
        
        # Convert back to Lorentz model
        return self.poincare_to_lorentz(res_p)
    
    def mobius_scalar_mul(self, r, x):
        """Möbius scalar multiplication (in Poincaré ball model)"""
        # Convert to Poincaré model
        x_p = self.lorentz_to_poincare(x)
        
        # Compute norm
        x_norm = torch.norm(x_p, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.eps, max=1.0-self.eps)
        
        # Compute scaling factor
        factor = torch.tanh(r * torch.atanh(x_norm)) / x_norm
        
        # Perform scaling
        res_p = factor * x_p
        
        # Convert back to Lorentz model
        return self.poincare_to_lorentz(res_p)


class HybridGNNLayer(nn.Module):
    """Hybrid Euclidean-Hyperbolic Graph Neural Network Layer"""
    
    def __init__(self, manifold, alpha=0.5, skip_connect=True):
        """
        Initialize hybrid GNN layer
        
        Args:
            manifold: Hyperbolic space operation object
            alpha: Mixing coefficient (0=pure Euclidean, 1=pure hyperbolic)
            skip_connect: Whether to use skip connections
        """
        super(HybridGNNLayer, self).__init__()
        self.manifold = manifold
        self.alpha = alpha
        self.skip_connect = skip_connect
    
    def forward(self, euclidean_x, lorentz_x, adj):
        """
        Hybrid GNN forward propagation
        
        Args:
            euclidean_x: Euclidean space representation
            lorentz_x: Lorentz space representation
            adj: Normalized adjacency matrix (sparse)
            
        Returns:
            Updated Euclidean and Lorentz representations
        """
        # Euclidean space message passing (standard GCN)
        euclidean_out = torch.sparse.mm(adj, euclidean_x)
        
        # Hyperbolic space message passing
        # 1. Map to tangent space
        tangent_x = self.manifold.log_map_zero(lorentz_x)
        
        # 2. Message passing in tangent space
        tangent_out = torch.sparse.mm(adj, tangent_x)
        
        # 3. Map back to hyperbolic space
        lorentz_out = self.manifold.exp_map_zero(tangent_out)
        
        # Skip connections
        if self.skip_connect:
            # Euclidean residual connection
            euclidean_out = 0.5 * euclidean_out + 0.5 * euclidean_x
            
            # Hyperbolic residual connection (using Möbius addition)
            l_skip = self.manifold.mobius_scalar_mul(0.5, lorentz_x)
            l_out = self.manifold.mobius_scalar_mul(0.5, lorentz_out)
            lorentz_out = self.manifold.mobius_addition(l_out, l_skip)
        
        return euclidean_out, lorentz_out

class TuckerScoring(nn.Module):
    """Tucker decomposition based knowledge graph scoring component"""
    
    def __init__(self, entity_dim, relation_dim, core_size=64, dropout=0.2):
        """
        Initialize TuckER scoring component
        
        Args:
            entity_dim: Entity embedding dimension
            relation_dim: Relation embedding dimension
            core_size: Core tensor dimension (can be smaller than embedding dimension to reduce parameters)
            dropout: Dropout rate
        """
        super(TuckerScoring, self).__init__()
        
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.core_size = core_size
        
        # Core tensor - simplified version to reduce parameters
        self.W = nn.Parameter(torch.FloatTensor(relation_dim, core_size, core_size))
        nn.init.xavier_uniform_(self.W)
        
        # Projection matrices (if embedding dimension differs from core tensor size)
        self.W_e = nn.Parameter(torch.FloatTensor(entity_dim, core_size))
        self.W_r = nn.Parameter(torch.FloatTensor(relation_dim, core_size))
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_r)
        
        # Batch normalization and dropout layers
        self.bn_h = nn.BatchNorm1d(core_size)
        self.bn_r = nn.BatchNorm1d(core_size)
        self.bn_t = nn.BatchNorm1d(core_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, r, t=None, all_entities=None, batch_process=False):
        """
        Compute TuckER scores
        
        Args:
            h: Head entity embeddings [batch_size, entity_dim]
            r: Relation embeddings [batch_size, relation_dim]
            t: Tail entity embeddings [batch_size, entity_dim], for computing specific triplet scores
            all_entities: All entity embeddings [n_entities, entity_dim], for full ranking computation
            batch_process: Whether to use batch processing mode (for large-scale evaluation)
        
        Returns:
            Scores: If t is not None, returns [batch_size]; otherwise returns [batch_size, n_entities]
        """
        batch_size = h.size(0)
        
        # Project to core tensor space
        h_proj = torch.matmul(h, self.W_e)  # [batch_size, core_size]
        r_proj = torch.matmul(r, self.W_r)  # [batch_size, core_size]
        
        # Apply batch normalization and dropout
        if h_proj.size(0) > 1:  # Ensure batch size > 1 for BatchNorm
            h_proj = self.bn_h(h_proj)
            r_proj = self.bn_r(r_proj)
        
        h_proj = self.dropout(h_proj)
        r_proj = self.dropout(r_proj)
        
        # Project relation to core tensor plane
        W_r = torch.matmul(r_proj, self.W.view(self.relation_dim, -1))
        W_r = W_r.view(-1, self.core_size, self.core_size)
        
        # Head entity projection multiplied by core tensor
        h_proj = h_proj.unsqueeze(1)  # [batch_size, 1, core_size]
        h_W_r = torch.bmm(h_proj, W_r)  # [batch_size, 1, core_size]
        h_W_r = h_W_r.squeeze(1)  # [batch_size, core_size]
        
        if t is not None:
            # Compute specific triplet scores
            t_proj = torch.matmul(t, self.W_e)  # [batch_size, core_size]
            if t_proj.size(0) > 1:  # Ensure batch size > 1
                t_proj = self.bn_t(t_proj)
            t_proj = self.dropout(t_proj)
            
            # Compute scores
            scores = torch.sum(h_W_r * t_proj, dim=1)  # [batch_size]
            return scores
        
        elif all_entities is not None:
            # Full ranking mode - compute scores with all entities
            e_proj = torch.matmul(all_entities, self.W_e)  # [n_entities, core_size]
            if e_proj.size(0) > 1:  # Ensure batch size > 1
                e_proj = self.bn_t(e_proj)
            e_proj = self.dropout(e_proj)
            
            if batch_process and batch_size > 100:
                # Batch computation to save memory
                sub_batch_size = 100
                n_sub_batches = (batch_size - 1) // sub_batch_size + 1
                all_scores = []
                
                for i in range(n_sub_batches):
                    start = i * sub_batch_size
                    end = min((i + 1) * sub_batch_size, batch_size)
                    sub_h_W_r = h_W_r[start:end]
                    
                    # Batch compute scores
                    sub_scores = torch.matmul(sub_h_W_r, e_proj.t())  # [sub_batch, n_entities]
                    all_scores.append(sub_scores)
                
                # Merge results
                scores = torch.cat(all_scores, dim=0)  # [batch_size, n_entities]
            else:
                # Directly compute all scores
                scores = torch.matmul(h_W_r, e_proj.t())  # [batch_size, n_entities]
                
            return scores
        
        else:
            raise ValueError("Must provide either tail entity embeddings (t) or all entity embeddings (all_entities)")


class KGAggregator(nn.Module):
    """Knowledge Graph Aggregator Layer, inspired by KGAT"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(KGAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        
        # Transformation matrices
        self.W_self = nn.Linear(input_dim, output_dim)
        self.W_neigh = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()
    
    def forward(self, entity_embs, adj_matrix):
        """
        Knowledge graph aggregation forward propagation
        
        Args:
            entity_embs: Entity embeddings [n_entities, embed_dim]
            adj_matrix: Normalized adjacency matrix [n_entities, n_entities]
        """
        # Self transformation
        self_embeddings = self.W_self(entity_embs)
        
        # Neighbor aggregation (efficient operation through sparse matrix multiplication)
        neigh_embeddings = torch.sparse.mm(adj_matrix, entity_embs)
        neigh_embeddings = self.W_neigh(neigh_embeddings)
        
        # Combine self and neighbor information
        output = self.activation(self_embeddings + neigh_embeddings)
        output = self.dropout(output)
        
        return output


class SVDKGContrastiveLearning(nn.Module):
    """Contrastive learning module for SVD and KG embeddings"""
    
    def __init__(self, svd_dim, kg_dim, proj_dim, temperature=0.1):
        super(SVDKGContrastiveLearning, self).__init__()
        
        # Projection network - SVD space
        self.svd_projector = nn.Sequential(
            nn.Linear(svd_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Projection network - KG space
        self.kg_projector = nn.Sequential(
            nn.Linear(kg_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.temperature = temperature
    
    def forward(self, svd_embeds, kg_embeds):
        """
        Compute contrastive loss between SVD and KG embeddings
        
        Args:
            svd_embeds: SVD-based embeddings [batch_size, svd_dim]
            kg_embeds: KG-based embeddings [batch_size, kg_dim]
        """
        # Project to common space
        svd_proj = self.svd_projector(svd_embeds)
        kg_proj = self.kg_projector(kg_embeds)
        
        # Normalize embeddings
        svd_proj = F.normalize(svd_proj, p=2, dim=1)
        kg_proj = F.normalize(kg_proj, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(svd_proj, kg_proj.T) / self.temperature
        
        # Labels are diagonal elements (positive pairs)
        batch_size = svd_embeds.size(0)
        labels = torch.arange(batch_size, device=svd_embeds.device)
        
        # Bidirectional InfoNCE loss
        loss_svd_to_kg = F.cross_entropy(sim_matrix, labels)
        loss_kg_to_svd = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_svd_to_kg + loss_kg_to_svd) / 2, svd_proj, kg_proj


class KGHybridGeoSVDGCN(KnowledgeRecommender):
    """Knowledge Graph Enhanced Hybrid Geometric SVDGCN Recommendation Model"""
    
    input_type = InputType.PAIRWISE
    type = ModelType.KNOWLEDGE
    
    def __init__(self, config, dataset):
        super(KGHybridGeoSVDGCN, self).__init__(config, dataset)
        
        # Setup logger
        self.logger = logging.getLogger()
        
        # Get parameters from config
        self.embedding_size = config['embedding_size']
        self.beta = config['beta']  # SVD weight coefficient
        self.req_vec = config['req_vec']  # Number of SVD vectors
        self.curvature = config['curvature']  # Hyperbolic space curvature
        self.hyperbolic_ratio = config['hyperbolic_ratio']  # Hyperbolic space weight
        self.pop_scale = config['pop_scale']  # Popularity scaling coefficient
        self.num_layers = config['num_layers']  # Number of GNN layers
        self.use_graph = config['use_graph']  # Whether to use graph structure
        self.reg_weight = config['reg_weight']  # Regularization coefficient
        self.svd_path = config['svd_path']  # SVD file path
        self.use_mmap = config['use_mmap']  # Whether to use memory mapping
        self.batch_size = config['sparse_graph_batch_size']  # Batch size
        
        # Knowledge graph related configuration
        self.kg_embedding_size = config['kg_embedding_size']
        self.kg_weight = config['kg_weight']  # Knowledge graph loss weight
        self.cl_weight = config['cl_weight']  # Contrastive learning weight
        self.cl_temperature = config['cl_temperature']
        self.kg_n_layers = config['kg_n_layers']
        self.long_tail_threshold = config['long_tail_threshold']  # Long-tail item threshold
        
        # Print main parameters
        self.logger.info(f"KGHybridGeoSVDGCN parameters:")
        self.logger.info(f"- embedding_size: {self.embedding_size}")
        self.logger.info(f"- kg_embedding_size: {self.kg_embedding_size}")
        self.logger.info(f"- curvature: {self.curvature}")
        self.logger.info(f"- hyperbolic_ratio: {self.hyperbolic_ratio}")
        self.logger.info(f"- pop_scale: {self.pop_scale}")
        self.logger.info(f"- num_layers: {self.num_layers}")
        
        # Build interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        self.logger.info(f"Dataset size: {self.n_users} users, {self.n_items} items")
        
        # Check SVD path
        if not os.path.exists(self.svd_path):
            self.logger.warning(f"SVD path does not exist: {self.svd_path}")
            os.makedirs(self.svd_path, exist_ok=True)
            raise ValueError(f"SVD files not found, please run preprocessing script to generate SVD files")
        
        # Create hyperbolic space tools
        self.manifold = LorentzManifold(c=self.curvature)
        
        # Create graph neural network layers
        if self.use_graph:
            self.create_adj_matrix()
            self.gnn_layers = nn.ModuleList([
                HybridGNNLayer(self.manifold, alpha=self.hyperbolic_ratio)
                for _ in range(self.num_layers)
            ])
        
        # Load SVD decomposition results
        self.load_svd()
        
        # Initialize feature projection matrix
        self.FS = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.FloatTensor(self.req_vec, self.embedding_size)
            )
        )
        
        # Initialize item popularity
        self.compute_item_popularity()
        
        # Create item mixing coefficient parameters
        self.item_mix_weights = nn.Parameter(torch.zeros(self.n_items))
        
        # Create hyperbolic fine-tuning parameters
        self.hyper_bias = nn.Parameter(torch.zeros(1, self.embedding_size + 1))
        
        # Create training curve parameters
        self.hybrid_temp = nn.Parameter(torch.tensor(2.0))
        
        # Initialize knowledge graph components
        self.init_kg_from_recbole(dataset)
        
        # Create KG adjacency matrix for efficient computation
        self.create_kg_adjacency()
        
        # Add KG-GNN layers
        self.kg_aggregator_layers = nn.ModuleList([
            KGAggregator(
                self.kg_embedding_size,
                self.kg_embedding_size,
                dropout=config['kg_dropout']
            )
            for _ in range(self.kg_n_layers)
        ])

        # Add TuckER scoring component
        self.use_tucker = config['use_tucker']  # Enable TuckER by default
        self.tucker_weight = config['tucker_weight']  # TuckER scoring weight
        self.tucker_core_size = config['tucker_core_size']  # Core tensor size
        self.tucker_dropout = config['tucker_dropout']  # Dropout rate
        
        self.tucker_scoring = TuckerScoring(
            entity_dim=self.kg_embedding_size,
            relation_dim=self.kg_embedding_size,
            core_size=self.tucker_core_size,
            dropout=self.tucker_dropout
        )
        
        # Define user-item relation ID for recommendation
        self.user_item_relation_id = 0  # "like" relation ID
        
        # Update logging
        self.logger.info(f"TuckER configuration: core size={self.tucker_core_size}, weight={self.tucker_weight}")
        
        # Contrastive learning component
        self.contrastive_learning = SVDKGContrastiveLearning(
            svd_dim=self.embedding_size,
            kg_dim=self.kg_embedding_size,
            proj_dim=config['cl_proj_dim'],
            temperature=self.cl_temperature
        )
        
        # SVD feature attention layer - use SVD features to guide knowledge graph representation
        self.svd_attention = nn.Sequential(
            nn.Linear(self.req_vec, 128),
            nn.ReLU(),
            nn.Linear(128, self.kg_embedding_size)
        )
        
        # KG feature fusion layer - integrate knowledge graph representation into final representation
        self.kg_fusion = nn.Sequential(
            nn.Linear(self.embedding_size + self.kg_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size)
        )
        
        # Adaptive weight for long-tail items
        self.adaptive_weight = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Loss functions
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # Apply parameter initialization
        self.apply(xavier_normal_initialization)
        
        self.logger.info(f"KGHybridGeoSVDGCN model initialization completed")
        self.config = config
        self.dataset = dataset

    def create_kg_adjacency(self):
        """Create knowledge graph adjacency matrix"""
        self.logger.info("Creating KG adjacency matrix...")
        
        if len(self.kg_triples) == 0:
            self.logger.warning("No KG triples, cannot create KG adjacency matrix")
            # Create empty adjacency matrix
            self.kg_adj_tensor = torch.sparse.FloatTensor(
                torch.LongTensor([[0], [0]]),
                torch.FloatTensor([0]),
                torch.Size([self.n_entities, self.n_entities])
            ).to(self.device)
            return
            
        # Extract triples
        head = self.kg_triples[:, 0].cpu().numpy()
        relation = self.kg_triples[:, 1].cpu().numpy()
        tail = self.kg_triples[:, 2].cpu().numpy()
        
        # Create adjacency matrix (value 1, can use relation types if needed)
        adj_size = self.n_entities
        row = np.concatenate([head, tail])
        col = np.concatenate([tail, head])
        data = np.ones_like(row, dtype=np.float32)
        
        # Create sparse matrix for efficiency
        kg_adj = sp.coo_matrix((data, (row, col)), shape=(adj_size, adj_size))
        
        # Create GNN normalized adjacency matrix
        rowsum = np.array(kg_adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        # Normalized adjacency: D^(-0.5) * A * D^(-0.5)
        norm_kg_adj = d_mat_inv.dot(kg_adj).dot(d_mat_inv)
        
        # Convert to PyTorch sparse tensor
        norm_kg_adj = norm_kg_adj.tocoo()
        indices = torch.LongTensor([norm_kg_adj.row, norm_kg_adj.col])
        values = torch.FloatTensor(norm_kg_adj.data)
        
        self.kg_adj_tensor = torch.sparse.FloatTensor(
            indices, values, torch.Size([adj_size, adj_size])
        ).to(self.device)
        
        self.logger.info(f"KG adjacency matrix creation completed, size: {adj_size} x {adj_size}")
    
    def _weight_func(self, sig):
        """Apply exponential weight to singular values"""
        return torch.exp(self.beta * sig)
    
    def create_adj_matrix(self):
        """Create normalized adjacency matrix"""
        self.logger.info("Building graph adjacency matrix...")
        
        # Get interaction data
        user_np, item_np = self.interaction_matrix.row, self.interaction_matrix.col
        
        # Total number of nodes
        n_nodes = self.n_users + self.n_items
        
        # Create bipartite graph indices
        user_np = user_np.astype(np.int64)
        item_np = item_np.astype(np.int64) + self.n_users  # Item index offset
        
        # Create edge indices and weights (bidirectional)
        row = np.concatenate([user_np, item_np])
        col = np.concatenate([item_np, user_np])
        data = np.ones(len(row), dtype=np.float32)
        
        # Create sparse adjacency matrix
        adj_mat = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        
        # Normalization: add self-loops and perform symmetric normalization
        adj_mat_self_loop = adj_mat + sp.eye(n_nodes, dtype=np.float32)
        
        # Compute inverse square root of degree matrix
        rowsum = np.array(adj_mat_self_loop.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        # Normalized adjacency matrix: D^(-0.5) * A * D^(-0.5)
        norm_adj = d_mat_inv.dot(adj_mat_self_loop).dot(d_mat_inv)
        
        # Convert to PyTorch sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        
        self.adj_matrix = torch.sparse.FloatTensor(
            indices, values, torch.Size([n_nodes, n_nodes])
        ).to(self.device)
        
        self.logger.info(f"Adjacency matrix construction completed, size: {n_nodes} x {n_nodes}")
    
    def compute_item_popularity(self):
        """Compute item popularity"""
        self.logger.info("Computing item popularity...")
        
        # Use interaction matrix to compute item popularity (interaction count)
        item_count = np.bincount(self.interaction_matrix.col, minlength=self.n_items)
        
        # Log transformation to reduce extreme value impact
        item_pop = np.log1p(item_count)
        
        # Normalize to [0,1]
        if item_pop.max() > 0:
            item_pop = item_pop / item_pop.max()
        
        # Convert to tensor
        self.item_popularity = torch.FloatTensor(item_pop).to(self.device)
        
        # Compute item popularity statistics
        self.logger.info(f"Item popularity statistics: min={item_pop.min():.4f}, "
                         f"max={item_pop.max():.4f}, "
                         f"mean={item_pop.mean():.4f}")
    
    def get_popularity_weights(self):
        """Generate popularity-based mixing weights"""
        # Use Sigmoid function to map item popularity to [0,1] interval
        # Higher popularity leads to larger Euclidean weights
        weights = torch.sigmoid((self.item_popularity - 0.5) * self.pop_scale + self.item_mix_weights)
        
        # Mixing temperature parameter controls mixing sharpness
        temp = F.softplus(self.hybrid_temp)
        weights = torch.sigmoid((weights - 0.5) * temp)
        
        return weights
    
    def load_svd(self):
        """Load SVD decomposition results"""
        self.logger.info("Loading SVD decomposition files...")
        
        try:
            # Load SVD singular values
            svd_values_path = os.path.join(self.svd_path, 'svd_value.npy')
            if not os.path.exists(svd_values_path):
                raise FileNotFoundError(f"SVD values file not found: {svd_values_path}")
            
            self.svd_values = torch.FloatTensor(
                np.load(svd_values_path)[:self.req_vec]
            ).to(self.device)
            
            # Apply SVD filter
            svd_filter = self._weight_func(self.svd_values)
            
            # Load user and item vectors
            if self.use_mmap:
                # Use memory mapping to load large files
                user_vector_mmap = np.load(os.path.join(self.svd_path, 'svd_u.npy'), mmap_mode='r')
                item_vector_mmap = np.load(os.path.join(self.svd_path, 'svd_v.npy'), mmap_mode='r')
                
                # Load to GPU in batches
                batch_size = self.batch_size
                
                # User vectors
                self.user_vector = torch.zeros((self.n_users, self.req_vec), device=self.device)
                for i in range(0, self.n_users, batch_size):
                    end_idx = min(i + batch_size, self.n_users)
                    batch_data = user_vector_mmap[i:end_idx, :self.req_vec]
                    self.user_vector[i:end_idx] = torch.FloatTensor(batch_data).to(self.device) * svd_filter
                
                # Item vectors
                self.item_vector = torch.zeros((self.n_items, self.req_vec), device=self.device)
                for i in range(0, self.n_items, batch_size):
                    end_idx = min(i + batch_size, self.n_items)
                    batch_data = item_vector_mmap[i:end_idx, :self.req_vec]
                    self.item_vector[i:end_idx] = torch.FloatTensor(batch_data).to(self.device) * svd_filter
            else:
                # Load entire files directly
                self.user_vector = torch.FloatTensor(
                    np.load(os.path.join(self.svd_path, 'svd_u.npy'))[:self.n_users, :self.req_vec]
                ).to(self.device) * svd_filter
                
                self.item_vector = torch.FloatTensor(
                    np.load(os.path.join(self.svd_path, 'svd_v.npy'))[:self.n_items, :self.req_vec]
                ).to(self.device) * svd_filter

            if len(self.item_vector) < self.n_items:
                self.logger.warning(f"Warning: SVD item vector count ({len(self.item_vector)}) less than item count ({self.n_items})!")
                # Add zero vectors for missing items
                padding = torch.zeros((self.n_items - len(self.item_vector), self.req_vec), device=self.device)
                self.item_vector = torch.cat([self.item_vector, padding], dim=0)
            
            self.logger.info(f"SVD data loading completed: {self.req_vec} dimensions")
            
        except Exception as e:
            self.logger.error(f"SVD file loading failed: {e}")
            raise ValueError(f"SVD file loading error: {e}. Please run preprocessing script to generate SVD files.")
    
    def init_kg_from_recbole(self, dataset):
        """Initialize knowledge graph information from RecBole dataset, supports loading pretrained embeddings"""
        self.logger.info("Initializing knowledge graph from RecBole dataset...")
        
        # Check if pretrained embedding path is specified
        kg_embed_path = None
        if hasattr(dataset.config, 'kg_embed_path'):
            kg_embed_path = dataset.config['kg_embed_path']
        
        # Explicitly log whether pretrained embeddings will be used
        if kg_embed_path:
            self.logger.info(f"Detected pretrained knowledge graph embedding path: {kg_embed_path}")
        else:
            self.logger.info("No pretrained knowledge graph embedding path specified, will use random initialization")
        
        try:
            # Load directly from file, avoid intermediate processing
            kg_file = os.path.join(dataset.dataset_path, f"{dataset.dataset_name}.kg")
            
            if os.path.exists(kg_file):
                self.logger.info(f"Loading knowledge graph directly from file: {kg_file}")
                self.kg_triples = []
                
                # Track ID information
                all_heads = set()
                all_tails = set()
                all_relations = set()
                
                with open(kg_file, 'r') as f:
                    header = f.readline().strip().split('\t')
                    
                    # Determine column indices
                    try:
                        head_idx = header.index('head_id:token')
                        relation_idx = header.index('relation_id:token')
                        tail_idx = header.index('tail_id:token')
                    except ValueError:
                        self.logger.warning("Standard column names not found, trying alternative format")
                        try:
                            head_idx = header.index('head_id')
                            relation_idx = header.index('relation_id')
                            tail_idx = header.index('tail_id')
                        except ValueError:
                            self.logger.warning("Using positional indices")
                            head_idx, relation_idx, tail_idx = 0, 1, 2
                    
                    # Read and process triples
                    line_count = 0
                    valid_count = 0
                    
                    for line in f:
                        line_count += 1
                        if line_count % 1000000 == 0:
                            self.logger.info(f"Processed {line_count} lines...")
                            
                        parts = line.strip().split('\t')
                        if len(parts) <= max(head_idx, relation_idx, tail_idx):
                            continue
                            
                        try:
                            h = int(parts[head_idx])
                            r = int(parts[relation_idx])
                            t = int(parts[tail_idx])
                            
                            self.kg_triples.append((h, r, t))
                            
                            all_heads.add(h)
                            all_relations.add(r)
                            all_tails.add(t)
                            
                            valid_count += 1
                        except (ValueError, IndexError):
                            continue
                
                if valid_count == 0:
                    raise ValueError("No valid knowledge graph triples")
                
                # Determine number of entities and relations
                all_entities = all_heads.union(all_tails)
                self.n_entities = max(all_entities) + 1
                self.n_relations = max(all_relations) + 1
                
                self.logger.info(f"Knowledge graph statistics: {len(self.kg_triples)} triples, entity ID range [{min(all_entities)}-{max(all_entities)}], relation ID range [{min(all_relations)}-{max(all_relations)}]")
                
                # Ensure item_to_entity mapping uses correct entity IDs
                link_file = os.path.join(dataset.dataset_path, f"{dataset.dataset_name}.link")
                if os.path.exists(link_file):
                    self.logger.info(f"Loading item-entity links: {link_file}")
                    self.item_to_entity = {}
                    
                    with open(link_file, 'r') as f:
                        header = f.readline().strip().split('\t')
                        
                        # Determine column indices
                        try:
                            item_idx = header.index('item_id:token')
                            entity_idx = header.index('entity_id:token')
                        except ValueError:
                            self.logger.warning("Standard column names not found, trying alternative format")
                            try:
                                item_idx = header.index('item_id')
                                entity_idx = header.index('entity_id')
                            except ValueError:
                                self.logger.warning("Using positional indices")
                                item_idx, entity_idx = 0, 1
                        
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) <= max(item_idx, entity_idx):
                                continue
                                
                            try:
                                item_id = int(parts[item_idx])
                                entity_id = int(parts[entity_idx])
                                self.item_to_entity[item_id] = entity_id
                            except (ValueError, IndexError):
                                continue
                
                # Convert triples to tensor
                self.kg_triples = torch.LongTensor(self.kg_triples).to(self.device)
                
                # Initialize embeddings
                # Ensure embedding dimensions are large enough to accommodate all entity IDs
                self.entity_embedding = nn.Embedding(self.n_entities, self.kg_embedding_size)
                self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
                
                # Try to load pretrained embeddings
                kg_embeds_loaded = False
                if kg_embed_path:
                    self.logger.info(f"Attempting to load pretrained knowledge graph embeddings: {kg_embed_path}")
                    
                    try:
                        # Check if pretrained files exist
                        entity_embed_path = os.path.join(kg_embed_path, 'entity_embeddings.npy')
                        relation_embed_path = os.path.join(kg_embed_path, 'relation_embeddings.npy')
                        
                        if not os.path.exists(entity_embed_path):
                            self.logger.warning(f"Entity embedding file does not exist: {entity_embed_path}")
                        if not os.path.exists(relation_embed_path):
                            self.logger.warning(f"Relation embedding file does not exist: {relation_embed_path}")
                        
                        if os.path.exists(entity_embed_path) and os.path.exists(relation_embed_path):
                            # Load pretrained embeddings
                            pretrained_entity_embeds = np.load(entity_embed_path)
                            pretrained_relation_embeds = np.load(relation_embed_path)
                            
                            self.logger.info(f"Pretrained embedding dimensions: entity={pretrained_entity_embeds.shape}, relation={pretrained_relation_embeds.shape}")
                            
                            # Handle dimension mismatch
                            pretrained_dim = pretrained_entity_embeds.shape[1]
                            if pretrained_dim != self.kg_embedding_size:
                                self.logger.warning(f"Pretrained embedding dimension ({pretrained_dim}) does not match configured dimension ({self.kg_embedding_size})")
                                self.logger.info(f"Adjusting knowledge graph embedding dimension to: {pretrained_dim}")
                                self.kg_embedding_size = pretrained_dim
                                
                                # Recreate embedding layers
                                self.entity_embedding = nn.Embedding(self.n_entities, self.kg_embedding_size)
                                self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
                            
                            # Copy pretrained embeddings
                            with torch.no_grad():
                                # Copy entity embeddings
                                copy_size_entity = min(self.n_entities, pretrained_entity_embeds.shape[0])
                                self.entity_embedding.weight.data[:copy_size_entity].copy_(
                                    torch.from_numpy(pretrained_entity_embeds[:copy_size_entity])
                                )
                                
                                # Copy relation embeddings
                                copy_size_relation = min(self.n_relations, pretrained_relation_embeds.shape[0])
                                self.relation_embedding.weight.data[:copy_size_relation].copy_(
                                    torch.from_numpy(pretrained_relation_embeds[:copy_size_relation])
                                )
                            
                            self.logger.info(f"Successfully loaded pretrained embeddings: {copy_size_entity}/{self.n_entities} entities, {copy_size_relation}/{self.n_relations} relations")
                            kg_embeds_loaded = True
                            
                            # Whether to freeze pretrained embeddings
                            freeze_kg_embeds = False
                            if hasattr(dataset.config, 'freeze_kg_embeds'):
                                freeze_kg_embeds = dataset.config['freeze_kg_embeds']
                            
                            if freeze_kg_embeds:
                                self.logger.info("Freezing pretrained knowledge graph embeddings")
                                self.entity_embedding.weight.requires_grad = False
                                self.relation_embedding.weight.requires_grad = False
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load pretrained embeddings: {e}")
                
                # If pretrained embeddings were not successfully loaded, use random initialization
                if not kg_embeds_loaded:
                    self.logger.info("Using random initialization for knowledge graph embeddings")
                    nn.init.xavier_uniform_(self.entity_embedding.weight)
                    nn.init.xavier_uniform_(self.relation_embedding.weight)
                
                self.logger.info(f"Knowledge graph initialization completed: {len(self.kg_triples)} triples, {self.n_entities} entities, {self.n_relations} relations")
            else:
                raise FileNotFoundError(f"Knowledge graph file does not exist: {kg_file}")
                
        except Exception as e:
            self.logger.error(f"Knowledge graph initialization failed: {e}")
            # Create empty knowledge graph components to ensure model can still run
            self.n_entities = self.n_items + 1
            self.n_relations = 1
            self.kg_triples = torch.LongTensor([]).to(self.device)
            self.item_to_entity = {i: i for i in range(self.n_items)}
            
            # Initialize empty embeddings
            self.entity_embedding = nn.Embedding(self.n_entities, self.kg_embedding_size)
            self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
            
            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)
            
            self.logger.warning("Running in mode without knowledge graph")
    
    def get_item_kg_embeddings(self, item_ids, kg_entity_embs=None):
        """Get knowledge graph embeddings for items (efficient batch processing)"""
        # If kg_entity_embs not provided, use entity embeddings
        if kg_entity_embs is None:
            kg_entity_embs = self.entity_embedding.weight
        
        # Map item IDs to entity IDs (batch processing)
        entity_ids = []
        for item_id in item_ids.cpu().tolist():
            if item_id < self.n_items and item_id in self.item_to_entity:
                entity_id = self.item_to_entity[item_id]
                # Ensure entity ID is within valid range
                if entity_id < self.n_entities:
                    entity_ids.append(entity_id)
                else:
                    entity_ids.append(0)  # Use default entity ID
            else:
                entity_ids.append(0)  # Use default entity ID
        
        entity_ids = torch.LongTensor(entity_ids).to(item_ids.device)
        
        # Get embeddings for these entities
        return kg_entity_embs[entity_ids]

    def get_entity_embeddings_for_items(self, item_ids):
        """Get entity embeddings for items, used for TuckER scoring"""
        return self.get_item_kg_embeddings(item_ids)

    def get_entity_embeddings_for_users(self, user_ids):
        """Get entity embeddings for users, used for TuckER scoring
        
        Here we assume users also have corresponding entity IDs, or we use average item embeddings to represent users
        """
        if hasattr(self, 'user_to_entity'):
            # If user-entity mapping exists, use it directly
            entity_ids = []
            for user_id in user_ids.cpu().tolist():
                entity_id = self.user_to_entity.get(user_id, 0)
                if entity_id >= self.n_entities:
                    entity_id = 0
                entity_ids.append(entity_id)
            
            entity_ids = torch.tensor(entity_ids, device=user_ids.device)
            return self.entity_embedding(entity_ids)
        else:
            # If no user-entity mapping, use special handling
            # Option 1: Use randomly initialized user embeddings
            batch_size = len(user_ids)
            user_entity_embs = torch.empty(batch_size, self.kg_embedding_size, device=user_ids.device)
            nn.init.normal_(user_entity_embs, mean=0.0, std=0.01)
            
            # Could also use other approaches, such as average embeddings of user-interacted items
            
            return user_entity_embs
    
    def process_kg(self):
        """Process knowledge graph to get enhanced entity embeddings"""
        entity_embs = self.entity_embedding.weight
        
        # Apply KG-GNN layers
        for layer in self.kg_aggregator_layers:
            entity_embs = layer(entity_embs, self.kg_adj_tensor)
        
        return entity_embs
    
    def calculate_kg_loss(self):
        """Calculate TransE loss function (batch processing for efficiency)"""
        if len(self.kg_triples) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Sample a batch of triples
        batch_size = min(1024, len(self.kg_triples))
        indices = torch.randperm(len(self.kg_triples))[:batch_size]
        batch_triples = self.kg_triples[indices]
        
        # Extract head entities, relations, and tail entities
        heads = batch_triples[:, 0]
        relations = batch_triples[:, 1]
        tails = batch_triples[:, 2]
        
        # Randomly replace some head or tail entities to generate negative samples
        neg_heads = heads.clone()
        neg_tails = tails.clone()
        
        # 50% probability to replace head entity, 50% probability to replace tail entity
        head_mask = torch.rand(batch_size, device=self.device) < 0.5
        
        # Generate random entity IDs
        head_corrupted = torch.randint(0, self.n_entities, (batch_size,), device=self.device)
        neg_heads[head_mask] = head_corrupted[head_mask]
        
        tail_mask = ~head_mask
        tail_corrupted = torch.randint(0, self.n_entities, (batch_size,), device=self.device)
        neg_tails[tail_mask] = tail_corrupted[tail_mask]
        
        # Get embeddings
        head_embs = self.entity_embedding(heads)
        rel_embs = self.relation_embedding(relations)
        tail_embs = self.entity_embedding(tails)
        
        neg_head_embs = self.entity_embedding(neg_heads)
        neg_tail_embs = self.entity_embedding(neg_tails)
        
        # Compute scores: ||h + r - t||
        pos_scores = torch.norm(head_embs + rel_embs - tail_embs, p=2, dim=1)
        neg_scores = torch.norm(neg_head_embs + rel_embs - neg_tail_embs, p=2, dim=1)
        
        # Use margin ranking loss
        margin = 1.0
        kg_loss = torch.mean(torch.relu(margin + pos_scores - neg_scores))
        
        return kg_loss
    
    def forward(self):
        """Enhanced forward propagation, efficient KG processing"""
        # 1. SVD embeddings
        euclidean_user = self.user_vector.mm(self.FS)
        euclidean_item = self.item_vector.mm(self.FS)
        
        # 2. Lorentz embeddings
        lorentz_user = self.manifold.euclidean_to_lorentz(euclidean_user)
        lorentz_item = self.manifold.euclidean_to_lorentz(euclidean_item)
        
        # 3. If using graph structure, apply GNN
        if self.use_graph:
            # Concatenate user and item embeddings
            euc_all = torch.cat([euclidean_user, euclidean_item], dim=0)
            lor_all = torch.cat([lorentz_user, lorentz_item], dim=0)
            
            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                euc_all, lor_all = gnn_layer(euc_all, lor_all, self.adj_matrix)
            
            # Separate user and item embeddings
            euclidean_user, euclidean_item = torch.split(euc_all, [self.n_users, self.n_items])
            lorentz_user, lorentz_item = torch.split(lor_all, [self.n_users, self.n_items])
        
        # 4. Apply hyperbolic bias
        lorentz_user = lorentz_user + self.hyper_bias
        lorentz_item = lorentz_item + self.hyper_bias
        
        # Ensure hyperbolic representations are on manifold
        lorentz_user = self.manifold.project_to_lorentz(lorentz_user)
        lorentz_item = self.manifold.project_to_lorentz(lorentz_item)
        
        # 5. Get popularity weights
        pop_weights = self.get_popularity_weights()
        
        # 6. Process knowledge graph embeddings
        kg_entity_embs = self.process_kg()
        
        # 7. Get KG embeddings for items
        item_ids = torch.arange(self.n_items, device=self.device)
        kg_item_embs = self.get_item_kg_embeddings(item_ids, kg_entity_embs)
        
        # 8. SVD-guided KG attention
        svd_features = self.item_vector
        kg_attention_weights = torch.sigmoid(self.svd_attention(svd_features))
        enhanced_kg_embs = kg_item_embs * kg_attention_weights
        
        # 9. Prepare for adaptive fusion
        # Store original Euclidean embeddings
        original_euclidean_item = euclidean_item.clone()
        
        # 10. Adaptive fusion based on item popularity
        # Long-tail items get more weight from KG embeddings
        kg_weights = 1 - pop_weights  # Inverse of popularity
        
        # 11. Fusion through MLP
        combined_embs = torch.cat([euclidean_item, enhanced_kg_embs], dim=1)
        fused_item_embs = self.kg_fusion(combined_embs)
        
        # 12. Apply adaptive weights - reshape for broadcasting
        kg_weights = kg_weights.unsqueeze(1)  # [n_items, 1]
        euclidean_item = (1 - kg_weights) * original_euclidean_item + kg_weights * fused_item_embs
        
        # 13. Update Lorentz embeddings
        enhanced_lorentz_item = self.manifold.euclidean_to_lorentz(euclidean_item)
        enhanced_lorentz_item = self.manifold.project_to_lorentz(enhanced_lorentz_item)
        
        return euclidean_user, euclidean_item, lorentz_user, enhanced_lorentz_item, pop_weights, enhanced_kg_embs
    
    def calculate_loss(self, interaction):
        """Calculate loss, add TuckER scoring component"""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        # Get embeddings
        euclidean_user, euclidean_item, lorentz_user, lorentz_item, pop_weights, kg_item_embs = self.forward()
        
        # Extract specific embeddings
        eu_u = euclidean_user[user]
        eu_pos = euclidean_item[pos_item]
        eu_neg = euclidean_item[neg_item]
        
        lo_u = lorentz_user[user]
        lo_pos = lorentz_item[pos_item]
        lo_neg = lorentz_item[neg_item]
        
        # Get popularity weights for specific items
        pos_weights = pop_weights[pos_item].unsqueeze(-1)  # [batch_size, 1]
        neg_weights = pop_weights[neg_item].unsqueeze(-1)  # [batch_size, 1]
        
        # 1. Euclidean space BPR loss
        eu_pos_scores = torch.sum(eu_u * eu_pos, dim=1)
        eu_neg_scores = torch.sum(eu_u * eu_neg, dim=1)
        euclidean_loss = self.mf_loss(eu_pos_scores, eu_neg_scores)
        
        # 2. Hyperbolic space loss
        # Use distance computation (smaller is better)
        lo_pos_dist = self.manifold.lorentz_distance(lo_u, lo_pos)
        lo_neg_dist = self.manifold.lorentz_distance(lo_u, lo_neg)
        # Convert to scores (negative of distance, larger is better)
        lo_pos_scores = -lo_pos_dist
        lo_neg_scores = -lo_neg_dist
        hyperbolic_loss = self.mf_loss(lo_pos_scores, lo_neg_scores)
        
        # 3. Mixed loss: dynamically mix Euclidean and hyperbolic losses for each item
        # Positive sample mixing
        pos_eu = pos_weights * eu_pos_scores
        pos_lo = (1 - pos_weights) * lo_pos_scores
        pos_mixed = pos_eu + pos_lo
        
        # Negative sample mixing
        neg_eu = neg_weights * eu_neg_scores
        neg_lo = (1 - neg_weights) * lo_neg_scores
        neg_mixed = neg_eu + neg_lo
        
        # Mixed BPR loss
        mixed_loss = self.mf_loss(pos_mixed, neg_mixed)
        
        # 4. Embedding regularization loss
        reg_loss = self.reg_loss(eu_u, eu_pos, eu_neg)
        
        # 5. Knowledge graph loss
        kg_loss = self.calculate_kg_loss()
        
        # 6. TuckER loss - if knowledge graph data is available and TuckER is enabled
        tucker_loss = torch.tensor(0.0, device=self.device)
        if self.use_tucker and len(self.kg_triples) > 0:
            # Get entity embeddings for users and items
            user_entity_embs = self.get_entity_embeddings_for_users(user)
            pos_entity_embs = self.get_entity_embeddings_for_items(pos_item)
            neg_entity_embs = self.get_entity_embeddings_for_items(neg_item)
            
            # Get relation embeddings ("like" relation)
            relation_embs = self.relation_embedding(
                torch.full_like(user, self.user_item_relation_id, device=self.device)
            )
            
            # Compute TuckER scores
            pos_tucker_scores = self.tucker_scoring(user_entity_embs, relation_embs, pos_entity_embs)
            neg_tucker_scores = self.tucker_scoring(user_entity_embs, relation_embs, neg_entity_embs)
            
            # TuckER BPR loss
            tucker_loss = self.mf_loss(pos_tucker_scores, neg_tucker_scores)
        
        # 7. Contrastive learning loss
        # Get SVD embeddings for positive sample items
        svd_pos_embs = self.item_vector[pos_item].mm(self.FS)
        # Get KG embeddings for positive sample items
        kg_pos_embs = kg_item_embs[pos_item]
        # Compute contrastive loss
        cl_loss, _, _ = self.contrastive_learning(svd_pos_embs, kg_pos_embs)
        
        # 8. Recommendation loss
        rec_loss = (
            (1 - self.hyperbolic_ratio) * euclidean_loss + 
            self.hyperbolic_ratio * hyperbolic_loss + 
            mixed_loss + 
            self.reg_weight * reg_loss
        )
        
        # Add TuckER loss (if enabled)
        if self.use_tucker:
            rec_loss = rec_loss + self.tucker_weight * tucker_loss
        
        # 9. Total loss
        total_loss = rec_loss + self.kg_weight * kg_loss + self.cl_weight * cl_loss
        
        return total_loss
    
    def predict(self, interaction):
        """Prediction function integrating TuckER scoring"""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # Get embeddings
        euclidean_user, euclidean_item, lorentz_user, lorentz_item, pop_weights, _ = self.forward()
        
        # Extract specific embeddings
        eu_u = euclidean_user[user]
        eu_i = euclidean_item[item]
        
        lo_u = lorentz_user[user]
        lo_i = lorentz_item[item]
        
        # Get item weights
        item_w = pop_weights[item].unsqueeze(-1)
        
        # 1. Euclidean score
        eu_score = torch.sum(eu_u * eu_i, dim=1)
        
        # 2. Hyperbolic score
        lo_score = -self.manifold.lorentz_distance(lo_u, lo_i)
        
        # 3. Mixed score
        mixed_score = item_w * eu_score + (1 - item_w) * lo_score
        
        # 4. TuckER score (if enabled and knowledge graph data available)
        if self.use_tucker and len(self.kg_triples) > 0:
            # Get entity embeddings for users and items
            user_entity_embs = self.get_entity_embeddings_for_users(user)
            item_entity_embs = self.get_entity_embeddings_for_items(item)
            
            # Get relation embeddings ("like" relation)
            relation_embs = self.relation_embedding(
                torch.full_like(user, self.user_item_relation_id, device=self.device)
            )
            
            # Compute TuckER score
            tucker_score = self.tucker_scoring(user_entity_embs, relation_embs, item_entity_embs)
            
            # Fuse scores - weighted combination
            final_score = mixed_score + self.tucker_weight * tucker_score
        else:
            final_score = mixed_score
        
        return final_score

    
    def full_sort_predict(self, interaction):
        """Improved full sort prediction - intelligent caching based on rounds"""
        user = interaction[self.USER_ID]
    
        # Get current epoch information for cache update decision
        # Try to get current epoch from different possible locations
        current_epoch = None
        
        # Get current epoch from trainer
        if hasattr(self, 'cur_epoch'):
            current_epoch = self.cur_epoch
        # If model itself tracks epoch
        elif hasattr(self, 'epoch'):
            current_epoch = self.epoch 
        # Get epoch information from interaction object
        elif hasattr(interaction, 'epoch'):
            current_epoch = interaction.epoch
        # If exact epoch cannot be obtained, use timestamp as substitute
        else:
            import time
            current_epoch = int(time.time() / 60)  # Refresh cache every minute
        
        # Check if cache needs update
        cache_needs_update = False
        if not hasattr(self, '_cached_epoch') or self._cached_epoch != current_epoch:
            self._cached_epoch = current_epoch
            cache_needs_update = True
        
        if not hasattr(self, '_cached_embeddings') or self._cached_embeddings is None or cache_needs_update:
            # Log cache update
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.info(f"Updating evaluation cache (epoch/timestamp: {current_epoch})")
            elif cache_needs_update:
                print(f"Updating evaluation cache (epoch/timestamp: {current_epoch})")
            
            # Update cache
            self._cached_embeddings = self.forward()
        
        # Use cache
        euclidean_user, euclidean_item, lorentz_user, lorentz_item, pop_weights, _ = self._cached_embeddings
    
        # Compute batch scores based on cached embeddings
        batch_size = min(1024, len(user))
        n_batches = (len(user) - 1) // batch_size + 1
        
        all_scores = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(user))
            user_batch = user[start:end]
        
            # Extract user embeddings
            eu_u = euclidean_user[user_batch]
            lo_u = lorentz_user[user_batch]
        
            # 1. Euclidean scores
            eu_scores = torch.matmul(eu_u, euclidean_item.t())
        
            # 2. Hyperbolic scores - optimized computation
            curr_batch_size = eu_u.size(0)
        
            # 2.1 Time component product
            t_u = lo_u[:, 0].unsqueeze(1)
            t_i = lorentz_item[:, 0].unsqueeze(0)
            t_prod = -torch.matmul(t_u, t_i)
        
            # 2.2 Space component product
            s_prod = torch.matmul(lo_u[:, 1:], lorentz_item[:, 1:].t())
        
            # 2.3 Minkowski inner product
            lo_scores = t_prod + s_prod
        
            # 3. Mixed scores using popularity weights
            pop_w = pop_weights.unsqueeze(0).expand(curr_batch_size, -1)
            batch_scores = pop_w * eu_scores + (1 - pop_w) * lo_scores
            
            # 4. TuckER score (if enabled and knowledge graph data available)
            if self.use_tucker and len(self.kg_triples) > 0:
                # Get user entity embeddings
                user_entity_embs = self.get_entity_embeddings_for_users(user_batch)
                
                # Get relation embeddings ("like" relation)
                relation_embs = self.relation_embedding(
                    torch.full((curr_batch_size,), self.user_item_relation_id, 
                              device=self.device)
                )
                
                # Get all item entity embeddings
                if not hasattr(self, '_cached_item_entity_embs'):
                    self._cached_item_entity_embs = torch.zeros(
                        self.n_items, self.kg_embedding_size, device=self.device
                    )
                    for item_id, entity_id in self.item_to_entity.items():
                        if item_id < self.n_items and entity_id < self.n_entities:
                            self._cached_item_entity_embs[item_id] = self.entity_embedding.weight[entity_id]
                
                # Compute TuckER scores - using full ranking mode
                try:
                    tucker_scores = self.tucker_scoring(
                        user_entity_embs, relation_embs, 
                        all_entities=self._cached_item_entity_embs,
                        batch_process=True
                    )
                    
                    # Fuse scores
                    batch_scores = batch_scores + self.tucker_weight * tucker_scores
                except Exception as e:
                    self.logger.warning(f"TuckER scoring computation failed: {e}")
        
            all_scores.append(batch_scores)
        
        # Merge results
        if len(all_scores) > 1:
            return torch.cat(all_scores, dim=0)
        else:
            return all_scores[0]

    def _batch_score(self, user_batch, eu_all, ei_all, lu_all, li_all, pop_w):
        """Optimized batch scoring computation"""
        # Extract user embeddings
        eu_u = eu_all[user_batch]
        lo_u = lu_all[user_batch]
    
        # Compute Euclidean scores - using matrix multiplication
        eu_scores = torch.matmul(eu_u, ei_all.t())
    
        # Optimized hyperbolic score computation 
        t_u = lo_u[:, 0].unsqueeze(1)
        t_i = li_all[:, 0].unsqueeze(0)
        t_prod = -torch.matmul(t_u, t_i)
    
        s_prod = torch.matmul(lo_u[:, 1:], li_all[:, 1:].t())
        lo_scores = t_prod + s_prod
    
        # Efficient mixing
        curr_batch_size = eu_u.size(0)
        batch_pop_w = pop_w.unsqueeze(0).expand(curr_batch_size, -1)
    
        # Separate sparse and dense computations during mixing
        return batch_pop_w * eu_scores + (1 - batch_pop_w) * lo_scores