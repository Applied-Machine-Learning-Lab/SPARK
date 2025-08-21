import os
import argparse
import torch
import logging
from logging import getLogger
import pandas as pd

from recbole.utils import init_seed, set_color
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer

# Import model class
from KG_model import KGHybridGeoSVDGCN

def main():
    """Run KGHybridGeoSVDGCN model (new architecture)"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run knowledge graph enhanced model')
    parser.add_argument('--dataset', type=str, default='amazon-book', help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Dataset path')
    parser.add_argument('--svd_path', type=str, default='./preprocessed', help='SVD preprocessing file path')
    parser.add_argument('--kg_embed_path', type=str, default=None, help='Pretrained knowledge graph embedding path, containing entity_embeddings.npy and relation_embeddings.npy')
    
    # Basic model parameters
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--kg_embedding_size', type=int, default=64, help='Knowledge graph embedding dimension')
    parser.add_argument('--kg_weight', type=float, default=0.01, help='Knowledge graph loss weight')
    parser.add_argument('--long_tail_threshold', type=float, default=0.3, help='Long-tail item threshold')
    parser.add_argument('--req_vec', type=int, default=256, help='Number of SVD features to use')
    parser.add_argument('--pop_scale', type=float, default=10.0, help='Popularity scaling coefficient')
    parser.add_argument('--hyperbolic_ratio', type=float, default=0.3, help='Hyperbolic space weight')
    parser.add_argument('--curvature', type=float, default=0.8, help='Hyperbolic space curvature')
    parser.add_argument('--freeze_kg_embeds', action='store_true', help='Whether to freeze pretrained knowledge graph embeddings')

    parser.add_argument('--tucker_weight', type=float, default=0.5, help='TuckER scoring weight')
    parser.add_argument('--tucker_core_size', type=int, default=64, help='TuckER core tensor dimension')
    parser.add_argument('--tucker_dropout', type=float, default=0.2, help='TuckER component dropout rate')
    parser.add_argument('--use_tucker', action='store_true', help='Whether to use TuckER scoring signal')
    
    # KG aggregation layer parameters
    parser.add_argument('--kg_n_layers', type=int, default=2, help='Number of knowledge graph aggregation layers')
    parser.add_argument('--kg_dropout', type=float, default=0.1, help='Knowledge graph aggregation layer dropout rate')
    
    # Contrastive learning parameters
    parser.add_argument('--cl_weight', type=float, default=0.1, help='Contrastive learning loss weight')
    parser.add_argument('--cl_temperature', type=float, default=0.1, help='Contrastive learning temperature parameter')
    parser.add_argument('--cl_proj_dim', type=int, default=128, help='Contrastive learning projection dimension')
    
    # New fusion module parameters
    parser.add_argument('--fusion_hidden_size', type=int, default=256, help='Fusion layer hidden dimension')
    parser.add_argument('--use_attention', action='store_true', help='Whether to use SVD-guided attention mechanism')
    
    # GNN layer parameters
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--use_graph', action='store_true', default=True, help='Whether to use graph structure')
    
    # Training parameters
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--sparse_batch_size', type=int, default=2048, help='Batch size for sparse graph operations')
    parser.add_argument('--batch_debug', action='store_true', help='Whether to output batch processing debug information')

    
    
    # Evaluation parameters
    parser.add_argument('--eval_step', type=int, default=1, help='How many epochs between evaluations')
    parser.add_argument('--topk', type=str, default='[5, 10, 20]', help='K values for recommendation evaluation')
    
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = getLogger()

    # Add file handler to save logs
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{args.dataset}_kghybridgeosvdgcn_{args.embedding_size}d.log"
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Log file will be saved to: {log_path}")
    
    # Print running parameters
    logger.info(f"Running KGHybridGeoSVDGCN (new architecture) model")
    logger.info(f"Running parameters: {args}")
    
    # SVD path
    svd_path = os.path.join(args.svd_path, args.dataset)
    if not os.path.exists(svd_path):
        logger.error(f"SVD path does not exist: {svd_path}")
        return
    
    # Knowledge graph embedding path validation
    use_pretrained_kg = False
    if args.kg_embed_path and os.path.exists(args.kg_embed_path):
        entity_path = os.path.join(args.kg_embed_path, 'entity_embeddings.npy')
        relation_path = os.path.join(args.kg_embed_path, 'relation_embeddings.npy')
        
        if os.path.exists(entity_path) and os.path.exists(relation_path):
            use_pretrained_kg = True
            logger.info(f"Will use pretrained knowledge graph embeddings: {args.kg_embed_path}")
        else:
            logger.warning(f"Pretrained embedding files incomplete, will use random initialization: {args.kg_embed_path}")
    else:
        logger.info("Using random initialization for knowledge graph embeddings")
    
    # Parse list parameters
    try:
        topk = eval(args.topk)
        if not isinstance(topk, list):
            topk = [5, 10, 20]
            logger.warning(f"Invalid topk format, using default: {topk}")
    except:
        topk = [5, 10, 20]
        logger.warning(f"Invalid topk format, using default: {topk}")
    
    # Build configuration dictionary
    config_dict = {
        # Dataset parameters
        'dataset': args.dataset,
        'data_path': args.dataset_path,
        
        # Training parameters
        'epochs': args.epochs,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.batch_size * 2,
        'learning_rate': args.learning_rate,
        'eval_step': args.eval_step,
        'stopping_step': 10,
        'neg_sampling': {'uniform': 1},
        'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'group_by': 'user'},
        
        # Evaluation parameters
        'metrics': ['Recall', 'NDCG', 'MRR', 'Hit', 'Precision'],
        'topk': topk,
        'valid_metric': 'Recall@10',
        
        # Environment parameters
        'seed': 2023,
        'gpu_id': args.gpu_id,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'reproducibility': True,
        'show_progress': True,
        'batch_debug': args.batch_debug,
        
        # Model basic parameters
        'embedding_size': args.embedding_size,
        'reg_weight': 0.01,
        'beta': 2.0,
        'req_vec': args.req_vec,
        'svd_path': svd_path,
        'curvature': args.curvature,
        'hyperbolic_ratio': args.hyperbolic_ratio,
        'pop_scale': args.pop_scale,
        'use_graph': args.use_graph,
        'use_mmap': False,
        'sparse_graph_batch_size': args.sparse_batch_size,
        'num_layers': args.num_layers,
        
        # Knowledge graph parameters
        'kg_embedding_size': args.kg_embedding_size,
        'kg_weight': args.kg_weight,
        'long_tail_threshold': args.long_tail_threshold,
        'kg_embed_path': args.kg_embed_path if use_pretrained_kg else None,
        'freeze_kg_embeds': args.freeze_kg_embeds if use_pretrained_kg else False,

        # TuckER framework parameters
        'tucker_weight': args.tucker_weight,
        'tucker_core_size': args.tucker_core_size,
        'tucker_dropout': args.tucker_dropout,
        'use_tucker': args.use_tucker,
        
        # KG aggregation layer parameters
        'kg_n_layers': args.kg_n_layers,
        'kg_dropout': args.kg_dropout,
        
        # Contrastive learning parameters
        'cl_weight': args.cl_weight,
        'cl_temperature': args.cl_temperature,
        'cl_proj_dim': args.cl_proj_dim,
        
        # Fusion module parameters
        'fusion_hidden_size': args.fusion_hidden_size,
        'use_attention': args.use_attention,
        
        # Field mapping
        'field_separator': '\t',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'HEAD_ENTITY_ID_FIELD': 'head_id',
        'RELATION_ID_FIELD': 'relation_id',
        'TAIL_ENTITY_ID_FIELD': 'tail_id',
        'ENTITY_ID_FIELD': 'entity_id',
        'NEG_PREFIX': 'neg_',
        
        # Data type settings
        'field2type': {
            'user_id:token': 'int',
            'item_id:token': 'int',
            'head_id:token': 'int',
            'relation_id:token': 'int',
            'tail_id:token': 'int',
            'entity_id:token': 'int'
        },
        
        # Additional feature parameters
        'additional_feat_suffix': ['kg', 'link'],
        
        # Load column configuration
        'load_col': {
            'inter': ['user_id', 'item_id'],
            'kg': ['head_id', 'relation_id', 'tail_id'],
            'link': ['item_id', 'entity_id']
        }
    }
    
    # Create configuration
    config = Config(model=KGHybridGeoSVDGCN, config_dict=config_dict)
    
    # Initialize random seed
    init_seed(config['seed'], config['reproducibility'])
    
    # Try to manually preprocess data files - check type issues
    try:
        kg_file = os.path.join(args.dataset_path, args.dataset, f"{args.dataset}.kg")
        link_file = os.path.join(args.dataset_path, args.dataset, f"{args.dataset}.link")
        
        # Check if files exist and try to read to understand data types
        if os.path.exists(kg_file):
            logger.info(f"Checking knowledge graph file: {kg_file}")
            df_kg = pd.read_csv(kg_file, sep='\t')
            logger.info(f"KG file columns: {df_kg.columns.tolist()}")
            logger.info(f"KG file size: {len(df_kg)} rows x {len(df_kg.columns)} columns")
            logger.info(f"KG file data types:\n{df_kg.dtypes}")
            
            # Ensure ID columns are integers
            for col in ['head_id', 'relation_id', 'tail_id']:
                if col in df_kg.columns:
                    try:
                        df_kg[col] = df_kg[col].astype('int64')
                    except:
                        logger.warning(f"Column {col} cannot be converted to integer, trying to clean")
                        df_kg[col] = pd.to_numeric(df_kg[col], errors='coerce')
                        df_kg = df_kg.dropna(subset=[col])
                        df_kg[col] = df_kg[col].astype('int64')
            
            # Save fixed file
            temp_kg_file = kg_file + '.fixed'
            df_kg.to_csv(temp_kg_file, sep='\t', index=False)
            os.rename(temp_kg_file, kg_file)
            logger.info(f"Fixed and saved KG file: {kg_file}")
        
        if os.path.exists(link_file):
            logger.info(f"Checking link file: {link_file}")
            df_link = pd.read_csv(link_file, sep='\t')
            logger.info(f"Link file columns: {df_link.columns.tolist()}")
            logger.info(f"Link file size: {len(df_link)} rows")
            logger.info(f"Link file data types:\n{df_link.dtypes}")
            
            # Ensure ID columns are integers
            for col in ['item_id', 'entity_id']:
                if col in df_link.columns:
                    try:
                        df_link[col] = df_link[col].astype('int64')
                    except:
                        logger.warning(f"Column {col} cannot be converted to integer, trying to clean")
                        df_link[col] = pd.to_numeric(df_link[col], errors='coerce')
                        df_link = df_link.dropna(subset=[col])
                        df_link[col] = df_link[col].astype('int64')
            
            # Save fixed file
            temp_link_file = link_file + '.fixed'
            df_link.to_csv(temp_link_file, sep='\t', index=False)
            os.rename(temp_link_file, link_file)
            logger.info(f"Fixed and saved Link file: {link_file}")
    except Exception as e:
        logger.error(f"Error occurred while preprocessing data files: {e}")

    kg_file = os.path.join(args.dataset_path, args.dataset, f"{args.dataset}.kg")
    if os.path.exists(kg_file):
        logger.info(f"Reading knowledge graph file sample directly: {kg_file}")
        with open(kg_file, 'r') as f:
            header = f.readline().strip()
            logger.info(f"Header: {header}")
            for i in range(5):
                line = f.readline().strip()
                logger.info(f"Line {i+1}: {line}")
    
    # Create dataset
    logger.info("Starting dataset creation...")
    dataset = create_dataset(config)
    logger.info(f"Dataset creation completed, users: {dataset.user_num}, items: {dataset.item_num}")
    
    # Check knowledge graph data
    if hasattr(dataset, 'kg_feat') and dataset.kg_feat is not None:
        logger.info(f"Knowledge graph data loaded successfully! Number of triples: {len(dataset.kg_feat)}")
        # Check data types
        if isinstance(dataset.kg_feat, pd.DataFrame):
            logger.info(f"KG data types: \n{dataset.kg_feat.dtypes}")
        
        if hasattr(dataset, 'link_feat') and dataset.link_feat is not None:
            logger.info(f"Item-entity link data loaded successfully! Number of links: {len(dataset.link_feat)}")
            # Check data types
            if isinstance(dataset.link_feat, pd.DataFrame):
                logger.info(f"Link data types: \n{dataset.link_feat.dtypes}")
    else:
        logger.warning("Knowledge graph data not loaded!")
    
    # Add exception handling for safer data preparation
    try:
        # Dataset splitting
        logger.info("Starting data preparation...")
        print("dataset type:",type(dataset))
        train_data, valid_data, test_data = data_preparation(config, dataset)
        # print("train_data:",type
        logger.info(f"Data preparation completed, train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")
    except TypeError as e:
        logger.error(f"Type error occurred during data preparation: {e}")
        logger.info("Try modifying RecBole source code to fix this issue...")
        
        # Suggest solution for users to modify RecBole source code
        source_file = os.path.join(os.path.dirname(torch.__file__), "../site-packages/recbole/data/dataset/dataset.py")
        line_number = 2136  # Based on error message
        
        fix_message = f"""
        Please manually modify RecBole source code file:
        File: {source_file}
        Line number: ~{line_number}
        
        Before:
        new_data[k] = torch.LongTensor(value)
        
        After:
        try:
            new_data[k] = torch.LongTensor(value)
        except TypeError:
            print(f"Column {{k}} conversion error, type is {{type(value)}}, trying forced conversion...")
            value = pd.to_numeric(pd.Series(value), errors='coerce').fillna(0).astype(int).values
            new_data[k] = torch.LongTensor(value)
        
        Please re-run this script after modification.
        """
        logger.info(fix_message)
        return
    
    # Initialize model
    logger.info("Starting model initialization...")
    model = KGHybridGeoSVDGCN(config, dataset).to(config['device'])
    logger.info("Model initialization completed")

    # Add after model initialization is completed
    logger.info("Checking item-entity mapping coverage")
    total_items = model.n_items 
    mapped_items = len([k for k, v in model.item_to_entity.items() if v < model.n_entities - 1])
    logger.info(f"Total items: {total_items}, mapped items: {mapped_items}, coverage: {mapped_items/total_items:.2%}")
    
    # Check knowledge graph triples
    logger.info(f"Loaded triple count: {len(model.kg_triples)}")
    if len(model.kg_triples) > 0:
        # Show first 5 triple examples
        logger.info(f"Triple examples: {model.kg_triples[:5]}")
    
    # Check new architecture configuration
    logger.info("New architecture configuration:")
    logger.info(f"- Hyperbolic space curvature: {config['curvature']}")
    logger.info(f"- Hyperbolic space weight: {config['hyperbolic_ratio']}")
    logger.info(f"- Number of GNN layers: {config['num_layers']}")
    
    # Check KG aggregation configuration
    logger.info("Knowledge graph aggregation configuration:")
    logger.info(f"- Number of KG aggregation layers: {config['kg_n_layers']}")
    logger.info(f"- KG dropout rate: {config['kg_dropout']}")
    
    # Check contrastive learning configuration
    logger.info("Contrastive learning configuration:")
    logger.info(f"- Temperature parameter: {config['cl_temperature']}")
    logger.info(f"- Loss weight: {config['cl_weight']}")
    logger.info(f"- Projection dimension: {config['cl_proj_dim']}")
    
    # Check fusion configuration
    logger.info("Fusion layer configuration:")
    logger.info(f"- Fusion hidden dimension: {config['fusion_hidden_size']}")
    logger.info(f"- Use attention: {config['use_attention']}")

    logger.info("TuckER framework configuration:")
    logger.info(f"- TuckER enabled: {config['use_tucker']}")
    logger.info(f"- TuckER scoring weight: {config['tucker_weight']}")
    logger.info(f"- Core tensor dimension: {config['tucker_core_size']}")
    logger.info(f"- Dropout rate: {config['tucker_dropout']}")
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}, trainable parameters: {trainable_params:,}")
    
    # Get trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # Model training
    logger.info("Starting model training...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    
    # Model evaluation
    logger.info("Starting model evaluation...")
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )
    
    # Print results
    logger.info(set_color("Best validation result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Test result", "yellow") + f": {test_result}")
    
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_score_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    main()