import argparse
import configparser
import preprocess.load_data as preprocess
from model.GPro_LLM import *
from copy import deepcopy
from utils import *
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np
import os

# Set random seed
randomSeed = 202408
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)


def train_stage1(model, dl_train, dl_val, config, device, params_path):
    """
    Stage 1: Pretrain the backbone network
    """
    print("=" * 60)
    print("Stage 1: Pretraining Backbone Network")
    print("=" * 60)
    
    max_epochs = config['max_epochs_stage1']
    learning_rate = config['learning_rate']
    regularization = config['regularization']
    patience = config['patience']
    display_step = config['display_step']
    downstream = config['downstream']
    
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate, amsgrad=True)
    
    best_hit20 = -np.inf
    best_model = deepcopy(model.state_dict())
    best_epoch = -1
    impatient = 0
    start = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        batch_cnt = 0
        epoch_loss = 0
        
        for batch in tqdm(dl_train, desc=f"Epoch {epoch+1}/{max_epochs}"):
            opt.zero_grad()
            batch_cnt += 1
            
            if batch.X_all_loc.shape[1] >= 700:
                continue
            
            loss, top_k_pred, _ = model(batch, mode='train', downstream=downstream)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if downstream != 'TPP':
                val_loss, hit_ratio_val, mrr_val = get_s_baselines_total_loss_s_for_MobilityLLM_DOWN(
                    dl_val, model, downstream=downstream
                )
                
                if (hit_ratio_val[19] - best_hit20) < 1e-4:
                    impatient += 1
                    if best_hit20 < hit_ratio_val[19]:
                        best_hit20 = hit_ratio_val[19]
                        best_model = deepcopy(model.state_dict())
                        best_epoch = epoch
                else:
                    best_hit20 = hit_ratio_val[19]
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
                    impatient = 0
                
                if epoch % display_step == 0:
                    print(f'Epoch {epoch+1:4d}, train_loss={epoch_loss/batch_cnt:.4f}, '
                          f'val_loss={val_loss:.4f}, val_hit_20={hit_ratio_val[19]:.4f}, '
                          f'best_hit_20={best_hit20:.4f}')
                
                if impatient >= patience:
                    print(f'Early stopping at epoch {epoch+1}, best epoch at {best_epoch+1}')
                    break
    
    # Save best model
    stage1_model_path = os.path.join(params_path, 'stage1_pretrained.model')
    torch.save(best_model, stage1_model_path)
    print(f"Stage 1 training completed. Best epoch: {best_epoch+1}")
    print(f"Pretrained model saved to {stage1_model_path}")
    print(f"Training time: {time.time() - start:.2f}s")
    
    # Load best model for representation extraction
    model.load_state_dict(best_model)
    
    return model


def extract_representations_stage1(model, dl_train, device, save_dir):
    """
    Extract H and V representations from training data using pretrained model
    """
    print("=" * 60)
    print("Extracting Representations for Prototype Generation")
    print("=" * 60)
    
    model.eval()
    all_H = []
    all_V = []
    
    with torch.no_grad():
        for batch in tqdm(dl_train, desc="Extracting"):
            H, V, _, _ = model.extract_representations(batch)
            all_H.append(H.cpu().numpy())
            all_V.append(V.cpu().numpy())
    
    # Concatenate all representations
    all_H = np.concatenate(all_H, axis=0)
    all_V = np.concatenate(all_V, axis=0)
    
    # Save representations
    rep_path = os.path.join(save_dir, 'representations.pkl')
    rep_dict = {'H': all_H, 'V': all_V}
    with open(rep_path, 'wb') as f:
        pickle.dump(rep_dict, f)
    
    print(f"Saved representations to {rep_path}")
    print(f"H shape: {all_H.shape}, V shape: {all_V.shape}")
    
    return all_H, all_V


def generate_prototypes(H, V, num_prototypes, save_path, method='kmeans'):
    """
    Generate global prototypes using various clustering methods
    
    Args:
        H: H representations (N, hidden_size)
        V: V representations (N, hidden_size)
        num_prototypes: Number of prototypes/clusters
        save_path: Path to save prototypes
        method: Clustering method ('kmeans', 'kmeans++', 'spectral', 'dbscan', 'gmm')
    """
    print("=" * 60)
    print(f"Generating Global Prototypes using {method.upper()}")
    print("=" * 60)
    
    from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"Clustering {V.shape[0]} samples...")
    
    # Perform clustering on V (preference space)
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=num_prototypes, init='random', random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(V)
        centers_V = clusterer.cluster_centers_
        
    elif method == 'kmeans++':
        clusterer = KMeans(n_clusters=num_prototypes, init='k-means++', random_state=42, n_init=50)
        cluster_labels = clusterer.fit_predict(V)
        centers_V = clusterer.cluster_centers_
        
    elif method == 'spectral':
        clusterer = SpectralClustering(
            n_clusters=num_prototypes, 
            affinity='nearest_neighbors',
            n_neighbors=50,
            assign_labels='kmeans',
            random_state=42
        )
        cluster_labels = clusterer.fit_predict(V)
        centers_V = np.zeros((num_prototypes, V.shape[1]))
        for i in range(num_prototypes):
            mask = cluster_labels == i
            if mask.sum() > 0:
                centers_V[i] = V[mask].mean(axis=0)
            else:
                centers_V[i] = V.mean(axis=0)
                
    elif method == 'dbscan':
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=20)
        neigh.fit(V[:min(5000, len(V))])
        distances, _ = neigh.kneighbors(V[:min(5000, len(V))])
        k_dist = np.sort(distances[:, 19])
        eps = k_dist[int(len(k_dist) * 0.9)]
        
        clusterer = DBSCAN(eps=eps, min_samples=10)
        cluster_labels = clusterer.fit_predict(V)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"DBSCAN found {n_clusters} clusters (eps={eps:.4f})")
        
        centers_V = np.zeros((n_clusters, V.shape[1]))
        for i in range(n_clusters):
            mask = cluster_labels == i
            if mask.sum() > 0:
                centers_V[i] = V[mask].mean(axis=0)
                
        if -1 in cluster_labels:
            noise_mask = cluster_labels == -1
            for idx in np.where(noise_mask)[0]:
                distances = np.linalg.norm(centers_V - V[idx], axis=1)
                cluster_labels[idx] = np.argmin(distances)
                
        num_prototypes = n_clusters
        
    elif method == 'gmm':
        clusterer = GaussianMixture(
            n_components=num_prototypes,
            covariance_type='full',
            random_state=42,
            max_iter=200,
            n_init=10
        )
        cluster_labels = clusterer.fit_predict(V)
        centers_V = clusterer.means_
        
        print(f"GMM converged: {clusterer.converged_}, n_iter: {clusterer.n_iter_}")
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Compute silhouette score if more than 1 cluster
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(V, cluster_labels)
        print(f"Silhouette Score: {sil_score:.4f}")
    
    # prototype_V: cluster centers in preference space (nodes in hypergraph)
    prototype_V = centers_V
    
    # prototype_H: cluster-wise mean of H (for reference)
    prototype_H = np.zeros((num_prototypes, H.shape[1]))
    for i in range(num_prototypes):
        mask = cluster_labels == i
        if mask.sum() > 0:
            prototype_H[i] = H[mask].mean(axis=0)
        else:
            prototype_H[i] = H.mean(axis=0)
    
    # Compute inter-cluster neighbors for hypergraph construction
    # Each prototype connects to its 2 most similar prototypes (cosine similarity)
    proto_sim = cosine_similarity(prototype_V)  # [K, K]
    np.fill_diagonal(proto_sim, -np.inf)
    inter_cluster_neighbors = np.argsort(proto_sim, axis=1)[:, -2:]  # [K, 2]
    print(f"Inter-cluster neighbors: {inter_cluster_neighbors.tolist()}")
    
    # Save prototypes with inter-cluster structure
    prototype_dict = {
        'prototype_H': prototype_H,
        'prototype_V': prototype_V,
        'cluster_labels': cluster_labels,
        'inter_cluster_neighbors': inter_cluster_neighbors,
        'clustering_method': method
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(prototype_dict, f)
    
    print(f"Saved prototypes to {save_path}")
    print(f"Prototype H shape: {prototype_H.shape}")
    print(f"Prototype V shape: {prototype_V.shape}")
    
    # Print cluster statistics
    for i in range(num_prototypes):
        count = (cluster_labels == i).sum()
        print(f"  Prototype {i}: {count} samples ({100*count/len(cluster_labels):.2f}%)")
    
    return prototype_H, prototype_V


def train_stage2(model, dl_train, dl_val, config, device, params_path):
    """
    Stage 2: Finetune with hypergraph prototype enhancement
    """
    print("=" * 60)
    print("Stage 2: Finetuning with Hypergraph Prototype Enhancement")
    print("=" * 60)
    
    max_epochs = config['max_epochs_stage2']
    learning_rate = config['learning_rate_stage2']
    regularization = config['regularization']
    patience = config['patience']
    display_step = config['display_step']
    downstream = config['downstream']
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          weight_decay=regularization, lr=learning_rate, amsgrad=True)
    
    best_hit20 = -np.inf
    best_model = deepcopy(model.state_dict())
    best_epoch = -1
    impatient = 0
    start = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        batch_cnt = 0
        epoch_loss = 0
        
        for batch in tqdm(dl_train, desc=f"Epoch {epoch+1}/{max_epochs}"):
            opt.zero_grad()
            batch_cnt += 1
            
            if batch.X_all_loc.shape[1] >= 700:
                continue
            
            loss, top_k_pred, _ = model(batch, mode='train', downstream=downstream)
            loss.backward()
            clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if downstream != 'TPP':
                val_loss, hit_ratio_val, mrr_val = get_s_baselines_total_loss_s_for_MobilityLLM_DOWN(
                    dl_val, model, downstream=downstream
                )
                
                if (hit_ratio_val[19] - best_hit20) < 1e-4:
                    impatient += 1
                    if best_hit20 < hit_ratio_val[19]:
                        best_hit20 = hit_ratio_val[19]
                        best_model = deepcopy(model.state_dict())
                        best_epoch = epoch
                else:
                    best_hit20 = hit_ratio_val[19]
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
                    impatient = 0
                
                if epoch % display_step == 0:
                    print(f'Epoch {epoch+1:4d}, train_loss={epoch_loss/batch_cnt:.4f}, '
                          f'val_loss={val_loss:.4f}, val_mrr={mrr_val:.4f}, '
                          f'val_hit_1={hit_ratio_val[0]:.4f}, val_hit_20={hit_ratio_val[19]:.4f}, '
                          f'best_hit_20={best_hit20:.4f}')
                
                if impatient >= patience:
                    print(f'Early stopping at epoch {epoch+1}, best epoch at {best_epoch+1}')
                    break
    
    # Save best model
    stage2_model_path = os.path.join(params_path, 'stage2_finetuned.model')
    torch.save(best_model, stage2_model_path)
    print(f"Stage 2 training completed. Best epoch: {best_epoch+1}")
    print(f"Finetuned model saved to {stage2_model_path}")
    print(f"Training time: {time.time() - start:.2f}s")
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config/GPro_LLM_bkc_POI.conf', type=str,
                        help="configuration file path")
    parser.add_argument("--dataroot", default='data/', type=str,
                        help="data root directory")
    parser.add_argument("--model_class", default='TinyLlama-1_1B', type=str, 
                        help="model class")
    parser.add_argument("--device", default='0', type=str, help="GPU device")
    parser.add_argument("--stage", default='both', type=str, 
                        choices=['pretrain', 'extract', 'prototype', 'finetune', 'both'],
                        help="Training stage: pretrain, extract, prototype, finetune, or both")
    parser.add_argument("--clustering", default='kmeans++', type=str,
                        choices=['kmeans', 'kmeans++', 'spectral', 'dbscan', 'gmm'],
                        help="Clustering method for prototype generation")
    parser.add_argument("--num_hgnn_layers", default=2, type=int,
                        help="Number of HGNN convolution layers")
    args = parser.parse_args()
    clustering_method = args.clustering
    num_hgnn_layers = args.num_hgnn_layers
    
    # Load config
    config_file = args.config
    data_root = args.dataroot
    model_class = args.model_class
    device_id = args.device
    stage = args.stage
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    data_config = config['Data']
    training_config = config['Training']
    model_config = config['Model']
    gpro_config = config['GPro_LLM']
    
    # Setup device
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_id}" if USE_CUDA else "cpu")
    print(f"Using device: {device}")
    
    # Data config
    dataset_name = data_config['dataset_name']
    split_save = bool(int(data_config['split_save']))
    
    # Training config
    mode = training_config['mode'].strip()
    regularization = float(training_config['regularization'])
    learning_rate = float(training_config['learning_rate'])
    learning_rate_stage2 = float(training_config.get('learning_rate_stage2', learning_rate * 0.1))
    max_epochs_stage1 = int(training_config['max_epochs_stage1'])
    max_epochs_stage2 = int(training_config['max_epochs_stage2'])
    display_step = int(training_config['display_step'])
    patience = int(training_config['patience'])
    batch_size = int(training_config['batch_size'])
    
    # Model config
    loc_emb_size = int(model_config['loc_emb_size'])
    tim_emb_size = int(model_config['tim_emb_size'])
    user_emb_size = int(model_config['user_emb_size'])
    hidden_size = int(model_config['hidden_size'])
    category_size = int(model_config['category_size'])
    geohash_size = int(model_config['geohash_size'])
    learnable_param_size = int(model_config.get('learnable_param_size', 1))
    downstream = model_config['downstream']
    
    # GPro-LLM config
    num_prototypes = int(gpro_config['num_prototypes'])
    num_hgnn_layers = int(gpro_config.get('num_hgnn_layers', 2))
    hgnn_dropout = float(gpro_config.get('hgnn_dropout', 0.5))
    gate_hidden_size = int(gpro_config.get('gate_hidden_size', 128))
    
    # Create directories
    params_path = os.path.join('experiments', f'GPro_LLM_{dataset_name}')
    os.makedirs(params_path, exist_ok=True)
    
    prototype_path = os.path.join(params_path, 'prototypes.pkl')
    rep_path = os.path.join(params_path, 'representations.pkl')
    
    # Load data
    print("Loading dataset...")
    data_train, data_val, data_test, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt, category_vector = \
        preprocess.load_dataset_for_MobilityLLM(
            dataset_name, save_split=split_save, data_root=data_root, device=device
        )
    
    collate = preprocess.collate_session_based
    dl_train = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    dl_val = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True
    )
    dl_test = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True
    )
    
    # Get timestamp statistics for TPP
    trainY_tau_mean, trainY_tau_std = data_train.get_tau_log_mean_std_Y()
    
    # ==================== Stage 1: Pretrain ====================
    if stage in ['pretrain', 'both']:
        print("\n" + "=" * 60)
        print("STARTING STAGE 1: PRETRAINING")
        print("=" * 60)
        
        general_config = GPro_LLM_ModelConfig(
            loc_size=int(data_train.venue_cnt),
            tim_size=48,
            uid_size=int(data_train.user_cnt),
            tim_emb_size=tim_emb_size,
            loc_emb_size=loc_emb_size,
            hidden_size=hidden_size,
            user_emb_size=user_emb_size,
            model_class=model_class,
            device=device,
            geohash_size=geohash_size,
            category_size=category_size,
            learnable_param_size=learnable_param_size,
            downstream=downstream,
            stage='pretrain',
            num_prototypes=num_prototypes,
            num_hgnn_layers=num_hgnn_layers,
            hgnn_dropout=hgnn_dropout
        )
        
        model = GPro_LLM(general_config).to(device)
        print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        train_config = {
            'max_epochs_stage1': max_epochs_stage1,
            'learning_rate': learning_rate,
            'regularization': regularization,
            'patience': patience,
            'display_step': display_step,
            'downstream': downstream
        }
        model = train_stage1(model, dl_train, dl_val, train_config, device, params_path)
        
        print("\nStage 1 completed!")
    
    # ==================== Extract Representations ====================
    if stage in ['extract', 'both']:
        print("\n" + "=" * 60)
        print("EXTRACTING REPRESENTATIONS")
        print("=" * 60)
        
        general_config = GPro_LLM_ModelConfig(
            loc_size=int(data_train.venue_cnt),
            tim_size=48,
            uid_size=int(data_train.user_cnt),
            tim_emb_size=tim_emb_size,
            loc_emb_size=loc_emb_size,
            hidden_size=hidden_size,
            user_emb_size=user_emb_size,
            model_class=model_class,
            device=device,
            geohash_size=geohash_size,
            category_size=category_size,
            learnable_param_size=learnable_param_size,
            downstream=downstream,
            stage='pretrain'
        )
        
        model = GPro_LLM(general_config).to(device)
        stage1_model_path = os.path.join(params_path, 'stage1_pretrained.model')
        model.load_state_dict(torch.load(stage1_model_path))
        print(f"Loaded pretrained model from {stage1_model_path}")
        
        H, V = extract_representations_stage1(model, dl_train, device, params_path)
    
    # ==================== Generate Prototypes ====================
    if stage in ['prototype', 'both']:
        print("\n" + "=" * 60)
        print("GENERATING GLOBAL PROTOTYPES")
        print("=" * 60)
        
        with open(rep_path, 'rb') as f:
            rep_dict = pickle.load(f)
        H, V = rep_dict['H'], rep_dict['V']
        
        prototype_H, prototype_V = generate_prototypes(
            H, V, num_prototypes, prototype_path, method=clustering_method
        )
    
    # ==================== Stage 2: Finetune ====================
    if stage in ['finetune', 'both']:
        print("\n" + "=" * 60)
        print("STARTING STAGE 2: FINETUNING WITH HYPERGRAPH")
        print("=" * 60)
        
        if not os.path.exists(prototype_path):
            raise FileNotFoundError(f"Prototypes not found at {prototype_path}. "
                                    f"Please run prototype generation stage first.")
        
        general_config = GPro_LLM_ModelConfig(
            loc_size=int(data_train.venue_cnt),
            tim_size=48,
            uid_size=int(data_train.user_cnt),
            tim_emb_size=tim_emb_size,
            loc_emb_size=loc_emb_size,
            hidden_size=hidden_size,
            user_emb_size=user_emb_size,
            model_class=model_class,
            device=device,
            geohash_size=geohash_size,
            category_size=category_size,
            learnable_param_size=learnable_param_size,
            downstream=downstream,
            stage='finetune',
            freeze_frontend=True,
            num_prototypes=num_prototypes,
            num_hgnn_layers=num_hgnn_layers,
            hgnn_dropout=hgnn_dropout,
            gate_hidden_size=gate_hidden_size,
            prototype_path=prototype_path
        )
        
        model = GPro_LLM(general_config).to(device)
        
        # Optionally load stage 1 weights for backend
        stage1_model_path = os.path.join(params_path, 'stage1_pretrained.model')
        if os.path.exists(stage1_model_path):
            model.load_state_dict(torch.load(stage1_model_path), strict=False)
            print(f"Loaded stage 1 weights from {stage1_model_path}")
        
        train_config = {
            'max_epochs_stage2': max_epochs_stage2,
            'learning_rate_stage2': learning_rate_stage2,
            'regularization': regularization,
            'patience': patience,
            'display_step': display_step,
            'downstream': downstream
        }
        model = train_stage2(model, dl_train, dl_val, train_config, device, params_path)
        
        print("\nStage 2 completed!")
    
    # ==================== Final Evaluation ====================
    if stage in ['finetune', 'both']:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        
        stage2_model_path = os.path.join(params_path, 'stage2_finetuned.model')
        model.load_state_dict(torch.load(stage2_model_path))
        model.eval()
        
        with torch.no_grad():
            if downstream != 'TPP':
                train_loss, train_hit, train_mrr = get_s_baselines_total_loss_s_for_MobilityLLM_DOWN(
                    dl_train, model, downstream=downstream
                )
                val_loss, val_hit, val_mrr = get_s_baselines_total_loss_s_for_MobilityLLM_DOWN(
                    dl_val, model, downstream=downstream
                )
                test_loss, test_hit, test_mrr = get_s_baselines_total_loss_s_for_MobilityLLM_DOWN(
                    dl_test, model, downstream=downstream
                )
                
                print('\nDataset\t loss\t hit_1\t hit_3\t hit_5\t hit_10\t hit_20\t MRR\t')
                print('Train:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t' % (
                    train_loss, train_hit[0], train_hit[2], train_hit[4], train_hit[9], train_hit[19], train_mrr))
                print('Val:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t' % (
                    val_loss, val_hit[0], val_hit[2], val_hit[4], val_hit[9], val_hit[19], val_mrr))
                print('Test:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t' % (
                    test_loss, test_hit[0], test_hit[2], test_hit[4], test_hit[9], test_hit[19], test_mrr))


if __name__ == '__main__':
    main()
