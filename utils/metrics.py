"""
Metrics utilities for federated autoencoder evaluation
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F


def calculate_reconstruction_metrics(model, data_loader, device):
    """
    Calculate reconstruction-based metrics for autoencoder
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing reconstruction metrics
    """
    model.eval()
    
    reconstruction_errors = []
    total_loss = 0.0
    num_samples = 0
    
    criterion = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            reconstructed, latent = model(data)
            
            # Calculate per-sample reconstruction error
            batch_errors = criterion(reconstructed, data).view(data.size(0), -1).mean(dim=1)
            reconstruction_errors.extend(batch_errors.cpu().numpy())
            
            # Calculate total loss
            total_loss += F.mse_loss(reconstructed, data).item() * data.size(0)
            num_samples += data.size(0)
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Calculate reconstruction statistics
    avg_loss = total_loss / num_samples
    avg_reconstruction_error = np.mean(reconstruction_errors)
    std_reconstruction_error = np.std(reconstruction_errors)
    median_reconstruction_error = np.median(reconstruction_errors)
    
    # Calculate reconstruction quality metrics (lower is better)
    # Use percentiles to define "good" vs "poor" reconstruction
    percentile_25 = np.percentile(reconstruction_errors, 25)
    percentile_75 = np.percentile(reconstruction_errors, 75)
    
    # Define good reconstruction as bottom 25% of errors
    good_reconstruction = reconstruction_errors <= percentile_25
    poor_reconstruction = reconstruction_errors >= percentile_75
    
    # For autoencoder evaluation, we'll use a more meaningful approach:
    # Compare reconstruction quality across different error thresholds
    
    # Method 1: Use median as threshold (more stable than mean)
    median_threshold = np.median(reconstruction_errors)
    better_than_median = (reconstruction_errors <= median_threshold).astype(int)
    
    # Method 2: Use a stricter threshold (25th percentile) for "good" reconstructions
    strict_threshold = np.percentile(reconstruction_errors, 25)
    high_quality = (reconstruction_errors <= strict_threshold).astype(int)
    
    # Calculate "precision" as: how many predicted good are actually good
    # This is more like "consistency" - if we predict good, how often is it actually good?
    
    # For binary classification metrics, we need to define what we're classifying
    # Let's classify: "Is this reconstruction better than average?"
    
    # Ground truth: better than median (50% of samples)
    true_better_than_median = better_than_median
    
    # Prediction: better than 40th percentile (slightly more lenient)
    prediction_threshold = np.percentile(reconstruction_errors, 40)
    predicted_better = (reconstruction_errors <= prediction_threshold).astype(int)
    
    # Calculate metrics - but note these are somewhat artificial for autoencoders
    accuracy = accuracy_score(true_better_than_median, predicted_better)
    precision = precision_score(true_better_than_median, predicted_better, average='binary', zero_division=0)
    recall = recall_score(true_better_than_median, predicted_better, average='binary', zero_division=0)
    f1 = f1_score(true_better_than_median, predicted_better, average='binary', zero_division=0)
    
    # Add a note about what these metrics mean
    classification_note = (
        "Note: Classification metrics compare 40th vs 50th percentile thresholds. "
        "Perfect scores may indicate threshold alignment rather than model quality."
    )
    
    return {
        'loss': avg_loss,
        'reconstruction_error': avg_reconstruction_error,
        'reconstruction_std': std_reconstruction_error,
        'reconstruction_median': median_reconstruction_error,
        'reconstruction_25th': percentile_25,
        'reconstruction_75th': percentile_75,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_samples': num_samples,
        'good_reconstructions': np.sum(good_reconstruction),
        'poor_reconstructions': np.sum(poor_reconstruction),
        'classification_note': classification_note,
        'better_than_median_count': np.sum(better_than_median),
        'high_quality_count': np.sum(high_quality)
    }


def calculate_latent_classification_metrics(model, data_loader, device):
    """
    Calculate classification metrics using latent space representations
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing latent space metrics
    """
    model.eval()
    
    latent_features = []
    true_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Get latent representations
            _, latent = model(data)
            latent_features.append(latent.cpu().numpy())
            true_labels.extend(labels)
    
    # Combine all latent features
    latent_features = np.vstack(latent_features)
    
    # Encode string labels to numeric
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(true_labels)
    unique_labels = np.unique(numeric_labels)
    n_classes = len(unique_labels)
    
    print(f"   🔍 Latent analysis: {n_classes} unique classes found")
    print(f"   📊 Class distribution: {dict(zip(label_encoder.classes_, np.bincount(numeric_labels)))}")
    print(f"   🎯 Latent features shape: {latent_features.shape}")
    print(f"   📐 Latent mean: {np.mean(latent_features, axis=0)[:5]}...")  # Show first 5 dims
    print(f"   📏 Latent std: {np.std(latent_features, axis=0)[:5]}...")   # Show first 5 dims
    
    # Check if latent features are all the same (indicating untrained model)
    latent_variance = np.var(latent_features, axis=0).mean()
    print(f"   📊 Average latent variance: {latent_variance:.6f}")
    
    if latent_variance < 1e-6:
        print(f"   ⚠️ WARNING: Very low latent variance - model may not be trained properly!")
        return {
            'latent_accuracy': 0.0,
            'latent_precision': 0.0,
            'latent_recall': 0.0,
            'latent_f1_score': 0.0,
            'silhouette_score': 0.0,
            'n_clusters': n_classes,
            'latent_dim': latent_features.shape[1],
            'cluster_quality': 'untrained_model'
        }
    
    if n_classes == 1:
        # Single class case - no meaningful classification possible
        return {
            'latent_accuracy': 0.0,  # No classification possible
            'latent_precision': 0.0,
            'latent_recall': 0.0,
            'latent_f1_score': 0.0,
            'silhouette_score': 0.0,  # No clustering possible
            'n_clusters': n_classes,
            'latent_dim': latent_features.shape[1],
            'cluster_quality': 'single_class'
        }
    
    # Perform clustering
    try:
        # Use the actual number of classes for clustering
        print(f"   🔄 Running K-means clustering with {n_classes} clusters...")
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        cluster_predictions = kmeans.fit_predict(latent_features)
        
        unique_clusters = len(set(cluster_predictions))
        print(f"   📊 Found {unique_clusters} unique cluster assignments")
        print(f"   🎯 Cluster distribution: {dict(zip(*np.unique(cluster_predictions, return_counts=True)))}")
        
        # Calculate silhouette score for cluster quality
        if n_classes > 1 and len(set(cluster_predictions)) > 1:
            silhouette = silhouette_score(latent_features, cluster_predictions)
            print(f"   📏 Silhouette score: {silhouette:.4f}")
        else:
            silhouette = 0.0
            print(f"   ⚠️  Cannot calculate silhouette score: insufficient clusters")
        
        # For meaningful classification metrics, we need to align cluster labels with true labels
        # This is a complex problem, so we'll use a simpler approach:
        # Calculate how well the clustering separates the true classes
        
        # Method 1: Direct comparison (may not be meaningful due to label permutation)
        accuracy_direct = accuracy_score(numeric_labels, cluster_predictions)
        
        # Method 2: Best possible alignment between clusters and true labels
        try:
            from scipy.optimize import linear_sum_assignment
            from sklearn.metrics import confusion_matrix
            
            # Create confusion matrix
            cm = confusion_matrix(numeric_labels, cluster_predictions)
            
            # Find best assignment using Hungarian algorithm
            if cm.shape[0] == cm.shape[1]:  # Same number of clusters and classes
                row_ind, col_ind = linear_sum_assignment(-cm)  # Negative for maximization
                aligned_predictions = np.zeros_like(cluster_predictions)
                for i, j in zip(row_ind, col_ind):
                    aligned_predictions[cluster_predictions == j] = i
                
                # Calculate metrics with aligned labels
                accuracy = accuracy_score(numeric_labels, aligned_predictions)
                precision = precision_score(numeric_labels, aligned_predictions, average='weighted', zero_division=0)
                recall = recall_score(numeric_labels, aligned_predictions, average='weighted', zero_division=0)
                f1 = f1_score(numeric_labels, aligned_predictions, average='weighted', zero_division=0)
                cluster_quality = 'aligned'
                print(f"   ✅ Aligned clustering metrics calculated: acc={accuracy:.4f}, f1={f1:.4f}")
            else:
                # Different number of clusters and classes - use direct comparison
                accuracy = accuracy_direct
                precision = precision_score(numeric_labels, cluster_predictions, average='weighted', zero_division=0)
                recall = recall_score(numeric_labels, cluster_predictions, average='weighted', zero_division=0)
                f1 = f1_score(numeric_labels, cluster_predictions, average='weighted', zero_division=0)
                cluster_quality = 'unaligned'
                print(f"   ⚠️  Unaligned clustering metrics: acc={accuracy:.4f}, f1={f1:.4f}")
        except ImportError:
            print(f"   ⚠️ scipy not available, using direct comparison")
            # Fallback to direct comparison without alignment
            accuracy = accuracy_direct
            precision = precision_score(numeric_labels, cluster_predictions, average='weighted', zero_division=0)
            recall = recall_score(numeric_labels, cluster_predictions, average='weighted', zero_division=0)
            f1 = f1_score(numeric_labels, cluster_predictions, average='weighted', zero_division=0)
            cluster_quality = 'direct'
            print(f"   📊 Direct clustering metrics: acc={accuracy:.4f}, f1={f1:.4f}")
        
    except Exception as e:
        print(f"   ⚠️ Clustering failed: {e}")
        accuracy = precision = recall = f1 = silhouette = 0.0
        cluster_quality = 'failed'
    
    return {
        'latent_accuracy': accuracy,
        'latent_precision': precision,
        'latent_recall': recall,
        'latent_f1_score': f1,
        'silhouette_score': silhouette,
        'n_clusters': n_classes,
        'latent_dim': latent_features.shape[1],
        'cluster_quality': cluster_quality
    }


def calculate_comprehensive_metrics(model, data_loader, device):
    """
    Calculate comprehensive metrics for autoencoder evaluation
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing all metrics
    """
    print(f"   🔄 Calculating reconstruction metrics...")
    recon_metrics = calculate_reconstruction_metrics(model, data_loader, device)
    
    print(f"   🧠 Calculating latent space metrics...")
    latent_metrics = calculate_latent_classification_metrics(model, data_loader, device)
    
    # Combine all metrics
    comprehensive_metrics = {
        **recon_metrics,
        **latent_metrics
    }
    
    return comprehensive_metrics


def print_metrics_summary(metrics, subversion_name):
    """Print a formatted summary of metrics"""
    print(f"\n📊 Metrics Summary for {subversion_name}:")
    print("=" * 60)
    
    # Reconstruction metrics
    print(f"🔄 Reconstruction Loss:     {metrics['loss']:.6f}")
    print(f"📏 Reconstruction Error:    {metrics['reconstruction_error']:.6f} ± {metrics['reconstruction_std']:.6f}")
    print(f"📊 Reconstruction Median:   {metrics['reconstruction_median']:.6f}")
    print(f"📈 25th/75th Percentile:    {metrics['reconstruction_25th']:.6f} / {metrics['reconstruction_75th']:.6f}")
    print(f"✅ Good Reconstructions:    {metrics['good_reconstructions']}/{metrics['num_samples']} ({100*metrics['good_reconstructions']/metrics['num_samples']:.1f}%)")
    print(f"❌ Poor Reconstructions:    {metrics['poor_reconstructions']}/{metrics['num_samples']} ({100*metrics['poor_reconstructions']/metrics['num_samples']:.1f}%)")
    print(f"🎯 Better than Median:     {metrics['better_than_median_count']}/{metrics['num_samples']} ({100*metrics['better_than_median_count']/metrics['num_samples']:.1f}%)")
    print(f"⭐ High Quality (top 25%):  {metrics['high_quality_count']}/{metrics['num_samples']} ({100*metrics['high_quality_count']/metrics['num_samples']:.1f}%)")
    
    # Classification metrics with explanation
    print(f"\n🎯 Reconstruction Classification Metrics:")
    print(f"   Accuracy:               {metrics['accuracy']:.4f}")
    print(f"   Precision:              {metrics['precision']:.4f}")
    print(f"   Recall:                 {metrics['recall']:.4f}")
    print(f"   F1-Score:               {metrics['f1_score']:.4f}")
    print(f"   ℹ️  {metrics['classification_note']}")
    
    # Latent space metrics
    print(f"\n🧠 Latent Space Analysis:")
    print(f"   Latent Accuracy:        {metrics['latent_accuracy']:.4f}")
    print(f"   Latent Precision:       {metrics['latent_precision']:.4f}")
    print(f"   Latent Recall:          {metrics['latent_recall']:.4f}")
    print(f"   Latent F1-Score:        {metrics['latent_f1_score']:.4f}")
    print(f"   Silhouette Score:       {metrics['silhouette_score']:.4f}")
    print(f"   Clusters Found:         {metrics['n_clusters']}")
    print(f"   Latent Dimension:       {metrics['latent_dim']}")
    print(f"   Cluster Quality:        {metrics['cluster_quality']}")
    
    # Interpretation guide
    print(f"\n📖 Interpretation Guide:")
    print(f"   • Reconstruction Loss: Lower = better image reconstruction")
    print(f"   • Latent Accuracy: How well clustering separates kidney stone classes")
    print(f"   • Silhouette Score: Quality of latent space clustering (higher = better)")
    print(f"   • Perfect precision (1.0) in reconstruction metrics may indicate")
    print(f"     threshold alignment rather than exceptional model performance")
    
    print(f"\n📦 Total Samples:          {metrics['num_samples']}")
    print("=" * 60) 