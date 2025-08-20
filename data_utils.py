# data_utils.py
"""
Utilities for data processing, caching, and preparation.
"""
import pickle
import numpy as np
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *
from collision_model import CollisionDataset

def save_collision_data(collision_data, filename):
    """
    Save collision data to file.
    Args:
        collision_data (list): List of dictionaries containing collision pair data.
        filename (str): The filename to save the data to.
    Returns:
        None
    """
    with open(filename, 'wb') as f:
        pickle.dump(collision_data, f)
    print(f"Saved {len(collision_data)} collision pairs to {filename}")

def load_collision_data(filename):
    """
    Load collision data from file.
    Args:
        filename (str): The filename to load the data from.
    Returns:
        collision_data (list): List of dictionaries containing collision pair data.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            collision_data = pickle.load(f)
        print(f"Loaded {len(collision_data)} collision pairs from {filename}")
        return collision_data
    else:
        print(f"File {filename} not found")
        return None

def deduplicate_satellites(df):
    """
    Remove duplicate satellites, keeping the first occurrence.
    Args:
        df (pd.DataFrame): DataFrame containing satellite features.
    Returns:
        df_unique (pd.DataFrame): Deduplicated DataFrame.
    """
    print(f"Before deduplication: {len(df)} satellites")
    df_unique = df.drop_duplicates(subset=['name'], keep='first')
    print(f"After deduplication: {len(df_unique)} satellites")
    return df_unique

def prepare_training_data(collision_data):
    """Prepare features and targets for multi-task learning with log-space probabilities"""
    
    features = []
    collision_probs = []
    risk_classes = []
    
    for pair in collision_data:
        # Features: skip collision_prob (index 0), take rest
        features.append(pair['features'][1:])
        
        # Targets
        collision_probs.append(pair['features'][0])
        risk_classes.append(pair['risk_class'])
    
    features = np.array(features)
    collision_probs = np.array(collision_probs)
    risk_classes = np.array(risk_classes)
    
    # Convert probabilities to log10 space
    # Add small epsilon to avoid log(0)
    epsilon = 1e-12
    collision_probs_safe = collision_probs + epsilon
    log_collision_probs = np.log10(collision_probs_safe)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Collision probability targets shape: {collision_probs.shape}")
    print(f"Log collision probability targets shape: {log_collision_probs.shape}")
    print(f"Risk class targets shape: {risk_classes.shape}")
    
    return features, log_collision_probs, risk_classes

def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset.
    Args:
        labels (np.ndarray): Array of labels for the dataset.
    Returns:
        weights (torch.FloatTensor): Class weights for the dataset.
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for class_idx in sorted(class_counts.keys()):
        weight = total_samples / (len(class_counts) * class_counts[class_idx])
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def load_preprocessed_data():
    """Load preprocessed collision data and convert to log-space"""
    print("Loading preprocessed data...")
    
    # Load cached collision data
    collision_data = load_collision_data(COLLISION_DATA_CACHE)
    if collision_data is None:
        raise FileNotFoundError("No preprocessed data found. Run preprocess.py first!")
    
    # Use existing prepare_training_data function
    features, log_collision_probs, risk_classes = prepare_training_data(collision_data)
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    print(f"Risk distribution: {np.bincount(risk_classes)}")

    return features, log_collision_probs, risk_classes  # Return log probs instead!

def create_data_splits(features, collision_probs, risk_classes, 
                      test_size=0.2, val_size=0.2, random_state=42):
    """Create train/val/test splits with stratification"""
    
    # First split: train+val vs test (stratified by risk class)
    X_temp, X_test, y_prob_temp, y_prob_test, y_class_temp, y_class_test = train_test_split(
        features, collision_probs, risk_classes,
        test_size=test_size, random_state=random_state, stratify=risk_classes
    )
    
    # Second split: train vs val (stratified by risk class)
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_prob_train, y_prob_val, y_class_train, y_class_val = train_test_split(
        X_temp, y_prob_temp, y_class_temp,
        test_size=val_size_adjusted, random_state=random_state, stratify=y_class_temp
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    print(f"Train risk distribution: {np.bincount(y_class_train)}")
    print(f"Val risk distribution: {np.bincount(y_class_val)}")
    print(f"Test risk distribution: {np.bincount(y_class_test)}")
    
    return (X_train, X_val, X_test, 
            y_prob_train, y_prob_val, y_prob_test,
            y_class_train, y_class_val, y_class_test)

def create_data_loaders(X_train, X_val, X_test, 
                       y_prob_train, y_prob_val, y_prob_test,
                       y_class_train, y_class_val, y_class_test,
                       batch_size=64):
    """Create data loaders with feature scaling"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = CollisionDataset(X_train_scaled, y_prob_train, y_class_train)
    val_dataset = CollisionDataset(X_val_scaled, y_prob_val, y_class_val)
    test_dataset = CollisionDataset(X_test_scaled, y_prob_test, y_class_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler