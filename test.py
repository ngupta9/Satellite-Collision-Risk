# test.py
"""
Standalone test script for evaluating trained collision risk model
"""
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, 
                           mean_squared_error, r2_score)
from torch.utils.data import DataLoader
import os

# Import your modules
from collision_model import CollisionRiskModel, CollisionDataset
from data_utils import load_preprocessed_data, create_data_splits
from config import *
from visualization import plot_test_results

def get_test_split():
    """
    Get the exact test split used in training
    """
    # Use existing data loading function
    features, log_collision_probs, risk_classes = load_preprocessed_data()
    
    (_, _, X_test, 
     _, _, y_log_prob_test,
     _, _, y_class_test) = create_data_splits(
        features, log_collision_probs, risk_classes,
        test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    return X_test, y_log_prob_test, y_class_test

def evaluate_model():
    """
    Main evaluation function
    """
    print("="*60)
    print("COLLISION RISK MODEL - TEST EVALUATION")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first!")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    X_test, y_log_prob_test, y_class_test = get_test_split()
    
    # Load model and scaler
    print("\nLoading trained model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Initialize model
    model = CollisionRiskModel(input_dim=7, hidden_dim=128, num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load scaler and transform test data
    scaler = checkpoint['scaler']
    X_test_scaled = scaler.transform(X_test)
    
    # Create test dataset and loader
    test_dataset = CollisionDataset(X_test_scaled, y_log_prob_test, y_class_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Evaluate
    print("\nRunning evaluation...")
    all_prob_preds = []
    all_log_prob_preds = []
    all_log_prob_targets = []
    all_class_preds = []
    all_class_targets = []
    
    with torch.no_grad():
        for features, log_prob_targets, class_targets in test_loader:
            features = features.to(device)
            log_prob_targets = log_prob_targets.to(device)
            class_targets = class_targets.to(device)
            
            # Get predictions
            prob_pred, class_pred, log_prob_pred = model(features)
            
            # Store predictions
            all_prob_preds.extend(prob_pred.cpu().numpy())
            all_log_prob_preds.extend(log_prob_pred.cpu().numpy())
            all_log_prob_targets.extend(log_prob_targets.cpu().numpy())
            
            # Class predictions
            class_pred_labels = torch.argmax(class_pred, dim=1).cpu().numpy()
            all_class_preds.extend(class_pred_labels)
            all_class_targets.extend(class_targets.cpu().numpy())
    
    # Convert to arrays
    prob_preds = np.array(all_prob_preds)
    log_prob_preds = np.array(all_log_prob_preds)
    log_prob_targets = np.array(all_log_prob_targets)
    class_preds = np.array(all_class_preds)
    class_targets = np.array(all_class_targets)
    
    # Convert log targets to linear for some metrics
    prob_targets = np.power(10, log_prob_targets)
    
    # Print metrics
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    # Regression metrics
    print("\nRegression Performance (Collision Probability):")
    print("-" * 40)
    
    # Linear space
    mse_linear = mean_squared_error(prob_targets, prob_preds)
    r2_linear = r2_score(prob_targets, prob_preds)
    print(f"Linear Space:")
    print(f"  MSE: {mse_linear:.2e}")
    print(f"  R²:  {r2_linear:.4f}")
    
    # Log space
    mse_log = mean_squared_error(log_prob_targets, log_prob_preds)
    r2_log = r2_score(log_prob_targets, log_prob_preds)
    print(f"\nLog Space:")
    print(f"  MSE: {mse_log:.4f}")
    print(f"  R²:  {r2_log:.4f}")
    
    # Order of magnitude analysis
    log_errors = np.abs(log_prob_preds - log_prob_targets)
    print(f"\nPrediction Accuracy by Order of Magnitude:")
    print(f"  Within 0.5 orders: {(log_errors < 0.5).mean():.1%}")
    print(f"  Within 1.0 orders: {(log_errors < 1.0).mean():.1%}")
    
    # Classification metrics
    print("\n" + "="*60)
    print("Classification Performance (Risk Level):")
    print("-" * 40)
    
    accuracy = accuracy_score(class_targets, class_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    print("\nPer-Class Performance:")
    print(classification_report(class_targets, class_preds,
                              target_names=['Low Risk', 'Medium Risk', 'High Risk'],
                              digits=4))
    
    # Risk distribution
    print("Test Set Risk Distribution:")
    unique, counts = np.unique(class_targets, return_counts=True)
    risk_names = ['Low', 'Medium', 'High']
    for risk_class, count in zip(unique, counts):
        pred_count = np.sum(class_preds == risk_class)
        print(f"  {risk_names[risk_class]:6}: Actual={count:4d} ({count/len(class_targets)*100:5.1f}%), "
              f"Predicted={pred_count:4d} ({pred_count/len(class_targets)*100:5.1f}%)")
    
    # Visualization
    print("\nGenerating visualizations...")
    plot_test_results(
        log_prob_targets, log_prob_preds, 
        prob_targets, prob_preds,
        class_targets, class_preds, 
        accuracy, r2_linear, r2_log,
        save_path=PLOT_TEST_FILE
    )
    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate_model()