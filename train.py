# train.py
"""
Training pipeline for satellite collision risk prediction
Multi-task learning: probability regression + risk classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from config import *
from tqdm import tqdm
from training_metrics import MetricsTracker
from collision_model import CollisionRiskModel
from data_utils import calculate_class_weights, load_preprocessed_data, create_data_splits, create_data_loaders

def train_epoch(model, train_loader, log_prob_criterion, class_criterion, optimizer, device):
    """Train for one epoch with log-space probabilities"""
    model.train()
    total_log_prob_loss = 0
    total_class_loss = 0
    total_loss = 0
    
    for features, log_prob_targets, class_targets in tqdm(train_loader, desc="Training"):
        features = features.to(device)
        log_prob_targets = log_prob_targets.to(device)
        class_targets = class_targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        prob_pred, class_pred, log_prob_pred = model(features)
        
        # Compute losses
        log_prob_loss = log_prob_criterion(log_prob_pred, log_prob_targets)  # Loss in log space
        class_loss = class_criterion(class_pred, class_targets)
        log_prob_loss_norm = log_prob_loss / 2.0  # Fixed scaling
        loss = log_prob_loss_norm + class_loss  # Equal weighting

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        
        total_log_prob_loss += log_prob_loss.item()
        total_class_loss += class_loss.item()
        total_loss += loss.item()
    
    avg_log_prob_loss = total_log_prob_loss / len(train_loader)
    avg_class_loss = total_class_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    
    return avg_log_prob_loss, avg_class_loss, avg_total_loss

def validate_epoch(model, val_loader, log_prob_criterion, class_criterion, device):
    """Validate for one epoch with log-space probabilities"""
    model.eval()
    total_log_prob_loss = 0
    total_class_loss = 0
    total_loss = 0
    
    prob_preds = []
    prob_targets = []
    class_preds = []
    class_targets = []
    
    with torch.no_grad():
        for features, log_prob_target, class_target in val_loader:
            features = features.to(device)
            log_prob_target = log_prob_target.to(device)
            class_target = class_target.to(device)
            
            # Forward pass
            prob_pred, class_pred, log_prob_pred = model(features)
            
            # Compute losses
            log_prob_loss = log_prob_criterion(log_prob_pred, log_prob_target)
            class_loss = class_criterion(class_pred, class_target)
            log_prob_loss_norm = log_prob_loss / 2.0  # Fixed scaling
            loss = log_prob_loss_norm + class_loss  # Equal weighting

            total_log_prob_loss += log_prob_loss.item()
            total_class_loss += class_loss.item()
            total_loss += loss.item()
            
            # Store predictions for metrics (convert log targets back to linear)
            prob_preds.extend(prob_pred.cpu().numpy())
            linear_targets = torch.pow(10, log_prob_target).cpu().numpy()
            prob_targets.extend(linear_targets)

            # Store class predictions
            class_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
            class_targets.extend(class_target.cpu().numpy())
    
    avg_log_prob_loss = total_log_prob_loss / len(val_loader)
    avg_class_loss = total_class_loss / len(val_loader)
    avg_total_loss = total_loss / len(val_loader)
    
    # Compute metrics in linear space
    prob_mse = mean_squared_error(prob_targets, prob_preds)
    prob_r2 = r2_score(prob_targets, prob_preds)
    class_acc = accuracy_score(class_targets, class_preds)
    
    return avg_log_prob_loss, avg_class_loss, avg_total_loss, prob_mse, prob_r2, class_acc

def train_model():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    features, log_collision_probs, risk_classes = load_preprocessed_data()

    # Create splits
    (X_train, X_val, X_test, 
    y_log_prob_train, y_log_prob_val, y_log_prob_test,  # Clear naming
    y_class_train, y_class_val, y_class_test) = create_data_splits(
        features, log_collision_probs, risk_classes, 
        test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Create data loaders
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        X_train, X_val, X_test, 
        y_log_prob_train, y_log_prob_val, y_log_prob_test,  # Pass log probs
        y_class_train, y_class_val, y_class_test,
        batch_size=BATCH_SIZE
    )
    
    # Initialize model
    model = CollisionRiskModel(input_dim=7, hidden_dim=128, num_classes=3)
    model.to(device)
    
    # Loss functions
    log_prob_criterion = nn.MSELoss()  # MSE in log space
    
    # Class weights for imbalanced classification
    class_weights = calculate_class_weights(y_class_train)
    class_weights = class_weights.to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5)
    
    # Training loop
    num_epochs = NUM_EPOCHS
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training metrics tracker
    tracker = MetricsTracker()
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Train
        train_prob_loss, train_class_loss, train_total_loss = train_epoch(
            model, train_loader, log_prob_criterion, class_criterion, optimizer, device
        )
        
        # Validate
        val_prob_loss, val_class_loss, val_total_loss, prob_mse, prob_r2, class_acc = validate_epoch(
            model, val_loader, log_prob_criterion, class_criterion, device
        )
        
        # Store metrics
        tracker.update(epoch+1, 
                       {'total_loss': train_total_loss}, 
                       {'r2': prob_r2, 'accuracy': class_acc, 'total_loss': val_total_loss})

        # Scheduler step
        scheduler.step(val_total_loss)
        
        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'scaler': scaler
            }, MODEL_PATH)
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Prob: {train_prob_loss:.6f}, Class: {train_class_loss:.4f}, Total: {train_total_loss:.4f}")
            print(f"  Val   - Prob: {val_prob_loss:.6f}, Class: {val_class_loss:.4f}, Total: {val_total_loss:.4f}")
            print(f"  Val   - MSE: {prob_mse:.8f}, R²: {prob_r2:.4f}, Acc: {class_acc:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_losses.append(train_total_loss)
        val_losses.append(val_total_loss)
    
    # Plot training curves
    tracker.plot_metrics(PLOT_TRAINING_FILE)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved as '{MODEL_PATH}'")

    return model, scaler, test_loader

if __name__ == "__main__":
    model, scaler, test_loader = train_model()