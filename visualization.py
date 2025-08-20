# visualization.py
"""
Visualization and analysis functions for collision data
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_collision_probability_analysis(collision_data, save_plot=True, 
                                        output_file='collision_probability_analysis.png'):
    """
    Plot collision probability distribution with risk thresholds
    Args:
        collision_data (list): List of collision data dictionaries.
        save_plot (bool): Whether to save the plot as a file.
    Returns:
        None
    """
    
    # Extract data
    probabilities = [pair['features'][0] for pair in collision_data]  
    distances = [pair['features'][1] for pair in collision_data]      
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution
    plt.subplot(2, 2, 1)
    plt.hist(probabilities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Collision Probability')
    plt.ylabel('Number of Satellite Pairs')
    plt.title('Distribution of Collision Probabilities')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x=1e-4, color='red', linestyle='--', label='High Risk Threshold (1e-4)')
    plt.axvline(x=1e-6, color='orange', linestyle='--', label='Medium Risk Threshold (1e-6)')
    plt.legend()
    
    # Plot 2: Log scale
    plt.subplot(2, 2, 2)
    plt.hist(probabilities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Collision Probability')
    plt.ylabel('Number of Satellite Pairs')
    plt.title('Collision Probability (Log-Log Scale)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x=1e-4, color='red', linestyle='--', label='High Risk (1e-4)')
    plt.axvline(x=1e-6, color='orange', linestyle='--', label='Medium Risk (1e-6)')
    plt.legend()
    
    # Plot 3: Distance vs Probability
    plt.subplot(2, 2, 3)
    plt.scatter(distances, probabilities, alpha=0.6, s=20)
    plt.xlabel('Closest Approach Distance (km)')
    plt.ylabel('Collision Probability')
    plt.title('Distance vs Collision Probability')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=1e-6, color='orange', linestyle='--', alpha=0.7)
    
    # Plot 4: Risk distribution
    plt.subplot(2, 2, 4)
    risk_counts = [0, 0, 0]
    risk_labels = ['Low Risk\n(< 1e-6)', 'Medium Risk\n(1e-6 to 1e-4)', 'High Risk\n(> 1e-4)']
    
    for pair in collision_data:
        risk_counts[pair['risk_class']] += 1
    
    colors = ['green', 'orange', 'red']
    plt.bar(risk_labels, risk_counts, color=colors, alpha=0.7)
    plt.ylabel('Number of Satellite Pairs')
    plt.title('Risk Category Distribution')
    plt.yscale('log')
    
    for i, count in enumerate(risk_counts):
        if count > 0:
            plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot as {output_file}")

    plt.show()
    
    # Print statistics
    probabilities = np.array(probabilities)
    print(f"\nCollision Probability Statistics:")
    print(f"Min probability: {probabilities.min():.2e}")
    print(f"Max probability: {probabilities.max():.2e}")
    print(f"Mean probability: {probabilities.mean():.2e}")
    print(f"Median probability: {np.median(probabilities):.2e}")
    
    # Risk distribution
    high_risk = np.sum(probabilities > 1e-4)
    medium_risk = np.sum((probabilities > 1e-6) & (probabilities <= 1e-4))
    low_risk = np.sum(probabilities <= 1e-6)
    
    print(f"\nRisk Distribution:")
    print(f"High risk (> 1e-4): {high_risk} pairs ({high_risk/len(probabilities)*100:.1f}%)")
    print(f"Medium risk (1e-6 to 1e-4): {medium_risk} pairs ({medium_risk/len(probabilities)*100:.1f}%)")
    print(f"Low risk (< 1e-6): {low_risk} pairs ({low_risk/len(probabilities)*100:.1f}%)")

def plot_test_results(log_prob_targets, log_prob_preds, prob_targets, prob_preds,
                     class_targets, class_preds, accuracy, r2_linear, r2_log,
                     save_path='test_evaluation.png'):
    """
    Comprehensive visualization of model test results
    
    Args:
        log_prob_targets: Actual log10 probabilities
        log_prob_preds: Predicted log10 probabilities  
        prob_targets: Actual linear probabilities
        prob_preds: Predicted linear probabilities
        class_targets: Actual risk classes
        class_preds: Predicted risk classes
        accuracy: Classification accuracy
        r2_linear: R² score in linear space
        r2_log: R² score in log space
        save_path: Path to save the plot
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, r2_score
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Predicted vs Actual (Log Space)
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(log_prob_targets, log_prob_preds, 
                         c=class_targets, cmap='viridis', alpha=0.6, s=20)
    ax1.plot([log_prob_targets.min(), log_prob_targets.max()], 
            [log_prob_targets.min(), log_prob_targets.max()], 
            'r--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Log₁₀ Probability')
    ax1.set_ylabel('Predicted Log₁₀ Probability')
    ax1.set_title(f'Log Space Predictions (R²={r2_log:.3f})')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Risk Class')
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted vs Actual (Linear Space)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(prob_targets, prob_preds, c=class_targets, 
               cmap='viridis', alpha=0.6, s=20)
    ax2.plot([prob_targets.min(), prob_targets.max()], 
            [prob_targets.min(), prob_targets.max()], 'r--', label='Perfect')
    ax2.set_xlabel('Actual Probability')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title(f'Linear Space Predictions (R²={r2_linear:.3f})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals plot
    ax3 = plt.subplot(2, 3, 3)
    residuals = log_prob_preds - log_prob_targets
    ax3.scatter(log_prob_targets, residuals, alpha=0.5, s=10)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5)
    ax3.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Actual Log₁₀ Probability')
    ax3.set_ylabel('Residual (Pred - Actual)')
    ax3.set_title('Residual Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    ax4.set_xlabel('Prediction Error (Log Space)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Error Distribution (μ={residuals.mean():.3f}, σ={residuals.std():.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm = confusion_matrix(class_targets, class_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Med', 'High'],
                yticklabels=['Low', 'Med', 'High'], ax=ax5)
    ax5.set_title(f'Confusion Matrix (Acc={accuracy:.3f})')
    ax5.set_ylabel('Actual Risk')
    ax5.set_xlabel('Predicted Risk')
    
    # 6. Per-class regression performance
    ax6 = plt.subplot(2, 3, 6)
    risk_names = ['Low', 'Medium', 'High']
    for risk_class in range(3):
        mask = class_targets == risk_class
        if mask.sum() > 0:
            class_r2 = r2_score(log_prob_targets[mask], log_prob_preds[mask])
            ax6.bar(risk_class, class_r2, label=f'{risk_names[risk_class]} (n={mask.sum()})')
    ax6.set_xlabel('Risk Class')
    ax6.set_ylabel('R² Score')
    ax6.set_title('Regression Performance by Risk Class')
    ax6.set_xticks([0, 1, 2])
    ax6.set_xticklabels(['Low', 'Medium', 'High'])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test results visualization saved to {save_path}")
    
    plt.show()