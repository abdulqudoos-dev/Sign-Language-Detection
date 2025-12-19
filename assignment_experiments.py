"""
ASSIGNMENT-SPECIFIC EXPERIMENTS FOR SIGN LANGUAGE DIGIT RECOGNITION

This file contains focused experiments for each hyperparameter mentioned in the assignment.
Each experiment can be run independently by uncommenting the relevant section.

PARAMETERS TO ANALYZE (as per assignment requirements):
1. Batch Size
2. Learning Rate  
3. Epochs
4. Regularization through Dropout Layers - Dropout Ratio
5. Regularization through Early Stopping - Patience Value
6. Regularization through L1/L2 - Lambda Value
7. Normalization of Data

EVALUATION METRICS (as per assignment requirements):
a. Confusion Matrix
b. Accuracy
c. Precision & Recall
d. Area Under the Curve (AUC-ROC)
"""

import sys
import os
sys.path.append('.')  # Add current directory to path

# Import the main recognizer class
from sign_language_main import SignLanguageDigitRecognizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def experiment_batch_size():
    """
    EXPERIMENT 1: BATCH SIZE ANALYSIS
    Tests different batch sizes: [16, 32, 64, 128]
    """
    print("="*60)
    print("EXPERIMENT 1: BATCH SIZE ANALYSIS")
    print("="*60)
    
    # Batch sizes to test
    batch_sizes = [16, 32, 64, 128]
    results = []
    
    # Load data once for all experiments
    recognizer = SignLanguageDigitRecognizer()
    recognizer.load_and_preprocess_data(normalize=True)
    
    for batch_size in batch_sizes:
        print(f"\nTesting Batch Size: {batch_size}")
        print("-" * 30)
        
        # Create new model instance
        exp_recognizer = SignLanguageDigitRecognizer()
        exp_recognizer.X_train = recognizer.X_train
        exp_recognizer.X_test = recognizer.X_test
        exp_recognizer.y_train = recognizer.y_train
        exp_recognizer.y_test = recognizer.y_test
        
        # Create model with fixed parameters (only batch_size varies)
        exp_recognizer.create_cnn_model(
            dropout_rate=0.5,
            l2_reg=0.01,
            learning_rate=0.001,
            architecture='default'
        )
        
        # Train with current batch size
        history = exp_recognizer.train_model(
            batch_size=batch_size,
            epochs=50,
            patience=10
        )
        
        # Evaluate
        metrics = exp_recognizer.evaluate_model()
        
        if metrics:
            results.append({
                'Batch_Size': batch_size,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Training_Time_Epochs': len(history.history['loss'])
            })
    
    # Plot results
    if results:
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(results_df['Batch_Size'], results_df['Test_Accuracy'], 'bo-')
        axes[0].set_xlabel('Batch Size')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy vs Batch Size')
        axes[0].grid(True)
        
        axes[1].plot(results_df['Batch_Size'], results_df['Test_Loss'], 'ro-')
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('Test Loss vs Batch Size')
        axes[1].grid(True)
        
        axes[2].plot(results_df['Batch_Size'], results_df['Training_Time_Epochs'], 'go-')
        axes[2].set_xlabel('Batch Size')
        axes[2].set_ylabel('Epochs to Convergence')
        axes[2].set_title('Training Time vs Batch Size')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nBatch Size Experiment Results:")
        print(results_df)
        results_df.to_csv('batch_size_experiment.csv', index=False)

def experiment_learning_rate():
    """
    EXPERIMENT 2: LEARNING RATE ANALYSIS
    Tests different learning rates: [0.0001, 0.001, 0.01, 0.1]
    """
    print("="*60)
    print("EXPERIMENT 2: LEARNING RATE ANALYSIS")
    print("="*60)
    
    # Learning rates to test
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    results = []
    
    # Load data once
    recognizer = SignLanguageDigitRecognizer()
    recognizer.load_and_preprocess_data(normalize=True)
    
    for lr in learning_rates:
        print(f"\nTesting Learning Rate: {lr}")
        print("-" * 30)
        
        # Create new model instance
        exp_recognizer = SignLanguageDigitRecognizer()
        exp_recognizer.X_train = recognizer.X_train
        exp_recognizer.X_test = recognizer.X_test
        exp_recognizer.y_train = recognizer.y_train
        exp_recognizer.y_test = recognizer.y_test
        
        # Create model with current learning rate
        exp_recognizer.create_cnn_model(
            dropout_rate=0.5,
            l2_reg=0.01,
            learning_rate=lr,  # This is what we're testing
            architecture='default'
        )
        
        # Train model
        history = exp_recognizer.train_model(
            batch_size=32,
            epochs=50,
            patience=15 if lr <= 0.001 else 10  # More patience for lower LR
        )
        
        # Evaluate
        metrics = exp_recognizer.evaluate_model()
        
        if metrics:
            results.append({
                'Learning_Rate': lr,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Final_Val_Accuracy': history.history['val_accuracy'][-1],
                'Best_Val_Accuracy': max(history.history['val_accuracy'])
            })
    
    # Plot results
    if results:
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].semilogx(results_df['Learning_Rate'], results_df['Test_Accuracy'], 'bo-')
        axes[0].set_xlabel('Learning Rate (log scale)')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy vs Learning Rate')
        axes[0].grid(True)
        
        axes[1].semilogx(results_df['Learning_Rate'], results_df['Best_Val_Accuracy'], 'go-', label='Best')
        axes[1].semilogx(results_df['Learning_Rate'], results_df['Final_Val_Accuracy'], 'ro-', label='Final')
        axes[1].set_xlabel('Learning Rate (log scale)')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Validation Accuracy vs Learning Rate')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nLearning Rate Experiment Results:")
        print(results_df)
        results_df.to_csv('learning_rate_experiment.csv', index=False)

def experiment_dropout_regularization():
    """
    EXPERIMENT 3: DROPOUT REGULARIZATION ANALYSIS
    Tests different dropout rates: [0.0, 0.2, 0.5, 0.8]
    """
    print("="*60)
    print("EXPERIMENT 3: DROPOUT REGULARIZATION ANALYSIS")
    print("="*60)
    
    # Dropout rates to test
    dropout_rates = [0.0, 0.2, 0.5, 0.8]
    results = []
    
    # Load data once
    recognizer = SignLanguageDigitRecognizer()
    recognizer.load_and_preprocess_data(normalize=True)
    
    for dropout in dropout_rates:
        print(f"\nTesting Dropout Rate: {dropout}")
        print("-" * 30)
        
        # Create new model instance
        exp_recognizer = SignLanguageDigitRecognizer()
        exp_recognizer.X_train = recognizer.X_train
        exp_recognizer.X_test = recognizer.X_test
        exp_recognizer.y_train = recognizer.y_train
        exp_recognizer.y_test = recognizer.y_test
        
        # Create model with current dropout rate
        exp_recognizer.create_cnn_model(
            dropout_rate=dropout,  # This is what we're testing
            l2_reg=0.01,
            learning_rate=0.001,
            architecture='default'
        )
        
        # Train model
        history = exp_recognizer.train_model(
            batch_size=32,
            epochs=50,
            patience=10
        )
        
        # Evaluate
        metrics = exp_recognizer.evaluate_model()
        
        # Calculate overfitting metric (difference between train and val accuracy)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        overfitting_gap = train_acc - val_acc
        
        if metrics:
            results.append({
                'Dropout_Rate': dropout,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Final_Train_Accuracy': train_acc,
                'Final_Val_Accuracy': val_acc,
                'Overfitting_Gap': overfitting_gap
            })
    
    # Plot results
    if results:
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(results_df['Dropout_Rate'], results_df['Test_Accuracy'], 'bo-')
        axes[0].set_xlabel('Dropout Rate')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy vs Dropout Rate')
        axes[0].grid(True)
        
        axes[1].plot(results_df['Dropout_Rate'], results_df['Final_Train_Accuracy'], 'go-', label='Train')
        axes[1].plot(results_df['Dropout_Rate'], results_df['Final_Val_Accuracy'], 'ro-', label='Validation')
        axes[1].set_xlabel('Dropout Rate')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Train vs Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].plot(results_df['Dropout_Rate'], results_df['Overfitting_Gap'], 'mo-')
        axes[2].set_xlabel('Dropout Rate')
        axes[2].set_ylabel('Overfitting Gap (Train - Val)')
        axes[2].set_title('Overfitting vs Dropout Rate')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nDropout Regularization Experiment Results:")
        print(results_df)
        results_df.to_csv('dropout_experiment.csv', index=False)

def experiment_early_stopping_patience():
    """
    EXPERIMENT 4: EARLY STOPPING PATIENCE ANALYSIS
    Tests different patience values: [5, 10, 15, 20]
    """
    print("="*60)
    print("EXPERIMENT 4: EARLY STOPPING PATIENCE ANALYSIS")
    print("="*60)
    
    # Patience values to test
    patience_values = [5, 10, 15, 20]
    results = []
    
    # Load data once
    recognizer = SignLanguageDigitRecognizer()
    recognizer.load_and_preprocess_data(normalize=True)
    
    for patience in patience_values:
        print(f"\nTesting Early Stopping Patience: {patience}")
        print("-" * 30)
        
        # Create new model instance
        exp_recognizer = SignLanguageDigitRecognizer()
        exp_recognizer.X_train = recognizer.X_train
        exp_recognizer.X_test = recognizer.X_test
        exp_recognizer.y_train = recognizer.y_train
        exp_recognizer.y_test = recognizer.y_test
        
        # Create model with fixed parameters
        exp_recognizer.create_cnn_model(
            dropout_rate=0.5,
            l2_reg=0.01,
            learning_rate=0.001,
            architecture='default'
        )
        
        # Train with current patience value
        history = exp_recognizer.train_model(
            batch_size=32,
            epochs=100,  # High epochs to see patience effect
            patience=patience  # This is what we're testing
        )
        
        # Evaluate
        metrics = exp_recognizer.evaluate_model()
        
        if metrics:
            results.append({
                'Patience': patience,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Epochs_Trained': len(history.history['loss']),
                'Best_Val_Accuracy': max(history.history['val_accuracy']),
                'Final_Val_Accuracy': history.history['val_accuracy'][-1]
            })
    
    # Plot results
    if results:
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(results_df['Patience'], results_df['Test_Accuracy'], 'bo-')
        axes[0].set_xlabel('Early Stopping Patience')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy vs Patience')
        axes[0].grid(True)
        
        axes[1].plot(results_df['Patience'], results_df['Epochs_Trained'], 'ro-')
        axes[1].set_xlabel('Early Stopping Patience')
        axes[1].set_ylabel('Epochs Trained')
        axes[1].set_title('Training Duration vs Patience')
        axes[1].grid(True)
        
        axes[2].plot(results_df['Patience'], results_df['Best_Val_Accuracy'], 'go-', label='Best')
        axes[2].plot(results_df['Patience'], results_df['Final_Val_Accuracy'], 'mo-', label='Final')
        axes[2].set_xlabel('Early Stopping Patience')
        axes[2].set_ylabel('Validation Accuracy')
        axes[2].set_title('Validation Accuracy vs Patience')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nEarly Stopping Patience Experiment Results:")
        print(results_df)
        results_df.to_csv('patience_experiment.csv', index=False)

def experiment_l2_regularization():
    """
    EXPERIMENT 5: L2 REGULARIZATION ANALYSIS
    Tests different L2 regularization values: [0.0, 0.001, 0.01, 0.1]
    """
    print("="*60)
    print("EXPERIMENT 5: L2 REGULARIZATION ANALYSIS")
    print("="*60)
    
    # L2 regularization values to test
    l2_values = [0.0, 0.001, 0.01, 0.1]
    results = []
    
    # Load data once
    recognizer = SignLanguageDigitRecognizer()
    recognizer.load_and_preprocess_data(normalize=True)
    
    for l2_reg in l2_values:
        print(f"\nTesting L2 Regularization: {l2_reg}")
        print("-" * 30)
        
        # Create new model instance
        exp_recognizer = SignLanguageDigitRecognizer()
        exp_recognizer.X_train = recognizer.X_train
        exp_recognizer.X_test = recognizer.X_test
        exp_recognizer.y_train = recognizer.y_train
        exp_recognizer.y_test = recognizer.y_test
        
        # Create model with current L2 regularization
        exp_recognizer.create_cnn_model(
            dropout_rate=0.5,
            l2_reg=l2_reg,  # This is what we're testing
            learning_rate=0.001,
            architecture='default'
        )
        
        # Train model
        history = exp_recognizer.train_model(
            batch_size=32,
            epochs=50,
            patience=10
        )
        
        # Evaluate
        metrics = exp_recognizer.evaluate_model()
        
        # Calculate overfitting metric
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        overfitting_gap = train_acc - val_acc
        
        if metrics:
            results.append({
                'L2_Regularization': l2_reg,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Final_Train_Accuracy': train_acc,
                'Final_Val_Accuracy': val_acc,
                'Overfitting_Gap': overfitting_gap
            })
    
    # Plot results
    if results:
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].semilogx(results_df['L2_Regularization'], results_df['Test_Accuracy'], 'bo-')
        axes[0].set_xlabel('L2 Regularization (log scale)')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy vs L2 Regularization')
        axes[0].grid(True)
        
        axes[1].semilogx(results_df['L2_Regularization'], results_df['Final_Train_Accuracy'], 'go-', label='Train')
        axes[1].semilogx(results_df['L2_Regularization'], results_df['Final_Val_Accuracy'], 'ro-', label='Validation')
        axes[1].set_xlabel('L2 Regularization (log scale)')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Train vs Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].semilogx(results_df['L2_Regularization'], results_df['Overfitting_Gap'], 'mo-')
        axes[2].set_xlabel('L2 Regularization (log scale)')
        axes[2].set_ylabel('Overfitting Gap')
        axes[2].set_title('Overfitting vs L2 Regularization')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nL2 Regularization Experiment Results:")
        print(results_df)
        results_df.to_csv('l2_regularization_experiment.csv', index=False)

def experiment_data_normalization():
    """
    EXPERIMENT 6: DATA NORMALIZATION ANALYSIS
    Tests with and without data normalization
    """
    print("="*60)
    print("EXPERIMENT 6: DATA NORMALIZATION ANALYSIS")
    print("="*60)
    
    normalization_options = [True, False]
    results = []
    
    for normalize in normalization_options:
        print(f"\nTesting Data Normalization: {normalize}")
        print("-" * 30)
        
        # Create new recognizer instance
        recognizer = SignLanguageDigitRecognizer()
        
        # Load data with/without normalization
        recognizer.load_and_preprocess_data(normalize=normalize)
        
        # Create model with fixed parameters
        recognizer.create_cnn_model(
            dropout_rate=0.5,
            l2_reg=0.01,
            learning_rate=0.001,
            architecture='default'
        )
        
        # Train model
        history = recognizer.train_model(
            batch_size=32,
            epochs=50,
            patience=10
        )
        
        # Evaluate
        metrics = recognizer.evaluate_model()
        
        if metrics:
            results.append({
                'Data_Normalized': normalize,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Training_Stability': np.std(history.history['val_accuracy']),
                'Final_Val_Accuracy': history.history['val_accuracy'][-1]
            })
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        
        print("\nData Normalization Experiment Results:")
        print(results_df)
        results_df.to_csv('normalization_experiment.csv', index=False)
        
        # Bar plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        categories = ['Normalized', 'Not Normalized']
        
        axes[0].bar(categories, results_df['Test_Accuracy'])
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy: Normalized vs Not Normalized')
        
        axes[1].bar(categories, results_df['Test_Loss'])
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('Test Loss: Normalized vs Not Normalized')
        
        axes[2].bar(categories, results_df['Training_Stability'])
        axes[2].set_ylabel('Validation Accuracy Std Dev')
        axes[2].set_title('Training Stability: Normalized vs Not Normalized')
        
        plt.tight_layout()
        plt.show()

def experiment_training_data_size():
    """
    EXPERIMENT 7: TRAINING DATA SIZE ANALYSIS
    Tests different proportions of training data: [0.5, 0.6, 0.7, 0.8, 0.9]
    """
    print("="*60)
    print("EXPERIMENT 7: TRAINING DATA SIZE ANALYSIS")
    print("="*60)
    
    # Training data proportions to test (1 - test_size)
    train_proportions = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for train_prop in train_proportions:
        test_size = 1 - train_prop
        print(f"\nTesting Training Data Proportion: {train_prop:.1f} (Test Size: {test_size:.1f})")
        print("-" * 50)
        
        # Create new recognizer instance
        recognizer = SignLanguageDigitRecognizer()
        
        # Load data with current train/test split
        recognizer.load_and_preprocess_data(normalize=True, test_size=test_size)
        
        print(f"Training samples: {len(recognizer.X_train)}, Test samples: {len(recognizer.X_test)}")
        
        # Create model with fixed parameters
        recognizer.create_cnn_model(
            dropout_rate=0.5,
            l2_reg=0.01,
            learning_rate=0.001,
            architecture='default'
        )
        
        # Train model
        history = recognizer.train_model(
            batch_size=32,
            epochs=50,
            patience=10
        )
        
        # Evaluate
        metrics = recognizer.evaluate_model()
        
        if metrics:
            results.append({
                'Train_Proportion': train_prop,
                'Training_Samples': len(recognizer.X_train),
                'Test_Samples': len(recognizer.X_test),
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Loss': metrics['test_loss'],
                'Macro_AUC': metrics['macro_auc'],
                'Best_Val_Accuracy': max(history.history['val_accuracy'])
            })
    
    # Plot results
    if results:
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(results_df['Train_Proportion'], results_df['Test_Accuracy'], 'bo-')
        axes[0].set_xlabel('Training Data Proportion')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy vs Training Data Size')
        axes[0].grid(True)
        
        axes[1].plot(results_df['Training_Samples'], results_df['Test_Accuracy'], 'ro-')
        axes[1].set_xlabel('Number of Training Samples')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Test Accuracy vs Training Sample Count')
        axes[1].grid(True)
        
        axes[2].plot(results_df['Train_Proportion'], results_df['Best_Val_Accuracy'], 'go-', label='Val Accuracy')
        axes[2].plot(results_df['Train_Proportion'], results_df['Test_Accuracy'], 'ro-', label='Test Accuracy')
        axes[2].set_xlabel('Training Data Proportion')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Validation vs Test Accuracy')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nTraining Data Size Experiment Results:")
        print(results_df)
        results_df.to_csv('training_data_size_experiment.csv', index=False)

# Main execution function for assignment experiments
def run_assignment_experiments():
    """
    Run all experiments required for the assignment
    
    INSTRUCTIONS FOR STUDENTS:
    1. Uncomment the experiments you want to run
    2. Each experiment will generate:
       - Detailed console output with results
       - Visualization plots
       - CSV files with numerical results
    3. You can run experiments individually or all together
    """
    
    print("SIGN LANGUAGE DIGIT RECOGNITION - ASSIGNMENT EXPERIMENTS")
    print("="*70)
    print("This script will run comprehensive experiments for all hyperparameters")
    print("mentioned in your assignment requirements.")
    print("="*70)
    
    # Uncomment the experiments you want to run:
    
    # EXPERIMENT 1: Batch Size Analysis
    experiment_batch_size()
    
    # EXPERIMENT 2: Learning Rate Analysis  
    experiment_learning_rate()
    
    # EXPERIMENT 3: Dropout Regularization Analysis
    experiment_dropout_regularization()
    
    # EXPERIMENT 4: Early Stopping Patience Analysis
    experiment_early_stopping_patience()
    
    # EXPERIMENT 5: L2 Regularization Analysis
    experiment_l2_regularization()
    
    # EXPERIMENT 6: Data Normalization Analysis
    experiment_data_normalization()
    
    # EXPERIMENT 7: Training Data Size Analysis
    experiment_training_data_size()
    
    print("\n" + "="*70)
    print("ALL ASSIGNMENT EXPERIMENTS COMPLETED!")
    print("="*70)
    print("Generated files:")
    print("- batch_size_experiment.csv")
    print("- learning_rate_experiment.csv")
    print("- dropout_experiment.csv")
    print("- patience_experiment.csv")
    print("- l2_regularization_experiment.csv")
    print("- normalization_experiment.csv")
    print("- training_data_size_experiment.csv")
    print("\nAll visualizations have been displayed.")
    print("Use these results for your assignment analysis and report!")

if __name__ == "__main__":
    # Run all assignment experiments
    run_assignment_experiments()