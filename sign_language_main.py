import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
import warnings
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

class SignLanguageDigitRecognizer:
    """
    A comprehensive class for Sign Language Digit Recognition
    This class handles data loading, preprocessing, model creation, training, and evaluation
    """
    
    def __init__(self, dataset_path='Dataset', img_size=(64, 64), output_dir=None):
        """
        Initialize the recognizer with dataset path and image size
        
        Args:
            dataset_path (str): Path to the dataset folder containing digit folders (0-9)
            img_size (tuple): Target size for images (height, width)
            output_dir (str): Directory to save results (if None, creates timestamped folder)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None
        
        # Create output directory for this run
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"Results_Run_{timestamp}"
        else:
            self.output_dir = output_dir
            
        self.create_output_structure()
    
    def create_output_structure(self):
        """Create organized folder structure for saving results"""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.output_dir, "Plots")
        self.models_dir = os.path.join(self.output_dir, "Models")
        self.results_dir = os.path.join(self.output_dir, "Results")
        self.logs_dir = os.path.join(self.output_dir, "Logs")
        
        for subdir in [self.plots_dir, self.models_dir, self.results_dir, self.logs_dir]:
            os.makedirs(subdir, exist_ok=True)
            
        print(f"Created output directory: {self.output_dir}")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Models: {self.models_dir}")
        print(f"  - Results: {self.results_dir}")
        print(f"  - Logs: {self.logs_dir}")
        
    def load_and_preprocess_data(self, normalize=True, test_size=0.2):
        """
        Load images from dataset folders and preprocess them
        
        Args:
            normalize (bool): Whether to normalize pixel values to [0,1]
            test_size (float): Proportion of data to use for testing
        
        Returns:
            tuple: Preprocessed training and testing data
        """
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Load images from each digit folder (0-9)
        for digit in range(10):
            digit_folder = os.path.join(self.dataset_path, str(digit))
            
            if not os.path.exists(digit_folder):
                print(f"Warning: Folder {digit_folder} does not exist!")
                continue
                
            print(f"Loading images for digit {digit}...")
            
            # Get all image files in the digit folder
            image_files = [f for f in os.listdir(digit_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(digit_folder, img_file)
                
                try:
                    # Load and resize image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size
                    img = cv2.resize(img, self.img_size)
                    
                    images.append(img)
                    labels.append(digit)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images with shape {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Normalize pixel values if requested
        if normalize:
            X = X / 255.0
            print("Pixel values normalized to [0, 1]")
        
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}, Testing set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def visualize_sample_images(self, num_samples=20, save_plot=True):
        """
        Display sample images from the dataset
        
        Args:
            num_samples (int): Number of sample images to display
            save_plot (bool): Whether to save the plot to file
        """
        if self.X_train is None:
            print("Please load data first using load_and_preprocess_data()")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Select random samples from each class
        samples_per_class = max(1, num_samples // 10)
        
        for digit in range(10):
            # Get indices of images for this digit
            digit_indices = np.where(self.y_train == digit)[0]
            
            if len(digit_indices) > 0:
                # Randomly select sample indices
                selected_indices = np.random.choice(
                    digit_indices, 
                    min(samples_per_class, len(digit_indices)), 
                    replace=False
                )
                
                for i, idx in enumerate(selected_indices):
                    plt.subplot(samples_per_class, 10, digit + 1 + i * 10)
                    plt.imshow(self.X_train[idx])
                    plt.title(f'Digit: {digit}')
                    plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle('Sample Images from Sign Language Digits Dataset', y=1.02)
        
        if save_plot:
            plot_path = os.path.join(self.plots_dir, "sample_images.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Sample images plot saved to: {plot_path}")
        
        plt.show()
    
    def create_cnn_model(self, 
                        dropout_rate=0.5, 
                        l2_reg=0.01, 
                        learning_rate=0.001,
                        architecture='default'):
        """
        Create a Convolutional Neural Network for digit recognition
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            l2_reg (float): L2 regularization strength
            learning_rate (float): Learning rate for optimizer
            architecture (str): Model architecture type ('default', 'deep', 'wide')
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        print(f"Creating CNN model with architecture: {architecture}")
        
        model = keras.Sequential()
        
        if architecture == 'default':
            # Default architecture - moderate complexity with better initialization
            model.add(layers.Input(shape=(*self.img_size, 3)))
            model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.3))  # Lower dropout in early layers
            
            model.add(layers.Conv2D(64, (3, 3), activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.5))
            
            model.add(layers.Conv2D(128, (3, 3), activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.7))
            
        elif architecture == 'deep':
            # Deeper architecture - more layers
            model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                                  input_shape=(*self.img_size, 3),
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.3))
            
            model.add(layers.Conv2D(64, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.5))
            
            model.add(layers.Conv2D(128, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.7))
            
        elif architecture == 'wide':
            # Wider architecture - more filters
            model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                                  input_shape=(*self.img_size, 3),
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.3))
            
            model.add(layers.Conv2D(128, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate * 0.5))
            
            model.add(layers.Conv2D(256, (3, 3), activation='relu',
                                  kernel_regularizer=l2(l2_reg)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate))
        
        # Flatten and add dense layers with better initialization
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu', 
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(256, activation='relu', 
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate * 0.5))
        
        # Output layer for 10 classes (digits 0-9) with proper initialization
        model.add(layers.Dense(10, activation='softmax', 
                              kernel_initializer='glorot_uniform'))
        
        # Compile the model with better optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, 
                                          beta_1=0.9, 
                                          beta_2=0.999, 
                                          epsilon=1e-7),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model created successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, 
                   batch_size=32, 
                   epochs=100, 
                   patience=10,
                   validation_split=0.2,
                   reduce_lr_patience=5):
        """
        Train the CNN model with various regularization techniques
        
        Args:
            batch_size (int): Training batch size
            epochs (int): Maximum number of epochs
            patience (int): Early stopping patience
            validation_split (float): Proportion of training data for validation
            reduce_lr_patience (int): Patience for learning rate reduction
        
        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            print("Please create a model first using create_cnn_model()")
            return None
        
        if self.X_train is None:
            print("Please load data first using load_and_preprocess_data()")
            return None
        
        print(f"Training model with batch_size={batch_size}, epochs={epochs}")
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def plot_training_history(self, config_name="model", save_plot=True):
        """
        Plot training and validation accuracy/loss curves
        
        Args:
            config_name (str): Name of the configuration for saving
            save_plot (bool): Whether to save the plot to file
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.plots_dir, f"training_history_{config_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {plot_path}")
        
        plt.show()
    
    def evaluate_model(self, config_name="model", save_plots=True):
        """
        Comprehensive evaluation of the trained model
        
        Args:
            config_name (str): Name of the configuration for saving
            save_plots (bool): Whether to save plots to file
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        if self.model is None or self.X_test is None:
            print("Please train the model and load data first.")
            return None
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred_prob = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate basic metrics
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        classification_rep = classification_report(self.y_test, y_pred)
        print(classification_rep)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_plots:
            cm_path = os.path.join(self.plots_dir, f"confusion_matrix_{config_name}.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to: {cm_path}")
        
        plt.show()
        
        # Calculate precision, recall, and F1-score for each class
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average=None
        )
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'Class': range(10),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })
        
        print("\nPer-class Metrics:")
        print(metrics_df.round(4))
        
        # ROC Curve and AUC (One-vs-Rest for multiclass)
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(self.y_test)
        
        plt.figure(figsize=(12, 10))
        
        # Calculate ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(10):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.subplot(2, 5, i + 1)
            plt.plot(fpr[i], tpr[i], color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc[i]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Digit {i}')
            plt.legend(loc="lower right")
        
        plt.tight_layout()
        
        if save_plots:
            roc_path = os.path.join(self.plots_dir, f"roc_curves_{config_name}.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves plot saved to: {roc_path}")
        
        plt.show()
        
        # Calculate macro-average AUC
        macro_auc = np.mean(list(roc_auc.values()))
        print(f"\nMacro-average AUC: {macro_auc:.4f}")
        
        # Return all metrics
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'per_class_metrics': metrics_df,
            'roc_auc_scores': roc_auc,
            'macro_auc': macro_auc,
            'y_true': self.y_test,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
        
        return results
    
    def save_model(self, config_name="model"):
        """Save the trained model"""
        if self.model is not None:
            model_path = os.path.join(self.models_dir, f"model_{config_name}.h5")
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")
    
    def save_results_summary(self, results, config_name="model"):
        """Save detailed results to text file"""
        if results is None:
            return
            
        results_path = os.path.join(self.results_dir, f"results_{config_name}.txt")
        
        with open(results_path, 'w') as f:
            f.write(f"RESULTS FOR CONFIGURATION: {config_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
            f.write(f"Test Loss: {results['test_loss']:.4f}\n")
            f.write(f"Macro AUC: {results['macro_auc']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write(str(results['classification_report']))
            f.write("\n\n")
            
            f.write("Per-class Metrics:\n")
            f.write(str(results['per_class_metrics']))
            f.write("\n\n")
            
            f.write("ROC AUC Scores:\n")
            for i, auc_score in results['roc_auc_scores'].items():
                f.write(f"Digit {i}: {auc_score:.4f}\n")
        
        print(f"Results summary saved to: {results_path}")

def hyperparameter_analysis():
    """
    Comprehensive hyperparameter analysis for the sign language digit recognition model
    This function tests different combinations of hyperparameters and compares their performance
    """
    print("Starting Hyperparameter Analysis...")
    print("="*50)
    
    # Initialize the recognizer with organized output structure
    recognizer = SignLanguageDigitRecognizer()
    
    # Load and preprocess data
    recognizer.load_and_preprocess_data(normalize=True)
    
    # Display sample images
    recognizer.visualize_sample_images()
    
    # Define hyperparameter combinations to test
    hyperparameter_configs = [
        {
            'name': 'Baseline',
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'High Learning Rate',
            'batch_size': 32,
            'learning_rate': 0.003,  # Moderate increase
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'Low Learning Rate',
            'batch_size': 32,
            'learning_rate': 0.0003,  # Moderate decrease
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'architecture': 'default',
            'patience': 15  # Increased patience for slower convergence
        },
        {
            'name': 'Large Batch Size',
            'batch_size': 128,  # Increased batch size
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'l2_reg': 0.01,
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'Small Batch Size',
            'batch_size': 16,   # Decreased batch size
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'l2_reg': 0.01,
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'High Dropout',
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.8,  # Increased dropout
            'l2_reg': 0.01,
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'Low Dropout',
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,  # Decreased dropout
            'l2_reg': 0.01,
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'High L2 Regularization',
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'l2_reg': 0.1,   # Increased L2 regularization
            'architecture': 'default',
            'patience': 10
        },
        {
            'name': 'Deep Architecture',
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'l2_reg': 0.01,
            'architecture': 'deep',  # Deeper network
            'patience': 15
        },
        {
            'name': 'Wide Architecture',
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'l2_reg': 0.01,
            'architecture': 'wide',  # Wider network
            'patience': 10
        }
    ]
    
    # Store results for comparison
    results_summary = []
    
    for i, config in enumerate(hyperparameter_configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(hyperparameter_configs)}: {config['name']}")
        print(f"{'='*60}")
        print(f"Parameters: {config}")
        
        try:
            # Reset the recognizer for this configuration
            recognizer.model = None
            recognizer.history = None
            
            # Set random seeds for reproducibility
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # Create model with current configuration
            recognizer.create_cnn_model(
                dropout_rate=config['dropout_rate'],
                l2_reg=config['l2_reg'],
                learning_rate=config['learning_rate'],
                architecture=config['architecture']
            )
            
            # Train model
            history = recognizer.train_model(
                batch_size=config['batch_size'],
                epochs=50,  # Reduced epochs for faster experimentation
                patience=config['patience']
            )
            
            # Evaluate model
            results = recognizer.evaluate_model(config_name=config['name'])
            
            # Plot training history
            print(f"\nTraining History for {config['name']}:")
            recognizer.plot_training_history(config_name=config['name'])
            
            # Save model and results
            recognizer.save_model(config_name=config['name'])
            recognizer.save_results_summary(results, config_name=config['name'])
            
            # Store results
            if results:
                results_summary.append({
                    'Configuration': config['name'],
                    'Test_Accuracy': results['test_accuracy'],
                    'Test_Loss': results['test_loss'],
                    'Macro_AUC': results['macro_auc'],
                    'Epochs_Trained': len(history.history['loss']),
                    'Best_Val_Accuracy': max(history.history['val_accuracy']),
                    **config
                })
            
        except Exception as e:
            print(f"Error with configuration {config['name']}: {e}")
            continue
    
    # Compare results
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        
        print("\n" + "="*80)
        print("HYPERPARAMETER ANALYSIS SUMMARY")
        print("="*80)
        
        # Sort by test accuracy
        comparison_df_sorted = comparison_df.sort_values('Test_Accuracy', ascending=False)
        
        print("\nRanking by Test Accuracy:")
        display_cols = ['Configuration', 'Test_Accuracy', 'Test_Loss', 'Macro_AUC', 
                       'Best_Val_Accuracy', 'Epochs_Trained']
        print(comparison_df_sorted[display_cols].round(4))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Test Accuracy comparison
        axes[0,0].bar(comparison_df['Configuration'], comparison_df['Test_Accuracy'])
        axes[0,0].set_title('Test Accuracy Comparison')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Test Loss comparison
        axes[0,1].bar(comparison_df['Configuration'], comparison_df['Test_Loss'])
        axes[0,1].set_title('Test Loss Comparison')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Macro AUC comparison
        axes[1,0].bar(comparison_df['Configuration'], comparison_df['Macro_AUC'])
        axes[1,0].set_title('Macro AUC Comparison')
        axes[1,0].set_ylabel('AUC')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Epochs trained comparison
        axes[1,1].bar(comparison_df['Configuration'], comparison_df['Epochs_Trained'])
        axes[1,1].set_title('Epochs Trained Comparison')
        axes[1,1].set_ylabel('Epochs')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save comparison plots
        comparison_plot_path = os.path.join(recognizer.plots_dir, "hyperparameter_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {comparison_plot_path}")
        
        plt.show()
        
        # Save results to CSV
        csv_path = os.path.join(recognizer.results_dir, 'hyperparameter_analysis_results.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Get best configuration
        best_config = comparison_df_sorted.iloc[0]
        
        # Save final summary
        summary_path = os.path.join(recognizer.results_dir, 'final_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("HYPERPARAMETER ANALYSIS FINAL SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total configurations tested: {len(comparison_df)}\n")
            f.write(f"Best performing configuration: {best_config['Configuration']}\n")
            f.write(f"Best test accuracy: {best_config['Test_Accuracy']:.4f}\n")
            f.write(f"Best test loss: {best_config['Test_Loss']:.4f}\n")
            f.write(f"Best macro AUC: {best_config['Macro_AUC']:.4f}\n\n")
            f.write("All results:\n")
            f.write(comparison_df_sorted[display_cols].round(4).to_string())
        
        print(f"Final summary saved to: {summary_path}")
        print(f"\nAll results organized in: {recognizer.output_dir}")
        
        # Provide recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        print(f"Best performing configuration: {best_config['Configuration']}")
        print(f"Test Accuracy: {best_config['Test_Accuracy']:.4f}")
        print(f"Parameters: Batch Size={best_config['batch_size']}, "
              f"Learning Rate={best_config['learning_rate']}, "
              f"Dropout={best_config['dropout_rate']}")

if __name__ == "__main__":
    """
    Main execution function
    
    FOR ASSIGNMENT SUBMISSION:
    To complete your assignment, modify these parameters in the hyperparameter_analysis() function:
    
    1. BATCH SIZE: Change 'batch_size' values in hyperparameter_configs (e.g., 16, 32, 64, 128)
    2. LEARNING RATE: Modify 'learning_rate' values (e.g., 0.0001, 0.001, 0.01)
    3. EPOCHS: Adjust 'epochs' parameter in train_model() calls (e.g., 50, 100, 200)
    4. DROPOUT RATIO: Change 'dropout_rate' values (e.g., 0.2, 0.5, 0.8)
    5. EARLY STOPPING PATIENCE: Modify 'patience' values (e.g., 5, 10, 15, 20)
    6. L2 REGULARIZATION: Adjust 'l2_reg' values (e.g., 0.001, 0.01, 0.1)
    7. DATA NORMALIZATION: Toggle 'normalize' parameter in load_and_preprocess_data()
    
    The code will automatically:
    - Load and visualize your dataset
    - Train models with different hyperparameters
    - Generate all required evaluation metrics
    - Create comprehensive plots and comparisons
    - Save results to CSV for analysis
    """
    
    # Run comprehensive hyperparameter analysis
    hyperparameter_analysis()
    
    print("\n" + "="*60)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated outputs:")
    print("- Sample dataset visualizations")
    print("- Training/validation curves for each configuration")
    print("- Confusion matrices")
    print("- ROC curves and AUC scores")
    print("- Comprehensive performance comparison")
    print("- CSV file with all results")
    print("- Recommendations for best hyperparameters")
