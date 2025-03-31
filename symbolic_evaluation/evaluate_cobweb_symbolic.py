 import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json
import os
from cobweb_symbolic import CobwebSymbolic
import untils

# Helper functions for evaluation
def safe_categorize(model, sample):
    """
    Safely categorize a sample using the model tree, with error handling.
    
    Args:
        model: Trained CobwebSymbolic model
        sample: Input sample with label
        
    Returns:
        Node in the tree or None if categorization failed
    """
    try:
        # Method 1: Try using _cobweb_categorize directly
        if hasattr(model.tree, '_cobweb_categorize'):
            return model.tree._cobweb_categorize(sample)
        # Method 2: Try using categorize method
        elif hasattr(model.tree, 'categorize'):
            return model.tree.categorize(sample)
        # Method 3: Try calling predict (which might use cobweb internally)
        elif hasattr(model, 'predict'):
            prediction = model.predict(sample[:-1])  # Exclude label for predict
            return {"prediction": prediction}
        else:
            print("  No categorization method found")
            return None
    except Exception as e:
        print(f"  Categorization error: {e}")
        return None

def visualize_confusion_matrix(test_labels, predictions, title):
    """
    Visualize confusion matrix for model evaluation.
    
    Args:
        test_labels: True labels
        predictions: Predicted labels
        title: Title for the confusion matrix plot
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def preprocess_dataset(dataset_name, split_classes=[0, 1, 2, 3]):
    """
    Load and preprocess dataset with symbolic feature encoding.
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
        split_classes: List of classes to use
        
    Returns:
        train_loader, test_loader, image_shape
    """
    print(f"Loading and preprocessing {dataset_name} dataset...")
    
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data/MNIST', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data/MNIST', train=False, download=True, transform=transform)
        image_shape = (28, 28)
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10('data/CIFAR10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data/CIFAR10', train=False, download=True, transform=transform)
        image_shape = (32, 32, 3)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Filter for specified classes
    train_dataset = untils.filter_by_label(train_dataset, split_classes, rename_labels=True)
    test_dataset = untils.filter_by_label(test_dataset, split_classes, rename_labels=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Print class distribution
    class_counts = {}
    for _, label in train_dataset:
        label_value = label.item() if hasattr(label, 'item') else int(label)
        class_counts[label_value] = class_counts.get(label_value, 0) + 1
    
    print(f"Training set class distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  Class {class_idx}: {count} samples")
    
    return train_loader, test_loader, image_shape

def train_symbolic_cobweb(train_loader, image_shape, depth=4, epochs=1):
    """
    Train a symbolic CobWeb model incrementally.
    
    Args:
        train_loader: DataLoader with training data
        image_shape: Shape of input images
        depth: Maximum depth of the CobWeb tree
        epochs: Number of training epochs
        
    Returns:
        trained model
    """
    print("Training symbolic CobWeb model...")
    
    if len(image_shape) == 2:
        input_dim = image_shape[0] * image_shape[1]
    else:
        input_dim = image_shape[0] * image_shape[1] * image_shape[2]
    
    # Initialize model
    model = CobwebSymbolic(input_dim=input_dim, depth=depth)
    
    # Train incrementally
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for data, label in tqdm(train_loader, desc="Training"):
            # Flatten the input data
            x = data.view(-1).numpy()
            # Get label
            # y = label.item() if hasattr(label, 'item') else int(label)
            
            # Include label at the end of the input vector
            # x_with_label = np.concatenate([x, np.array([y])])
            
            # Incrementally fit the tree
            model.tree.ifit(x_with_label)
    
    return model

def save_model_and_results(model, results, filename):
    """
    Save the trained model and evaluation results.
    
    Args:
        model: Trained CobwebSymbolic model
        results: Evaluation results dictionary
        filename: Base filename for saving
    """
    # Create directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save model
    model_file = f"results/{filename}_model.json"
    try:
        model.save_tree_to_json(model_file)
        print(f"Model saved to {model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Save results
    results_file = f"results/{filename}_results.json"
    
    # Make all results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    # Write to file
    try:
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        print("Saving simplified results instead...")
        
        # Fallback to basic results
        simple_results = {}
        for method, result in results.items():
            if isinstance(result, dict) and "accuracy" in result:
                simple_results[method] = {"accuracy": float(result["accuracy"])}
            elif isinstance(result, dict) and "error" in result:
                simple_results[method] = {"error": str(result["error"])}
            else:
                simple_results[method] = {"result": "unknown"}
        
        with open(results_file, 'w') as f:
            json.dump(simple_results, f)

# Evaluation methods
def evaluate_with_averaging(model, train_loader, test_loader, image_shape, num_classes, 
                           samples_per_class, visualize=True):
    """
    Evaluation using sample points averaging: Select samples from each class, 
    find nodes in the tree, average them, and classify test samples.
    
    Args:
        model: Trained CobwebSymbolic model
        train_loader: DataLoader with training data
        test_loader: DataLoader with test data
        image_shape: Shape of input images
        num_classes: Number of classes in the dataset
        samples_per_class: Number of sample points to take per class
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating with averaging approach (samples_per_class={samples_per_class})...")
    
    # 1. Collect training samples by class
    class_samples = {cls: [] for cls in range(num_classes)}
    
    for data, label in tqdm(train_loader, desc="Collecting samples by class"):
        x = data.view(-1).numpy()
        y = label.item() if hasattr(label, 'item') else int(label)
        
        if y < num_classes:  # Ensure valid class
            class_samples[y].append(x)
    
    # 2. Select samples for each class
    selected_samples = {}
    for cls in range(num_classes):
        if class_samples[cls]:
            # If samples_per_class is -1, use all samples
            if samples_per_class == -1:
                selected_samples[cls] = class_samples[cls]
                print(f"Class {cls}: Using all {len(selected_samples[cls])} samples")
            # Otherwise, select random samples if enough are available
            elif len(class_samples[cls]) >= samples_per_class:
                indices = np.random.choice(len(class_samples[cls]), samples_per_class, replace=False)
                selected_samples[cls] = [class_samples[cls][i] for i in indices]
                print(f"Class {cls}: Selected {len(selected_samples[cls])} samples")
            else:
                # Use all available samples if less than requested
                selected_samples[cls] = class_samples[cls]
                print(f"Class {cls}: Using all {len(selected_samples[cls])} samples (fewer than requested)")
        else:
            print(f"Warning: No samples found for class {cls}")
            selected_samples[cls] = []
    
    # 3. Calculate average for each class
    class_averages = {}
    for cls in range(num_classes):
        if selected_samples[cls]:
            # Calculate average sample
            avg_sample = np.mean(selected_samples[cls], axis=0)
            
            # Store average representation
            class_averages[cls] = avg_sample
            print(f"Class {cls}: Calculated average from {len(selected_samples[cls])} samples")
        else:
            print(f"Warning: Cannot calculate average for class {cls}")
    
    # Exit if no valid classes found
    if len(class_averages) == 0:
        print("No valid class averages found, cannot proceed with evaluation")
        return {"accuracy": 0.0, "error": "No valid class averages"}
    
    # 4. Visualize class averages
    if visualize:
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Class Averages (samples_per_class={samples_per_class})", fontsize=16)
        
        for cls in range(num_classes):
            if cls in class_averages:
                plt.subplot(1, num_classes, cls + 1)
                
                # Reshape average sample
                if len(image_shape) == 2:
                    img = class_averages[cls].reshape(image_shape)
                    plt.imshow(img, cmap='gray')
                else:
                    img = class_averages[cls].reshape(image_shape)
                    plt.imshow(img)
                
                plt.title(f"Class {cls}")
                plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    # 5. Evaluate on test data
    test_labels = []
    predictions = []
    
    for data, label in tqdm(test_loader, desc="Evaluating test data"):
        x = data.view(-1).numpy()
        y = label.item() if hasattr(label, 'item') else int(label)
        test_labels.append(y)
        
        # Find closest class average
        best_class = None
        best_distance = float('inf')
        
        for cls, avg_sample in class_averages.items():
            try:
                dist = np.linalg.norm(x - avg_sample)
                if dist < best_distance:
                    best_distance = dist
                    best_class = cls
            except ValueError:
                # Skip if shapes don't match
                continue
        
        # Use most common class as fallback
        if best_class is None:
            best_class = np.argmax(np.bincount(test_labels))
        
        predictions.append(best_class)
    
    # 6. Calculate accuracy and visualize results
    test_labels = np.array(test_labels)
    predictions = np.array(predictions)
    accuracy = accuracy_score(test_labels, predictions)
    
    if visualize:
        visualize_confusion_matrix(
            test_labels, predictions, f'Confusion Matrix (Averaging, samples={samples_per_class})'
        )
    
    print(f"Averaging Approach (samples={samples_per_class}) Accuracy: {accuracy:.4f}")
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(test_labels, predictions).tolist()
    }

def evaluate_class_centroids(model, train_loader, test_loader, image_shape, num_classes, visualize=True):
    """
    Evaluate using centroids (average of all samples) from each class.
    
    Args:
        model: Trained CobwebSymbolic model
        train_loader: DataLoader with training data
        test_loader: DataLoader with test data
        image_shape: Shape of input images
        num_classes: Number of classes in the dataset
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating with class centroids approach...")
    
    # 1. Collect all training samples by class
    class_samples = {cls: [] for cls in range(num_classes)}
    
    for data, label in tqdm(train_loader, desc="Collecting all class samples"):
        x = data.view(-1).numpy()
        y = label.item() if hasattr(label, 'item') else int(label)
        
        if y < num_classes:  # Ensure valid class
            class_samples[y].append(x)
    
    # Print stats about collected samples
    for cls in range(num_classes):
        print(f"Class {cls}: Collected {len(class_samples[cls])} samples")
    
    # 2. Calculate class centroids (average of all samples in each class)
    class_centroids = {}
    for cls in range(num_classes):
        if len(class_samples[cls]) > 0:
            class_centroids[cls] = np.mean(class_samples[cls], axis=0)
    
    # Visualize class centroids
    if visualize:
        plt.figure(figsize=(15, 5))
        plt.suptitle("Class Centroids (All Data)", fontsize=16)
        
        for cls in range(num_classes):
            if cls in class_centroids:
                plt.subplot(1, num_classes, cls + 1)
                
                # Reshape centroid
                if len(image_shape) == 2:
                    img = class_centroids[cls].reshape(image_shape)
                    plt.imshow(img, cmap='gray')
                else:
                    img = class_centroids[cls].reshape(image_shape)
                    plt.imshow(img)
                
                plt.title(f"Class {cls} Centroid")
                plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    # 3. Evaluate on test data using centroids
    test_labels = []
    predictions = []
    
    for data, label in tqdm(test_loader, desc="Evaluating with centroids"):
        x = data.view(-1).numpy()
        y = label.item() if hasattr(label, 'item') else int(label)
        test_labels.append(y)
        
        # Find closest centroid
        best_class = None
        best_distance = float('inf')
        
        for cls, centroid in class_centroids.items():
            try:
                dist = np.linalg.norm(x - centroid)
                if dist < best_distance:
                    best_distance = dist
                    best_class = cls
            except ValueError:
                continue
        
        # Use most common class as fallback
        if best_class is None:
            best_class = np.argmax(np.bincount(test_labels))
        
        predictions.append(best_class)
    
    # 4. Calculate accuracy
    test_labels = np.array(test_labels)
    predictions = np.array(predictions)
    
    accuracy = accuracy_score(test_labels, predictions)
    
    # 5. Visualize results
    if visualize:
        visualize_confusion_matrix(
            test_labels, predictions, 'Confusion Matrix (Class Centroids)'
        )
    
    print(f"Class Centroids Accuracy: {accuracy:.4f}")
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(test_labels, predictions).tolist()
    }

def main():
    """Main function to run the symbolic CobWeb evaluation."""
    # Configuration parameters
    datasets = ['mnist', 'cifar10']
    split_classes = [0, 1, 2, 3]  # First 4 classes
    
    # Default parameters
    default_depth = 5
    default_epochs = 3
    
    # Sample sizes to test
    sample_sizes = [30, 100, 500, 5000, -1]  # -1 means use all available samples
    
    for dataset_name in datasets:
        print(f"\n\n{'='*50}")
        print(f"Evaluating symbolic CobWeb on {dataset_name.upper()}")
        print(f"{'='*50}\n")
        
        # 1. Load and preprocess dataset
        train_loader, test_loader, image_shape = preprocess_dataset(dataset_name, split_classes)
        
        # 2. Train model
        model = train_symbolic_cobweb(train_loader, image_shape, depth=default_depth, epochs=default_epochs)
        
        # 3. Evaluate with different approaches
        results = {}
        
        # Test different sample sizes for averaging approach
        for samples in sample_sizes:
            method_name = f"averaging_samples_{samples}" if samples != -1 else "averaging_all_samples"
            print(f"\nEvaluating averaging approach with {samples if samples != -1 else 'all'} samples per class...")
            
            try:
                results[method_name] = evaluate_with_averaging(
                    model, train_loader, test_loader, image_shape, len(split_classes),
                    samples_per_class=samples, visualize=True
                )
            except Exception as e:
                print(f"Error with averaging evaluation (samples={samples}): {e}")
                results[method_name] = {"accuracy": 0.0, "error": str(e)}
        
        # Evaluate with class centroids approach
        print("\nEvaluating with class centroids approach...")
        try:
            results["class_centroids"] = evaluate_class_centroids(
                model, train_loader, test_loader, image_shape, len(split_classes), visualize=True
            )
        except Exception as e:
            print(f"Error with class centroids evaluation: {e}")
            results["class_centroids"] = {"error": str(e)}
        
        # 4. Save model and results
        save_model_and_results(model, results, f"{dataset_name}_symbolic_evaluation")
        
        # 5. Print summary
        print("\n" + "="*50)
        print(f"SUMMARY OF RESULTS FOR {dataset_name.upper()}:")
        print("="*50)
        
        for samples in sample_sizes:
            method_name = f"averaging_samples_{samples}" if samples != -1 else "averaging_all_samples"
            if "error" not in results[method_name]:
                print(f"Averaging ({samples if samples != -1 else 'all'} samples): {results[method_name]['accuracy']:.4f}")
            else:
                print(f"Averaging ({samples if samples != -1 else 'all'} samples): Failed")
        
        if "error" not in results["class_centroids"]:
            print(f"Class Centroids: {results['class_centroids']['accuracy']:.4f}")
        else:
            print(f"Class Centroids: Failed")
        
        print("\n\n")

if __name__ == '__main__':
    main() 