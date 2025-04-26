import torch
import torch.nn.functional as F
from tqdm.auto import tqdm # Optional: for progress bar
import numpy as np

# --- Main Soft Dendrogram Purity Function (Updated) ---

def compute_soft_dendrogram_purity(model, test_dataloader, annotation_matrix, device, epsilon=1e-9):
    """
    Computes Dendrogram Purity using a soft approach based on joint probabilities.
    Assumes annotation_matrix columns already represent P(class_k | node_c).

    Args:
        model: The PyTorch model (must be in eval mode).
        test_dataloader: DataLoader for the test set (yields batches of (x, y)).
        annotation_matrix: Numpy array or Torch tensor (n_classes x n_nodes).
                           Each column c represents the distribution P(class | node c)
                           and should sum to 1.
        device: The device to run computations on ('cuda' or 'cpu').
        epsilon (float): Small value to prevent division by zero during normalization.

    Returns:
        float: The computed Soft Dendrogram Purity score.
    """
    model.eval()
    model.to(device)

    if isinstance(annotation_matrix, np.ndarray):
        # Ensure float and move to device
        node_purity_matrix = torch.from_numpy(annotation_matrix).float().to(device)
    else:
        # Ensure it's on the correct device
        node_purity_matrix = annotation_matrix.float().to(device)


    n_classes = node_purity_matrix.shape[0]
    n_nodes = node_purity_matrix.shape[1]

    # --- Precomputation ---
    # No normalization needed based on the user's clarification.
    # We directly use annotation_matrix as the node purity information P(k|c).
    print("Using provided annotation_matrix directly as node purities P(k|c).")

    # Optional Check: Verify columns sum to 1 (approximately)
    col_sums = node_purity_matrix.sum(dim=0)
    if not torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5):
        print("Warning: Columns of annotation_matrix do not all sum to 1. Check input.")
        print(f"Min sum: {col_sums.min()}, Max sum: {col_sums.max()}")


    # --- Process Test Data ---
    print("Processing test data to get probability distributions (pcx)...")
    all_pcx = []
    all_true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_dataloader, desc="Evaluating Test Set"):
            x_batch = x_batch.to(device)
            # Assuming model forward pass returns pcx as the 6th element
            try:
                outputs = model(x_batch)
                if len(outputs) < 6:
                     raise ValueError(f"Model output tuple has length {len(outputs)}, expected at least 6.")
                pcx_batch = outputs[5] # Probability of cluster c given x
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print("Check model definition and output structure.")
                return None

            # Check pcx_batch shape
            if pcx_batch.shape[1] != n_nodes:
                print(f"Error: Model output pcx_batch has {pcx_batch.shape[1]} columns, but expected {n_nodes} nodes.")
                return None

            all_pcx.append(pcx_batch.cpu())
            all_true_labels.append(y_batch.cpu())

    try:
        all_pcx = torch.cat(all_pcx).to(device) # Move final tensor to computation device
        all_true_labels = torch.cat(all_true_labels)
    except RuntimeError as e:
         print(f"Error concatenating tensors (likely GPU memory issue): {e}")
         print("Try running on CPU or processing in smaller chunks if possible.")
         return None

    n_test = len(all_true_labels)
    print(f"Processed {n_test} test samples.")

    # --- Calculate Soft Dendrogram Purity ---
    print("Calculating Soft Dendrogram Purity (iterating over pairs)...")
    total_purity_sum = 0.0
    total_pairs = 0

    for k in tqdm(range(n_classes), desc="Processing Classes"):
        indices_k = (all_true_labels == k).nonzero(as_tuple=True)[0]
        N_k = len(indices_k)

        if N_k < 2:
            continue

        num_pairs_k = N_k * (N_k - 1) / 2
        total_pairs += num_pairs_k
        class_purity_sum = 0.0

        pcx_k = all_pcx[indices_k].to(device)
        # Get node purities P(k|c) for the current class k
        purities_k = node_purity_matrix[k, :] # Shape: (n_nodes,)

        # Iterate over unique pairs within the class - O(N_k^2 * n_nodes)
        for i_idx in range(N_k):
            P_i = pcx_k[i_idx, :] # Probs for point i
            for j_idx in range(i_idx + 1, N_k):
                P_j = pcx_k[j_idx, :] # Probs for point j

                # Calculate joint probability (element-wise product)
                joint_p = P_i * P_j # Shape: (n_nodes,)

                # Normalize to get weights w(c|xi, xj)
                joint_p_sum = joint_p.sum()
                if joint_p_sum < epsilon:
                    weights = torch.zeros_like(joint_p)
                else:
                    weights = joint_p / joint_p_sum

                # Calculate weighted purity for the pair: sum( weights * P(k|c) )
                pair_purity = torch.dot(weights, purities_k) # Dot product
                class_purity_sum += pair_purity.item() # Accumulate scalar value

        total_purity_sum += class_purity_sum

    if total_pairs == 0:
        print("Warning: No valid pairs found to calculate purity.")
        return 0.0

    final_soft_dendrogram_purity = total_purity_sum / total_pairs
    print("Calculation complete.")

    return final_soft_dendrogram_purity

# --- Example Usage (Should work as before, assuming dummy_annotation columns sum to 1) ---
if __name__ == '__main__':
    # 1. Define a Dummy Model (same as before)
    class DummyHierarchicalModel(torch.nn.Module):
        def __init__(self, n_features, n_nodes):
            super().__init__()
            self.n_nodes = n_nodes
            self.fc = torch.nn.Linear(n_features, n_nodes)
        def forward(self, x):
            node_logits = self.fc(x)
            pcx = F.softmax(node_logits, dim=1)
            dummy_tensor = torch.tensor(0.0, device=x.device); dummy_int = 0
            return dummy_tensor, dummy_tensor, dummy_tensor, dummy_int, dummy_int, pcx, dummy_tensor, dummy_tensor

    # 2. Setup Parameters (same as before)
    N_TEST = 100
    N_FEATURES = 20
    N_CLASSES = 5
    N_LEAVES = 8
    N_NODES = 2 * N_LEAVES - 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3. Create Dummy Data and DataLoader (same as before)
    X_test = torch.randn(N_TEST, N_FEATURES)
    true_labels_test = torch.randint(0, N_CLASSES, (N_TEST,))
    test_dataset = torch.utils.data.TensorDataset(X_test, true_labels_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # 4. Create Dummy Annotation Matrix (Normalized)
    # Now ensure columns sum to 1
    raw_annotation = torch.rand(N_CLASSES, N_NODES) * 10
    raw_annotation[0, 0] = 50; raw_annotation[1, 1] = 50
    raw_annotation[2, N_LEAVES] = 40; raw_annotation[3, N_NODES-1] = 60
    # Normalize columns
    col_sums = raw_annotation.sum(dim=0, keepdim=True)
    col_sums[col_sums == 0] = 1.0 # Avoid division by zero
    dummy_annotation_normalized = raw_annotation / col_sums


    # 5. Instantiate Model (same as before)
    model = DummyHierarchicalModel(N_FEATURES, N_NODES)

    # 6. Compute Soft Purity
    print(f"Using device: {DEVICE}")
    soft_purity_score = compute_soft_dendrogram_purity(model, test_loader, dummy_annotation_normalized, DEVICE)

    if soft_purity_score is not None:
        print(f"\nComputed Soft Dendrogram Purity: {soft_purity_score:.4f}")



import torch
import torch.nn.functional as F
from tqdm.auto import tqdm # Optional: for progress bar
import numpy as np

# --- Main Soft Dendrogram Purity Function (Test Data Only) ---

def compute_soft_dendrogram_purity_test_only(model, test_dataloader, device, epsilon=1e-9):
    """
    Computes Soft Dendrogram Purity using only test data.
    Node purities P(k|c) are calculated on-the-fly based on expected counts
    derived from the model's probabilistic outputs on the test set.

    Args:
        model: The PyTorch model (must be in eval mode).
        test_dataloader: DataLoader for the test set (yields batches of (x, y)).
                         The dataset should provide true class labels y.
        device: The device to run computations on ('cuda' or 'cpu').
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The computed Soft Dendrogram Purity score, or None if an error occurs.
    """
    model.eval()
    model.to(device)

    n_classes = 10
    n_nodes = None

    # --- Process Test Data ---
    print("Processing test data to get probability distributions (pcx)...")
    all_pcx = []
    all_true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_dataloader, desc="Evaluating Test Set"):
            # Determine n_classes dynamically from labels seen
            if n_classes is None:
                # Assuming labels are 0-indexed
                n_classes = int(y_batch.max().item()) + 1
            else:
                n_classes = max(n_classes, int(y_batch.max().item()) + 1)

            x_batch = x_batch.to(device)
            # Assuming model forward pass returns pcx as the 6th element
            try:
                outputs = model(x_batch)
                if len(outputs) < 6:
                     raise ValueError(f"Model output tuple has length {len(outputs)}, expected at least 6.")
                pcx_batch = outputs[5] # Probability of cluster c given x
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print("Check model definition and output structure.")
                return None

            # Determine n_nodes from the first batch
            if n_nodes is None:
                n_nodes = pcx_batch.shape[1]
            elif pcx_batch.shape[1] != n_nodes:
                 print(f"Error: Inconsistent number of nodes in model output ({pcx_batch.shape[1]} vs {n_nodes}).")
                 return None

            all_pcx.append(pcx_batch.cpu())
            all_true_labels.append(y_batch.cpu()) # Store labels on CPU

    if n_nodes is None or n_classes is None:
        print("Error: Could not determine number of nodes or classes from test data.")
        return None

    try:
        all_pcx = torch.cat(all_pcx).float() # Ensure float
        all_true_labels = torch.cat(all_true_labels)
    except RuntimeError as e:
         print(f"Error concatenating tensors (likely memory issue): {e}")
         print("Try reducing batch size or running on a machine with more memory.")
         return None

    n_test = len(all_true_labels)
    print(f"Processed {n_test} test samples. Found {n_classes} classes and {n_nodes} nodes.")

    # Move final tensors to computation device
    all_pcx = all_pcx.to(device)
    all_true_labels = all_true_labels.to(device) # Keep labels on device for filtering

    # --- Calculate Node Purities based on Test Set Assignments (On-the-Fly) ---
    print("Calculating node purities based on test set expected counts...")
    test_node_purity = torch.zeros((n_classes, n_nodes), dtype=torch.float32, device=device)

    # Calculate Expected Total Count per Node E[Nc]
    # Sum probabilities p(c|xi) over all test points i
    expected_total_count_per_node = torch.sum(all_pcx, dim=0) # Shape: (n_nodes,)

    # Calculate Expected Class Count per Node E[Nk,c]
    for k in range(n_classes):
        # Find indices of test points belonging to class k
        indices_k = (all_true_labels == k).nonzero(as_tuple=True)[0]
        if len(indices_k) > 0:
            # Sum probabilities p(c|xi) only for points xi in class k
            expected_class_count_per_node = torch.sum(all_pcx[indices_k, :], dim=0) # Shape: (n_nodes,)

            # Calculate Purity P_test(k|c) = E[Nk,c] / E[Nc]
            # Avoid division by zero
            denominator = expected_total_count_per_node + epsilon
            test_node_purity[k, :] = expected_class_count_per_node / denominator
        # else: purity remains 0 for class k if no samples of k are present

    print("Node purities calculated.")

    # --- Calculate Soft Dendrogram Purity ---
    print("Calculating Soft Dendrogram Purity (iterating over pairs)...")
    total_purity_sum = 0.0
    total_pairs = 0

    for k in tqdm(range(n_classes), desc="Processing Classes"):
        # Find indices for class k again (needed for pair iteration)
        # This could be pre-calculated and stored if memory allows
        indices_k = (all_true_labels == k).nonzero(as_tuple=True)[0]
        N_k = len(indices_k)

        if N_k < 2:
            continue

        num_pairs_k = N_k * (N_k - 1) / 2
        total_pairs += num_pairs_k
        class_purity_sum = 0.0

        # Get pcx for points in class k
        pcx_k = all_pcx[indices_k] # Shape: (Nk, n_nodes)
        # Get node purities for class k (calculated from test set)
        purities_k_vector = test_node_purity[k, :] # Shape: (n_nodes,)

        # Iterate over unique pairs within the class - O(N_k^2 * n_nodes)
        # This loop can be slow for large Nk
        for i_idx in range(N_k):
            P_i = pcx_k[i_idx, :] # Probs for point i
            for j_idx in range(i_idx + 1, N_k):
                P_j = pcx_k[j_idx, :] # Probs for point j

                # Calculate joint probability (element-wise product)
                joint_p = P_i * P_j # Shape: (n_nodes,)

                # Normalize to get weights w(c|xi, xj)
                joint_p_sum = joint_p.sum()
                if joint_p_sum < epsilon:
                    # If joint probability is negligible, contribution is 0
                    pair_purity = 0.0
                else:
                    weights = joint_p / joint_p_sum
                    # Calculate weighted purity for the pair: sum( weights * P_test(k|c) )
                    pair_purity = torch.dot(weights, purities_k_vector).item() # Use .item() for scalar

                class_purity_sum += pair_purity # Accumulate scalar value

        total_purity_sum += class_purity_sum

    if total_pairs == 0:
        print("Warning: No valid pairs found to calculate purity (test set might be too small or lack classes with >= 2 points).")
        return 0.0

    final_soft_dendrogram_purity = total_purity_sum / total_pairs
    print("Calculation complete.")

    return final_soft_dendrogram_purity

# --- Example Usage ---
if __name__ == '__main__':
    # This is a placeholder example. Replace with your actual model, data, etc.

    # 1. Define a Dummy Model (same as before)
    class DummyHierarchicalModel(torch.nn.Module):
        def __init__(self, n_features, n_nodes):
            super().__init__()
            self.n_nodes = n_nodes
            self.fc = torch.nn.Linear(n_features, n_nodes)
        def forward(self, x):
            node_logits = self.fc(x)
            pcx = F.softmax(node_logits, dim=1)
            dummy_tensor = torch.tensor(0.0, device=x.device); dummy_int = 0
            # Ensure output tuple matches expected structure
            return dummy_tensor, dummy_tensor, dummy_tensor, dummy_int, dummy_int, pcx, dummy_tensor, dummy_tensor

    # 2. Setup Parameters
    N_TEST = 150       # Increased size slightly
    N_FEATURES = 20
    N_CLASSES = 5
    N_LEAVES = 16       # Increased hierarchy size slightly
    N_NODES = 2 * N_LEAVES - 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3. Create Dummy Data and DataLoader
    X_test = torch.randn(N_TEST, N_FEATURES)
    # Ensure labels cover 0 to N_CLASSES-1 and have enough samples per class
    labels_per_class = N_TEST // N_CLASSES
    remainder = N_TEST % N_CLASSES
    true_labels_list = []
    for i in range(N_CLASSES):
        count = labels_per_class + (1 if i < remainder else 0)
        true_labels_list.extend([i] * count)
    true_labels_test = torch.tensor(true_labels_list, dtype=torch.long)
    # Shuffle data
    perm = torch.randperm(N_TEST)
    X_test = X_test[perm]
    true_labels_test = true_labels_test[perm]

    test_dataset = torch.utils.data.TensorDataset(X_test, true_labels_test)
    # Use a reasonable batch size
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # 4. No Annotation Matrix needed here

    # 5. Instantiate Model
    model = DummyHierarchicalModel(N_FEATURES, N_NODES)

    # 6. Compute Soft Purity (Test Only)
    print(f"Using device: {DEVICE}")
    soft_purity_score = compute_soft_dendrogram_purity_test_only(model, test_loader, DEVICE)

    if soft_purity_score is not None:
        print(f"\nComputed Soft Dendrogram Purity (Test Only): {soft_purity_score:.4f}")
    else:
        print("\nCalculation failed.")