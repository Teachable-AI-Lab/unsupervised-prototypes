import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import untils
from cobweb.cobweb_continuous import CobwebContinuousTree
import json
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from copy import deepcopy

class CobwebSymbolic():
    def __init__(self, input_dim, depth=5):
        self.input_dim = input_dim
        self.depth = depth
        self.tree = CobwebContinuousTree(size=self.input_dim, covar_from=2, depth=self.depth, branching_factor=10)

    def train(self, train_data, epochs=10):
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        for epoch in range(epochs):
            for (x, y) in tqdm(train_loader):
                x = x.view(-1).numpy()
                self.tree.ifit(x)

    def save_tree_to_json(self, filename):
        # Convert tree to serializable format
        def convert_node_to_dict(node):
            if node is None:
                return None
            
            node_dict = {
                "mean": node.mean.tolist() if isinstance(node.mean, np.ndarray) else node.mean,
                "sum_sq": node.sum_sq.tolist() if isinstance(node.sum_sq, np.ndarray) else node.sum_sq,
                "count": node.count.tolist() if isinstance(node.count, np.ndarray) else node.count,
                "children": []
            }
            
            if hasattr(node, 'children'):
                for child in node.children:
                    node_dict["children"].append(convert_node_to_dict(child))
            
            return node_dict

        # Convert the tree to a dictionary
        tree_dict = convert_node_to_dict(self.tree.root)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(tree_dict, f)

    def load_tree_in_torch(self, filename):
        with open(filename, 'r') as f:
            temp_tree = json.load(f)

        pq = [temp_tree]

        while True:
            curr = pq.pop(0)
            curr["mean"] = torch.tensor(curr["mean"])
            curr["sum_sq"] = torch.tensor(curr["sum_sq"])
            curr["count"] = torch.tensor(curr["count"])
            curr["logvar"] = torch.log(curr["sum_sq"] / curr["count"])
            
            if "children" not in curr or not curr["children"]:
                break
                
            for child in curr["children"]:
                pq.append(child)

        self.tree = temp_tree
    
    def tensor_to_base64(self, tensor, shape, cmap="gray", normalize=False):
        array = tensor.numpy().reshape(shape)
        if normalize:
            plt.imshow(array, cmap=cmap, aspect="auto")
        else:
            plt.imshow(array, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        plt.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def viz_cobweb_tree(self, viz_filename):
        temp_tree = deepcopy(self.tree)
        pq = [temp_tree]
        while True:
            curr = pq.pop(0)
            curr["image"] = self.tensor_to_base64(torch.tensor(curr["mean"]), (28, 28), cmap="inferno", normalize=True)
            curr.pop("mean")
            curr.pop("sum_sq")
            curr.pop("count")
            curr.pop("logvar")
            
            if "children" not in curr or not curr["children"]:
                break
                
            for child in curr["children"]:
                pq.append(child)
                
        with open(f'{viz_filename}.json', 'w') as f:
            json.dump(temp_tree, f)