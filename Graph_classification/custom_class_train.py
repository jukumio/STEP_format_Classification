#! /usr/bin/env python
from GCN import *
from datetime import datetime
from utils.my_utils import *
from utils.util import *
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import math
from train_utils import *
import glob
from collections import defaultdict
import networkx as nx
import numpy as np


torch.manual_seed(124)
np.random.seed(124)


class Graph:
    """
    Graph class to store graph data compatible with train_utils.py functions
    """
    def __init__(self):
        self.g = []  # List of nodes
        self.node_features = None  # Node features (numpy array)
        self.edge_mat = None  # Edge matrix (numpy array)
        self.label = None  # Graph label


# Parameters
# ==================================================
parser = ArgumentParser("GCN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="/outputs", help="")
parser.add_argument("--dataset", default="./dataset/Split/graph/8class/", help="Path to the dataset with class folders")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--dropout", default=0.5, type=float, help="")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The calculations will be performed on the device:", device)

# save paths
base_dir = "/3D_STEP_Classification/Graph_classification"
model_name = str(datetime.today().strftime('%m-%d'))
out_dir = os.path.join(base_dir, args.run_folder)
if not os.path.exists(out_dir + "/Models/"):
    os.makedirs(out_dir + "/Models/")
save_path = out_dir + "/Models/" + model_name
print("Results will be saved in:", out_dir)
print("    The model will be saved as:", save_path)
print("Settings:",args)


def extract_node_features_from_graphml(G):
    """
    Extract node features from GraphML graph
    This function should be customized based on your GraphML structure
    """
    nodes = list(G.nodes(data=True))
    if len(nodes) == 0:
        return np.array([[0.0]])  # Default feature for empty graphs
    
    # Get all possible feature keys from all nodes
    all_keys = set()
    for _, node_data in nodes:
        all_keys.update(node_data.keys())
    
    # Remove non-numeric keys or keys that shouldn't be features
    feature_keys = []
    for key in sorted(all_keys):
        # Skip certain keys that are typically not features
        if key.lower() in ['id', 'label', 'name', 'type']:
            continue
        feature_keys.append(key)
    
    # If no valid feature keys found, use node degree as feature
    if not feature_keys:
        print("No feature attributes found in GraphML, using node degree as feature")
        node_features = []
        for node_id, _ in nodes:
            degree = G.degree(node_id)
            node_features.append([float(degree)])
        return np.array(node_features, dtype=np.float32)
    
    # Extract features for each node
    node_features = []
    for node_id, node_data in nodes:
        features = []
        for key in feature_keys:
            if key in node_data:
                try:
                    # Try to convert to float
                    value = float(node_data[key])
                    features.append(value)
                except (ValueError, TypeError):
                    # If conversion fails, use 0.0
                    features.append(0.0)
            else:
                # Missing attribute, use 0.0
                features.append(0.0)
        node_features.append(features)
    
    return np.array(node_features, dtype=np.float32)


def load_single_graphml(file_path, label):
    """
    Load a single GraphML file and convert it to Graph object
    """
    try:
        # Load GraphML using networkx
        G = nx.read_graphml(file_path)
        
        # Create Graph object
        graph = Graph()
        
        # Extract nodes
        nodes = list(G.nodes())
        graph.g = nodes  # Store node list
        
        if len(nodes) == 0:
            print(f"Warning: Empty graph in {file_path}")
            # Create minimal graph structure for empty graphs
            graph.node_features = np.array([[0.0]], dtype=np.float32)
            graph.edge_mat = np.array([[0], [0]], dtype=np.int64)  # Self-loop for single node
            graph.label = label
            return graph
        
        # Extract node features
        graph.node_features = extract_node_features_from_graphml(G)
        
        # Extract edges
        edges = list(G.edges())
        if len(edges) == 0:
            # Create self-loops if no edges exist
            edge_list = [[i, i] for i in range(len(nodes))]
        else:
            # Create node to index mapping
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            edge_list = []
            
            for edge in edges:
                src, dst = edge
                if src in node_to_idx and dst in node_to_idx:
                    edge_list.append([node_to_idx[src], node_to_idx[dst]])
                    # Add reverse edge for undirected graph
                    if [node_to_idx[dst], node_to_idx[src]] not in edge_list:
                        edge_list.append([node_to_idx[dst], node_to_idx[src]])
        
        # Convert to numpy array and transpose for pytorch geometric format
        graph.edge_mat = np.array(edge_list, dtype=np.int64).T
        
        graph.label = label
        
        return graph
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_folder_based_data(dataset_path):
    """
    Load data from folder structure: dataset_path/class_name/split/files
    """
    # Get all class folders
    class_folders = [f for f in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, f))]
    class_folders.sort()  # Ensure consistent ordering
    
    print(f"Found classes: {class_folders}")
    
    # Create class to label mapping
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_folders)}
    num_classes = len(class_folders)
    
    print(f"Class to label mapping: {class_to_label}")
    
    # Load data for each split
    train_graphs = []
    valid_graphs = []
    test_graphs = []
    
    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        label = class_to_label[class_name]
        
        # Load train data
        train_path = os.path.join(class_path, 'train')
        if os.path.exists(train_path):
            train_files = glob.glob(os.path.join(train_path, '*.graphml'))
            print(f"Loading {len(train_files)} training files from {class_name}")
            for file_path in train_files:
                graph = load_single_graphml(file_path, label)
                if graph is not None:
                    train_graphs.append(graph)
        
        # Load validation data
        valid_path = os.path.join(class_path, 'valid')
        if os.path.exists(valid_path):
            valid_files = glob.glob(os.path.join(valid_path, '*.graphml'))
            print(f"Loading {len(valid_files)} validation files from {class_name}")
            for file_path in valid_files:
                graph = load_single_graphml(file_path, label)
                if graph is not None:
                    valid_graphs.append(graph)
        
        # Load test data
        test_path = os.path.join(class_path, 'test')
        if os.path.exists(test_path):
            test_files = glob.glob(os.path.join(test_path, '*.graphml'))
            print(f"Loading {len(test_files)} test files from {class_name}")
            for file_path in test_files:
                graph = load_single_graphml(file_path, label)
                if graph is not None:
                    test_graphs.append(graph)
    
    print(f"Loaded {len(train_graphs)} training graphs")
    print(f"Loaded {len(valid_graphs)} validation graphs") 
    print(f"Loaded {len(test_graphs)} test graphs")
    
    return train_graphs, valid_graphs, test_graphs, num_classes, class_to_label


def print_data_composition_by_class(graphs, class_to_label):
    """
    Print data composition by class
    """
    label_to_class = {v: k for k, v in class_to_label.items()}
    class_counts = defaultdict(int)
    
    for graph in graphs:
        class_name = label_to_class[graph.label]
        class_counts[class_name] += 1
    
    total_graphs = len(graphs)
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_graphs) * 100 if total_graphs > 0 else 0
        print(f"  {class_name}: {count} graphs ({percentage:.1f}%)")


# Load Graph data
# ==================================================
print("Loading data...")
use_degree_as_tag = False

# Check if dataset path exists
if not os.path.exists(args.dataset):
    raise ValueError(f"Dataset path does not exist: {args.dataset}")

# Load data from folder structure
train_graphs, valid_graphs, test_graphs, num_classes, class_to_label = load_folder_based_data(args.dataset)

# Check if we have any data
if len(train_graphs) == 0 and len(valid_graphs) == 0 and len(test_graphs) == 0:
    raise ValueError("No graphs found in the dataset!")

print("\n" + "="*50)
print("DATA SUMMARY")
print("="*50)
print("# training graphs: ", len(train_graphs))
if len(train_graphs) > 0:
    print_data_composition_by_class(train_graphs, class_to_label)

print("# validation graphs: ", len(valid_graphs))
if len(valid_graphs) > 0:
    print_data_composition_by_class(valid_graphs, class_to_label)

print("# test graphs: ", len(test_graphs))
if len(test_graphs) > 0:
    print_data_composition_by_class(test_graphs, class_to_label)

# Get feature dimension from available graphs
feature_dim_size = None
for graphs in [train_graphs, valid_graphs, test_graphs]:
    if len(graphs) > 0:
        feature_dim_size = graphs[0].node_features.shape[1]
        break

if feature_dim_size is None:
    raise ValueError("No graphs found to determine feature dimension!")

print(f"Feature dimension size: {feature_dim_size}")
print(f"Number of classes: {num_classes}")
print("="*50)
print("Loading data... finished!")

# Model
# =============================================================
print("\nCreating model...")
# Create a GCN model
model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Calculate batches per epoch based on training data
if len(train_graphs) > 0:
    num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
else:
    num_batches_per_epoch = 1

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Main process
# =============================================================
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

# Save class mapping
import json
with open(os.path.join(out_dir, 'class_mapping.json'), 'w') as f:
    json.dump(class_to_label, f, indent=2)

print(f"Class mapping saved to: {os.path.join(out_dir, 'class_mapping.json')}")

train_losses = []
train_accuracy = []
valid_losses = []
valid_accuracy = []
valid_accuracy_x_class = []

best_loss = math.inf
best_accuracy = 0

# Check if we have training data
if len(train_graphs) == 0:
    print("Warning: No training data available!")
    exit(1)

# Train loop
print(f"\nStarting training for {args.num_epochs} epochs...")
print("="*50)

for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    
    # train model
    train(mmodel=model, optimizer=optimizer, train_graphs=train_graphs, 
          batch_size=args.batch_size, num_classes=num_classes, device=device)
    
    # evaluate on train data
    train_loss, train_acc, _ = evaluate(mmodel=model, current_graphs=train_graphs, 
                                       batch_size=args.batch_size, num_classes=num_classes, 
                                       device=device, out_dir=out_dir)
    
    # evaluate on validation data if available
    if len(valid_graphs) > 0:
        valid_loss, valid_acc, valid_acc_x_class = evaluate(mmodel=model, current_graphs=valid_graphs, 
                                                           batch_size=args.batch_size, num_classes=num_classes, 
                                                           device=device, out_dir=out_dir)
        print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | train acc {:5.2f} | valid loss {:5.2f} | valid acc {:5.2f} |'.format(
            epoch, (time.time() - epoch_start_time), train_loss, train_acc*100, valid_loss, valid_acc*100))
    else:
        # Use training data for validation if no validation set
        valid_loss, valid_acc, valid_acc_x_class = train_loss, train_acc, [train_acc*100]*num_classes
        print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | train acc {:5.2f} | (no validation set) |'.format(
            epoch, (time.time() - epoch_start_time), train_loss, train_acc*100))

    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracy.append(valid_acc)
    valid_accuracy_x_class.append(valid_acc_x_class)

    # Make a step of the optimizer if the mean of the last 6 epochs were better than the current epoch
    if epoch > 5 and train_losses[-1] > np.mean(train_losses[-6:-1]):
        scheduler.step()
        print("Scheduler step")
        
    # save if best performance ever
    if best_accuracy < valid_acc or (best_accuracy == valid_acc and best_loss > valid_loss):
        print("Save at epoch: {:3d} at valid loss: {:5.2f} and valid accuracy: {:5.2f}".format(
            epoch, valid_loss, valid_acc*100))
        best_accuracy = valid_acc
        best_loss = valid_loss
        torch.save(model.state_dict(), save_path)
        torch.save(model, save_path + "_full.pt")
        
    write_acc.write('epoch ' + str(epoch) + ' acc ' + str(valid_acc*100) + '%\n')

# Plot results
# =============================================================
print("\nGenerating plots...")
valid_accuracy_x_class = np.array(valid_accuracy_x_class).T

# plot training flow
plot_training_flow(ys=[train_losses, valid_losses], names=["train", "validation"], 
                  path=out_dir, fig_name="/losses_flow", y_axis="Loss")
plot_training_flow(ys=[np.array(train_accuracy)*100, np.array(valid_accuracy)*100], 
                  names=["train","validation"], path=out_dir, fig_name="/accuracy_flow", y_axis="Accuracy")

# Evaluate on test data if available
if len(test_graphs) > 0:
    print("\nEvaluating on test data...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, _ = evaluate(mmodel=model, current_graphs=test_graphs, 
                                     batch_size=args.batch_size, num_classes=num_classes, 
                                     device=device, out_dir=out_dir, last_round=True)
    print("Evaluate: loss on test: ", test_loss, " and accuracy: ", test_acc * 100)
else:
    test_acc = 0
    print("No test data available for evaluation")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
if len(test_graphs) > 0:
    print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Number of classes: {num_classes}")
print(f"Classes: {list(class_to_label.keys())}")
print(f"Model saved to: {save_path}")
print(f"Class mapping saved to: {os.path.join(out_dir, 'class_mapping.json')}")
print("="*50)

write_acc.close()