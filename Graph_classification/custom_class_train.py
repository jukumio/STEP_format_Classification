#!/usr/bin/env python
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
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import json

try:
    import OCC
    from OCC.Core import STEPControl_Reader, TopExp_Explorer, TopAbs_VERTEX, TopAbs_EDGE
    from OCC.Core import BRep_Tool, TopoDS
    HAS_OCC = True
except ImportError:
    print("OpenCASCADE (python-opencascade) not available. Will try alternative STEP parsing.")
    HAS_OCC = False

try:
    import FreeCAD
    HAS_FREECAD = True
except ImportError:
    HAS_FREECAD = False

torch.manual_seed(124)
np.random.seed(124)


class Graph:
    """Graph class to store graph data compatible with train_utils.py functions"""
    def __init__(self):
        self.g = []  # List of nodes
        self.node_features = None  # Node features (numpy array)
        self.edge_mat = None  # Edge matrix (numpy array)
        self.label = None  # Graph label


class GraphDataProcessor:
    """Centralized graph data processing with consistent feature dimensions"""
    
    def __init__(self, target_feature_dim=None, min_feature_dim=8):
        self.target_feature_dim = target_feature_dim
        self.min_feature_dim = min_feature_dim
        self.feature_keys_cache = {}
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'empty_graphs': 0,
            'feature_dims_found': Counter()
        }
    
    def _extract_graphml_features(self, G):
        """Extract node features from GraphML graph with caching (compatible with original behavior)"""
        nodes = list(G.nodes(data=True))
        if len(nodes) == 0:
            return self._create_empty_features(1)
        
        # Get all possible feature keys from all nodes
        cache_key = id(G)  # Simple cache based on graph identity
        if cache_key not in self.feature_keys_cache:
            all_keys = set()
            for _, node_data in nodes:
                all_keys.update(node_data.keys())
            
            # Filter feature keys (using original logic - more permissive)
            feature_keys = []
            for key in sorted(all_keys):
                # Skip certain keys that are typically not features (original logic)
                if key.lower() in ['id', 'label', 'name', 'type']:
                    continue
                feature_keys.append(key)
            
            self.feature_keys_cache[cache_key] = feature_keys
        
        feature_keys = self.feature_keys_cache[cache_key]
        
        # Use node degree if no valid features found (original behavior)
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
                        # Try to convert to float (original logic)
                        value = float(node_data[key])
                        features.append(value)
                    except (ValueError, TypeError):
                        # If conversion fails, use 0.0 (original logic)
                        features.append(0.0)
                else:
                    # Missing attribute, use 0.0
                    features.append(0.0)
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)
    
    def _extract_structural_features(self, G):
        """Extract structural features when no explicit features are available"""
        nodes = list(G.nodes())
        if len(nodes) == 0:
            return self._create_empty_features(1)
        
        node_features = []
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        
        # Try to compute centrality measures (may be slow for large graphs)
        try:
            if len(nodes) <= 1000:  # Only for reasonably sized graphs
                betweenness = nx.betweenness_centrality(G)
                closeness = nx.closeness_centrality(G)
            else:
                betweenness = {node: 0.0 for node in nodes}
                closeness = {node: 0.0 for node in nodes}
        except:
            betweenness = {node: 0.0 for node in nodes}
            closeness = {node: 0.0 for node in nodes}
        
        for node in nodes:
            features = [
                float(degrees.get(node, 0)),           # degree
                float(clustering.get(node, 0.0)),      # clustering coefficient
                float(betweenness.get(node, 0.0)),     # betweenness centrality
                float(closeness.get(node, 0.0)),       # closeness centrality
            ]
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)
    
    def _extract_step_features(self, G):
        """Extract features from STEP graph with consistent dimensionality"""
        if G is None or len(G.nodes()) == 0:
            return self._create_empty_features(1)
        
        nodes = list(G.nodes(data=True))
        node_features = []
        
        # Define consistent feature extraction
        for node_id, node_data in nodes:
            features = []
            
            # Basic structural features
            features.append(float(node_data.get('degree', G.degree(node_id))))
            features.append(float(node_data.get('content_length', 0)))
            features.append(float(node_data.get('num_count', 0)))
            
            # Numeric features (standardized to 8 values)
            for i in range(8):
                features.append(float(node_data.get(f'num_{i}', 0.0)))
            
            # Entity type encoding
            entity_type = node_data.get('entity_type', 'UNKNOWN')
            type_hash = abs(hash(entity_type)) % 1000 / 1000.0
            features.append(type_hash)
            
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)
    
    def _create_empty_features(self, num_nodes=1):
        """Create consistent empty features based on target dimension"""
        if self.target_feature_dim is None:
            # Return minimal features that will be normalized later
            return np.zeros((num_nodes, self.min_feature_dim), dtype=np.float32)
        else:
            return np.zeros((num_nodes, self.target_feature_dim), dtype=np.float32)
    
    def _is_numeric(self, value):
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _parse_step_file(self, file_path):
        """Parse STEP file and return NetworkX graph"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract entities using regex
            import re
            entities = re.findall(r'#(\d+)\s*=\s*([^;]+);', content)
            
            if len(entities) == 0:
                return None
            
            # Create graph
            G = nx.Graph()
            entity_dict = {}
            
            for entity_id, entity_content in entities:
                entity_dict[entity_id] = entity_content.strip()
                G.add_node(int(entity_id))
            
            # Add edges based on entity references
            for entity_id, entity_content in entities:
                refs = re.findall(r'#(\d+)', entity_content)
                for ref in refs:
                    if ref in entity_dict and ref != entity_id:
                        G.add_edge(int(entity_id), int(ref))
            
            # Add node attributes
            for node in G.nodes():
                entity_content = entity_dict.get(str(node), "")
                
                features = {}
                features['entity_type'] = entity_content.split('(')[0] if '(' in entity_content else 'UNKNOWN'
                features['degree'] = G.degree(node)
                features['content_length'] = len(entity_content)
                
                # Extract numbers with better parsing
                numbers = re.findall(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', entity_content)
                features['num_count'] = len(numbers)
                
                # Store up to 8 numeric values
                if numbers:
                    try:
                        numeric_values = [float(n.split('E')[0].split('e')[0]) for n in numbers[:8]]
                        for i, val in enumerate(numeric_values):
                            features[f'num_{i}'] = val
                    except:
                        pass
                
                G.nodes[node].update(features)
            
            return G
            
        except Exception as e:
            print(f"Error parsing STEP file {file_path}: {e}")
            return None
    
    def determine_target_dimension(self, all_graphs):
        """Determine the target feature dimension based on all graphs"""
        if self.target_feature_dim is not None:
            return self.target_feature_dim
        
        if not all_graphs:
            return self.min_feature_dim
        
        # Collect all dimensions
        dims = []
        for graph in all_graphs:
            if graph is not None and graph.node_features is not None:
                dims.append(graph.node_features.shape[1])
        
        if not dims:
            return self.min_feature_dim
        
        # Use the most common dimension, but ensure it's at least min_feature_dim
        dim_counts = Counter(dims)
        most_common_dim = dim_counts.most_common(1)[0][0]
        target_dim = max(most_common_dim, self.min_feature_dim)
        
        print(f"Determined target feature dimension: {target_dim}")
        print(f"Dimension distribution: {dict(dim_counts)}")
        
        self.target_feature_dim = target_dim
        return target_dim
    
    def normalize_features(self, graph):
        """Normalize a single graph's features to target dimension"""
        if graph is None or graph.node_features is None:
            return graph
        
        current_dim = graph.node_features.shape[1]
        num_nodes = graph.node_features.shape[0]
        
        if current_dim == self.target_feature_dim:
            return graph
        
        if current_dim < self.target_feature_dim:
            # Pad with zeros
            padding = np.zeros((num_nodes, self.target_feature_dim - current_dim), dtype=np.float32)
            graph.node_features = np.concatenate([graph.node_features, padding], axis=1)
        else:
            # Truncate (with warning)
            if current_dim > self.target_feature_dim:
                print(f"Warning: Truncating features from {current_dim} to {self.target_feature_dim}")
            graph.node_features = graph.node_features[:, :self.target_feature_dim]
        
        return graph
    
    def process_file(self, file_path, label):
        """Process a single file and return Graph object"""
        self.stats['files_processed'] += 1
        
        try:
            if file_path.lower().endswith(('.stp', '.step')):
                return self._process_step_file(file_path, label)
            elif file_path.lower().endswith(('.graphml', '.xml', '.gml')):
                return self._process_graphml_file(file_path, label)
            else:
                print(f"Unsupported file format: {file_path}")
                self.stats['files_failed'] += 1
                return None
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            self.stats['files_failed'] += 1
            return None
    
    def _process_graphml_file(self, file_path, label):
        """Process GraphML file"""
        try:
            G = nx.read_graphml(file_path)
            return self._create_graph_object(G, label, self._extract_graphml_features)
        except Exception as e:
            print(f"Error loading GraphML {file_path}: {e}")
            self.stats['files_failed'] += 1
            return None
    
    def _process_step_file(self, file_path, label):
        """Process STEP file"""
        try:
            G = self._parse_step_file(file_path)
            if G is None:
                self.stats['files_failed'] += 1
                return None
            return self._create_graph_object(G, label, self._extract_step_features)
        except Exception as e:
            print(f"Error loading STEP {file_path}: {e}")
            self.stats['files_failed'] += 1
            return None
    
    def _create_graph_object(self, G, label, feature_extractor):
        """Create Graph object from NetworkX graph"""
        graph = Graph()
        
        if G is None or len(G.nodes()) == 0:
            print(f"Warning: Empty graph encountered")
            self.stats['empty_graphs'] += 1
            
            # Create minimal valid graph
            graph.g = [0]
            graph.node_features = self._create_empty_features(1)
            graph.edge_mat = np.array([[0], [0]], dtype=np.int64)
            graph.label = label
            return graph
        
        # Extract nodes and features
        nodes = list(G.nodes())
        graph.g = list(range(len(nodes)))  # Use indices instead of original node IDs
        graph.node_features = feature_extractor(G)
        
        # Track feature dimension
        if graph.node_features is not None:
            self.stats['feature_dims_found'][graph.node_features.shape[1]] += 1
        
        # Extract edges with proper indexing
        edges = list(G.edges())
        if len(edges) == 0:
            # Create self-loops for isolated nodes
            edge_list = [[i, i] for i in range(len(nodes))]
        else:
            # Map original node IDs to indices
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            edge_list = []
            
            for src, dst in edges:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edge_list.append([src_idx, dst_idx])
                    # Add reverse edge for undirected graph
                    if src_idx != dst_idx:
                        edge_list.append([dst_idx, src_idx])
        
        # Convert to proper format
        if edge_list:
            graph.edge_mat = np.array(edge_list, dtype=np.int64).T
        else:
            graph.edge_mat = np.array([[0], [0]], dtype=np.int64)
        
        graph.label = label
        return graph
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("DATA PROCESSING STATISTICS")
        print("="*50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files failed: {self.stats['files_failed']}")
        print(f"Empty graphs: {self.stats['empty_graphs']}")
        print(f"Feature dimensions found: {dict(self.stats['feature_dims_found'])}")
        if self.target_feature_dim:
            print(f"Target feature dimension: {self.target_feature_dim}")
        print("="*50)


def find_files_recursively(directory, extensions=['.graphml', '.xml', '.gml', '.stp', '.step']):
    """Recursively find files with given extensions"""
    found_files = []
    if not os.path.exists(directory):
        return found_files
        
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                found_files.append(os.path.join(root, file))
    return found_files


def analyze_dataset_structure(dataset_path):
    """Analyze dataset structure and return comprehensive information"""
    print(f"Analyzing dataset structure at: {dataset_path}")
    print(f"Absolute path: {os.path.abspath(dataset_path)}")
    print(f"Path exists: {os.path.exists(dataset_path)}")
    
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist!")
        return None, None, {}
    
    # Find all graph files
    all_files = find_files_recursively(dataset_path)
    print(f"Found {len(all_files)} graph files total")
    
    # Analyze file types
    file_types = Counter()
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        file_types[ext] += 1
    
    analysis = {
        'total_files': len(all_files),
        'file_types': dict(file_types),
        'sample_files': all_files[:5]
    }
    
    # Check directory structure
    items = os.listdir(dataset_path)
    class_folders = [f for f in items if os.path.isdir(os.path.join(dataset_path, f))]
    
    structures = []
    if class_folders:
        # Check if class/split/files structure exists
        has_split_structure = False
        for class_folder in class_folders:
            class_path = os.path.join(dataset_path, class_folder)
            class_items = os.listdir(class_path)
            if any(split in class_items for split in ['train', 'valid', 'test', 'validation']):
                has_split_structure = True
                break
        
        if has_split_structure:
            structures.append("class/split/files")
        else:
            structures.append("class/files_directly")
    
    if any(split in items for split in ['train', 'valid', 'test', 'validation']):
        structures.append("split/class/files")
    
    analysis['structures'] = structures
    analysis['class_folders'] = class_folders
    
    print(f"File types: {dict(file_types)}")
    print(f"Detected structures: {structures}")
    print(f"Class folders: {class_folders}")
    
    return all_files, structures, analysis


def load_dataset(dataset_path, processor):
    """Load dataset with improved structure handling"""
    # Analyze dataset first
    all_files, structures, analysis = analyze_dataset_structure(dataset_path)
    
    if all_files is None:
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    if len(all_files) == 0:
        raise ValueError(f"No graph files found in dataset: {dataset_path}")
    
    # Get class information
    class_folders = analysis['class_folders']
    if not class_folders:
        raise ValueError("No class folders found in dataset")
    
    class_folders.sort()
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_folders)}
    
    print(f"Found classes: {class_folders}")
    print(f"Class to label mapping: {class_to_label}")
    
    # Load data based on structure
    train_graphs, valid_graphs, test_graphs = [], [], []
    
    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        label = class_to_label[class_name]
        
        print(f"\nProcessing class: {class_name}")
        
        # Check for split structure
        has_splits = any(os.path.exists(os.path.join(class_path, split)) 
                        for split in ['train', 'valid', 'validation', 'test'])
        
        if has_splits:
            # Load from split folders
            for split_name, graph_list in [('train', train_graphs), 
                                          ('valid', valid_graphs), 
                                          ('validation', valid_graphs),
                                          ('test', test_graphs)]:
                split_path = os.path.join(class_path, split_name)
                if os.path.exists(split_path):
                    split_files = find_files_recursively(split_path)
                    print(f"Loading {len(split_files)} {split_name} files from {class_name}")
                    
                    for file_path in split_files:
                        graph = processor.process_file(file_path, label)
                        if graph is not None:
                            graph_list.append(graph)
        else:
            # Load all files and split randomly
            class_files = find_files_recursively(class_path)
            print(f"Loading {len(class_files)} files directly from {class_name}")
            
            # Random split (70/15/15)
            np.random.shuffle(class_files)
            n_files = len(class_files)
            n_train = int(0.7 * n_files)
            n_valid = int(0.15 * n_files)
            
            splits = [
                (class_files[:n_train], train_graphs),
                (class_files[n_train:n_train+n_valid], valid_graphs),
                (class_files[n_train+n_valid:], test_graphs)
            ]
            
            for files, graph_list in splits:
                for file_path in files:
                    graph = processor.process_file(file_path, label)
                    if graph is not None:
                        graph_list.append(graph)
    
    return train_graphs, valid_graphs, test_graphs, len(class_folders), class_to_label


def normalize_all_graphs(train_graphs, valid_graphs, test_graphs, processor):
    """Normalize all graphs to consistent feature dimensions"""
    print("\nNormalizing feature dimensions...")
    
    # Determine target dimension from all graphs
    all_graphs = train_graphs + valid_graphs + test_graphs
    target_dim = processor.determine_target_dimension(all_graphs)
    
    # Normalize all graphs
    for graph in all_graphs:
        processor.normalize_features(graph)
    
    # Verify consistency
    dims = [g.node_features.shape[1] for g in all_graphs if g.node_features is not None]
    unique_dims = set(dims)
    
    if len(unique_dims) > 1:
        raise ValueError(f"Feature normalization failed! Still have dimensions: {unique_dims}")
    
    print(f"Successfully normalized all graphs to {target_dim} dimensions")
    return target_dim


def print_data_summary(graphs, class_to_label, split_name):
    """Print data composition by class"""
    if not graphs:
        print(f"# {split_name} graphs: 0")
        return
    
    label_to_class = {v: k for k, v in class_to_label.items()}
    class_counts = defaultdict(int)
    
    for graph in graphs:
        class_name = label_to_class[graph.label]
        class_counts[class_name] += 1
    
    total = len(graphs)
    print(f"# {split_name} graphs: {total}")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {class_name}: {count} graphs ({percentage:.1f}%)")


# Main execution starts here
# ==================================================

# Parameters
parser = ArgumentParser("GCN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--run_folder", default="../", help="Output folder")
parser.add_argument("--dataset", default="/Users/juheon/Desktop/newdataset/Split/8class/", help="Path to the dataset")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
parser.add_argument("--target_feature_dim", default=None, type=int, help="Target feature dimension (auto-detect if None)")
parser.add_argument("--min_feature_dim", default=8, type=int, help="Minimum feature dimension")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Calculations will be performed on device:", device)

# Setup output paths
model_name = str(datetime.today().strftime('%m-%d'))
out_dir = os.path.abspath(args.run_folder)
os.makedirs(out_dir + "/Models/", exist_ok=True)
os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

save_path = out_dir + "/Models/" + model_name
print(f"Results will be saved in: {out_dir}")
print(f"Model will be saved as: {save_path}")
print(f"Settings: {args}")

# Initialize data processor
processor = GraphDataProcessor(
    target_feature_dim=args.target_feature_dim,
    min_feature_dim=args.min_feature_dim
)

# Load and process data
print("\n" + "="*50)
print("LOADING DATA")
print("="*50)

train_graphs, valid_graphs, test_graphs, num_classes, class_to_label = load_dataset(args.dataset, processor)

# Print processing statistics
processor.print_stats()

# Normalize all graphs to consistent dimensions
feature_dim_size = normalize_all_graphs(train_graphs, valid_graphs, test_graphs, processor)

# Print data summary
print("\n" + "="*50)
print("DATA SUMMARY")
print("="*50)
print_data_summary(train_graphs, class_to_label, "training")
print_data_summary(valid_graphs, class_to_label, "validation")
print_data_summary(test_graphs, class_to_label, "test")

print(f"Feature dimension: {feature_dim_size}")
print(f"Number of classes: {num_classes}")
print("="*50)

# Verify we have training data
if len(train_graphs) == 0:
    raise ValueError("No training data available!")

# Save class mapping
with open(os.path.join(out_dir, 'class_mapping.json'), 'w') as f:
    json.dump(class_to_label, f, indent=2)

# Create model
print("\nCreating model...")
model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

num_batches_per_epoch = max(1, int((len(train_graphs) - 1) / args.batch_size) + 1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Training setup
checkpoint_prefix = os.path.join(out_dir, "checkpoints", "model")
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

train_losses, train_accuracy = [], []
valid_losses, valid_accuracy = [], []
valid_accuracy_x_class = []
best_loss, best_accuracy = math.inf, 0

# Training loop
print(f"\nStarting training for {args.num_epochs} epochs...")
print("="*50)

for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    
    # Train model
    train(mmodel=model, optimizer=optimizer, train_graphs=train_graphs, 
          batch_size=args.batch_size, num_classes=num_classes, device=device)
    
    # Evaluate on training data
    train_loss, train_acc, _ = evaluate(mmodel=model, current_graphs=train_graphs, 
                                       batch_size=args.batch_size, num_classes=num_classes, 
                                       device=device, out_dir=out_dir)
    
    # Evaluate on validation data
    if len(valid_graphs) > 0:
        valid_loss, valid_acc, valid_acc_x_class = evaluate(mmodel=model, current_graphs=valid_graphs, 
                                                           batch_size=args.batch_size, num_classes=num_classes, 
                                                           device=device, out_dir=out_dir)
        print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | train acc {:5.2f} | valid loss {:5.2f} | valid acc {:5.2f} |'.format(
            epoch, (time.time() - epoch_start_time), train_loss, train_acc*100, valid_loss, valid_acc*100))
    else:
        valid_loss, valid_acc, valid_acc_x_class = train_loss, train_acc, [train_acc*100]*num_classes
        print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | train acc {:5.2f} | (no validation set) |'.format(
            epoch, (time.time() - epoch_start_time), train_loss, train_acc*100))

    # Store metrics
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracy.append(valid_acc)
    valid_accuracy_x_class.append(valid_acc_x_class)

    # Learning rate scheduling
    if epoch > 5 and train_losses[-1] > np.mean(train_losses[-6:-1]):
        scheduler.step()
        print("Scheduler step")
        
    # Save best model
    if best_accuracy < valid_acc or (best_accuracy == valid_acc and best_loss > valid_loss):
        print("Save at epoch: {:3d} at valid loss: {:5.2f} and valid accuracy: {:5.2f}".format(
            epoch, valid_loss, valid_acc*100))
        best_accuracy = valid_acc
        best_loss = valid_loss
        torch.save(model.state_dict(), save_path)
        torch.save(model, save_path + "_full.pt")
        
        write_acc.write('epoch ' + str(epoch) + ' acc ' + str(valid_acc*100) + '%\n')

# Generate plots
print("\nGenerating plots...")
valid_accuracy_x_class = np.array(valid_accuracy_x_class).T

# Plot training flow
plot_training_flow(ys=[train_losses, valid_losses], names=["train", "validation"], 
                  path=out_dir, fig_name="/losses_flow", y_axis="Loss")
plot_training_flow(ys=[np.array(train_accuracy)*100, np.array(valid_accuracy)*100], 
                  names=["train","validation"], path=out_dir, fig_name="/accuracy_flow", y_axis="Accuracy")

# Final evaluation on test data
if len(test_graphs) > 0:
    print("\nEvaluating on test data...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, _ = evaluate(mmodel=model, current_graphs=test_graphs, 
                                     batch_size=args.batch_size, num_classes=num_classes, 
                                     device=device, out_dir=out_dir, last_round=True)
    print("Test evaluation - loss: {:.4f}, accuracy: {:.2f}%".format(test_loss, test_acc * 100))
else:
    test_acc = 0
    print("No test data available for evaluation")

# Final results
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
if len(test_graphs) > 0:
    print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Number of classes: {num_classes}")
print(f"Classes: {list(class_to_label.keys())}")
print(f"Feature dimension used: {feature_dim_size}")
print(f"Model saved to: {save_path}")
print(f"Class mapping saved to: {os.path.join(out_dir, 'class_mapping.json')}")

# Print final processing statistics
processor.print_stats()
print("="*50)

write_acc.close()
print("Training completed successfully!")