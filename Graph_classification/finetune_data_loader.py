import os
import networkx as nx
import numpy as np
import json
from utils.util import S2VGraph

def load_finetune_data_with_mapping(dataset_path, class_mapping_path, degree_as_tag=False, use_existing_splits=True):
    """
    Load data from directory structure with predefined class mapping.
    
    Args:
        dataset_path: Path to dataset (e.g., "dataset/Split/graph/scaled_8class/")
        class_mapping_path: Path to class_mapping.json file
        degree_as_tag: Whether to use node degree as features
        use_existing_splits: If True, loads data respecting train/valid/test splits
                           If False, loads all data together for random splitting
    
    Returns:
        If use_existing_splits=True: (train_graphs, valid_graphs, test_graphs), num_classes
        If use_existing_splits=False: graphs, num_classes
    """
    # Load predefined class mapping
    with open(class_mapping_path, 'r') as f:
        class_to_label = json.load(f)
    
    print(f"Using predefined class mapping: {class_to_label}")
    
    g_list = []
    train_graphs = []
    valid_graphs = []
    test_graphs = []
    feat_dict = {}
    
    dataset_path = dataset_path.rstrip('/')
    if not dataset_path.endswith('/'):
        dataset_path += '/'
    
    print(f"Loading data from: {dataset_path}")
    
    # Get all class directories that exist in the mapping
    available_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Filter to only include classes that are in our mapping
    valid_class_dirs = [d for d in available_dirs if d in class_to_label]
    missing_classes = set(class_to_label.keys()) - set(available_dirs)
    extra_dirs = set(available_dirs) - set(class_to_label.keys())
    
    if missing_classes:
        print(f"Warning: Classes in mapping but not found in dataset: {missing_classes}")
    if extra_dirs:
        print(f"Warning: Directories found but not in mapping (will be ignored): {extra_dirs}")
    
    if not valid_class_dirs:
        print(f"No valid class directories found in {dataset_path}")
        return ([], [], []), len(class_to_label) if use_existing_splits else [], len(class_to_label)
    
    print(f"Processing classes: {valid_class_dirs}")
    
    # Process each valid class
    for class_name in valid_class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        label = class_to_label[class_name]  # Use predefined mapping
        
        print(f"Loading class: {class_name} (label: {label})")
        
        if use_existing_splits:
            # Load from train/valid/test subdirectories
            for split in ['train', 'valid', 'test']:
                split_path = os.path.join(class_path, split)
                if not os.path.exists(split_path):
                    print(f"Warning: {split} directory not found for class {class_name}")
                    continue
                
                graphs = load_graphs_from_directory(split_path, label, class_name, feat_dict)
                
                if split == 'train':
                    train_graphs.extend(graphs)
                elif split == 'valid':
                    valid_graphs.extend(graphs)
                elif split == 'test':
                    test_graphs.extend(graphs)
                
                print(f"  {split}: {len(graphs)} graphs")
        else:
            # Load all graphs from all subdirectories together
            for split in ['train', 'valid', 'test']:
                split_path = os.path.join(class_path, split)
                if os.path.exists(split_path):
                    graphs = load_graphs_from_directory(split_path, label, class_name, feat_dict)
                    g_list.extend(graphs)
    
    # Post-process graphs (same as original)
    all_graphs = train_graphs + valid_graphs + test_graphs if use_existing_splits else g_list
    
    for g in all_graphs:
        # Create node ID mapping (convert string IDs to integers)
        dict_node_id = {}
        for node in g.g:
            if node not in dict_node_id:
                dict_node_id[node] = len(dict_node_id)

        # Build neighbors list
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            int_i = dict_node_id[i]
            int_j = dict_node_id[j]
            g.neighbors[int_i].append(int_j)
            g.neighbors[int_j].append(int_i)
        
        # Calculate max neighbor count
        degree_list = [len(neighbors) for neighbors in g.neighbors]
        g.max_neighbor = max(degree_list) if degree_list else 0

        # Create edge matrix
        edges = []
        for pair in g.g.edges():
            g1, g2 = pair
            edges.append([dict_node_id[g1], dict_node_id[g2]])
        edges.extend([[j, i] for i, j in edges])  # Add reverse edges
        
        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1, 0)) if edges else np.empty((2, 0), dtype=np.int32)

    # Handle degree as tag option
    if degree_as_tag:
        for g in all_graphs:
            g.node_tags = list(dict(g.g.degree).values())

    # Create feature vectors (one-hot encoding of node tags)
    tagset = set([])
    for g in all_graphs:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in all_graphs:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
        if g.node_tags:  # Only if we have node tags
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    num_classes = len(class_to_label)  # Use mapping size, not discovered classes
    print(f'# classes (from mapping): {num_classes}')
    print(f'# maximum node tag: {len(tagset)}')
    
    if use_existing_splits:
        print(f"# train graphs: {len(train_graphs)}")
        print(f"# valid graphs: {len(valid_graphs)}")
        print(f"# test graphs: {len(test_graphs)}")
        print(f"# total graphs: {len(all_graphs)}")
        return (train_graphs, valid_graphs, test_graphs), num_classes
    else:
        print(f"# total graphs: {len(g_list)}")
        return g_list, num_classes


def load_graphs_from_directory(directory_path, class_label, class_name, feat_dict):
    """Load all .graphml files from a directory."""
    graphs = []
    
    if not os.path.exists(directory_path):
        return graphs
    
    files = [f for f in os.listdir(directory_path) if f.endswith('.graphml')]
    
    for file in files:
        try:
            file_path = os.path.join(directory_path, file)
            g = nx.read_graphml(file_path)
            
            node_tags = []
            for node in g:
                # Get node type/label - adjust this based on your GraphML structure
                node_lab = g.nodes[node].get("type", "default")  # Default if no 'type' attribute
                
                if node_lab not in feat_dict:
                    feat_dict[node_lab] = len(feat_dict)
                node_tags.append(feat_dict[node_lab])
            
            # Create S2VGraph object
            graph_obj = S2VGraph(g, class_label, node_tags, name_graph=file)
            graphs.append(graph_obj)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    return graphs


def print_data_composition(graphs, class_mapping_path, title="Data composition"):
    """Print data composition by class using original class names."""
    print(f"\n{title}:")
    
    # Load class mapping for reverse lookup
    with open(class_mapping_path, 'r') as f:
        class_to_label = json.load(f)
    
    # Create reverse mapping
    label_to_class = {v: k for k, v in class_to_label.items()}
    
    class_counts = {}
    for g in graphs:
        class_name = label_to_class.get(g.label, f"Unknown_{g.label}")
        if class_name not in class_counts:
            class_counts[class_name] = []
        class_counts[class_name].append(g)
    
    for class_name in sorted(class_counts.keys()):
        graphs_in_class = class_counts[class_name]
        print(f"  {class_name}: {len(graphs_in_class)} graphs")