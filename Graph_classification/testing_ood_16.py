#!/usr/bin/env python
"""
Integrated Graph Classification Testing and Analysis Script
Tests model and generates comprehensive analysis with confusion matrices
"""

import torch
import numpy as np
import os
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Import training modules
from GCN import *
from utils.my_utils import *
from utils.util import *
from train_utils import *

# Set seeds for reproducibility
torch.manual_seed(124)
np.random.seed(124)
warnings.filterwarnings('ignore')


class Graph:
    """Simple graph data structure"""
    def __init__(self):
        self.g = []
        self.node_features = None
        self.edge_mat = None
        self.label = None


class GraphProcessor:
    """Graph processing with proper type handling"""
    
    def __init__(self, target_feature_dim=None):
        self.target_feature_dim = target_feature_dim
        self.stats = {'processed': 0, 'failed': 0, 'empty': 0}
    
    def _safe_convert_features(self, features):
        """Convert features to numpy float32 safely"""
        if features is None:
            return np.zeros((1, self.target_feature_dim or 8), dtype=np.float32)
        
        try:
            features_np = np.array(features, dtype=np.float32)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert features to numpy array: {e}")
            return np.zeros((1, self.target_feature_dim or 8), dtype=np.float32)
        
        if features_np.size == 0:
            return np.zeros((1, self.target_feature_dim or 8), dtype=np.float32)
        
        features_np = np.nan_to_num(features_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if len(features_np.shape) == 1:
            features_np = features_np.reshape(1, -1)
        elif len(features_np.shape) == 0:
            features_np = np.array([[float(features_np)]], dtype=np.float32)
        
        return features_np
    
    def _safe_convert_edges(self, edges, num_nodes):
        """Convert edges to numpy int64 safely"""
        default_edges = np.array([[i, i] for i in range(num_nodes)], dtype=np.int64).T
        
        if edges is None:
            return default_edges
        
        try:
            if isinstance(edges, list):
                if not edges:
                    return default_edges
                edges_np = np.array(edges, dtype=np.int64)
            else:
                edges_np = np.array(edges, dtype=np.int64)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert edges to numpy array: {e}")
            return default_edges
        
        if edges_np.size == 0:
            return default_edges
        
        if len(edges_np.shape) == 1:
            if edges_np.size >= 2:
                edges_np = edges_np[:2].reshape(2, 1)
            else:
                return default_edges
        elif edges_np.shape[0] != 2:
            if edges_np.shape[1] == 2:
                edges_np = edges_np.T
            else:
                print(f"Warning: Unusual edge shape {edges_np.shape}, using self-loops")
                return default_edges
        
        edges_np = np.clip(edges_np, 0, num_nodes - 1)
        return edges_np
    
    def _extract_graphml_features(self, G):
        """Extract features from GraphML"""
        if len(G.nodes()) == 0:
            return self._safe_convert_features(None)
        
        nodes = list(G.nodes(data=True))
        node_features = []
        
        all_keys = set()
        for _, data in nodes:
            if isinstance(data, dict):
                all_keys.update(data.keys())
        
        feature_keys = [k for k in sorted(all_keys) 
                       if k.lower() not in ['id', 'label', 'name', 'type', 'node_id']]
        
        if not feature_keys:
            for node_id, _ in nodes:
                degree = G.degree(node_id)
                features = [float(degree)]
                node_features.append(features)
        else:
            for node_id, data in nodes:
                features = []
                for key in feature_keys:
                    try:
                        value = data.get(key, 0) if isinstance(data, dict) else 0
                        if isinstance(value, (int, float)):
                            features.append(float(value))
                        elif isinstance(value, str):
                            try:
                                features.append(float(value))
                            except ValueError:
                                features.append(float(abs(hash(value)) % 1000) / 1000.0)
                        else:
                            features.append(0.0)
                    except (ValueError, TypeError, AttributeError):
                        features.append(0.0)
                
                if not features:
                    features = [0.0]
                
                node_features.append(features)
        
        return self._safe_convert_features(node_features)
    
    def _extract_step_features(self, G):
        """Extract features from STEP graph"""
        if len(G.nodes()) == 0:
            return self._safe_convert_features(None)
        
        nodes = list(G.nodes(data=True))
        node_features = []
        
        for node_id, data in nodes:
            features = []
            
            try:
                degree = data.get('degree', G.degree(node_id)) if isinstance(data, dict) else G.degree(node_id)
                features.append(float(degree))
            except:
                features.append(0.0)
            
            try:
                content_len = data.get('content_length', 0) if isinstance(data, dict) else 0
                features.append(float(content_len))
            except:
                features.append(0.0)
            
            try:
                num_count = data.get('num_count', 0) if isinstance(data, dict) else 0
                features.append(float(num_count))
            except:
                features.append(0.0)
            
            for i in range(8):
                try:
                    val = data.get(f'num_{i}', 0.0) if isinstance(data, dict) else 0.0
                    features.append(float(val))
                except:
                    features.append(0.0)
            
            try:
                entity_type = data.get('entity_type', 'UNKNOWN') if isinstance(data, dict) else 'UNKNOWN'
                type_hash = abs(hash(str(entity_type))) % 1000 / 1000.0
                features.append(float(type_hash))
            except:
                features.append(0.0)
            
            node_features.append(features)
        
        return self._safe_convert_features(node_features)
    
    def _parse_step_file(self, file_path):
        """Parse STEP file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            import re
            entities = re.findall(r'#(\d+)\s*=\s*([^;]+);', content)
            
            if not entities:
                return None
            
            G = nx.Graph()
            entity_dict = {}
            
            for entity_id, entity_content in entities:
                try:
                    node_id = int(entity_id)
                    entity_dict[entity_id] = entity_content.strip()
                    G.add_node(node_id)
                except ValueError:
                    continue
            
            for entity_id, entity_content in entities:
                try:
                    refs = re.findall(r'#(\d+)', entity_content)
                    src_id = int(entity_id)
                    for ref in refs:
                        if ref in entity_dict and ref != entity_id:
                            dst_id = int(ref)
                            if G.has_node(src_id) and G.has_node(dst_id):
                                G.add_edge(src_id, dst_id)
                except (ValueError, KeyError):
                    continue
            
            for node in G.nodes():
                try:
                    content = entity_dict.get(str(node), "")
                    
                    features = {
                        'entity_type': content.split('(')[0] if '(' in content else 'UNKNOWN',
                        'degree': G.degree(node),
                        'content_length': len(content),
                    }
                    
                    numbers = re.findall(r'[-+]?\d*\.?\d+', content)
                    features['num_count'] = len(numbers)
                    
                    for i, num_str in enumerate(numbers[:8]):
                        try:
                            features[f'num_{i}'] = float(num_str)
                        except (ValueError, OverflowError):
                            features[f'num_{i}'] = 0.0
                    
                    G.nodes[node].update(features)
                except Exception as e:
                    G.nodes[node].update({
                        'entity_type': 'UNKNOWN',
                        'degree': G.degree(node),
                        'content_length': 0,
                        'num_count': 0
                    })
            
            return G
            
        except Exception as e:
            print(f"Error parsing STEP file: {e}")
            return None
    
    def normalize_features(self, graph):
        """Normalize features to target dimension"""
        if graph is None or graph.node_features is None or self.target_feature_dim is None:
            return
        
        try:
            current_dim = graph.node_features.shape[1]
            
            if current_dim < self.target_feature_dim:
                num_nodes = graph.node_features.shape[0]
                padding = np.zeros((num_nodes, self.target_feature_dim - current_dim), dtype=np.float32)
                graph.node_features = np.concatenate([graph.node_features, padding], axis=1)
            elif current_dim > self.target_feature_dim:
                graph.node_features = graph.node_features[:, :self.target_feature_dim]
        except Exception as e:
            print(f"Error normalizing features: {e}")
    
    def process_file(self, file_path, label=None):
        """Process a single file"""
        self.stats['processed'] += 1
        
        try:
            if file_path.lower().endswith(('.stp', '.step')):
                G = self._parse_step_file(file_path)
                if G is None:
                    self.stats['failed'] += 1
                    return None
                features = self._extract_step_features(G)
                
            elif file_path.lower().endswith(('.graphml', '.xml', '.gml')):
                try:
                    G = nx.read_graphml(file_path)
                except Exception as e:
                    print(f"Error reading GraphML file {file_path}: {e}")
                    self.stats['failed'] += 1
                    return None
                features = self._extract_graphml_features(G)
                
            else:
                print(f"Unsupported file format: {file_path}")
                self.stats['failed'] += 1
                return None
            
            nodes = list(G.nodes())
            if not nodes:
                print(f"Empty graph in file: {file_path}")
                self.stats['empty'] += 1
                return None
            
            graph = Graph()
            graph.g = list(range(len(nodes)))
            graph.node_features = features
            graph.label = label
            
            edges = list(G.edges())
            if not edges:
                edge_list = [[i, i] for i in range(len(nodes))]
            else:
                node_to_idx = {node: idx for idx, node in enumerate(nodes)}
                edge_list = []
                
                for src, dst in edges:
                    if src in node_to_idx and dst in node_to_idx:
                        src_idx = node_to_idx[src]
                        dst_idx = node_to_idx[dst]
                        edge_list.extend([[src_idx, dst_idx], [dst_idx, src_idx]])
                
                if not edge_list:
                    edge_list = [[i, i] for i in range(len(nodes))]
            
            graph.edge_mat = self._safe_convert_edges(edge_list, len(nodes))
            
            return graph
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            self.stats['failed'] += 1
            return None


class IntegratedAnalyzer:
    """Integrated testing and analysis class"""
    
    def __init__(self, model_path, run_folder, output_dir="analysis_results"):
        self.model_path = model_path
        self.run_folder = run_folder
        self.output_dir = output_dir
        self.model = None
        self.class_to_label = None
        self.label_to_class = None
        self.device = None
        self.processor = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model_and_config(self):
        """Load model and configuration from run_folder"""
        print(f"Loading model from: {self.model_path}")
        
        # Load class mapping from run_folder
        class_mapping_path = os.path.join(self.run_folder, 'class_mapping.json')
        if not os.path.exists(class_mapping_path):
            # Try alternative locations
            alt_paths = [
                os.path.join(os.path.dirname(self.model_path), 'class_mapping.json'),
                os.path.join(os.path.dirname(self.model_path), '..', 'class_mapping.json'),
            ]
            for alt_path in alt_paths:
                if os.path.exists(os.path.abspath(alt_path)):
                    class_mapping_path = os.path.abspath(alt_path)
                    break
            else:
                raise FileNotFoundError("Could not find class_mapping.json")
        
        with open(class_mapping_path, 'r') as f:
            self.class_to_label = json.load(f)
        
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        print(f"Loaded class mapping from: {class_mapping_path}")
        print(f"Classes: {list(self.class_to_label.keys())}")
        
        # Load model
        full_model_path = self.model_path + "_full.pt"
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model not found: {full_model_path}")
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(full_model_path, map_location=self.device)
        self.model.eval()
        
        # Infer feature dimension
        feature_dim = None
        try:
            for name, param in self.model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    feature_dim = param.shape[1]
                    break
            
            if feature_dim is None:
                first_param = next(self.model.parameters())
                if len(first_param.shape) == 2:
                    feature_dim = first_param.shape[1]
        except:
            feature_dim = 72
        
        self.processor = GraphProcessor(target_feature_dim=feature_dim)
        print(f"Model loaded - Classes: {len(self.class_to_label)}, Feature dim: {feature_dim}")
    
    def find_files(self, directory, extensions=['.graphml', '.xml', '.gml', '.stp', '.step']):
        """Find files with given extensions"""
        files = []
        if os.path.isfile(directory):
            return [directory]
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
        return files
    
    def extract_true_class_from_path(self, file_path):
        """Extract true class from file path using dynamic mapping"""
        path_parts = file_path.split('/')
        
        # Remove empty parts
        path_parts = [p for p in path_parts if p.strip()]
        
        # Define the mapping from folder keywords to model classes
        folder_to_class_mapping = {
            'bolts': 'Bolts',
            'pipes': 'Pipes', 
            'shaft': 'Shafts',
            'hinge': 'Hinges',
            'gear': 'Gears',
            'sprocket': 'Sprockets',
            'screw': 'Hex_head_screw',  # Default screw mapping, can be refined
            'screws': 'Hex_head_screw'   # Handle plural form
        }
        
        # Strategy 1: Look for '--' delimiter (primary strategy for your data)
        for part in path_parts:
            if '--' in part:
                class_key = part.split('--')[1].lower()  # Get part after '--' and make lowercase
                
                # Check if the key maps to a known class
                if class_key in folder_to_class_mapping:
                    mapped_class = folder_to_class_mapping[class_key]
                    if mapped_class in self.class_to_label:
                        return mapped_class
        
        # Strategy 2: Handle special cases like 'Screws/' directory
        for part in path_parts:
            part_lower = part.lower()
            if part_lower == 'screws':
                return 'Hex_head_screw'  # or determine based on subdirectory if needed
        
        # Strategy 3: Look for keywords anywhere in the path
        full_path_lower = file_path.lower()
        for keyword, mapped_class in folder_to_class_mapping.items():
            if keyword in full_path_lower and mapped_class in self.class_to_label:
                return mapped_class
        
        # If all strategies fail, return None and log
        print(f"WARNING: Could not determine class for path: {file_path}")
        return None

    def load_test_data(self, test_path):
        """Load test data and extract ground truth"""
        test_graphs = []
        file_paths = []
        true_labels = []
        
        if os.path.isfile(test_path):
            print(f"Processing single file: {test_path}")
            graph = self.processor.process_file(test_path)
            if graph is not None:
                test_graphs.append(graph)
                file_paths.append(test_path)
                true_class = self.extract_true_class_from_path(test_path)
                true_labels.append(self.class_to_label.get(true_class, None))
            return test_graphs, file_paths, true_labels
        
        if not os.path.isdir(test_path):
            raise ValueError(f"Test path does not exist: {test_path}")
        
        # DEBUG: Show directory structure
        print(f"Exploring test directory: {test_path}")
        try:
            items = os.listdir(test_path)
            print(f"Found {len(items)} items in directory:")
            for item in sorted(items):
                item_path = os.path.join(test_path, item)
                if os.path.isdir(item_path):
                    subfiles = []
                    for root, dirs, files in os.walk(item_path):
                        subfiles.extend([f for f in files if any(f.lower().endswith(ext) 
                                       for ext in ['.graphml', '.xml', '.gml', '.stp', '.step'])])
                    print(f"  📁 {item}/ ({len(subfiles)} graph files)")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"Error exploring directory: {e}")
        
        # Process all files and extract ground truth from paths
        files = self.find_files(test_path)
        print(f"\nFound {len(files)} total graph files to process")
        
        # DEBUG: Show file distribution by parent folder
        folder_file_count = {}
        for file_path in files[:20]:  # Show first 20 as example
            path_parts = file_path.split('/')
            try:
                testing_ood_index = path_parts.index('testing_ood')
                if testing_ood_index + 1 < len(path_parts):
                    parent_folder = path_parts[testing_ood_index + 1]
                    if parent_folder not in folder_file_count:
                        folder_file_count[parent_folder] = 0
                    folder_file_count[parent_folder] += 1
            except ValueError:
                pass
        
        if folder_file_count:
            print(f"Sample file distribution by folder:")
            for folder, count in sorted(folder_file_count.items()):
                print(f"  {folder}: {count} files (from first 20)")
        
        # Track class distribution for debugging
        class_distribution = {}
        
        for file_path in files:
            graph = self.processor.process_file(file_path)
            if graph is not None:
                test_graphs.append(graph)
                file_paths.append(file_path)
                
                # Extract ground truth from path
                true_class = self.extract_true_class_from_path(file_path)
                true_label = self.class_to_label.get(true_class, None)
                true_labels.append(true_label)
                
                # Track distribution
                if true_class not in class_distribution:
                    class_distribution[true_class] = 0
                class_distribution[true_class] += 1
                
                # Debug output for variety of samples
                if len(test_graphs) <= 3 or (len(test_graphs) % 100 == 0):
                    print(f"  Sample {len(test_graphs)}: {os.path.basename(file_path)}")
                    print(f"    Path: {file_path}")
                    print(f"    True class: {true_class} -> Label: {true_label}")
        
        print(f"\nSuccessfully loaded {len(test_graphs)} graphs from {len(file_paths)} files")
        print(f"Class distribution found:")
        for class_name, count in sorted(class_distribution.items()):
            label = self.class_to_label.get(class_name, "Unknown")
            percentage = count / len(test_graphs) * 100 if len(test_graphs) > 0 else 0
            print(f"  {class_name}: {count} files ({percentage:.1f}%) -> Label: {label}")
        
        return test_graphs, file_paths, true_labels
    
    def load_test_data(self, test_path):
        """Load test data and extract ground truth"""
        test_graphs = []
        file_paths = []
        true_labels = []
        
        if os.path.isfile(test_path):
            print(f"Processing single file: {test_path}")
            graph = self.processor.process_file(test_path)
            if graph is not None:
                test_graphs.append(graph)
                file_paths.append(test_path)
                true_class = self.extract_true_class_from_path(test_path)
                true_labels.append(self.class_to_label.get(true_class, None))
            return test_graphs, file_paths, true_labels
        
        if not os.path.isdir(test_path):
            raise ValueError(f"Test path does not exist: {test_path}")
        
        # Process all files and extract ground truth from paths
        files = self.find_files(test_path)
        print(f"Found {len(files)} files to process")
        
        for file_path in files:
            graph = self.processor.process_file(file_path)
            if graph is not None:
                test_graphs.append(graph)
                file_paths.append(file_path)
                
                # Extract ground truth from path
                true_class = self.extract_true_class_from_path(file_path)
                true_label = self.class_to_label.get(true_class, None)
                true_labels.append(true_label)
                
                if len(test_graphs) <= 5:  # Debug first few files
                    print(f"  File: {os.path.basename(file_path)}")
                    print(f"    Path: {file_path}")
                    print(f"    True class: {true_class} -> Label: {true_label}")
        
        print(f"Successfully loaded {len(test_graphs)} graphs from {len(file_paths)} files")
        return test_graphs, file_paths, true_labels
    
    def get_batch_data_custom(self, batch_graph):
        """Custom batch data processing matching train_utils.py"""
        X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
        X_concat = torch.from_numpy(X_concat).to(self.device)
        
        edge_mat_list = []
        start_idx = [0]
        
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        
        adjj = np.concatenate(edge_mat_list, 1)
        adjj = torch.from_numpy(adjj).to(self.device).to(torch.int64)
        
        graph_labels = np.array([graph.label if graph.label is not None else 0 for graph in batch_graph])
        graph_labels = torch.from_numpy(graph_labels).to(self.device)
        
        return X_concat, graph_labels, adjj
    
    def predict_graphs(self, graphs):
        """Make predictions on graphs"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        print(f"Making predictions on {len(graphs)} graphs...")
        
        with torch.no_grad():
            for i in range(len(graphs)):
                try:
                    batch_graphs = [graphs[i]]
                    
                    X_concat, graph_labels, adjj = self.get_batch_data_custom(batch_graphs)
                    
                    prediction_scores = self.model(adjj, X_concat)
                    
                    batch_predictions = prediction_scores.max(1, keepdim=True)[1]
                    batch_probs = torch.softmax(prediction_scores, dim=1)
                    
                    predictions.extend(batch_predictions.cpu().numpy().flatten())
                    probabilities.extend(batch_probs.cpu().numpy())
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Processed {i+1}/{len(graphs)} graphs")
                        
                except Exception as e:
                    print(f"Prediction failed for graph {i}: {e}")
                    predictions.append(0)
                    dummy_prob = np.zeros(len(self.class_to_label))
                    dummy_prob[0] = 1.0
                    probabilities.append(dummy_prob)
        
        return np.array(predictions), np.array(probabilities)
    
    def save_results_csv(self, predictions, probabilities, file_paths, true_labels, output_path):
        """Save results to CSV"""
        results = []
        
        for i in range(len(predictions)):
            try:
                pred = int(predictions[i])
                file_path = file_paths[i]
                
                # Get probability and confidence
                if i < len(probabilities):
                    prob = probabilities[i]
                    confidence = float(np.max(prob)) if prob.size > 0 else 0.0
                else:
                    prob = np.zeros(len(self.class_to_label))
                    confidence = 0.0
                
                # Create result entry
                result = {
                    'file_path': str(file_path),
                    'filename': os.path.basename(str(file_path)),
                    'predicted_class': self.label_to_class.get(pred, f'unknown_class_{pred}'),
                    'predicted_label': pred,
                    'confidence': confidence
                }
                
                # Add true label if available
                if i < len(true_labels) and true_labels[i] is not None:
                    true_label = int(true_labels[i])
                    result['true_class'] = self.label_to_class.get(true_label, f'unknown_class_{true_label}')
                    result['true_label'] = true_label
                    result['correct'] = (pred == true_label)
                
                # Add class probabilities
                for class_label, class_name in self.label_to_class.items():
                    if class_label < len(prob):
                        result[f'prob_{class_name}'] = float(prob[class_label])
                    else:
                        result[f'prob_{class_name}'] = 0.0
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing result {i}: {e}")
        
        # Save to CSV
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        return df
    
    def create_confusion_matrix(self, y_true, y_pred, normalize=None):
        """Create confusion matrix"""
        all_classes = sorted(list(set(y_true) | set(y_pred)))
        class_names = [self.label_to_class.get(i, f'Class_{i}') for i in all_classes]
        
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
            cm = np.nan_to_num(cm)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        
        return cm, class_names
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=None, figsize=(12, 10)):
        """Plot confusion matrix"""
        cm, class_names = self.create_confusion_matrix(y_true, y_pred, normalize=normalize)
        
        plt.figure(figsize=figsize)
        
        if normalize is not None:
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Proportion'})
            title = f'Normalized Confusion Matrix ({normalize})'
            filename = f'confusion_matrix_norm_{normalize}.png'
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'})
            title = 'Confusion Matrix'
            filename = 'confusion_matrix.png'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
        plt.close()
    
    def plot_class_distribution(self, y_true, y_pred, figsize=(14, 8)):
        """Plot class distribution comparison"""
        true_classes = [self.label_to_class.get(i, f'Class_{i}') for i in y_true]
        pred_classes = [self.label_to_class.get(i, f'Class_{i}') for i in y_pred]
        
        true_counts = pd.Series(true_classes).value_counts()
        pred_counts = pd.Series(pred_classes).value_counts()
        
        all_classes = sorted(list(set(true_classes) | set(pred_classes)))
        
        x = np.arange(len(all_classes))
        width = 0.35
        
        true_values = [true_counts.get(cls, 0) for cls in all_classes]
        pred_values = [pred_counts.get(cls, 0) for cls in all_classes]
        
        plt.figure(figsize=figsize)
        plt.bar(x - width/2, true_values, width, label='True', alpha=0.7, color='skyblue')
        plt.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.7, color='lightcoral')
        
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('True vs Predicted Class Distribution')
        plt.xticks(x, all_classes, rotation=45, ha='right')
        plt.legend()
        
        # Add value labels
        for i, (true_val, pred_val) in enumerate(zip(true_values, pred_values)):
            if true_val > 0:
                plt.text(i - width/2, true_val + 0.5, str(true_val), ha='center', va='bottom')
            if pred_val > 0:
                plt.text(i + width/2, pred_val + 0.5, str(pred_val), ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution: {save_path}")
        plt.close()
    
    def plot_per_class_accuracy(self, y_true, y_pred, figsize=(12, 6)):
        """Plot per-class accuracy"""
        unique_true = np.unique(y_true)
        class_accuracies = {}
        class_counts = {}
        
        for true_label in unique_true:
            mask = np.array(y_true) == true_label
            correct = np.array(y_pred)[mask] == true_label
            class_accuracies[true_label] = correct.mean() if len(correct) > 0 else 0
            class_counts[true_label] = len(correct)
        
        classes = [self.label_to_class.get(label, f'Class_{label}') for label in unique_true]
        accuracies = [class_accuracies[label] for label in unique_true]
        counts = [class_counts[label] for label in unique_true]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.7, edgecolor='darkblue')
        
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Per-class Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add accuracy and count labels
        for bar, acc, count in zip(bars, accuracies, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2f}\n(n={count})', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'per_class_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class accuracy: {save_path}")
        plt.close()
    
    def create_error_analysis(self, df):
        """Create detailed error analysis"""
        if 'true_class' not in df.columns or 'predicted_class' not in df.columns:
            print("Cannot perform error analysis without ground truth")
            return
        
        errors = df[df['predicted_class'] != df['true_class']].copy()
        
        if len(errors) == 0:
            print("No errors found - perfect accuracy!")
            return
        
        error_analysis = []
        for _, row in errors.iterrows():
            error_analysis.append({
                'filename': row['filename'],
                'file_path': row['file_path'],
                'true_class': row['true_class'],
                'predicted_class': row['predicted_class'],
                'confidence': row['confidence'],
                'error_type': f"{row['true_class']} → {row['predicted_class']}"
            })
        
        error_df = pd.DataFrame(error_analysis)
        error_types = error_df['error_type'].value_counts()
        
        # Save error analysis
        error_path = os.path.join(self.output_dir, 'error_analysis.csv')
        error_df.to_csv(error_path, index=False)
        
        # Create error summary
        summary_path = os.path.join(self.output_dir, 'error_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("ERROR ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Errors: {len(errors)} out of {len(df)} samples\n")
            f.write(f"Error Rate: {len(errors)/len(df)*100:.2f}%\n\n")
            f.write("Most Common Error Types:\n")
            f.write("-" * 30 + "\n")
            for error_type, count in error_types.head(10).items():
                f.write(f"{error_type}: {count} ({count/len(errors)*100:.1f}%)\n")
            
            f.write(f"\nConfidence Distribution of Errors:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean confidence: {errors['confidence'].mean():.3f}\n")
            f.write(f"Median confidence: {errors['confidence'].median():.3f}\n")
            f.write(f"Min confidence: {errors['confidence'].min():.3f}\n")
            f.write(f"Max confidence: {errors['confidence'].max():.3f}\n")
        
        print(f"Saved error analysis: {error_path}")
        print(f"Saved error summary: {summary_path}")
        
        return error_df
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate and save classification report"""
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        class_names = [self.label_to_class.get(i, f'Class_{i}') for i in unique_labels]
        
        report = classification_report(y_true, y_pred, labels=unique_labels, 
                                     target_names=class_names, digits=4)
        accuracy = accuracy_score(y_true, y_pred)
        
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Total Samples: {len(y_true)}\n")
            f.write(f"Number of Classes: {len(unique_labels)}\n\n")
            
            # Class distribution
            f.write("True Class Distribution:\n")
            f.write("-" * 25 + "\n")
            true_counts = pd.Series(y_true).value_counts().sort_index()
            for label, count in true_counts.items():
                class_name = self.label_to_class.get(label, f'Class_{label}')
                f.write(f"{class_name}: {count} ({count/len(y_true)*100:.1f}%)\n")
            f.write("\n")
            
            f.write("Detailed Classification Report:\n")
            f.write("-" * 35 + "\n")
            f.write(report)
        
        print(f"Saved classification report: {report_path}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        return report
    
    def run_integrated_analysis(self, test_path, output_csv="test_results.csv"):
        """Run integrated testing and analysis"""
        print("=" * 70)
        print("INTEGRATED GRAPH CLASSIFICATION TESTING AND ANALYSIS")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Run folder: {self.run_folder}")
        print(f"Test data: {test_path}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
        # Step 1: Load model and configuration
        print("\n1. Loading model and configuration...")
        self.load_model_and_config()
        
        # Step 2: Load test data
        print(f"\n2. Loading test data from {test_path}...")
        test_graphs, file_paths, true_labels = self.load_test_data(test_path)
        
        if not test_graphs:
            raise ValueError("No valid test graphs loaded!")
        
        print(f"Loaded {len(test_graphs)} valid graphs")
        
        # Step 3: Normalize features
        print("\n3. Normalizing feature dimensions...")
        for i, graph in enumerate(test_graphs):
            self.processor.normalize_features(graph)
            if i == 0:
                print(f"   Sample graph features shape: {graph.node_features.shape}")
        
        # Step 4: Make predictions
        print(f"\n4. Making predictions...")
        predictions, probabilities = self.predict_graphs(test_graphs)
        
        print(f"Generated predictions for {len(predictions)} graphs")
        
        # Step 5: Save results to CSV
        print(f"\n5. Saving results to CSV...")
        output_csv_path = os.path.join(self.output_dir, output_csv)
        results_df = self.save_results_csv(predictions, probabilities, file_paths, 
                                         true_labels, output_csv_path)
        
        # Step 6: Analysis and visualizations
        print(f"\n6. Generating analysis and visualizations...")
        
        # Check if we have ground truth
        valid_labels = [t for t in true_labels if t is not None]
        has_ground_truth = len(valid_labels) == len(true_labels) and len(valid_labels) > 0
        
        if has_ground_truth:
            print("Ground truth available - generating comprehensive analysis...")
            
            y_true = np.array(true_labels)
            y_pred = predictions
            
            # Generate confusion matrices
            self.plot_confusion_matrix(y_true, y_pred)
            self.plot_confusion_matrix(y_true, y_pred, normalize='true')
            
            # Generate other visualizations
            self.plot_class_distribution(y_true, y_pred)
            self.plot_per_class_accuracy(y_true, y_pred)
            
            # Generate reports
            self.generate_classification_report(y_true, y_pred)
            self.create_error_analysis(results_df)
            
            # Print summary
            accuracy = accuracy_score(y_true, y_pred)
            print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Print prediction summary
            unique_true, true_counts = np.unique(y_true, return_counts=True)
            unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
            
            print(f"\nTrue class distribution:")
            for label, count in zip(unique_true, true_counts):
                class_name = self.label_to_class.get(label, f'Class_{label}')
                print(f"  {class_name}: {count} ({count/len(y_true)*100:.1f}%)")
            
            print(f"\nPredicted class distribution:")
            for label, count in zip(unique_pred, pred_counts):
                class_name = self.label_to_class.get(label, f'Class_{label}')
                print(f"  {class_name}: {count} ({count/len(y_pred)*100:.1f}%)")
        
        else:
            print("No ground truth available - generating prediction summary only...")
            
            unique_pred, pred_counts = np.unique(predictions, return_counts=True)
            print(f"\nPrediction Distribution:")
            for label, count in zip(unique_pred, pred_counts):
                class_name = self.label_to_class.get(label, f'Class_{label}')
                percentage = count / len(predictions) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Step 7: Print final statistics
        print(f"\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Files processed: {self.processor.stats['processed']}")
        print(f"Files failed: {self.processor.stats['failed']}")
        print(f"Empty graphs: {self.processor.stats['empty']}")
        print(f"Valid predictions: {len(predictions)}")
        print(f"Results CSV: {output_csv_path}")
        print(f"Analysis outputs: {self.output_dir}")
        
        print("\n" + "=" * 70)
        print("INTEGRATED ANALYSIS COMPLETED!")
        print("=" * 70)
        
        return results_df


def main():
    parser = ArgumentParser("Integrated Graph Classification Testing and Analysis", 
                          formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--model_path", required=True, 
                       help="Path to trained model (without _full.pt extension)")
    parser.add_argument("--run_folder", required=True, 
                       help="Folder containing class_mapping.json and training config")
    parser.add_argument("--test_path", required=True, 
                       help="Path to test data (file or directory)")
    parser.add_argument("--output_dir", default="analysis_results",
                       help="Output directory for all results and visualizations")
    parser.add_argument("--output_csv", default="test_results.csv",
                       help="Name of output CSV file")
    
    args = parser.parse_args()
    
    try:
        analyzer = IntegratedAnalyzer(args.model_path, args.run_folder, args.output_dir)
        results_df = analyzer.run_integrated_analysis(args.test_path, args.output_csv)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("ANALYSIS FAILED!")
        print("=" * 70)
        
        raise


if __name__ == "__main__":
    main()