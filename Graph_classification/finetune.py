#! /usr/bin/env python
from GCN import *
from datetime import datetime
from utils.my_utils import *
from utils.util import *
from finetune_data_loader import load_finetune_data_with_mapping, print_data_composition
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import math
from train_utils import *
import torch.nn as nn

torch.manual_seed(124)
np.random.seed(124)

# Parameters
# ==================================================
parser = ArgumentParser("GCN Fine-tuning", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="New_Dataset/Graphml_Models/", help="Name of the new graph dataset.")
parser.add_argument("--class_mapping", default="class_mapping.json", help="Path to class mapping JSON file")
parser.add_argument("--pretrained_model", required=True, help="Path to pretrained model (.pt file)")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate (lower for fine-tuning)")
parser.add_argument("--feature_lr", default=0.00001, type=float, help="Learning rate for feature extractor")
parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=30, type=int, help="Number of training epochs")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--freeze_features", action='store_true', help="Freeze feature extraction layers")
parser.add_argument("--freeze_strategy", default="none", choices=["none", "partial", "full"], 
                   help="Freezing strategy: none, partial (freeze early layers), full (freeze all except classifier)")
parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")
parser.add_argument("--min_delta", default=0.001, type=float, help="Minimum improvement for early stopping")
parser.add_argument("--monitor", default="accuracy", choices=["accuracy", "loss"], help="Metric to monitor for early stopping")
parser.add_argument("--restore_best", action='store_true', help="Restore best weights after early stopping")

parser.add_argument("--use_existing_splits", action='store_true', default=True, 
                   help="Use existing train/valid/test splits from directories")
parser.add_argument("--random_split", action='store_true', 
                   help="Ignore existing splits and randomly split all data (overrides use_existing_splits)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The calculations will be performed on the device:", device)

# save paths
model_name = str(datetime.today().strftime('%m-%d')) + "_finetune"
out_dir = os.path.abspath(args.run_folder)  # Use args.run_folder instead of hardcoded
if not os.path.exists(out_dir + "/Models/"):
    os.makedirs(out_dir + "/Models/")
save_path = out_dir + "/Models/" + model_name
print("Results will be saved in:", out_dir)
print("    The model will be saved as:", save_path)
print("Settings:",args)

# Handle split logic
if args.random_split:
    args.use_existing_splits = False

# Load new dataset with predefined class mapping
# ==================================================
print("Loading new data with predefined class mapping...")
use_degree_as_tag = False

# Check if class mapping file exists
class_mapping_path = os.path.abspath(args.class_mapping)
if not os.path.exists(class_mapping_path):
    print(f"ERROR: Class mapping file not found: {class_mapping_path}")
    print("Please ensure the class_mapping.json file exists.")
    exit(1)

print(f"Using class mapping from: {class_mapping_path}")

if args.use_existing_splits:
    print("Using existing train/valid/test splits from directories")
    (train_graphs, valid_graphs, test_graphs), num_classes = load_finetune_data_with_mapping(
        args.dataset, class_mapping_path, use_degree_as_tag, use_existing_splits=True)
    
    if not train_graphs:
        print("ERROR: No training data loaded! Check your directory structure.")
        print("Expected structure: dataset_path/class_name/train|valid|test/file.graphml")
        exit(1)
        
else:
    print("Loading all data for random splitting")
    graphs, num_classes = load_finetune_data_with_mapping(
        args.dataset, class_mapping_path, use_degree_as_tag, use_existing_splits=False)
    
    if not graphs:
        print("ERROR: No data loaded! Check your directory structure.")
        exit(1)
        
    # Use original splitting logic
    fold = 0
    train_graphs, test_graphs = separate_data(graphs, fold)
    train_graphs, valid_graphs = split_data(train_graphs, perc=0.9)

# Print data composition with original class names
print_data_composition(train_graphs, class_mapping_path, "Training data")
print_data_composition(valid_graphs, class_mapping_path, "Validation data") 
print_data_composition(test_graphs, class_mapping_path, "Test data")

print("# training graphs: ", len(train_graphs))
print("# validation graphs: ", len(valid_graphs))
print("# test graphs: ", len(test_graphs))

if not train_graphs:
    print("ERROR: No training graphs available!")
    exit(1)

feature_dim_size = train_graphs[0].node_features.shape[1]
print("Loading new data... finished!")
print("New dataset has {} classes".format(num_classes))

# Load pretrained model and modify for new classes
# =============================================================
print("Loading pretrained model from:", args.pretrained_model)

# Check if the number of classes matches
import json
with open(class_mapping_path, 'r') as f:
    expected_classes = json.load(f)

print(f"Expected classes from mapping: {expected_classes}")
print(f"Number of classes from mapping: {len(expected_classes)}")
print(f"Number of classes detected: {num_classes}")

if len(expected_classes) != num_classes:
    print(f"WARNING: Class count mismatch!")
    print(f"  Expected from mapping: {len(expected_classes)}")
    print(f"  Detected from data: {num_classes}")
    print("Using the mapping size for model architecture...")
    num_classes = len(expected_classes)

# Load the full model first to get the architecture
pretrained_model = torch.load(args.pretrained_model + "_full.pt", map_location=device)
print("Original model loaded successfully")

# Create new model with same architecture but different output classes
model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=args.dropout).to(device)

# Load pretrained weights except for layers that have dimension mismatches
pretrained_dict = torch.load(args.pretrained_model, map_location=device)
model_dict = model.state_dict()

# Check for dimension mismatches and filter out incompatible layers
compatible_dict = {}
skipped_layers = []

for k, v in pretrained_dict.items():
    if k in model_dict:
        if v.shape == model_dict[k].shape:
            compatible_dict[k] = v
        else:
            skipped_layers.append(f"{k}: {v.shape} -> {model_dict[k].shape}")
    else:
        skipped_layers.append(f"{k}: not found in new model")

print("Skipped incompatible layers:")
for layer in skipped_layers:
    print(f"  {layer}")

print(f"Loading {len(compatible_dict)} out of {len(pretrained_dict)} layers from pretrained model")

# Update the model dictionary and load
model_dict.update(compatible_dict)
model.load_state_dict(model_dict)

print("Pretrained weights loaded successfully!")

# Print which layers will be trained from scratch
scratch_layers = [k for k in model_dict.keys() if k not in compatible_dict]
if scratch_layers:
    print("Layers training from scratch:")
    for layer in scratch_layers:
        print(f"  {layer}")
        
# Analyze what's being transferred
conv_layers = [k for k in compatible_dict.keys() if 'convolution' in k]
attention_layers = [k for k in compatible_dict.keys() if 'attention' in k]
classifier_layers = [k for k in scratch_layers if 'fully_connected' in k or 'scoring_layer' in k]

print(f"\nTransfer Learning Summary:")
print(f"  Transferred conv layers: {len(conv_layers)}")
print(f"  Transferred attention layers: {len(attention_layers)}")  
print(f"  New classifier layers: {len(classifier_layers)}")

if len(conv_layers) == 0:
    print("\n WARNING: No convolution layers transferred!")
    print("   This means feature extraction will be trained from scratch.")
    print("   Consider using a lower learning rate and more data.")

# Apply freezing strategy
# =============================================================
def apply_freezing_strategy(model, strategy):
    """Apply different freezing strategies to the model"""
    if strategy == "none":
        print("No layers frozen - training all parameters")
        return
    
    elif strategy == "full":
        # Freeze all layers except fully_connected_first and scoring_layer
        for name, param in model.named_parameters():
            if 'fully_connected_first' not in name and 'scoring_layer' not in name:
                param.requires_grad = False
        print("All layers frozen except fully_connected_first and scoring_layer")
    
    elif strategy == "partial":
        # Freeze GCN layers, fine-tune attention + classifier
        for name, param in model.named_parameters():
            if 'convolution' in name:  # Freeze convolution layers
                param.requires_grad = False
        print("Frozen convolution layers, training attention + classifier")

apply_freezing_strategy(model, args.freeze_strategy)

# Setup optimizer with different learning rates
# =============================================================
if args.freeze_strategy == "none" and args.feature_lr != args.learning_rate:
    # Different learning rates for feature extractor and classifier
    classifier_params = []
    feature_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fully_connected_first' in name or 'scoring_layer' in name:
                classifier_params.append(param)
            else:
                feature_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': feature_params, 'lr': args.feature_lr},
        {'params': classifier_params, 'lr': args.learning_rate}
    ])
    print(f"Using different learning rates: features={args.feature_lr}, classifier={args.learning_rate}")
else:
    # Single learning rate for all trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Using single learning rate: {args.learning_rate}")

num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch*5, gamma=0.5)

# Early Stopping Class
# =============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='accuracy', restore_best=True, verbose=True):
        """
        Early stopping to stop training when validation metric stops improving
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            monitor (str): Metric to monitor ('accuracy' or 'loss')
            restore_best (bool): Whether to restore best weights
            verbose (bool): Whether to print stopping info
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.wait = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, epoch, current_score, model):
        """
        Check if training should stop
        
        Returns:
            bool: True if training should stop
        """
        # For accuracy, higher is better. For loss, lower is better.
        if self.monitor == 'accuracy':
            score = current_score
            is_improvement = score > (self.best_score + self.min_delta) if self.best_score is not None else True
        else:  # loss
            score = -current_score  # Convert to "higher is better"
            is_improvement = score > (self.best_score + self.min_delta) if self.best_score is not None else True
        
        if self.best_score is None or is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights if restore_best is True
            if self.restore_best:
                self.best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            
            if self.verbose:
                metric_name = self.monitor
                actual_value = current_score
                print(f"New best {metric_name}: {actual_value:.4f} at epoch {epoch}")
        else:
            self.wait += 1
            if self.verbose and self.wait % 3 == 0:  # Print every 3 epochs of no improvement
                print(f"No improvement for {self.wait}/{self.patience} epochs")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"\n Early stopping triggered!")
                print(f"   Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch}")
                print(f"   Stopped at epoch: {epoch}")
                print(f"   Total patience reached: {self.patience}")
            return True
        
        return False
    
    def restore_best_weights(self, model):
        """Restore the best weights to the model"""
        if self.best_weights is not None:
            model.load_state_dict({key: value.to(model.device if hasattr(model, 'device') else 'cpu') 
                                 for key, value in self.best_weights.items()})
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch}")

# Print trainable parameters info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# Initialize Early Stopping
# =============================================================
early_stopping = EarlyStopping(
    patience=args.patience,
    min_delta=args.min_delta,
    monitor=args.monitor,
    restore_best=args.restore_best,
    verbose=True
)

print(f"Early Stopping Config:")
print(f"  - Patience: {args.patience} epochs")
print(f"  - Min Delta: {args.min_delta}")
print(f"  - Monitor: {args.monitor}")
print(f"  - Restore Best: {args.restore_best}")

# Main training process
# =============================================================
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_finetune_acc.txt', 'w')

train_losses = []
train_accuracy = []
valid_losses = []
valid_accuracy = []
valid_accuracy_x_class = []

best_loss = math.inf
best_accuracy = 0
training_stopped_early = False

# Fine-tuning loop
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    
    # train model
    train(mmodel=model, optimizer=optimizer, train_graphs=train_graphs, 
          batch_size=args.batch_size, num_classes=num_classes, device=device)
    
    # evaluate on train data
    train_loss, train_acc, _ = evaluate(mmodel=model, current_graphs=train_graphs, 
                                       batch_size=args.batch_size, num_classes=num_classes, 
                                       device=device, out_dir=out_dir)
    
    # evaluate on validation data
    valid_loss, valid_acc, valid_acc_x_class = evaluate(mmodel=model, current_graphs=valid_graphs, 
                                                        batch_size=args.batch_size, num_classes=num_classes, 
                                                        device=device, out_dir=out_dir)
    
    print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | valid acc {:5.2f} | '.format(
        epoch, (time.time() - epoch_start_time), train_loss, valid_loss, valid_acc*100))
    
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracy.append(valid_acc)
    valid_accuracy_x_class.append(valid_acc_x_class)
    
    # Learning rate scheduling
    if epoch > 5 and train_losses[-1] > np.mean(train_losses[-6:-1]):
        scheduler.step()
        print("Scheduler step - reducing learning rate")
    
    # Save best model (for backup)
    if best_accuracy < valid_acc or (best_accuracy == valid_acc and best_loss > valid_loss):
        best_accuracy = valid_acc
        best_loss = valid_loss
        torch.save(model.state_dict(), save_path)
        torch.save(model, save_path + "_full.pt")
    
    # Early stopping check
    monitor_value = valid_acc if args.monitor == 'accuracy' else valid_loss
    if early_stopping(epoch, monitor_value, model):
        training_stopped_early = True
        break
    
    write_acc.write('epoch ' + str(epoch) + ' acc ' + str(valid_acc*100) + '%\n')

# Handle early stopping restoration
if training_stopped_early and args.restore_best:
    print("\nRestoring best weights from early stopping...")
    early_stopping.restore_best_weights(model)
    
    # Re-evaluate with restored weights
    print("Re-evaluating with restored weights...")
    restored_train_loss, restored_train_acc, _ = evaluate(mmodel=model, current_graphs=train_graphs, 
                                                         batch_size=args.batch_size, num_classes=num_classes, 
                                                         device=device, out_dir=out_dir)
    restored_valid_loss, restored_valid_acc, _ = evaluate(mmodel=model, current_graphs=valid_graphs, 
                                                         batch_size=args.batch_size, num_classes=num_classes, 
                                                         device=device, out_dir=out_dir)
    
    print(f"Restored weights performance:")
    print(f"   Train - Loss: {restored_train_loss:.4f}, Acc: {restored_train_acc*100:.2f}%")
    print(f"   Valid - Loss: {restored_valid_loss:.4f}, Acc: {restored_valid_acc*100:.2f}%")
    
    # Save the restored model
    torch.save(model.state_dict(), save_path + "_early_stopped")
    torch.save(model, save_path + "_early_stopped_full.pt")

# Plot results and final evaluation
# =============================================================
valid_accuracy_x_class = np.array(valid_accuracy_x_class).T

# plot training flow
plot_training_flow(ys=[train_losses, valid_losses], names=["train", "validation"], 
                  path=out_dir, fig_name="/finetune_losses_flow", y_axis="Loss")
plot_training_flow(ys=[np.array(train_accuracy)*100, np.array(valid_accuracy)*100], 
                  names=["train","validation"], path=out_dir, fig_name="/finetune_accuracy_flow", y_axis="Accuracy")

# Evaluate on test data
if training_stopped_early and args.restore_best:
    test_model_path = save_path + "_early_stopped"
    print(f"Using early stopped model for final evaluation: {test_model_path}")
else:
    test_model_path = save_path
    model.load_state_dict(torch.load(save_path))
    print(f"Using best validation model for final evaluation: {test_model_path}")

test_loss, test_acc, _ = evaluate(mmodel=model, current_graphs=test_graphs, 
                                 batch_size=args.batch_size, num_classes=num_classes, 
                                 device=device, out_dir=out_dir, last_round=True)

print("\n" + "="*60)
print("FINE-TUNING COMPLETED!")
print("="*60)
if training_stopped_early:
    print(f"Training stopped early at epoch {early_stopping.stopped_epoch}")
    print(f"Best epoch: {early_stopping.best_epoch}")
else:
    print(f"Training completed full {args.num_epochs} epochs")

print(f"Final test results:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc*100:.2f}%")
print("="*60)

write_acc.close()