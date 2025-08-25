import os
import argparse
from .Parser.Make_graph import make_graph_simplex_direct


def make_graphh_dataset(path_stp, path_graph):
    """
    Convert STEP files to graphs with recursive folder processing
    Maintains the same directory structure in the output
    """
    if not path_stp.endswith("/"):
        path_stp = path_stp + "/"
    if not path_graph.endswith("/"):
        path_graph = path_graph + "/"

    if not os.path.exists(path_graph):
        os.makedirs(path_graph)
    
    print(f"Converting STEP files from: {path_stp}")
    print(f"Saving graphs to: {path_graph}")
    
    # Recursively process all directories and subdirectories
    for root, dirs, files in os.walk(path_stp):
        # Get relative path from the source directory
        rel_path = os.path.relpath(root, path_stp)
        
        # Skip the root directory itself
        if rel_path == '.':
            rel_path = ''
        
        print(f"\nProcessing directory: {root}")
        print(f"Relative path: {rel_path}")
        
        # Create corresponding output directory
        if rel_path:
            output_dir = os.path.join(path_graph, rel_path)
        else:
            output_dir = path_graph
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Process all STEP files in current directory
        step_files = [f for f in files if f.lower().endswith(('.stp', '.step'))]
        
        if step_files:
            print(f"Found {len(step_files)} STEP files in {root}")
            
            for i, file in enumerate(step_files, 1):
                print(f"  Processing file {i}/{len(step_files)}: {file}")
                
                try:
                    # Construct the relative file path for the make_graph function
                    if rel_path:
                        file_path = os.path.join(rel_path, file).replace('\\', '/')
                    else:
                        file_path = file
                    
                    # Call the graph generation function
                    g = make_graph_simplex_direct(
                        file_name=file_path,
                        graph_saves_base_paths=path_graph,
                        dataset_path=path_stp
                    )
                    
                    if g is not None:
                        print(f"    Successfully converted: {file}")
                    else:
                        print(f"    Warning: Failed to convert {file}")
                        
                except Exception as e:
                    print(f"    Error processing {file}: {str(e)}")
                    continue
        else:
            print(f"No STEP files found in {root}")
    
    print("\nConversion completed!")


def make_graphh_dataset_class_structure(path_stp, path_graph):
    """
    Alternative function for class/train-valid-test structure
    Processes: path_stp/class/split/*.stp -> path_graph/class/split/*.graphml
    """
    if not path_stp.endswith("/"):
        path_stp = path_stp + "/"
    if not path_graph.endswith("/"):
        path_graph = path_graph + "/"

    if not os.path.exists(path_graph):
        os.makedirs(path_graph)
    
    print(f"Converting STEP files from: {path_stp}")
    print(f"Saving graphs to: {path_graph}")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(path_stp) 
                  if os.path.isdir(os.path.join(path_stp, d))]
    
    total_files = 0
    converted_files = 0
    
    for class_dir in class_dirs:
        print(f"\n=== Processing class: {class_dir} ===")
        class_path = os.path.join(path_stp, class_dir)
        
        # Create output class directory
        output_class_dir = os.path.join(path_graph, class_dir)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        # Process train/valid/test subdirectories
        split_dirs = [d for d in os.listdir(class_path) 
                     if os.path.isdir(os.path.join(class_path, d))]
        
        for split_dir in split_dirs:
            print(f"\n  Processing split: {split_dir}")
            split_path = os.path.join(class_path, split_dir)
            
            # Create output split directory
            output_split_dir = os.path.join(output_class_dir, split_dir)
            if not os.path.exists(output_split_dir):
                os.makedirs(output_split_dir)
            
            # Get all STEP files in this split
            step_files = [f for f in os.listdir(split_path) 
                         if f.lower().endswith(('.stp', '.step'))]
            
            if step_files:
                print(f"    Found {len(step_files)} STEP files")
                total_files += len(step_files)
                
                for i, file in enumerate(step_files, 1):
                    print(f"    Processing {i}/{len(step_files)}: {file}")
                    
                    try:
                        # Construct relative path for make_graph function
                        rel_file_path = f"{class_dir}/{split_dir}/{file}"
                        
                        # Call the graph generation function
                        g = make_graph_simplex_direct(
                            file_name=rel_file_path,
                            graph_saves_base_paths=path_graph,
                            dataset_path=path_stp
                        )
                        
                        if g is not None:
                            converted_files += 1
                            print(f"      ✓ Successfully converted: {file}")
                        else:
                            print(f"      ✗ Failed to convert: {file}")
                            
                    except Exception as e:
                        print(f"      ✗ Error processing {file}: {str(e)}")
                        continue
            else:
                print(f"    No STEP files found in {split_path}")
    
    print(f"\n=== Conversion Summary ===")
    print(f"Total STEP files found: {total_files}")
    print(f"Successfully converted: {converted_files}")
    print(f"Failed conversions: {total_files - converted_files}")
    print("Conversion completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert STEP files to graphs")
    parser.add_argument("--path_step", required=True, help="Path to STEP files directory")
    parser.add_argument("--path_graph", required=True, help="Path to save graph files")
    parser.add_argument("--mode", choices=['recursive', 'class_structure'], 
                       default='class_structure',
                       help="Processing mode: 'recursive' for general recursive processing, "
                            "'class_structure' for class/train-valid-test structure")
    args = parser.parse_args()

    path_stp = args.path_step
    path_graph = args.path_graph
    mode = args.mode

    print(f"Starting conversion in '{mode}' mode...")
    
    if mode == 'recursive':
        make_graphh_dataset(path_stp, path_graph)
    elif mode == 'class_structure':
        make_graphh_dataset_class_structure(path_stp, path_graph)