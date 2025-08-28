import os
import argparse
from .Parser.Make_graph import make_graph_simplex_direct


def validate_step_file(file_path):
    """
    STEP 파일의 유효성을 검증하고 기본 정보를 출력
    """
    if not os.path.exists(file_path):
        return False, "파일이 존재하지 않습니다"
    
    if not os.path.isfile(file_path):
        return False, "디렉토리입니다, 파일이 아닙니다"
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "파일 크기가 0입니다"
    
    # 파일 읽기 권한 확인
    if not os.access(file_path, os.R_OK):
        return False, "파일 읽기 권한이 없습니다"
    
    # STEP 파일 내용 미리보기 (처음 몇 줄)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = []
            for i in range(5):  # 처음 5줄만 읽기
                line = f.readline().strip()
                if not line:
                    break
                first_lines.append(line)
        
        # STEP 파일의 기본 헤더 확인
        has_iso_header = any('ISO-10303' in line for line in first_lines)
        has_step_header = any(line.startswith('ISO-10303') for line in first_lines)
        
        info = {
            'size': file_size,
            'first_lines': first_lines,
            'has_iso_header': has_iso_header,
            'has_step_header': has_step_header
        }
        
        return True, info
        
    except Exception as e:
        return False, f"파일 읽기 오류: {str(e)}"


def analyze_class_structure(class_path, debug=False):
    """
    클래스 디렉토리의 구조를 분석해서 처리 방법 결정
    Returns: ('direct', step_files) or ('splits', split_dirs)
    """
    if debug:
        print(f"    Analyzing structure of: {class_path}")
    
    try:
        items = os.listdir(class_path)
    except Exception as e:
        if debug:
            print(f"    Error reading directory: {e}")
        return 'error', str(e)
    
    # 현재 디렉토리의 STEP 파일들
    step_files = [f for f in items if f.lower().endswith(('.stp', '.step'))]
    
    # 하위 디렉토리들
    subdirs = [d for d in items if os.path.isdir(os.path.join(class_path, d))]
    
    # 하위 디렉토리에 STEP 파일이 있는지 확인
    subdirs_with_steps = []
    for subdir in subdirs:
        subdir_path = os.path.join(class_path, subdir)
        try:
            subdir_files = os.listdir(subdir_path)
            subdir_steps = [f for f in subdir_files if f.lower().endswith(('.stp', '.step'))]
            if subdir_steps:
                subdirs_with_steps.append({
                    'name': subdir,
                    'path': subdir_path,
                    'step_files': subdir_steps
                })
        except Exception as e:
            if debug:
                print(f"    Error reading subdirectory {subdir}: {e}")
            continue
    
    if debug:
        print(f"    Direct STEP files: {len(step_files)}")
        print(f"    Subdirectories with STEP files: {len(subdirs_with_steps)}")
        if subdirs_with_steps:
            print(f"    Split directories: {[d['name'] for d in subdirs_with_steps]}")
    
    # 처리 방법 결정
    if subdirs_with_steps:
        # 하위 디렉토리에 STEP 파일이 있으면 split 구조로 처리
        return 'splits', subdirs_with_steps
    elif step_files:
        # 현재 디렉토리에 STEP 파일이 있으면 직접 처리
        return 'direct', step_files
    else:
        # STEP 파일이 없음
        return 'empty', []


def process_class_direct(class_name, class_path, output_class_dir, path_stp, path_graph, step_files, debug=False):
    """
    클래스 디렉토리에 직접 있는 STEP 파일들을 처리
    """
    if debug:
        print(f"  Processing class '{class_name}' with direct STEP files")
    
    successful = 0
    failed = 0
    
    for i, file in enumerate(step_files, 1):
        print(f"    Processing {i}/{len(step_files)}: {file}")
        
        full_file_path = os.path.join(class_path, file)
        
        if debug:
            print(f"      Full file path: {full_file_path}")
            
            # 파일 검증
            is_valid, validation_info = validate_step_file(full_file_path)
            if not is_valid:
                print(f"      ❌ File validation failed: {validation_info}")
                failed += 1
                continue
            else:
                print(f"      ✓ File is valid (size: {validation_info['size']} bytes)")
        
        try:
            # make_graph_simplex_direct 함수 호출
            rel_file_path = f"{class_name}/{file}"
            
            if debug:
                print(f"      Calling make_graph_simplex_direct with:")
                print(f"        - file_name: {rel_file_path}")
                print(f"        - graph_saves_base_paths: {path_graph}")
                print(f"        - dataset_path: {path_stp}")
            
            g = make_graph_simplex_direct(
                file_name=rel_file_path,
                graph_saves_base_paths=path_graph,
                dataset_path=path_stp
            )
            
            if g is not None:
                successful += 1
                print(f"      ✅ Successfully converted: {file}")
                if debug and hasattr(g, 'number_of_nodes'):
                    print(f"        - Graph nodes: {g.number_of_nodes()}")
                    print(f"        - Graph edges: {g.number_of_edges()}")
            else:
                failed += 1
                print(f"      ⚠️  Warning: make_graph_simplex_direct returned None for {file}")
                
                # 추가 디버깅: 예상 출력 파일 확인
                if debug:
                    expected_output = os.path.join(output_class_dir, file.rsplit('.', 1)[0] + '.graphml')
                    if os.path.exists(expected_output):
                        output_size = os.path.getsize(expected_output)
                        print(f"        - Output file exists: {expected_output}")
                        print(f"        - Output file size: {output_size} bytes")
                    else:
                        print(f"        - Expected output file not found: {expected_output}")
                
        except Exception as e:
            failed += 1
            print(f"      ❌ Error processing {file}: {str(e)}")
            if debug:
                import traceback
                print(f"      Full traceback: {traceback.format_exc()}")
            continue
    
    return successful, failed


def process_class_with_splits(class_name, class_path, output_class_dir, path_stp, path_graph, split_info, debug=False):
    """
    클래스 디렉토리의 split 구조 처리 (train/valid/test 등)
    """
    if debug:
        print(f"  Processing class '{class_name}' with split directories")
    
    successful = 0
    failed = 0
    
    for split_data in split_info:
        split_name = split_data['name']
        split_path = split_data['path']
        step_files = split_data['step_files']
        
        print(f"\n    Processing split: {split_name}")
        
        # Create output split directory
        output_split_dir = os.path.join(output_class_dir, split_name)
        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)
        
        print(f"      Found {len(step_files)} STEP files")
        
        for i, file in enumerate(step_files, 1):
            print(f"      Processing {i}/{len(step_files)}: {file}")
            
            full_file_path = os.path.join(split_path, file)
            
            if debug:
                print(f"        Full file path: {full_file_path}")
                
                # 파일 검증
                is_valid, validation_info = validate_step_file(full_file_path)
                if not is_valid:
                    print(f"        ❌ File validation failed: {validation_info}")
                    failed += 1
                    continue
                else:
                    print(f"        ✓ File is valid (size: {validation_info['size']} bytes)")
            
            try:
                # Construct relative path for make_graph function
                rel_file_path = f"{class_name}/{split_name}/{file}"
                
                if debug:
                    print(f"        Calling make_graph_simplex_direct with:")
                    print(f"          - file_name: {rel_file_path}")
                    print(f"          - graph_saves_base_paths: {path_graph}")
                    print(f"          - dataset_path: {path_stp}")
                
                # Call the graph generation function
                g = make_graph_simplex_direct(
                    file_name=rel_file_path,
                    graph_saves_base_paths=path_graph,
                    dataset_path=path_stp
                )
                
                if g is not None:
                    successful += 1
                    print(f"        ✅ Successfully converted: {file}")
                    if debug and hasattr(g, 'number_of_nodes'):
                        print(f"          - Graph nodes: {g.number_of_nodes()}")
                        print(f"          - Graph edges: {g.number_of_edges()}")
                else:
                    failed += 1
                    print(f"        ⚠️  Warning: make_graph_simplex_direct returned None for {file}")
                    
                    # 추가 디버깅: 예상 출력 파일 확인
                    if debug:
                        expected_output = os.path.join(output_split_dir, file.rsplit('.', 1)[0] + '.graphml')
                        if os.path.exists(expected_output):
                            output_size = os.path.getsize(expected_output)
                            print(f"          - Output file exists: {expected_output}")
                            print(f"          - Output file size: {output_size} bytes")
                        else:
                            print(f"          - Expected output file not found: {expected_output}")
                    
            except Exception as e:
                failed += 1
                print(f"        ❌ Error processing {file}: {str(e)}")
                if debug:
                    import traceback
                    print(f"        Full traceback: {traceback.format_exc()}")
                continue
    
    return successful, failed


def make_graphh_dataset_adaptive(path_stp, path_graph, debug=True):
    """
    적응형 STEP to Graph 변환기
    - 클래스별로 구조를 자동 감지
    - Split 디렉토리가 있으면 split 처리
    - 없으면 직접 파일 처리
    """
    if not path_stp.endswith("/"):
        path_stp = path_stp + "/"
    if not path_graph.endswith("/"):
        path_graph = path_graph + "/"

    if not os.path.exists(path_graph):
        os.makedirs(path_graph)
    
    print(f"🚀 Starting adaptive STEP to GraphML conversion")
    print(f"📁 Source directory: {path_stp}")
    print(f"💾 Output directory: {path_graph}")
    print(f"🐛 Debug mode: {debug}")
    
    # Get all class directories
    try:
        class_dirs = [d for d in os.listdir(path_stp) 
                      if os.path.isdir(os.path.join(path_stp, d))]
    except Exception as e:
        print(f"❌ Error reading source directory: {e}")
        return
    
    if not class_dirs:
        print(f"❌ No class directories found in {path_stp}")
        return
    
    print(f"📂 Found {len(class_dirs)} class directories: {class_dirs}")
    
    total_files = 0
    successful_conversions = 0
    failed_conversions = 0
    
    # 각 클래스별 처리 방법 요약
    processing_summary = []
    
    for class_dir in class_dirs:
        print(f"\n{'='*50}")
        print(f"🏷️  Processing class: {class_dir}")
        print(f"{'='*50}")
        
        class_path = os.path.join(path_stp, class_dir)
        
        # Create output class directory
        output_class_dir = os.path.join(path_graph, class_dir)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        # 클래스 구조 분석
        structure_type, structure_data = analyze_class_structure(class_path, debug)
        
        if structure_type == 'error':
            print(f"  ❌ Error analyzing class structure: {structure_data}")
            processing_summary.append({
                'class': class_dir,
                'type': 'error',
                'files': 0,
                'successful': 0,
                'failed': 0
            })
            continue
        elif structure_type == 'empty':
            print(f"  ⚠️  No STEP files found in class '{class_dir}'")
            processing_summary.append({
                'class': class_dir,
                'type': 'empty',
                'files': 0,
                'successful': 0,
                'failed': 0
            })
            continue
        
        # 파일 개수 계산
        if structure_type == 'direct':
            file_count = len(structure_data)
        else:  # splits
            file_count = sum(len(split_data['step_files']) for split_data in structure_data)
        
        total_files += file_count
        print(f"  📊 Structure type: {structure_type}")
        print(f"  📈 Total STEP files: {file_count}")
        
        # 적절한 처리 방법 선택
        if structure_type == 'direct':
            successful, failed = process_class_direct(
                class_dir, class_path, output_class_dir, path_stp, path_graph, structure_data, debug
            )
        else:  # splits
            successful, failed = process_class_with_splits(
                class_dir, class_path, output_class_dir, path_stp, path_graph, structure_data, debug
            )
        
        successful_conversions += successful
        failed_conversions += failed
        
        processing_summary.append({
            'class': class_dir,
            'type': structure_type,
            'files': file_count,
            'successful': successful,
            'failed': failed
        })
        
        print(f"  ✅ Class '{class_dir}' completed: {successful}/{file_count} successful")
    
    # 최종 요약
    print(f"\n{'='*60}")
    print("🎯 FINAL CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📋 Processing Details by Class:")
    for summary in processing_summary:
        if summary['type'] == 'error':
            print(f"  ❌ {summary['class']}: Error in processing")
        elif summary['type'] == 'empty':
            print(f"  ⚪ {summary['class']}: No STEP files found")
        else:
            success_rate = (summary['successful']/summary['files']*100) if summary['files'] > 0 else 0
            print(f"  📊 {summary['class']}: {summary['successful']}/{summary['files']} files ({success_rate:.1f}%) - {summary['type']} structure")
    
    print(f"\n🏆 Overall Statistics:")
    print(f"  📁 Total classes processed: {len(class_dirs)}")
    print(f"  📄 Total STEP files found: {total_files}")
    print(f"  ✅ Successfully converted: {successful_conversions}")
    print(f"  ❌ Failed conversions: {failed_conversions}")
    
    if total_files > 0:
        success_rate = (successful_conversions/total_files*100)
        print(f"  📈 Overall success rate: {success_rate:.1f}%")
    
    print(f"\n🎉 Conversion completed!")


def make_graphh_dataset(path_stp, path_graph, debug=True):
    """
    기존 재귀 처리 함수 (하위 호환성을 위해 유지)
    """
    print("⚠️  Using legacy recursive mode. Consider using 'adaptive' mode for better results.")
    return make_graphh_dataset_adaptive(path_stp, path_graph, debug)


def make_graphh_dataset_class_structure(path_stp, path_graph, debug=True):
    """
    기존 클래스 구조 함수 (하위 호환성을 위해 유지)
    """
    print("⚠️  Using legacy class_structure mode. Consider using 'adaptive' mode for better results.")
    return make_graphh_dataset_adaptive(path_stp, path_graph, debug)


def debug_make_graph_function(file_path, graph_path, dataset_path):
    """
    make_graph_simplex_direct 함수를 직접 테스트하는 디버깅 함수
    """
    print(f"\n=== DEBUGGING make_graph_simplex_direct ===")
    print(f"Input parameters:")
    print(f"  file_name: {file_path}")
    print(f"  graph_saves_base_paths: {graph_path}")
    print(f"  dataset_path: {dataset_path}")
    
    # 실제 파일 경로 확인
    full_path = os.path.join(dataset_path, file_path)
    print(f"  Computed full path: {full_path}")
    print(f"  File exists: {os.path.exists(full_path)}")
    
    if os.path.exists(full_path):
        print(f"  File size: {os.path.getsize(full_path)} bytes")
        
        # 파일 내용 미리보기
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                print(f"  First line: {first_line}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    # make_graph_simplex_direct 함수 호출
    try:
        result = make_graph_simplex_direct(
            file_name=file_path,
            graph_saves_base_paths=graph_path,
            dataset_path=dataset_path
        )
        print(f"  Function result: {result}")
        print(f"  Result type: {type(result)}")
        
        if result is not None and hasattr(result, 'number_of_nodes'):
            print(f"  Graph nodes: {result.number_of_nodes()}")
            print(f"  Graph edges: {result.number_of_edges()}")
            
        return result
        
    except Exception as e:
        print(f"  Exception occurred: {str(e)}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert STEP files to graphs - Adaptive version")
    parser.add_argument("--path_step", required=True, help="Path to STEP files directory")
    parser.add_argument("--path_graph", required=True, help="Path to save graph files")
    parser.add_argument("--mode", choices=['adaptive', 'recursive', 'class_structure', 'debug'], 
                       default='adaptive',
                       help="Processing mode: 'adaptive' (recommended) automatically detects structure, "
                            "'recursive' for general recursive processing, "
                            "'class_structure' for class/train-valid-test structure, "
                            "'debug' for single file debugging")
    parser.add_argument("--debug", action='store_true', help="Enable debug output")
    parser.add_argument("--test_file", help="Single file to test (for debug mode)")
    
    args = parser.parse_args()

    path_stp = args.path_step
    path_graph = args.path_graph
    mode = args.mode
    debug = args.debug

    print(f"🚀 Starting conversion in '{mode}' mode...")
    
    if mode == 'adaptive':
        make_graphh_dataset_adaptive(path_stp, path_graph, debug)
    elif mode == 'recursive':
        make_graphh_dataset(path_stp, path_graph, debug)
    elif mode == 'class_structure':
        make_graphh_dataset_class_structure(path_stp, path_graph, debug)
    elif mode == 'debug':
        if args.test_file:
            debug_make_graph_function(args.test_file, path_graph, path_stp)
        else:
            print("Debug mode requires --test_file parameter")