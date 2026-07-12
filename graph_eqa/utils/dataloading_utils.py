import csv, os, ast
import numpy as np
import json
from pathlib import Path
import click

def load_lhpr_vln_dataset_simple(cfg):
    """
    简化的LHPR-VLN数据加载函数
    专注于验证数据加载是否正确
    """
    click.secho("\n" + "="*80, fg="cyan")
    click.secho("LHPR-VLN Dataset Loading", fg="cyan", bold=True)
    click.secho("="*80, fg="cyan")
    
    # 设置数据路径
    if hasattr(cfg.data, 'task_data'):
        task_data_path = Path(cfg.data.task_data)
    else:
        # 默认路径
        task_data_path = Path(__file__).resolve().parent.parent / "LH-VLN" / "LHPR-VLN" / "huggingface" / "hub" / "datasets--Starry123--LHPR-VLN" / "snapshots" / "af66785e0b456639088d673bc100dcc63f2df997" / "task_data"
    
    if hasattr(cfg.data, 'step_task_data'):
        step_task_data_path = Path(cfg.data.step_task_data)
    else:
        step_task_data_path = Path(__file__).resolve().parent.parent / "LH-VLN" / "LHPR-VLN" / "huggingface" / "hub" / "datasets--Starry123--LHPR-VLN" / "snapshots" / "af66785e0b456639088d673bc100dcc63f2df997" / "step_task"
    
    click.secho(f"Task data path: {task_data_path}", fg="white")
    click.secho(f"Step task data path: {step_task_data_path}", fg="white")
    
    # 检查路径是否存在
    if not task_data_path.exists():
        click.secho(f"ERROR: Task data path does not exist: {task_data_path}", fg="red", bold=True)
        click.secho("Please check your configuration and dataset download.", fg="red")
        return []
    
    # 设置批次
    mode = cfg.dataset.mode if hasattr(cfg.dataset, 'mode') else 'test'
    click.secho(f"Mode: {mode}", fg="white")
    
    if mode == 'train':
        batches = cfg.dataset.train_batch if hasattr(cfg.dataset, 'train_batch') else ["batch_1", "batch_2", "batch_3", "batch_4", "batch_5", "batch_6"]
    elif mode == 'valid':
        batches = cfg.dataset.val_batch if hasattr(cfg.dataset, 'val_batch') else ["batch_7"]
    else:  # test
        batches = cfg.dataset.test_batch if hasattr(cfg.dataset, 'test_batch') else ["batch_8"]
    
    click.secho(f"Batches to load: {batches}", fg="white")
    
    all_tasks = []
    
    for batch in batches:
        batch_tasks = load_batch_tasks_simple(task_data_path, batch)
        all_tasks.extend(batch_tasks)
        
        if batch_tasks:
            click.secho(f"✓ Loaded {len(batch_tasks)} tasks from batch {batch}", fg="green")
            # 显示第一个任务的简要信息
            show_task_summary(batch_tasks[0])
        else:
            click.secho(f"✗ No tasks loaded from batch {batch}", fg="yellow")
    
    # 总体统计
    click.secho("\n" + "="*80, fg="cyan")
    click.secho("DATASET SUMMARY", fg="cyan", bold=True)
    click.secho("="*80, fg="cyan")
    click.secho(f"Total tasks loaded: {len(all_tasks)}", fg="white")
    
    if all_tasks:
        # 按子任务数量统计
        subtask_counts = {}
        for task in all_tasks:
            count = len(task.get('parsed_subtasks', []))
            subtask_counts[count] = subtask_counts.get(count, 0) + 1
        
        click.secho("\nSubtask count distribution:", fg="white")
        for count in sorted(subtask_counts.keys()):
            percentage = subtask_counts[count]/len(all_tasks)*100
            click.secho(f"  {count} subtasks: {subtask_counts[count]} tasks ({percentage:.1f}%)", fg="white")
        
        # 场景统计
        scenes = {}
        for task in all_tasks:
            scene = task.get('Scene', 'unknown')
            scenes[scene] = scenes.get(scene, 0) + 1
        
        click.secho(f"\nUnique scenes: {len(scenes)}", fg="white")
        if len(scenes) <= 10:
            click.secho("Top scenes:", fg="white")
            sorted_scenes = sorted(scenes.items(), key=lambda x: x[1], reverse=True)
            for scene, count in sorted_scenes[:5]:
                click.secho(f"  {scene}: {count} tasks", fg="white")
    
    return all_tasks

def load_batch_tasks_simple(task_data_path, batch):
    """加载单个批次的任务 """
    tasks = []
    batch_path = task_data_path / batch
    
    if not batch_path.exists():
        click.secho(f"ERROR: Batch path does not exist: {batch_path}", fg="red")
        return tasks
    
    # 遍历子文件夹（2, 3, 4, 5, 6, 7, 8表示子任务数量）
    for subtask_count_dir in batch_path.iterdir():
        if not subtask_count_dir.is_dir():
            continue
        
        subtask_count = subtask_count_dir.name
        if not subtask_count.isdigit():
            continue
        
        task_dirs = [d for d in subtask_count_dir.iterdir() if d.is_dir()]
        
        for task_dir in task_dirs:
            # 读取config.json
            config_file = task_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    
                    # 添加额外信息
                    task_data['task_dir'] = str(task_dir)
                    task_data['batch'] = batch
                    task_data['subtask_count'] = int(subtask_count)
                    
                    # 解析子任务
                    parsed_subtasks = parse_subtasks_simple(task_data.get('Subtask list', []))
                    task_data['parsed_subtasks'] = parsed_subtasks
                    
                    # ===== 新增：完整解析Object字段 =====
                    if 'Object' in task_data:
                        object_info = parse_object_field(task_data['Object'])
                        # 将解析后的信息添加到task_data中
                        task_data.update(object_info)
                    
                    # # ===== 新增：解析区域信息 =====
                    # if 'Object' in task_data:
                    #     task_data['regions_info'] = extract_regions_info(task_data['Object'])
                    
                    # # ===== 新增：创建对象-区域映射 =====
                    # if 'Object' in task_data:
                    #     task_data['object_region_map'] = create_object_region_mapping(task_data['Object'])
                    #     task_data['region_object_map'] = create_region_object_mapping(task_data['Object'])
                    
                    tasks.append(task_data)
                    
                except Exception as e:
                    click.secho(f"ERROR loading {config_file}: {e}", fg="red")
    
    return tasks

# ===== 新增：解析Object字段的函数 =====
def parse_object_field(object_list):
    """
    完整解析Object字段
    输入示例: [["teapot", "Region 0: Living room"], ["refrigerator", "Region 3: Kitchen"]]
    输出结构化的对象信息
    """
    parsed_info = {
        'objects': [],          # 所有对象名称列表
        'object_details': [],   # 每个对象的详细信息
        'regions': [],          # 所有区域ID列表
        'region_names': [],     # 所有区域名称列表
        'object_count': 0       # 对象数量
    }
    
    if not object_list:
        return parsed_info
    
    for obj_pair in object_list:
        if len(obj_pair) >= 2:
            obj_name = obj_pair[0]
            region_str = obj_pair[1]
            
            # 解析区域字符串，如 "Region 0: Living room"
            region_parts = region_str.split(': ')
            region_id = None
            region_name = region_str
            
            if len(region_parts) >= 2:
                # 提取区域ID，如从"Region 0"中提取"0"
                region_id_str = region_parts[0]  # "Region 0"
                # 提取数字部分
                for char in region_id_str:
                    if char.isdigit():
                        if region_id is None:
                            region_id = char
                        else:
                            region_id += char
                region_name = ': '.join(region_parts[1:])  # "Living room"
            
            # 添加到各个列表
            parsed_info['objects'].append(obj_name)
            
            obj_detail = {
                'name': obj_name,
                'region_raw': region_str,
                'region_id': region_id,
                'region_name': region_name,
                'full_pair': obj_pair
            }
            parsed_info['object_details'].append(obj_detail)
            
            if region_id:
                parsed_info['regions'].append(region_id)
                parsed_info['region_names'].append(region_name)
    
    parsed_info['object_count'] = len(object_list)
    return parsed_info

def parse_subtasks_simple(subtask_list):
    """解析子任务列表"""
    parsed = []
    for subtask in subtask_list:
        if subtask.startswith("Move_to"):
            # Move_to('cup_0')
            try:
                content = subtask[9:-2]  # 提取'cup_0'
                parts = content.split('_')
                if len(parts) >= 2:
                    obj_id = parts[0]
                    region_id = parts[1]
                    parsed.append({
                        'type': 'move_to',
                        'target': obj_id,
                        'region': region_id,
                        'raw': subtask
                    })
                else:
                    parsed.append({
                        'type': 'move_to_error',
                        'raw': subtask,
                        'error': f'Cannot parse: {subtask}'
                    })
            except:
                parsed.append({
                    'type': 'move_to_error',
                    'raw': subtask,
                    'error': 'Parsing error'
                })
        elif subtask.startswith("Grab"):
            # Grab('cup')
            try:
                obj = subtask[6:-2]  # 提取'cup'
                parsed.append({
                    'type': 'grab',
                    'target': obj,
                    'raw': subtask
                })
            except:
                parsed.append({
                    'type': 'grab_error',
                    'raw': subtask,
                    'error': 'Parsing error'
                })
        elif subtask.startswith("Release"):
            # Release('cup')
            try:
                obj = subtask[9:-2]  # 提取'cup'
                parsed.append({
                    'type': 'release',
                    'target': obj,
                    'raw': subtask
                })
            except:
                parsed.append({
                    'type': 'release_error',
                    'raw': subtask,
                    'error': 'Parsing error'
                })
        else:
            parsed.append({
                'type': 'unknown',
                'raw': subtask
            })
    return parsed

def show_task_summary(task_data):
    """显示任务摘要"""
    instruction = task_data.get('Task instruction', 'N/A')
    if len(instruction) > 80:
        instruction = instruction[:77] + "..."
    
    click.secho(f"\n  Sample task from batch {task_data.get('batch', 'N/A')}:", fg="yellow")
    click.secho(f"    Instruction: {instruction}", fg="yellow")
    click.secho(f"    Scene: {task_data.get('Scene', 'N/A')}", fg="yellow")
    click.secho(f"    Robot: {task_data.get('Robot', 'N/A')}", fg="yellow")
    click.secho(f"    Subtasks: {len(task_data.get('parsed_subtasks', []))}", fg="yellow")
