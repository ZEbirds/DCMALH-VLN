import json
import os
import glob
import numpy as np
from pathlib import Path

# =========================================================
# 1. 官方的算分器 (保持不变)
# =========================================================
class NavigationMetrics:
    def __init__(self):
        self.successes = []
        self.gt_steps = []
        self.gt_length = []
        self.error_length = []
        self.path_steps = []
        self.oracle_successes = []
        self.navigation_errors = []
        self.subtask_successes = []
        self.subtask_path_steps = []

    def add_sample(self, success, gt_step, path_step, oracle_success, navigation_error, 
                   subtask_successes, subtask_path_step, gt_length, error_length):
        self.successes.append(success)
        self.gt_steps.append(gt_step)
        self.path_steps.append(path_step)
        self.oracle_successes.append(oracle_success)
        self.navigation_errors.append(navigation_error)
        self.subtask_successes.append(subtask_successes)
        self.subtask_path_steps.append(subtask_path_step)
        self.gt_length.append(gt_length)
        self.error_length.append(error_length)

    def success_rate(self):
        return sum(self.successes) / len(self.successes) if len(self.successes) > 0 else 0

    def independent_success_rate(self):
        subtask_counts = [len(subtasks) for subtasks in self.subtask_successes]
        total_subtasks = sum(subtask_counts)
        total_successes = sum(sum(subtasks) for subtasks in self.subtask_successes)
        return total_successes / total_subtasks if total_subtasks > 0 else 0

    def conditional_success_rate(self):
        M = len(self.subtask_successes)
        if M == 0: return 0
        csr = 0
        for i in range(M):
            sr = 0
            N = len(self.subtask_successes[i])
            if N == 0: continue
            s = self.subtask_successes[i][0]
            sr += s*N
            if N == 1:
                csr += sr
                continue
            for j in self.subtask_successes[i][1:]:
                sr += j*(1+(N-1)*s)
                s = j
            csr += sr/(N**2)
        csr = csr/M
        return csr
    
    def navigation_error(self):
        return sum(self.navigation_errors) / len(self.navigation_errors) if len(self.navigation_errors) > 0 else 0

    def compute(self):
        return {
            "SR (Success Rate)": self.success_rate(),
            "ISR (Independent Success Rate)": self.independent_success_rate(),
            "CSR (Conditional Success Rate)": self.conditional_success_rate(),
            "NE (Navigation Error)": self.navigation_error()
        }


# =========================================================
# 2. 评估主逻辑 (新增批量处理与 TXT 日志保存)
# =========================================================
def evaluate_all(base_pred_dir, official_task_dir, output_txt_path, success_threshold=3.0):
    metrics = NavigationMetrics()
    pred_files = list(Path(base_pred_dir).rglob("task_execution_details.json"))
    total_tasks_evaluated = 0

    with open(output_txt_path, 'w', encoding='utf-8') as log_file:
        
        def log_print(msg=""):
            print(msg)
            log_file.write(msg + "\n")

        for pred_file in pred_files:
            task_folder_name = pred_file.parent.name
            log_print(f"▶ 开始评估任务: {task_folder_name}")

            with open(pred_file, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
                
            instruction = pred_data.get('instruction')
            subtask_results = pred_data.get('subtask_results', [])
            detailed_trajectory = pred_data.get('detailed_trajectory', [])
            
            if not subtask_results:
                log_print("  ⚠️ 警告: 该任务没有子任务执行记录，跳过。\n")
                continue

            # 关键修复：从 detailed_trajectory 提取每个子任务完成时的真实位置
            subtask_end_positions = {}  # {subtask_index: [x, y, z]}
            
            for step in detailed_trajectory:
                step_type = step.get('step_type', '')
                
                # 🚀 核心修复：兼容成功和失败两种情况，确保哪怕任务失败也能提取到停下的坐标
                if step_type.startswith('subtask_completed_') or step_type.startswith('subtask_failed_'):
                    completed_index = step['subtask_info'].get('completed_index')
                    failed_index = step['subtask_info'].get('failed_index')
                    
                    target_idx = completed_index if completed_index is not None else failed_index
                    
                    if target_idx is not None:
                        subtask_idx = target_idx + 1  # 转换为1-based索引
                        agent_pos = step.get('agent_position')
                        if agent_pos:
                            if isinstance(agent_pos, dict):
                                pos = [agent_pos['x'], agent_pos['y'], agent_pos['z']]
                            else:
                                pos = agent_pos
                            subtask_end_positions[subtask_idx] = pos
                            log_print(f"  ✅ 提取子任务 {subtask_idx} 结束位置: {pos}")
                            
            safe_instruction = instruction.replace('?', '').replace('/', '')
            search_pattern = f"{official_task_dir}/*/*/{safe_instruction}/success/trial_1/task.json"
            gt_files = glob.glob(search_pattern)
            
            if not gt_files:
                log_print(f"  ⚠️ 警告: 找不到真值文件，跳过指令 -> {instruction[:30]}...\n")
                continue
                
            gt_file = gt_files[0]
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            # 获取所有 trial 的终点坐标
            trials = gt_data['trial']
            trial_end_positions = {}
            for trial_name, trial_data in trials.items():
                if 'pos' in trial_data and trial_data['pos']:
                    trial_end_positions[trial_name] = trial_data['pos'][-1]
            
            log_print(f"  📂 真值文件: {gt_file}")
            log_print(f"  📍 Trial 终点坐标:")
            for trial_name, pos in trial_end_positions.items():
                log_print(f"    {trial_name}: {pos}")
            
            # =========================================================
            # 🚀 核心修复：遍历全量任务蓝图（Blueprint），而不仅仅是执行记录
            # =========================================================
            full_subtasks_blueprint = pred_data.get('subtasks', [])
            
            # 将智能体实际的执行结果建立索引映射，方便快速反查
            results_map = {res.get('subtask_index', idx+1): res for idx, res in enumerate(subtask_results)}
            
            my_successes = []
            my_nav_errors = []
            
            for idx, subtask_bp in enumerate(full_subtasks_blueprint):
                subtask_index = idx + 1
                subtask_type = subtask_bp.get('type', '')
                
                # 依然过滤掉原地的抓取和放置动作
                if subtask_type in ['grab', 'release']:
                    continue
                
                log_print(f"\n    🔹 评估全局子任务 {subtask_index} [{subtask_type}]:")
                
                # 检查智能体是否真的活到了执行这一步
                subtask_executed_log = results_map.get(subtask_index)
                
                if not subtask_executed_log:
                    # 智能体由于超时或崩溃压根没来得及做这步，直接强行判 0（未完成），计入分母
                    my_successes.append(0)
                    my_nav_errors.append(5.0) # 赋予未达到的惩罚距离基准
                    log_print(f"      ❌ 智能体未尝试该子任务（中途超时/物理崩溃），直接判定失败")
                    continue

                # ──── 如果智能体实际执行了，则走原本的真实物理距离评测 ────
                my_final_pos = subtask_end_positions.get(subtask_index)
                if not my_final_pos:
                    my_final_pos = subtask_executed_log.get('final_position')
                    log_print(f"      ⚠️ 未找到子任务 {subtask_index} 的结束位置，使用备用位置: {my_final_pos}")
                
                log_print(f"      预测坐标: {my_final_pos}")
                
                if not my_final_pos:
                    my_successes.append(0)
                    my_nav_errors.append(5.0)
                    log_print(f"      ❌ 坐标丢失，判定失败")
                    continue
                
                # 判定对应的真值 Trial 阶段
                if subtask_index <= 2: trial_name = 'trial_0'
                elif subtask_index <= 4: trial_name = 'trial_1'
                elif subtask_index <= 6: trial_name = 'trial_2'
                else: trial_name = 'trial_3'
                
                gt_target_pos = trial_end_positions.get(trial_name)
                if not gt_target_pos:
                    gt_target_pos = trials['trial_1']['pos'][-1]
                
                log_print(f"      真值坐标 ({trial_name} 终点): {gt_target_pos}")
                
                # 保持您原本的平面欧氏距离判定不变
                my_xy = np.array([my_final_pos[0], my_final_pos[2]])
                gt_xy = np.array([gt_target_pos[0], gt_target_pos[2]])
                dist = np.linalg.norm(my_xy - gt_xy)
                my_nav_errors.append(dist)
                
                log_print(f"      欧氏距离: {dist:.4f}m")
                
                # 同时验证物理距离以及智能体自身的结算报告
                if dist <= success_threshold:
                    my_successes.append(1)
                    log_print(f"      ✅ 判定成功")
                else:
                    my_successes.append(0)
                    log_print(f"      ❌ 判定失败")
            
            # 打印总结
            task_success = 1 if all(s == 1 for s in my_successes) else 0
            log_print(f"\n  📊 总体任务判定: {'🏆 成功' if task_success else '❌ 失败'}")
            log_print(f"  成功子任务: {sum(my_successes)}/{len(my_successes)}")
            log_print("-" * 60)

            metrics.add_sample(
                success=task_success,
                gt_step=sum(len(trial['pos']) for trial in trials.values()),
                path_step=len(detailed_trajectory),
                oracle_success=1,
                navigation_error=np.mean(my_nav_errors) if my_nav_errors else 0,
                subtask_successes=my_successes,
                subtask_path_step=[1]*len(my_successes),
                gt_length=[1]*len(my_successes),
                error_length=my_nav_errors
            )

            total_tasks_evaluated += 1
        # 打印最终结果
        results = metrics.compute()
        log_print("\n" + "="*60)
        log_print(" 🏆 最终评估成绩单 🏆 ")
        log_print("="*60)
        for k, v in results.items():
            if "Rate" in k:
                log_print(f" {k}: {v * 100:.2f}%")
            else:
                log_print(f" {k}: {v:.2f} meters")
        log_print("="*60)
        # log_print(f"  实际评估数量: {(total_tasks_evaluated)}")
        
    print(f"\n🎉 评估完成！结果保存在: {output_txt_path}")
    return total_tasks_evaluated

if __name__ == "__main__":
    # --- FIX: Use relative paths based on where the script is located ---
    import os
    
    # Get the directory where Evaluate.py is located (.../graph_eqa/scripts/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Assuming your outputs folder is one level up and then into outputs/lhpr_vln_verification/
    # This resolves to: /home/csl-p920/Zibo Zheng/ZiboZHeng/graph_eqa/outputs/lhpr_vln_verification
    BASE_PRED_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "lhpr_vln_verification_ours")
    
    # --- FIX: Update this path to where the Hugging Face dataset actually is on your machine! ---
    # For example, if it's in your graph_eqa folder:
    # OFFICIAL_TASK_DIR = os.path.join(SCRIPT_DIR, "..", "LH-VLN", "LHPR-VLN", "huggingface", "hub", "datasets--Starry123--LHPR-VLN", "snapshots", "af66785e0b456639088d673bc100dcc63f2df997", "task (1)")
    
    # Or if you know the exact absolute path, put it here:
    OFFICIAL_TASK_DIR = "/home/csl-p920/Zibo Zheng/ZiboZHeng/graph_eqa/LH-VLN/LHPR-VLN/huggingface/hub/datasets--Starry123--LHPR-VLN/snapshots/af66785e0b456639088d673bc100dcc63f2df997/task (1)" 
    
    # Specify the output txt file path.
    # We first make sure the BASE_PRED_DIR actually exists before trying to write into it.
    os.makedirs(BASE_PRED_DIR, exist_ok=True) 
    OUTPUT_TXT = os.path.join(BASE_PRED_DIR, "final_evaluation_results.txt")
    
    print(f"Target Prediction Directory: {os.path.abspath(BASE_PRED_DIR)}")
    print(f"Target Output Text File: {os.path.abspath(OUTPUT_TXT)}")
    
    evaluated_count = evaluate_all(
        base_pred_dir=BASE_PRED_DIR, 
        official_task_dir=OFFICIAL_TASK_DIR, 
        output_txt_path=OUTPUT_TXT,
        success_threshold=3
    )

    print(f"\n 本次评估共处理了 {evaluated_count} 个任务")