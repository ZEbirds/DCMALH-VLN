import json
import os
import glob
import re
import numpy as np
from pathlib import Path
from collections import Counter


# =========================================================
# 1. Metrics
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
        self.spls = []
        self.cgts = []  # NEW
        self.actual_path_steps = [] 

        # specific target failure stats
        self.target_stats = {}
        self.failure_reasons = Counter()

    def add_sample(
        self,
        success,
        gt_step,
        path_step,
        oracle_success,
        navigation_error,
        subtask_successes,
        subtask_path_step,
        gt_length,
        error_length,
        spl,
        cgt=None,
        actual_path_step=None,
    ):
        self.successes.append(success)
        self.gt_steps.append(gt_step)
        self.path_steps.append(path_step)
        self.actual_path_steps.append(actual_path_step)
        self.oracle_successes.append(oracle_success)
        self.navigation_errors.append(navigation_error)
        self.subtask_successes.append(subtask_successes)
        self.subtask_path_steps.append(subtask_path_step)
        self.gt_length.append(gt_length)
        self.error_length.append(error_length)
        self.spls.append(spl)
        if cgt is not None:
            self.cgts.append(cgt)

    def add_target_result(self, target, is_fail):
        if target not in self.target_stats:
            self.target_stats[target] = {"total": 0, "fail": 0}
        self.target_stats[target]["total"] += 1
        if is_fail:
            self.target_stats[target]["fail"] += 1

    def add_failure_reason(self, reason):
        if reason:
            self.failure_reasons[reason] += 1

    def success_rate(self):
        return sum(self.successes) / len(self.successes) if self.successes else 0

    def oracle_success_rate(self):
        return sum(self.oracle_successes) / len(self.oracle_successes) if self.oracle_successes else 0

    def spl_rate(self):
        return sum(self.spls) / len(self.spls) if self.spls else 0

    def independent_success_rate(self):
        subtask_counts = [len(subtasks) for subtasks in self.subtask_successes]
        total_subtasks = sum(subtask_counts)
        total_successes = sum(sum(subtasks) for subtasks in self.subtask_successes)
        return total_successes / total_subtasks if total_subtasks > 0 else 0

    def conditional_success_rate(self):
        M = len(self.subtask_successes)
        if M == 0:
            return 0
        csr = 0
        for i in range(M):
            sr = 0
            N = len(self.subtask_successes[i])
            if N == 0:
                continue
            s = self.subtask_successes[i][0]
            sr += s * N
            if N == 1:
                csr += sr
                continue
            for j in self.subtask_successes[i][1:]:
                sr += j * (1 + (N - 1) * s)
                s = j
            csr += sr / (N ** 2)
        csr = csr / M
        return csr

    def cgt_rate(self):
        return sum(self.cgts) / len(self.cgts) if self.cgts else 0

    def navigation_error(self):
        return sum(self.navigation_errors) / len(self.navigation_errors) if self.navigation_errors else 0

    def compute(self):
        return {
            "SR (Success Rate)": self.success_rate(),
            "ISR (Independent Success Rate)": self.independent_success_rate(),
            "CSR (Conditional Success Rate)": self.conditional_success_rate(),
            "CGT (CSR weighted by Ground Truth)": self.cgt_rate(),
            "OSR (Oracle Success Rate)": self.oracle_success_rate(),
            "SPL (Success weighted by Path Length)": self.spl_rate(),
            "NE (Navigation Error)": self.navigation_error(),
            "Actual Path Steps": safe_mean(self.actual_path_steps) if self.actual_path_steps else 0.0,
        }


# =========================================================
# 2. Helpers
# =========================================================
def calculate_path_length(points_list):
    if len(points_list) < 2:
        return 0.0
    pts = np.array(points_list)
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else 0.0


def format_or_dash(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.2f}%" if pct else f"{v:.2f}"


def normalize_region_label(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    s = s.replace("region", "").strip()
    return s or None


def map_subtask_index_to_trial_name(subtask_index):
    # keep your original mapping logic
    if subtask_index <= 2:
        return "trial_0"
    elif subtask_index <= 4:
        return "trial_1"
    elif subtask_index <= 6:
        return "trial_2"
    else:
        return "trial_3"


# =========================================================
# 3. CGT / CSR
# =========================================================
def compute_task_csr(subtask_successes):
    """
    Keep the same CSR style as your original code.
    """
    N = len(subtask_successes)
    if N == 0:
        return 0.0
    sr = 0.0
    s = subtask_successes[0]
    sr += s * N
    if N == 1:
        return sr / (N ** 2)
    for j in subtask_successes[1:]:
        sr += j * (1 + (N - 1) * s)
        s = j
    return sr / (N ** 2)


def compute_task_cgt(subtask_successes, subtask_gt_lengths):
    """
    Paper Eq.(4): CGT = weighted CSR by GT length.

    Since the paper's notation for s_{i-1} is not fully precise for i=0,
    we keep a consistent convention:
    - first step uses prev_s = 1
    - then follow the same recurrence style as CSR
    """
    N = len(subtask_successes)
    if N == 0 or len(subtask_gt_lengths) != N:
        return 0.0

    P = sum(subtask_gt_lengths)
    if P <= 0:
        return 0.0

    total = 0.0
    prev_s = 1
    for i, s in enumerate(subtask_successes):
        w = subtask_gt_lengths[i] / P
        cond = N if i == 0 else (1 + (N - 1) * prev_s)
        total += w * s * cond
        prev_s = s

    return total / N


# =========================================================
# 4. Config parsing: Move_to targets + regions
# =========================================================
def get_move_to_targets_and_regions_from_config(task_dir):
    """
    Read:
        task/.../config.json

    Return:
        targets: ['hoverboard', 'sink', 'towel', 'sink', ...]
        regions: ['bedroom', 'kitchen', 'bathroom', 'kitchen', ...]
    """
    config_path = os.path.join(task_dir, "config.json")
    if not os.path.exists(config_path):
        return [], []

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    subtask_list = config_data.get("Subtask list", [])
    object_list = config_data.get("Object", [])

    targets = []
    regions = []

    move_to_pattern = r"Move_to\('(.+?)'\)"
    move_idx = 0

    for subtask in subtask_list:
        match = re.match(move_to_pattern, subtask)
        if not match:
            continue

        if move_idx < len(object_list):
            obj_name = object_list[move_idx][0]
            region_raw = object_list[move_idx][1] if len(object_list[move_idx]) > 1 else None
            region_name = normalize_region_label(region_raw)
        else:
            raw = match.group(1)
            obj_name = raw.split("_")[0]
            region_name = None

        targets.append(str(obj_name).lower().strip())
        regions.append(region_name)
        move_idx += 1

    return targets, regions


# =========================================================
# 5. Scene graph helpers
# =========================================================
def find_scene_graph_json(task_pred_dir):
    """
    Search for a DSG / scene graph json under the prediction task directory.
    Works with layouts like:
        .../task_id/backend/dsg_with_mesh.json
        .../task_id/.../scene_graph.json
    """
    preferred_names = {
        "dsg_with_mesh.json",
        "dsg.json",
        "scene_graph.json",
        "scenegraph.json",
        "graph.json",
    }

    candidates = []
    for p in Path(task_pred_dir).rglob("*.json"):
        name = p.name.lower()
        if "task_execution_details" in name or "config" in name or "task" in name:
            continue
        if name in preferred_names or ("dsg" in name and "json" in name) or ("scene" in name and "graph" in name):
            candidates.append(str(p))

    if not candidates:
        return None

    # Prefer backend paths
    candidates.sort(key=lambda x: (0 if "backend" in x.lower() else 1, len(x)))
    return candidates[0]


def extract_object_num_from_scene_graph(scene_graph_data):
    """
    In your visualization code, object-level nodes are layer == 2.
    So we count layer-2 nodes as Obj Num.
    """
    nodes = scene_graph_data.get("nodes", [])
    if isinstance(nodes, dict):
        nodes = list(nodes.values())

    if isinstance(nodes, list) and nodes:
        layer2 = [n for n in nodes if isinstance(n, dict) and n.get("layer", 2) == 2]
        if layer2:
            return len(layer2)
        return len(nodes)

    # fallback recursive scan
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.lower() == "nodes" and isinstance(v, list):
                    return extract_object_num_from_scene_graph({"nodes": v})
                r = walk(v)
                if r is not None:
                    return r
        elif isinstance(x, list):
            for item in x:
                r = walk(item)
                if r is not None:
                    return r
        return None

    return walk(scene_graph_data)


def build_region_to_floor_map(scene_graph_data):
    """
    Try to infer region -> floor from any dicts containing both:
      region/room/area name + floor/level id
    If nothing exists, return {}.
    """
    region_to_floor = {}

    def visit(x):
        if isinstance(x, dict):
            # possible keys
            name_keys = ["region", "region_name", "room", "room_name", "area", "area_name", "name", "label"]
            floor_keys = ["floor", "floor_id", "level", "level_id", "storey", "storey_id"]

            name_val = None
            floor_val = None

            for nk in name_keys:
                if nk in x and x[nk] is not None:
                    name_val = x[nk]
                    break
            for fk in floor_keys:
                if fk in x and x[fk] is not None:
                    floor_val = x[fk]
                    break

            if name_val is not None and floor_val is not None:
                region_to_floor[normalize_region_label(name_val)] = str(floor_val)

            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(scene_graph_data)
    return region_to_floor


def extract_floor_span_from_scene_graph(scene_graph_data, region_labels):
    """
    If region->floor mapping exists, return number of distinct floors spanned.
    Otherwise return None.
    """
    region_to_floor = build_region_to_floor_map(scene_graph_data)
    if not region_to_floor:
        return None

    floors = set()
    for r in region_labels:
        if r is None:
            continue
        f = region_to_floor.get(normalize_region_label(r))
        if f is not None:
            floors.add(f)

    return len(floors) if floors else None


# =========================================================
# 6. Table 2 / Table 6 summarizers
# =========================================================
def summarize_bucket(records):
    if not records:
        return None

    sr = safe_mean([r["task_success"] for r in records])
    ne = safe_mean([r["nav_error"] for r in records])

    total_subtasks = sum(len(r["subtask_successes"]) for r in records)
    total_successes = sum(sum(r["subtask_successes"]) for r in records)
    isr = total_successes / total_subtasks if total_subtasks > 0 else 0.0

    csr = safe_mean([r["csr"] for r in records])
    cgt = safe_mean([r["cgt"] for r in records])

    return {"SR": sr, "NE": ne, "ISR": isr, "CSR": csr, "CGT": cgt}


def summarize_table6(records):
    if not records:
        return None

    return {
        "Task Steps": safe_mean([r["task_steps"] for r in records]),
        "Actual Path Steps": safe_mean([r["actual_path_steps"] for r in records]),
        "Nav Dis": safe_mean([r["nav_dis"] for r in records]),
        "Obj Num": safe_mean([r["obj_num"] for r in records]),
        "Floor Span": safe_mean([r["floor_span"] for r in records]) if any(r["floor_span"] is not None for r in records) else None,
        "Reentry Rate": safe_mean([r["reentry_rate"] for r in records]) * 100.0,
        "Area Association": safe_mean([r["area_association"] for r in records]),
    }


# =========================================================
# 7. Main evaluation
# =========================================================
def evaluate_all(base_pred_dir, official_task_dir, output_txt_path, success_threshold=3.0):
    metrics = NavigationMetrics()
    pred_files = list(Path(base_pred_dir).rglob("task_execution_details.json"))
    total_tasks_evaluated = 0

    # Table 2 / Table 6 buckets
    length_stats = {2: [], 3: [], 4: []}
    group_stats = {"2-3": [], "3-4": []}

    with open(output_txt_path, "w", encoding="utf-8") as log_file:

        def log_print(msg=""):
            print(msg)
            log_file.write(msg + "\n")

        for pred_file in pred_files:
            task_folder_name = pred_file.parent.name
            log_print(f"▶ 开始评估任务: {task_folder_name}")

            with open(pred_file, "r", encoding="utf-8") as f:
                pred_data = json.load(f)

            instruction = pred_data.get("instruction", "")
            subtask_results = pred_data.get("subtask_results", [])
            detailed_trajectory = pred_data.get("detailed_trajectory", [])
            actual_full_trajectory = pred_data.get("full_trajectory", [])

            if not subtask_results:
                log_print("  ⚠️ 警告: 该任务没有子任务执行记录，跳过。\n")
                continue

            # -------------------------------------------------
            # extract subtask end positions from predicted traj
            # -------------------------------------------------
            subtask_end_positions = {}
            for step in detailed_trajectory:
                step_type = step.get("step_type", "")
                if step_type.startswith("subtask_completed_") or step_type.startswith("subtask_failed_"):
                    completed_index = step["subtask_info"].get("completed_index")
                    failed_index = step["subtask_info"].get("failed_index")
                    target_idx = completed_index if completed_index is not None else failed_index

                    if target_idx is not None:
                        subtask_idx = target_idx + 1
                        agent_pos = step.get("agent_position")
                        if agent_pos:
                            if isinstance(agent_pos, dict):
                                pos = [agent_pos["x"], agent_pos["y"], agent_pos["z"]]
                            else:
                                pos = agent_pos
                            subtask_end_positions[subtask_idx] = pos
                            log_print(f"  ✅ 提取子任务 {subtask_idx} 结束位置: {pos}")

            # -------------------------------------------------
            # locate GT task file (keep your original path logic)
            # -------------------------------------------------
            safe_instruction = instruction.replace("?", "").replace("/", "")
            search_pattern = f"{official_task_dir}/*/*/{safe_instruction}/success/trial_1/task.json"
            gt_files = glob.glob(search_pattern)

            if not gt_files:
                log_print(f"  ⚠️ 警告: 找不到真值文件，跳过。\n")
                continue

            gt_file = gt_files[0]
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            trials = gt_data["trial"]
            trial_end_positions = {}
            trial_start_positions = {}
            trial_path_lengths = {}
            trial_step_counts = {}
            gt_total_path_length = 0.0

            for trial_name, trial_data in trials.items():
                if "pos" in trial_data and trial_data["pos"]:
                    trial_start_positions[trial_name] = trial_data["pos"][0]
                    trial_end_positions[trial_name] = trial_data["pos"][-1]
                    trial_path_lengths[trial_name] = calculate_path_length(trial_data["pos"])
                    trial_step_counts[trial_name] = len(trial_data["pos"])
                    gt_total_path_length += trial_path_lengths[trial_name]

            log_print(f"  📂 真值文件: {gt_file}")

            # -------------------------------------------------
            # config.json: get Move_to targets + regions
            # -------------------------------------------------
            task_dir = os.path.dirname(os.path.dirname(os.path.dirname(gt_file)))
            move_to_targets, move_to_regions = get_move_to_targets_and_regions_from_config(task_dir)

            if move_to_targets:
                log_print(f"  📌 Move_to Targets: {move_to_targets}")
            else:
                log_print("  ⚠️ 没有解析到 Move_to targets")

            # -------------------------------------------------
            # scene graph: object num / floor span
            # -------------------------------------------------
            scene_graph_path = find_scene_graph_json(pred_file.parent)
            scene_graph_data = None
            obj_num = None
            floor_span = None

            if scene_graph_path and os.path.exists(scene_graph_path):
                try:
                    with open(scene_graph_path, "r", encoding="utf-8") as f:
                        scene_graph_data = json.load(f)
                    obj_num = extract_object_num_from_scene_graph(scene_graph_data)
                    floor_span = extract_floor_span_from_scene_graph(scene_graph_data, move_to_regions)
                except Exception as e:
                    log_print(f"  ⚠️ 场景图读取失败: {e}")
                    scene_graph_data = None

            # -------------------------------------------------
            # evaluate only Move_to
            # -------------------------------------------------
            full_subtasks_blueprint = pred_data.get("subtasks", [])
            results_map = {res.get("subtask_index", idx + 1): res for idx, res in enumerate(subtask_results)}

            my_successes = []
            my_nav_errors = []
            expected_nav_count = 0
            task_failure_reason = None

            # per-task table stats
            per_subtask_gt_steps = []
            per_subtask_gt_lengths = []
            per_subtask_nav_dists = []

            nav_idx = 0  # counts Move_to only

            for idx, subtask_bp in enumerate(full_subtasks_blueprint):
                subtask_index = idx + 1
                subtask_type = subtask_bp.get("type", "")

                # only evaluate Move_to, ignore grab/release
                if subtask_type in ["grab", "release"]:
                    log_print(f"\n    🔹 子任务 {subtask_index} [{subtask_type}]: ➖ 纯语义动作，跳过评估不计入。")
                    continue

                expected_nav_count += 1
                nav_idx += 1

                subtask_executed_log = results_map.get(subtask_index)
                log_print(f"\n    🔹 子任务 {subtask_index} [{subtask_type}] (Move_to #{nav_idx}):")

                target_name = move_to_targets[nav_idx - 1] if nav_idx - 1 < len(move_to_targets) else f"move_to_{nav_idx}"
                target_region = move_to_regions[nav_idx - 1] if nav_idx - 1 < len(move_to_regions) else None

                if not subtask_executed_log:
                    log_print("      ❌ 未执行 (由于超时或物理崩溃跳过)")
                    if not task_failure_reason:
                        task_failure_reason = "中途崩溃/未走完全程 (Crash or Timeout)"
                    # still count as failure for target stats
                    metrics.add_target_result(target_name, is_fail=True)
                    continue

                my_final_pos = subtask_end_positions.get(subtask_index)
                if not my_final_pos:
                    my_final_pos = subtask_executed_log.get("final_position")

                if not my_final_pos:
                    my_successes.append(0)
                    my_nav_errors.append(5.0)
                    log_print("      ❌ 预测坐标丢失，判定失败")
                    if not task_failure_reason:
                        task_failure_reason = "坐标信息丢失 (Missing Coordinates)"
                    metrics.add_target_result(target_name, is_fail=True)
                    continue

                trial_name = map_subtask_index_to_trial_name(subtask_index)
                gt_target_pos = trial_end_positions.get(trial_name, trials["trial_1"]["pos"][-1])

                log_print(f"      目标: {target_name} | 区域: {target_region}")
                log_print(f"      预测坐标: {my_final_pos}")
                log_print(f"      真值坐标 ({trial_name} 终点): {gt_target_pos}")

                my_xy = np.array([my_final_pos[0], my_final_pos[2]])
                gt_xy = np.array([gt_target_pos[0], gt_target_pos[2]])
                dist = np.linalg.norm(my_xy - gt_xy)
                my_nav_errors.append(dist)

                log_print(f"      欧氏距离: {dist:.4f}m")

                # per-subtask GT stats for CGT / Table 6
                trial_data = trials.get(trial_name)
                if trial_data and "pos" in trial_data and trial_data["pos"]:
                    per_subtask_gt_steps.append(len(trial_data["pos"]))
                    per_subtask_gt_lengths.append(calculate_path_length(trial_data["pos"]))
                    # For Table 6 "Nav Dis", we use GT path length as a practical proxy
                    per_subtask_nav_dists.append(calculate_path_length(trial_data["pos"]))
                else:
                    per_subtask_gt_steps.append(None)
                    per_subtask_gt_lengths.append(None)
                    per_subtask_nav_dists.append(None)

                if dist <= success_threshold:
                    my_successes.append(1)
                    log_print("      ✅ 判定成功")
                    metrics.add_target_result(target_name, is_fail=False)
                else:
                    my_successes.append(0)
                    log_print("      ❌ 判定失败")
                    metrics.add_target_result(target_name, is_fail=True)
                    if not task_failure_reason:
                        task_failure_reason = f"导航误差超标 (Nav Error > {success_threshold}m)"

            # -------------------------------------------------
            # task-level success
            # -------------------------------------------------
            if expected_nav_count > 0 and len(my_successes) == expected_nav_count and all(s == 1 for s in my_successes):
                task_success = 1
                task_failure_reason = None
            else:
                task_success = 0
                if not task_failure_reason:
                    task_failure_reason = "未知失败 (Unknown Error)"

            if task_success == 0:
                metrics.add_failure_reason(task_failure_reason)

            # -------------------------------------------------
            # metrics / CGT
            # -------------------------------------------------
            actual_total_path_length = calculate_path_length(actual_full_trajectory)
            spl = float(task_success) if gt_total_path_length == 0 else task_success * (gt_total_path_length / max(gt_total_path_length, actual_total_path_length))

            oracle_success = 0
            sorted_trials = sorted(trial_end_positions.keys())
            if sorted_trials and actual_full_trajectory:
                final_goal_pos = trial_end_positions[sorted_trials[-1]]
                for path_pt in actual_full_trajectory:
                    if np.linalg.norm(np.array(path_pt) - np.array(final_goal_pos)) <= success_threshold:
                        oracle_success = 1
                        break

            # align lengths for CGT
            valid_gt_lengths = []
            valid_steps = []
            valid_nav_dis = []
            for stp, gl, nd in zip(per_subtask_gt_steps, per_subtask_gt_lengths, per_subtask_nav_dists):
                if gl is not None:
                    valid_gt_lengths.append(gl)
                if stp is not None:
                    valid_steps.append(stp)
                if nd is not None:
                    valid_nav_dis.append(nd)

            task_csr = compute_task_csr(my_successes)
            task_cgt = compute_task_cgt(my_successes, valid_gt_lengths) if len(valid_gt_lengths) == len(my_successes) else 0.0

            log_print(f"   📈 路径统计: 真值最优总长度: {gt_total_path_length:.2f}m | 实际走过总长度: {actual_total_path_length:.2f}m")
            log_print(f"   📊 任务判定: {'🏆 成功' if task_success else '❌ 失败'} | OSR 判定: {oracle_success} | SPL 得分: {spl:.4f}")
            log_print(f"   成功子任务(仅限执行过的导航): {sum(my_successes)}/{len(my_successes)}")
            log_print("-" * 60)

            # -------------------------------------------------
            # Table 6 style stats
            # -------------------------------------------------
            region_counter = Counter([r for r in move_to_regions if r is not None])
            area_association = max(region_counter.values()) if region_counter else None
            reentry_rate = 1.0 if any(v > 1 for v in region_counter.values()) else 0.0

            task_num_nav = len(move_to_targets) if move_to_targets else len(my_successes)

            # 计算实际轨迹步数
            actual_path_step = len(actual_full_trajectory) if actual_full_trajectory else 0

            task_record = {
                "task_success": task_success,
                "nav_error": float(np.mean(my_nav_errors)) if my_nav_errors else 0.0,
                "subtask_successes": my_successes,
                "task_steps": safe_mean(valid_steps),          # Table 6: Task Steps
                "actual_path_steps": actual_path_step,
                "nav_dis": safe_mean(valid_nav_dis),          # Table 6: Nav Dis
                "obj_num": obj_num,                            # Table 6: Obj Num
                "floor_span": floor_span,                      # Table 6: Floor Span
                "reentry_rate": reentry_rate,                  # Table 6: Reentry Rate
                "area_association": area_association,          # Table 6: Area Association
                "csr": task_csr,
                "cgt": task_cgt,
            }

            if task_num_nav in length_stats:
                length_stats[task_num_nav].append(task_record)

            # Table 2 style buckets (inclusive, as labeled in paper)
            if 2 <= task_num_nav <= 3:
                group_stats["2-3"].append(task_record)
            if 3 <= task_num_nav <= 4:
                group_stats["3-4"].append(task_record)

            metrics.add_sample(
                success=task_success,
                gt_step=sum(len(trial["pos"]) for trial in trials.values()),
                path_step=len(detailed_trajectory),
                actual_path_step=actual_path_step,
                oracle_success=oracle_success,
                navigation_error=np.mean(my_nav_errors) if my_nav_errors else 0,
                subtask_successes=my_successes,
                subtask_path_step=[1] * len(my_successes),
                gt_length=[1] * len(my_successes),
                error_length=my_nav_errors,
                spl=spl,
                cgt=task_cgt,
            )
            total_tasks_evaluated += 1

        # =========================================================
        # Final reports
        # =========================================================
        log_print("\n" + "=" * 60)
        log_print(" 🚨 失败率最高的导航目标 (Top Failed Targets, 至少出现 3 次) 🚨 ")
        log_print("=" * 60)

        target_stats_list = []
        for tgt, stats in metrics.target_stats.items():
            if stats["total"] >= 3:
                rate = stats["fail"] / stats["total"]
                target_stats_list.append((tgt, rate, stats["fail"], stats["total"]))

        target_stats_list.sort(key=lambda x: (-x[1], -x[3], -x[2]))

        if not target_stats_list:
            log_print("  (暂无出现次数 >= 3 的目标对象)")
        else:
            for idx, (tgt, rate, fails, total) in enumerate(target_stats_list, 1):
                log_print(f" {idx}. {tgt.ljust(15)} : 失败率 {rate * 100:>5.1f}% ({fails}/{total})")

        log_print("\n" + "=" * 60)
        log_print(" 🔍 任务全局失败原因诊断 (Failure Reason Breakdown) 🔍 ")
        log_print("=" * 60)
        total_fails = sum(metrics.failure_reasons.values())
        if total_fails == 0:
            log_print(" 完美！没有失败任务！")
        else:
            for reason, count in metrics.failure_reasons.most_common():
                pct = count / total_fails * 100
                log_print(f" - {reason}: {count} 次 ({pct:.1f}%)")

        log_print("\n" + "=" * 60)
        log_print(" 🏆 最终评估成绩单 (Final Metrics) 🏆 ")
        log_print("=" * 60)
        results = metrics.compute()
        for k, v in results.items():
            if "Rate" in k or "SPL" in k or "CGT" in k or "CSR" in k or "ISR" in k or "SR" in k or "OSR" in k:
                log_print(f" {k}: {v * 100:.2f}%")
            else:
                log_print(f" {k}: {v:.2f} meters")
        log_print("=" * 60)

        # =========================================================
        # Table 2 style summary
        # =========================================================
        log_print("\n" + "=" * 60)
        log_print(" 📊 Table 2 style: 2-3 subtasks / 3-4 subtasks ")
        log_print("=" * 60)
        for name in ["2-3", "3-4"]:
            s = summarize_bucket(group_stats[name])
            if s is None:
                log_print(f" {name}: (no samples)")
                continue
            log_print(
                f" {name} | "
                f"SR {s['SR'] * 100:.2f}% | "
                f"NE {s['NE']:.2f}m | "
                f"ISR {s['ISR'] * 100:.2f}% | "
                f"CSR {s['CSR'] * 100:.2f}% | "
                f"CGT {s['CGT'] * 100:.2f}%"
            )

        # =========================================================
        # Table 6 style summary
        # =========================================================
        log_print("\n" + "=" * 60)
        log_print(" 📊 Table 6 style: 2 / 3 / 4 subtasks ")
        log_print("=" * 60)
        for n in [2, 3, 4]:
            s = summarize_table6(length_stats[n])
            if s is None:
                log_print(f" {n} subtasks: (no samples)")
                continue

            floor_span_str = format_or_dash(s["Floor Span"]) if s["Floor Span"] is not None else "-"

            log_print(
                f" {n} subtasks | "
                f"Task Steps {format_or_dash(s['Task Steps'])} | "
                f"Actual Path Steps {format_or_dash(s['Actual Path Steps'])} | " 
                f"Nav Dis {format_or_dash(s['Nav Dis'])} | "
                f"Obj Num {format_or_dash(s['Obj Num'])} | "
                f"Floor Span {floor_span_str} | "
                f"Reentry Rate {format_or_dash(s['Reentry Rate'], pct=True)} | "
                f"Area Association {format_or_dash(s['Area Association'])}"
            )

    print(f"\n🎉 评估完成！结果保存在: {output_txt_path}")
    return total_tasks_evaluated


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_PRED_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "lhpr_vln_verification_baseline")
    OFFICIAL_TASK_DIR = "/home/csl-p920/Zibo Zheng/ZiboZHeng/graph_eqa/LH-VLN/LHPR-VLN/huggingface/hub/datasets--Starry123--LHPR-VLN/snapshots/af66785e0b456639088d673bc100dcc63f2df997/task (1)"

    os.makedirs(BASE_PRED_DIR, exist_ok=True)
    OUTPUT_TXT = os.path.join(BASE_PRED_DIR, "final_evaluation_results.txt")

    print(f"Target Prediction Directory: {os.path.abspath(BASE_PRED_DIR)}")
    print(f"Target Output Text File: {os.path.abspath(OUTPUT_TXT)}")

    evaluated_count = evaluate_all(
        base_pred_dir=BASE_PRED_DIR,
        official_task_dir=OFFICIAL_TASK_DIR,
        output_txt_path=OUTPUT_TXT,
        success_threshold=3,
    )

    print(f"\n 本次评估共处理了 {evaluated_count} 个任务")