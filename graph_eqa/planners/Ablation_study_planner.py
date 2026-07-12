import json
from enum import Enum
from typing import List, Union, Tuple, Optional, Dict
import time
import base64
from openai import OpenAI
from graph_eqa.utils.data_utils import get_latest_image
from pydantic import BaseModel, Field
import numpy as np
import re  # 用于解析房间和位置

# 🌟 导入独立的经验记忆模块
from experience_memory import VectorizedExperienceMemory

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class SemanticSimilarityResponse(BaseModel):
    object: str = Field(..., description="The object name")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity between target and object (0-1)")
    has_spatial_relation: bool = Field(..., description="Whether there is a spatial/containment relation (e.g., flower in vase)")
    explanation: str = Field(..., description="Reasoning for score and relation")

class BatchSemanticSimilarityResponse(BaseModel):
    responses: List[SemanticSimilarityResponse] = Field(..., description="List of similarity responses")

class TargetRoomResponse(BaseModel):
    room: Optional[str] = Field(None, description="The target room name extracted from the instruction and subtask, or None if not found")

class VLMPlannerLHPRVLNGPT:
    def __init__(self, cfg, sg_sim, instruction, subtasks, task_path, task_manager=None):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self._instruction = instruction
        self.subtasks = subtasks
        self.task_manager = task_manager
        self._output_path = task_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image
        self._history = ''
        self.full_plan = ''
        self._t = 0
        self._add_history = cfg.add_history
        self._outputs_to_save = [f'Instruction: {self._instruction} \n']
        # 添加visited_frontiers集合
        self.visited_frontiers = set()
        # 相似度阈值
        self.similarity_threshold = 0.6
        # 相似度缓存: (target, obj_name) -> (sim, has_relation, exp)
        self.similarity_cache = {}
        # 对象-房间映射缓存
        self.object_room_cache: Dict[str, str] = {}
        
        # ─── 🌟 [消融实验配置参数一键注入] ───
        self.disable_memory = getattr(cfg, 'disable_memory', False)                 # w/o Memory 消融开关
        self.disable_frontier_score = getattr(cfg, 'disable_frontier_score', False) # w/o Frontier Score 消融开关
        self.disable_safety_filter = getattr(cfg, 'disable_safety_filter', False)   # w/o Safety Filter 消融开关
        self.disable_scene_graph = getattr(cfg, 'disable_scene_graph', False)       # w/o Scene Graph 消融开关
        
        print(f"⚙️ [消融实验状态检查] "
              f"w/o Memory: {self.disable_memory} | "
              f"w/o Frontier Score: {self.disable_frontier_score} | "
              f"w/o Safety Filter: {self.disable_safety_filter} | "
              f"w/o Scene Graph: {self.disable_scene_graph}")
        
        # 初始化大脑皮层：向量化经验记忆模块
        self.experience_memory = VectorizedExperienceMemory(similarity_threshold=0.3)

    @property
    def t(self):
        return self._t

    def get_actions(self):
        object_node_list = Enum('object_node_list', {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)}, type=str)
        if len(self.sg_sim.frontier_node_ids) > 0:
            frontier_node_list = Enum('frontier_node_list', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        else:
            frontier_node_list = None

        room_node_list = Enum('room_node_list', {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)}, type=str)
        region_node_list = Enum('region_node_list', {ac: ac for ac in self.sg_sim.region_node_ids}, type=str)
        return frontier_node_list, room_node_list, region_node_list, object_node_list

    def get_object_room(self, obj_id: str) -> Optional[str]:
        """获取物体所属的房间节点ID"""
        if obj_id in self.object_room_cache:
            return self.object_room_cache[obj_id]
        if not hasattr(self.sg_sim, 'filtered_netx_graph'): return None
        if obj_id not in self.sg_sim.filtered_netx_graph: return None
        
        try:
            room_predecessors = [pred_id for pred_id in self.sg_sim.filtered_netx_graph.predecessors(obj_id) if 'room' in pred_id]
            if room_predecessors:
                self.object_room_cache[obj_id] = room_predecessors[0]
                return room_predecessors[0]
            
            region_predecessors = [pred_id for pred_id in self.sg_sim.filtered_netx_graph.predecessors(obj_id) if 'region' in pred_id]
            for region_id in region_predecessors:
                region_room_predecessors = [pred_id for pred_id in self.sg_sim.filtered_netx_graph.predecessors(region_id) if 'room' in pred_id]
                if region_room_predecessors:
                    self.object_room_cache[obj_id] = region_room_predecessors[0]
                    return region_room_predecessors[0]
            
            for room_id in self.sg_sim.room_node_ids:
                if self.sg_sim.filtered_netx_graph.has_edge(room_id, obj_id):
                    self.object_room_cache[obj_id] = room_id
                    return room_id
        except Exception as e:
            print(f"获取物体 {obj_id} 的房间时出错: {e}")
            return None
        return None

    def get_object_room_info(self, obj_id: str) -> Dict[str, str]:
        result = {
            "object_id": obj_id,
            "object_name": self.get_object_name(obj_id) or obj_id,  
            "room_id": None,
            "room_name": None
        }
        room_id = self.get_object_room(obj_id)
        if room_id:
            result["room_id"] = room_id
            result["room_name"] = self.get_room_name(room_id)
        return result

    def get_objects_in_room(self, room_id: str) -> List[str]:
        if not hasattr(self.sg_sim, 'filtered_netx_graph'): return []
        objects_in_room = []
        if self.sg_sim.include_regions:
            region_ids = [node_id for node_id in self.sg_sim.filtered_netx_graph.successors(room_id) if 'region' in node_id]
            for region_id in region_ids:
                object_ids = [node_id for node_id in self.sg_sim.filtered_netx_graph.successors(region_id) if 'object' in node_id]
                objects_in_room.extend(object_ids)
        else:
            object_ids = [node_id for node_id in self.sg_sim.filtered_netx_graph.successors(room_id) if 'object' in node_id]
            objects_in_room.extend(object_ids)
        return objects_in_room

    def get_room_name(self, room_id: str) -> Optional[str]:
        if not room_id: return None
        try:
            if hasattr(self.sg_sim, 'room_node_ids') and hasattr(self.sg_sim, 'room_node_names'):
                room_ids = list(self.sg_sim.room_node_ids)
                room_names = list(self.sg_sim.room_node_names)
                if room_id in room_ids:
                    return room_names[room_ids.index(room_id)]
        except: pass
        try:
            if hasattr(self.sg_sim, 'filtered_netx_graph') and room_id in self.sg_sim.filtered_netx_graph.nodes:
                node_data = self.sg_sim.filtered_netx_graph.nodes[room_id]
                if 'name' in node_data and node_data['name'] != 'room':
                    return node_data['name']
        except: pass
        return None

    def get_object_name(self, obj_id: str) -> Optional[str]:
        if not obj_id: return None
        try:
            if hasattr(self.sg_sim, 'object_node_ids') and hasattr(self.sg_sim, 'object_node_names'):
                object_ids = list(self.sg_sim.object_node_ids)
                object_names = list(self.sg_sim.object_node_names)
                if obj_id in object_ids:
                    return object_names[object_ids.index(obj_id)]
        except: pass
        return obj_id 

    def find_object_id_by_name(self, object_name: str) -> Optional[str]:
        if not hasattr(self.sg_sim, 'object_node_names') or not hasattr(self.sg_sim, 'object_node_ids'): return None
        for obj_id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            if object_name.lower() == name.lower(): return obj_id
        for obj_id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            if object_name.lower() in name.lower() or name.lower() in object_name.lower(): return obj_id
        return None

    def print_object_room_info(self, obj_id: str):
        info = self.get_object_room_info(obj_id)
        object_name = info.get('object_name', obj_id) 
        if info["room_id"]:
            print(f"物体 {object_name} ({obj_id}) 位于房间: {info['room_name']} ({info['room_id']})")
        else:
            print(f"物体 {object_name} ({obj_id}) 未找到所属房间")

    def print_all_objects_room_info(self):
        print("\n=== 所有物体的房间信息 ===")
        if not hasattr(self.sg_sim, 'object_node_ids'):
            print("警告: 场景图没有object_node_ids属性")
            return
        for obj_id in self.sg_sim.object_node_ids:
            try: self.print_object_room_info(obj_id)
            except Exception as e: print(f"处理物体 {obj_id} 时出错: {e}")
        print("========================\n")

    def batch_get_semantic_similarity(self, target, object_names, exp_context="") -> List[Tuple[str, float, bool, str]]:
        names_str = ', '.join([f"'{name}'" for name in object_names])
        prompt = f"{exp_context}\nCompute the semantic similarity between '{target}' and each of the following objects: {names_str}. Respond ONLY with JSON list."
        messages = [{"role": "system", "content": "You are a semantic analyzer."}, {"role": "user", "content": prompt}]
        try:
            completion = client.beta.chat.completions.parse(
                model=self._vlm_type, messages=messages, response_format=BatchSemanticSimilarityResponse, temperature=0.1
            )
            return [(res.object, res.similarity_score, res.has_spatial_relation, res.explanation) for res in completion.choices[0].message.parsed.responses]
        except Exception as e:
            print(f"Error in batch semantic similarity: {e}")
            return [(name, 0.0, False, "Error") for name in object_names]

    def get_semantic_similarity(self, target, object_name, exp_context="") -> Tuple[float, bool, str]:
        key = (target.lower(), object_name.lower(), hash(exp_context))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        
        prompt = f"{exp_context}\nCompute the semantic similarity between '{target}' and '{object_name}' on a scale of 0 to 1. Respond ONLY with JSON."
        messages = [{"role": "system", "content": "You are a semantic analyzer."}, {"role": "user", "content": prompt}]
        try:
            completion = client.beta.chat.completions.parse(
                model=self._vlm_type, messages=messages, response_format=SemanticSimilarityResponse, temperature=0.1
            )
            response = completion.choices[0].message.parsed
            result = (response.similarity_score, response.has_spatial_relation, response.explanation)
            self.similarity_cache[key] = result
            return result
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0, False, "Error"

    def get_current_room(self, agent_state):
        match = re.search(r'at room node: (room_\d+) with name (.*)', agent_state)
        if match: return match.group(1), match.group(2).strip()
        return None, None

    def parse_target_room(self, subtask_target):
        prompt = f"Extract target room name from instruction and subtask target.\nInstruction: {self._instruction}\nSubtask target: {subtask_target}"
        messages = [{"role": "system", "content": "You are a precise location extractor."}, {"role": "user", "content": prompt}]
        try:
            completion = client.beta.chat.completions.parse(model=self._vlm_type, messages=messages, response_format=TargetRoomResponse, temperature=0.1)
            return completion.choices[0].message.parsed.room
        except: return None

    def get_room_frontiers(self, room_id):
        graph_data = json.loads(self.sg_sim.scene_graph_str)
        room_frontiers = []
        for link in graph_data.get('links', []):
            source, target = link.get('source'), link.get('target')
            if source == room_id and 'frontier' in target: room_frontiers.append(target)
            elif target == room_id and 'frontier' in source: room_frontiers.append(source)
        return room_frontiers

    def get_closest_frontier(self, current_pos, frontiers):
        if not frontiers: return None
        valid, distances = [], []
        for f in frontiers:
            try:
                pos = self.sg_sim.get_position_from_id(f)
                valid.append(f)
                distances.append(np.linalg.norm(current_pos - pos))
            except: pass
        return valid[np.argmin(distances)] if distances else None

    def get_farthest_frontier(self, current_pos, frontiers):
        if not frontiers: return None
        valid, distances = [], []
        for f in frontiers:
            try:
                pos = self.sg_sim.get_position_from_id(f)
                valid.append(f)
                distances.append(np.linalg.norm(current_pos - pos))
            except: pass
        return valid[np.argmax(distances)] if distances else None

    def get_current_pos(self, agent_state):
        pos_str = re.search(r'position \[(.*?)\]', agent_state).group(1)
        coords = re.findall(r'np\.float64\((.*?)\)', pos_str)
        return np.array([float(c) for c in coords])

    def get_frontier_info(self, frontier_id: str) -> Tuple[float, str]:
        degree, room_name = 10.0, "unknown region"
        if not hasattr(self.sg_sim, 'filtered_netx_graph'): return degree, room_name
        graph = self.sg_sim.filtered_netx_graph
        if frontier_id not in graph: return degree, room_name
        try:
            regions = [p for p in graph.predecessors(frontier_id) if 'region' in p]
            if not regions: regions = [s for s in graph.successors(frontier_id) if 'region' in s]
            if regions:
                region_id = regions[0]
                degree = graph.nodes[region_id].get('exploration_degree', 10.0)
                rooms = [p for p in graph.predecessors(region_id) if 'room' in p]
                if rooms:
                    name = graph.nodes[rooms[0]].get('name', 'room')
                    room_name = name if name != 'room' else rooms[0]
        except Exception as e: print(f"Error getting frontier info: {e}")
        return degree, room_name

    # ─── 🚀 认知规划核心主函数 (全量对齐消融实验分支) ───
    def get_next_action(self, current_subtask_index=None, imgs_rgb=None, imgs_depth=None, intrinsics=None, extrinsics=None, frontier_nodes=None):
        print(f"\n=== Updating Scene Graph at t={self.t} ===")
        if imgs_rgb is None: imgs_rgb = []
        if frontier_nodes is None: frontier_nodes = []
        try:
            self.sg_sim.update(imgs_rgb=imgs_rgb, imgs_depth=imgs_depth, intrinsics=intrinsics, extrinsics=extrinsics, frontier_nodes=frontier_nodes)
        except Exception as e: print(f"Error updating scene graph: {e}")

        self.print_all_objects_room_info()
        current_subtask = self.task_manager.get_current_subtask() if self.task_manager else self.subtasks[current_subtask_index]
        if not current_subtask: return None, None, True, 1.0, "no_action"

        subtask_type = current_subtask['type']
        target = current_subtask.get('target', 'unknown')
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_pos = self.get_current_pos(agent_state)
        current_room_id, current_room_name = self.get_current_room(agent_state)
        target_room = self.parse_target_room(target)

        target_pose, target_id, action_output, is_confident, confidence_level = None, None, "no_action", False, 0.0

        if subtask_type in ['grab', 'release']:
            is_confident, confidence_level, action_output = True, 1.0, subtask_type
            for obj_id, obj_name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
                if target.lower() in obj_name.lower():
                    target_id = obj_id
                    break
            if target_id and not self.disable_memory:  # 🌟 长时记忆写回控制
                self.experience_memory.add_experience(
                    query=f"操作 {target}", thought=f"成功在 {current_room_name} 找到并操作了 {target}。", outcome="Success", graph_anchor=target_id
                )
        else: # move_to 导航原语
            # ─────────────────────────────────────────────────────────────────
            # 开关 A: w/o Memory 经验检索消融阻断
            # ─────────────────────────────────────────────────────────────────
            exp_context = ""
            if self.disable_memory:
                print("🧠 [消融激活] w/o Memory 分支: 彻底阻断历史长时经验库的检索注入。")
            else:
                def safe_get_anchor_pos(anchor_id):
                    try: return self.sg_sim.get_position_from_id(anchor_id)
                    except: return None
                experiences = self.experience_memory.retrieve_experience(
                    current_query=f"寻找{target}", current_pos=current_pos, get_anchor_pos_func=safe_get_anchor_pos, top_k=2
                )
                if experiences:
                    stable_exps = sorted(experiences, key=lambda x: x['id'])
                    exp_context = "Historical Experience for context:\n" + "\n".join([f"- {exp['thought']}" for exp in stable_exps]) + "\n"

            # ─────────────────────────────────────────────────────────────────
            # 开关 B: w/o Scene Graph 拓扑图谱消融阻断 (直接使 Branch A 物体评估流失效)
            # ─────────────────────────────────────────────────────────────────
            candidates = []
            global_best_object_id, global_max_score = None, -float('inf')

            if self.disable_scene_graph:
                print("🌐 [消融激活] w/o Scene Graph 分支: 强行脱敏已知拓扑实体，直接跳过 Branch A 物体探测评估。")
                has_confident_target = False
            else:
                # Full Framework / 其他消融环境下完备的已知图谱匹配打分
                uncached = [name for name in self.sg_sim.object_node_names if (target.lower(), name.lower(), hash(exp_context)) not in self.similarity_cache]
                if uncached:
                    batch_results = self.batch_get_semantic_similarity(target, uncached, exp_context)
                    for name, sim, has_relation, exp in batch_results:
                        self.similarity_cache[(target.lower(), name.lower(), hash(exp_context))] = (sim, has_relation, exp)

                for obj_id, obj_name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
                    sim, has_relation, exp = self.get_semantic_similarity(target, obj_name, exp_context)
                    room_id = self.get_object_room(obj_id)
                    room_name = self.get_room_name(room_id)
                    
                    room_sim = 0.0
                    if target_room and room_name and room_name not in ["unknown region", "room", "room_0"]:
                        r_sim, _, _ = self.get_semantic_similarity(target_room, room_name, exp_context)
                        room_sim = r_sim
                    elif not target_room: room_sim = 0.5 
                    
                    score = 0.7 * sim + 0.3 * room_sim
                    if has_relation: score = min(1.0, score + 0.2)

                    if score > global_max_score:
                        global_max_score, global_best_object_id = score, obj_id
                    if sim > self.similarity_threshold:
                        candidates.append((score, sim, room_sim, obj_id))

                has_confident_target = False
                if candidates:
                    candidates.sort(key=lambda x: (-x[1], -x[2]))
                    best_score, best_sim, best_room_match, best_object_id = candidates[0]
                    if best_score >= 0.6:
                        confidence_level, target_id, action_output, is_confident, has_confident_target = best_score, best_object_id, 'move_to', True, True
                        target_pose = self.sg_sim.get_position_from_id(best_object_id)

            # ─────────────────────────────────────────────────────────────────
            # 未发现确凿物体或处于三维场景图消融状态下，强制降级发起前沿探索边界挖掘
            # ─────────────────────────────────────────────────────────────────
            if not has_confident_target:
                if not hasattr(self, 'final_check_done'): self.final_check_done = False
                if not hasattr(self, 'last_target'): self.last_target = target
                if self.last_target != target:
                    if hasattr(self, 'visited_frontiers'): self.visited_frontiers.clear()
                    if hasattr(self, 'visited_frontiers_coords'): self.visited_frontiers_coords.clear()
                    self.final_check_done, self.last_target = False, target

                if not hasattr(self, 'visited_frontiers_coords'): self.visited_frontiers_coords = []
                all_frontiers = self.sg_sim.frontier_node_ids
                
                unvisited_frontiers = []
                for f_id in all_frontiers:
                    try:
                        f_pos = self.sg_sim.get_position_from_id(f_id)
                        if not any(np.linalg.norm(f_pos - v_pos) < 1.0 for v_pos in self.visited_frontiers_coords):
                            unvisited_frontiers.append(f_id)
                    except: pass
                
                if not unvisited_frontiers and all_frontiers:
                    if not self.final_check_done:
                        self.visited_frontiers_coords.clear()
                        unvisited_frontiers, self.final_check_done = all_frontiers, True
                    else: unvisited_frontiers = []

                if not unvisited_frontiers:  # 边界耗尽盲赌逻辑
                    if global_best_object_id and global_max_score > 0.1:
                        confidence_level = global_max_score
                        target_pose = self.sg_sim.get_position_from_id(global_best_object_id)
                        target_id, action_output, is_confident = global_best_object_id, 'move_to', True
                        if not self.disable_memory:
                            self.experience_memory.add_experience(
                                query=f"寻找 {target}", thought=f"全屋搜查未找到完美匹配，最终选择前往最相似物体 {global_best_object_id} 进行盲赌。", outcome="Gamble_Fallback", graph_anchor=global_best_object_id
                            )
                    else:
                        action_output, confidence_level, is_confident, target_pose, target_id = 'give_up', 0.0, False, None, None
                        if not self.disable_memory:
                            self.experience_memory.add_experience(
                                query=f"寻找 {target}", thought=f"在全屋彻底搜查后未找到 {target}。宣告放弃。", outcome="Failure", graph_anchor=current_room_id if current_room_id else "unknown"
                            )
                else:
                    # ─────────────────────────────────────────────────────────
                    # 开关 C: w/o Frontier Score 消融判断 / 开关 B 的连锁语义封锁
                    # ─────────────────────────────────────────────────────────
                    if self.disable_frontier_score or self.disable_scene_graph:
                        if self.disable_scene_graph:
                            print("🌐 [消融连锁] Branch B 触发 w/o Scene Graph 语义封锁: 强行将环境语义因子归零，退化为纯几何近邻探测。")
                        else:
                            print("🎯 [消融激活] w/o Frontier Score 分支: 彻底抽离 4D 融合打分，退化为纯几何近邻探测。")
                        
                        best_frontier, min_distance = None, float('inf')
                        for f_id in unvisited_frontiers:
                            try:
                                f_pos = self.sg_sim.get_position_from_id(f_id)
                                distance = np.linalg.norm(current_pos - f_pos)
                                if distance < min_distance:
                                    min_distance, best_frontier = distance, f_id
                            except: pass
                        if best_frontier:
                            target_id, action_output, confidence_level, is_confident = best_frontier, 'explore_frontier', 0.3, False
                            target_pose = self.sg_sim.get_position_from_id(best_frontier)
                            self.visited_frontiers_coords.append(target_pose)
                    else:
                        # Full Framework 完备四维信息增益与时序上下文打分模型
                        alpha, beta, gamma_room, gamma_obj, delta = 3.0, 2.0, 20.0, 35.0, 1.5
                        best_frontier, best_score = None, -float('inf')
                        
                        for f_id in unvisited_frontiers:
                            try:
                                f_pos = self.sg_sim.get_position_from_id(f_id)
                                distance = np.linalg.norm(current_pos - f_pos)
                                ig_score = self.sg_sim.get_frontier_ig(f_id)
                                room_id, room_name = self.sg_sim.get_room_for_frontier(f_id)
                                raw_room_val = self.sg_sim.get_room_unexplored_ratio(room_id)
                                room_unexplored_val = min(10.0, raw_room_val / 50.0)
                                
                                sim_room = 0.0
                                if room_name and room_name not in ["unknown region", "room", "room_0"]:
                                    r_sim, _, _ = self.get_semantic_similarity(target, room_name, exp_context)
                                    sim_room = r_sim
                                    
                                sim_obj_max = 0.0
                                nearby_objects = self.sg_sim.get_objects_near_frontier(f_id, radius=2.0)
                                for obj_name in nearby_objects:
                                    obj_sim, _, _ = self.get_semantic_similarity(target, obj_name, exp_context)
                                    if obj_sim > sim_obj_max: sim_obj_max = obj_sim
                                    
                                score = (alpha * ig_score) + (beta * room_unexplored_val) + (gamma_room * sim_room) + (gamma_obj * sim_obj_max) - (delta * distance)
                                if score > best_score:
                                    best_score, best_frontier = score, f_id
                            except: pass
                            
                        if best_frontier:
                            target_id, action_output, confidence_level, is_confident = best_frontier, 'explore_frontier', 0.4, False
                            target_pose = self.sg_sim.get_position_from_id(best_frontier)
                            self.visited_frontiers_coords.append(target_pose)

        self._t += 1
        # 🌟 [消融标志位随流向透传回运动控制层]
        return target_pose, target_id, is_confident, confidence_level, action_output, self.disable_safety_filter

    def get_current_state_prompt_for_subtask(self, scene_graph, agent_state, current_subtask):
        prompt = f"At t = {self.t}: \n CURRENT AGENT STATE: {agent_state}. \n SCENE GRAPH: {scene_graph}. \n CURRENT SUBTASK: {current_subtask}. \n "
        if self._add_history: prompt += f"HISTORY: {self._history}"
        return prompt

    def get_scene_graph_summary(self):
        rooms = self.sg_sim.room_node_names[:5] if hasattr(self.sg_sim, 'room_node_names') else []
        objects = self.sg_sim.object_node_names[:10] if hasattr(self.sg_sim, 'object_node_names') else []
        frontiers = [f for f in (self.sg_sim.frontier_node_ids[:5] if hasattr(self.sg_sim, 'frontier_node_ids') else []) if f not in self.visited_frontiers]
        agent_state = self.sg_sim.get_current_semantic_state_str()
        return json.dumps({'rooms': rooms, 'visible_objects': objects, 'frontiers': frontiers, 'agent_state': agent_state}, indent=2)

    def find_target_position(self, target_name):
        if not hasattr(self.sg_sim, 'object_node_ids') or not hasattr(self.sg_sim, 'object_node_names'): return None
        for obj_id, obj_name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            if target_name.lower() in obj_name.lower():
                try: return self.sg_sim.get_position_from_id(obj_id)
                except: continue
        return None

    def is_task_completed(self):
        return self.task_manager.is_task_completed() if self.task_manager else False