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

# 🌟 [新增] 导入独立的经验记忆模块
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
        
        # 🌟 [新增] 初始化大脑皮层：向量化经验记忆模块
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
        """
        获取物体所属的房间节点ID
        """
        # 检查缓存
        if obj_id in self.object_room_cache:
            return self.object_room_cache[obj_id]
        
        if not hasattr(self.sg_sim, 'filtered_netx_graph'):
            return None
        
        if obj_id not in self.sg_sim.filtered_netx_graph:
            return None
        
        # 方法1: 查找对象节点的直接前驱节点（房间）
        try:
            room_predecessors = [pred_id for pred_id in self.sg_sim.filtered_netx_graph.predecessors(obj_id) 
                            if 'room' in pred_id]
            if room_predecessors:
                self.object_room_cache[obj_id] = room_predecessors[0]
                return room_predecessors[0]
            
            # 方法2: 通过区域节点查找（当包含区域节点时）
            region_predecessors = [pred_id for pred_id in self.sg_sim.filtered_netx_graph.predecessors(obj_id) 
                                if 'region' in pred_id]
            for region_id in region_predecessors:
                # 查找区域的房间前驱
                region_room_predecessors = [pred_id for pred_id in self.sg_sim.filtered_netx_graph.predecessors(region_id) 
                                        if 'room' in pred_id]
                if region_room_predecessors:
                    self.object_room_cache[obj_id] = region_room_predecessors[0]
                    return region_room_predecessors[0]
            
            # 方法3: 从边关系中查找（适用于区域节点被移除的情况）
            for room_id in self.sg_sim.room_node_ids:
                if self.sg_sim.filtered_netx_graph.has_edge(room_id, obj_id):
                    self.object_room_cache[obj_id] = room_id
                    return room_id
        except Exception as e:
            print(f"获取物体 {obj_id} 的房间时出错: {e}")
            return None
        
        return None

    def get_object_room_info(self, obj_id: str) -> Dict[str, str]:
        """
        获取物体的完整房间信息
        """
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
        """
        获取指定房间中的所有物体节点ID
        """
        if not hasattr(self.sg_sim, 'filtered_netx_graph'):
            return []
        
        objects_in_room = []
        if self.sg_sim.include_regions:
            region_ids = [node_id for node_id in self.sg_sim.filtered_netx_graph.successors(room_id) 
                         if 'region' in node_id]
            for region_id in region_ids:
                object_ids = [node_id for node_id in self.sg_sim.filtered_netx_graph.successors(region_id) 
                            if 'object' in node_id]
                objects_in_room.extend(object_ids)
        else:
            object_ids = [node_id for node_id in self.sg_sim.filtered_netx_graph.successors(room_id) 
                         if 'object' in node_id]
            objects_in_room.extend(object_ids)
        return objects_in_room

    def get_room_name(self, room_id: str) -> Optional[str]:
        """获取房间名称"""
        if not room_id:
            return None
        try:
            if hasattr(self.sg_sim, 'room_node_ids') and hasattr(self.sg_sim, 'room_node_names'):
                room_ids = list(self.sg_sim.room_node_ids)
                room_names = list(self.sg_sim.room_node_names)
                if room_id in room_ids:
                    index = room_ids.index(room_id)
                    return room_names[index]
        except (ValueError, AttributeError) as e:
            pass
        try:
            if hasattr(self.sg_sim, 'filtered_netx_graph'):
                if room_id in self.sg_sim.filtered_netx_graph.nodes:
                    node_data = self.sg_sim.filtered_netx_graph.nodes[room_id]
                    if 'name' in node_data and node_data['name'] != 'room':
                        return node_data['name']
        except Exception as e:
            pass
        return None

    def get_object_name(self, obj_id: str) -> Optional[str]:
        """获取物体名称"""
        if not obj_id:
            return None
        try:
            if hasattr(self.sg_sim, 'object_node_ids') and hasattr(self.sg_sim, 'object_node_names'):
                object_ids = list(self.sg_sim.object_node_ids)
                object_names = list(self.sg_sim.object_node_names)
                if obj_id in object_ids:
                    index = object_ids.index(obj_id)
                    return object_names[index]
        except (ValueError, AttributeError) as e:
            pass
        return obj_id 

    def find_object_id_by_name(self, object_name: str) -> Optional[str]:
        """根据物体名称查找对应的物体节点ID"""
        if not hasattr(self.sg_sim, 'object_node_names') or not hasattr(self.sg_sim, 'object_node_ids'):
            return None
        for obj_id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            if object_name.lower() == name.lower():
                return obj_id
        for obj_id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            if object_name.lower() in name.lower() or name.lower() in object_name.lower():
                return obj_id
        return None

    def print_object_room_info(self, obj_id: str):
        """打印物体的房间信息"""
        info = self.get_object_room_info(obj_id)
        object_name = info.get('object_name', obj_id) 
        if info["room_id"]:
            room_name = info.get('room_name', '未知房间')
            print(f"物体 {object_name} ({obj_id}) 位于房间: {room_name} ({info['room_id']})")
        else:
            print(f"物体 {object_name} ({obj_id}) 未找到所属房间")

    def print_all_objects_room_info(self):
        """打印所有物体的房间信息"""
        print("\n=== 所有物体的房间信息 ===")
        if not hasattr(self.sg_sim, 'object_node_ids'):
            print("警告: 场景图没有object_node_ids属性")
            return
        for obj_id in self.sg_sim.object_node_ids:
            try:
                self.print_object_room_info(obj_id)
            except Exception as e:
                print(f"处理物体 {obj_id} 时出错: {e}")
        print("========================\n")

    # 🌟 [修改] 增加 exp_context 参数，接收记忆上下文
    def batch_get_semantic_similarity(self, target, object_names, exp_context="") -> List[Tuple[str, float, bool, str]]:
        """Batch compute semantic similarity for multiple objects"""
        names_str = ', '.join([f"'{name}'" for name in object_names])
        prompt = f"""
        {exp_context}
        Compute the semantic similarity between '{target}' and each of the following objects: {names_str}.
        For each, score on a scale of 0 to 1 (1.0 if identical, high if related), judge if there is spatial/containment relation (yes/no), and provide explanation.
        Respond ONLY with JSON list: [{{"object": "name1", "similarity_score": float, "has_spatial_relation": bool, "explanation": "string"}}, ...]
        """

        messages = [{"role": "system", "content": "You are a semantic analyzer. Utilize historical experience if provided to adjust your scoring."},
                    {"role": "user", "content": prompt}]

        try:
            completion = client.beta.chat.completions.parse(
                model=self._vlm_type,
                messages=messages,
                response_format=BatchSemanticSimilarityResponse,
                temperature=0.1
            )
            responses = completion.choices[0].message.parsed.responses
            results = []
            for res in responses:
                results.append((res.object, res.similarity_score, res.has_spatial_relation, res.explanation))
            return results
        except Exception as e:
            print(f"Error in batch semantic similarity: {e}")
            return [(name, 0.0, False, "Error") for name in object_names]

    # 🌟 [修改] 增加 exp_context 参数，并将 context 加入缓存的 Key
    def get_semantic_similarity(self, target, object_name, exp_context="") -> Tuple[float, bool, str]:
        """使用GPT计算语义相似度，使用缓存避免重复查询"""
        # 缓存键值加上 exp_context，防止不同的经验覆盖了同一个 target-object 打分
        key = (target.lower(), object_name.lower(), hash(exp_context))
        if key in self.similarity_cache:
            room_info = "未知房间"
            try:
                object_id = self.find_object_id_by_name(object_name)
                if object_id:
                    info = self.get_object_room_info(object_id)
                    room_info = info["room_name"] if info["room_name"] else "未知房间"
                else:
                    room_info = "未找到对象节点"
            except Exception as e:
                room_info = f"获取房间信息失败: {e}"
            
            similarity_score = self.similarity_cache[key][0]
            has_spatial_relation = self.similarity_cache[key][1]
            explanation = self.similarity_cache[key][2]
            
            print(f"Using cached similarity for '{target}' and '{object_name}' - 所属房间: {room_info}, 相似度: {similarity_score:.3f}, 空间关系: {has_spatial_relation}")
            return similarity_score, has_spatial_relation, explanation
        
        prompt = f"""
        {exp_context}
        Compute the semantic similarity between '{target}' and '{object_name}' on a scale of 0 to 1.
        - 1.0 if identical.
        - High if related (e.g., flower and vase = 0.7 if flower is in vase).
        - Also judge if there is spatial/containment relation (yes/no).
        Respond ONLY with JSON: {{"similarity_score": float, "has_spatial_relation": bool, "explanation": "string"}}
        """

        messages = [{"role": "system", "content": "You are a semantic analyzer. Utilize historical experience if provided to adjust your scoring."},
                    {"role": "user", "content": prompt}]

        try:
            completion = client.beta.chat.completions.parse(
                model=self._vlm_type,
                messages=messages,
                response_format=SemanticSimilarityResponse,
                temperature=0.1
            )
            response = completion.choices[0].message.parsed
            result = (response.similarity_score, response.has_spatial_relation, response.explanation)
            self.similarity_cache[key] = result
            
            room_info = "未知房间"
            try:
                object_id = self.find_object_id_by_name(object_name)
                if object_id:
                    info = self.get_object_room_info(object_id)
                    room_info = info["room_name"] if info["room_name"] else "未知房间"
                else:
                    room_info = "未找到对象节点"
            except Exception as e:
                room_info = f"获取房间信息失败: {e}"
            
            print(f"New similarity for '{target}' and '{object_name}' - 所属房间: {room_info}, 相似度: {response.similarity_score:.3f}, 空间关系: {response.has_spatial_relation}")
            return result
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0, False, "Error"

    def get_current_room(self, agent_state):
        """从agent_state提取当前房间"""
        match = re.search(r'at room node: (room_\d+) with name (.*)', agent_state)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def parse_target_room(self, subtask_target):
        """使用GPT从vlm_instruction和subtask target中解析当前子任务的目标房间"""
        prompt = f"""
        Extract the target room name from the following instruction and subtask target.
        Instruction: {self._instruction}
        Subtask target: {subtask_target}
        
        Focus on the last mentioned location that appears to be a room (e.g., 'bedroom', 'kitchen').
        If multiple, choose the destination room (e.g., in 'from bedroom to kitchen', choose 'kitchen').
        If no room is mentioned, return null.
        
        Respond ONLY with JSON: {{"room": "room_name" or null}}
        """

        messages = [
            {"role": "system", "content": "You are a precise extractor of location information from text."},
            {"role": "user", "content": prompt}
        ]

        try:
            completion = client.beta.chat.completions.parse(
                model=self._vlm_type,
                messages=messages,
                response_format=TargetRoomResponse,
                temperature=0.1
            )
            response = completion.choices[0].message.parsed
            return response.room
        except Exception as e:
            print(f"Error parsing target room with GPT: {e}")
            return None

    def get_room_frontiers(self, room_id):
        """获取房间内的frontier，通过解析 scene_graph_str 的 links"""
        graph_data = json.loads(self.sg_sim.scene_graph_str)
        room_frontiers = []
        for link in graph_data.get('links', []):
            source = link.get('source')
            target = link.get('target')
            if source == room_id and 'frontier' in target:
                room_frontiers.append(target)
            elif target == room_id and 'frontier' in source:
                room_frontiers.append(source)
        return room_frontiers

    def get_closest_frontier(self, current_pos, frontiers):
        """选择最近的frontier"""
        if not frontiers:
            return None
        valid = []
        distances = []
        for f in frontiers:
            try:
                pos = self.sg_sim.get_position_from_id(f)
                d = np.linalg.norm(current_pos - pos)
                valid.append(f)
                distances.append(d)
            except Exception:
                pass
        if not distances:
            return None
        closest_idx = np.argmin(distances)
        return valid[closest_idx]

    def get_farthest_frontier(self, current_pos, frontiers):
        """选择最远的frontier"""
        if not frontiers:
            return None
        valid = []
        distances = []
        for f in frontiers:
            try:
                pos = self.sg_sim.get_position_from_id(f)
                d = np.linalg.norm(current_pos - pos)
                valid.append(f)
                distances.append(d)
            except Exception:
                pass
        if not distances:
            return None
        farthest_idx = np.argmin(distances)
        return valid[farthest_idx]

    def get_current_pos(self, agent_state):
        """从agent_state提取当前位置，格式 [np.float64(x), np.float64(y), np.float64(z)] """
        pos_str = re.search(r'position \[(.*?)\]', agent_state).group(1)
        coords = re.findall(r'np\.float64\((.*?)\)', pos_str)
        return np.array([float(c) for c in coords])

    # =========================================================================
    # [新增] 辅助函数：提取前沿点 (Frontier) 的探索度及所属房间名称
    # =========================================================================
    def get_frontier_info(self, frontier_id: str) -> Tuple[float, str]:
        """
        获取前沿点对应的探索度和连接的房间名称
        
        Args:
            frontier_id: 前沿节点ID
        Returns:
            (exploration_degree, room_name)
        """
        degree = 10.0  # 默认基础探索度
        room_name = "unknown region"
        
        if not hasattr(self.sg_sim, 'filtered_netx_graph'):
            return degree, room_name
            
        graph = self.sg_sim.filtered_netx_graph
        if frontier_id not in graph:
            return degree, room_name
            
        try:
            # 1. 查找前沿点连接的区域 (Region)
            regions = [p for p in graph.predecessors(frontier_id) if 'region' in p]
            if not regions:
                regions = [s for s in graph.successors(frontier_id) if 'region' in s]
                
            if regions:
                region_id = regions[0]
                degree = graph.nodes[region_id].get('exploration_degree', 10.0)
                
                # 2. 查找区域连接的房间 (Room) -> region
                rooms = [p for p in graph.predecessors(region_id) if 'room' in p]
                if rooms:
                    room_id = rooms[0]
                    # 获取房间名，如果还是默认的 'room'，就返回 ID
                    name = graph.nodes[room_id].get('name', 'room')
                    room_name = name if name != 'room' else room_id
                    
        except Exception as e:
            print(f"提取前沿点 {frontier_id} 信息时出错: {e}")
            
        return degree, room_name
    # =========================================================================

    def get_next_action(self, current_subtask_index=None, imgs_rgb=None, imgs_depth=None, intrinsics=None, extrinsics=None, frontier_nodes=None):
        # 实时更新场景图 - 调用 sg_sim.update()
        print(f"\n=== Updating Scene Graph at t={self.t} ===")
        if imgs_rgb is None:
            imgs_rgb = []
            print("Warning: No RGB images provided for scene graph update. Using empty list.")
        if frontier_nodes is None:
            frontier_nodes = []
            print("Warning: No frontier nodes provided for scene graph update. Using empty list.")
        
        try:
            self.sg_sim.update(
                imgs_rgb=imgs_rgb,
                imgs_depth=imgs_depth,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                frontier_nodes=frontier_nodes
            )
            print("Scene Graph updated successfully.")
        except Exception as e:
            print(f"Error updating scene graph: {e}")

        # 打印所有物体的房间信息（调试用）
        self.print_all_objects_room_info()

        # Debugging : Print frontier nodes
        print(f"\n=== Frontier Information at t={self.t} ===")
        if frontier_nodes is not None and len(frontier_nodes) > 0:
            print(f"Raw frontier_nodes shape: {np.array(frontier_nodes).shape}")
            print(f"Raw frontier_nodes (first 5): {frontier_nodes[:5]}")
        else:
            print("No raw frontier nodes provided")

        print(f"Scene Graph Frontier Node IDs: {self.sg_sim.frontier_node_ids}")
        if hasattr(self.sg_sim, 'frontier_node_ids') and len(self.sg_sim.frontier_node_ids) > 0:
            print("Frontier Node Coordinates:")
            for frontier_id in self.sg_sim.frontier_node_ids:
                try:
                    pos = self.sg_sim.get_position_from_id(frontier_id)
                    print(f"  {frontier_id}: {pos}")
                except Exception as e:
                    print(f"  {frontier_id}: Could not get position - {e}")
        else:
            print("No frontier nodes in scene graph")
        print("========================\n")
        
        # Debugging: Print current scene graph summary
        print(f"Current Scene Graph Summary: {self.get_scene_graph_summary()}")
        print(f"Room Nodes: {self.sg_sim.room_node_ids} - Names: {self.sg_sim.room_node_names}")
        print(f"Object Nodes: {self.sg_sim.object_node_ids} - Names: {self.sg_sim.object_node_names}")
        print(f"Frontier Nodes: {self.sg_sim.frontier_node_ids}")
        print(f"Full Scene Graph JSON: {self.sg_sim.scene_graph_str[:500]}... (truncated)")  # Truncate for brevity
        print("========================\n")

        # 获取当前 subtask
        if self.task_manager:
            current_subtask = self.task_manager.get_current_subtask()
        else:
            current_subtask = self.subtasks[current_subtask_index] if current_subtask_index is not None else None

        if not current_subtask:
            return None, None, True, 1.0, "no_action"

        subtask_type = current_subtask['type']
        target = current_subtask.get('target', 'unknown')

        # 获取当前位置和房间
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_pos = self.get_current_pos(agent_state)
        current_room_id, current_room_name = self.get_current_room(agent_state)

        # 解析目标房间
        target_room = self.parse_target_room(target)

        # 决策逻辑
        target_pose = None
        target_id = None
        action_output = "no_action"
        is_confident = False
        confidence_level = 0.0

        if subtask_type in ['grab', 'release']:
            is_confident = True
            confidence_level = 1.0
            action_output = subtask_type
            best_object_id = None
            for obj_id, obj_name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
                if target.lower() in obj_name.lower():
                    best_object_id = obj_id
                    break
            if best_object_id:
                # target_pose = self.sg_sim.get_position_from_id(best_object_id)
                target_pose = None
                target_id = best_object_id
                
                # 🌟 [记忆闭环：写入成功经验]
                self.experience_memory.add_experience(
                    query=f"操作 {target}",
                    thought=f"成功在 {current_room_name} 找到并操作了 {target}。",
                    outcome="Success",
                    graph_anchor=best_object_id
                )
                
        else:  # move_to
            # =================================================================
            # 🌟 [Level 2: 兜底检索 (Memory Retrieval)]
            # 在寻找候选目标前，先获取过往经验，准备将其注入给 Planner
            # =================================================================
            def safe_get_anchor_pos(anchor_id):
                try: return self.sg_sim.get_position_from_id(anchor_id)
                except: return None

            experiences = self.experience_memory.retrieve_experience(
                current_query=f"寻找{target}",
                current_pos=current_pos,
                get_anchor_pos_func=safe_get_anchor_pos,
                top_k=2
            )
            
            exp_context = ""
            if experiences:
                # 排序确保拼接的 Prompt 字符串稳定，最大化利用缓存
                stable_exps = sorted(experiences, key=lambda x: x['id'])
                exp_context = "Historical Experience for context:\n" + "\n".join([f"- {exp['thought']}" for exp in stable_exps]) + "\n"

            uncached = []
            for obj_name in self.sg_sim.object_node_names:
                key = (target.lower(), obj_name.lower(), hash(exp_context))
                if key not in self.similarity_cache:
                    uncached.append(obj_name)

            if uncached:
                # 🌟 传入 exp_context 引导批量评估
                batch_results = self.batch_get_semantic_similarity(target, uncached, exp_context)
                for name, sim, has_relation, exp in batch_results:
                    key = (target.lower(), name.lower(), hash(exp_context))
                    self.similarity_cache[key] = (sim, has_relation, exp)

            candidates = []

            global_best_object_id = None
            global_max_score = -float('inf')
            
            for obj_id, obj_name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
                # 🌟 传入 exp_context 引导独立评估
                sim, has_relation, exp = self.get_semantic_similarity(target, obj_name, exp_context)
                room_id = self.get_object_room(obj_id)
                room_name = self.get_room_name(room_id)
                
                # =========================================================
                # [继承上一轮改进] 使用 VLM 进行房间软语义匹配
                # =========================================================
                room_sim = 0.0
                if target_room and room_name and room_name not in ["unknown region", "room", "room_0"]:
                    # 🌟 传入 exp_context 引导房间评估
                    r_sim, _, _ = self.get_semantic_similarity(target_room, room_name, exp_context)
                    room_sim = r_sim
                elif not target_room:
                    room_sim = 0.5 
                
                score = 0.7 * sim + 0.3 * room_sim
                
                if has_relation:
                    score = min(1.0, score + 0.2)

                if score > global_max_score:
                    global_max_score = score
                    global_best_object_id = obj_id

                if sim > self.similarity_threshold:
                    candidates.append((score, sim, room_sim, obj_id))

            has_confident_target = False
            if candidates:
                candidates.sort(key=lambda x: (-x[1], -x[2]))
                best_score, best_sim, best_room_match, best_object_id = candidates[0]
                
                # =========================================================
                # 🚀 [新增] 移动阈值拦截机制
                # 只有最高分 >= 0.6，才真正执行 move_to
                # =========================================================
                if best_score >= 0.6:
                    confidence_level = best_score
                    target_pose = self.sg_sim.get_position_from_id(best_object_id)
                    target_id = best_object_id
                    action_output = 'move_to'
                    is_confident = True
                    has_confident_target = True
                    print(f"✅ 锁定高置信度目标 {best_object_id} (得分 {best_score:.2f} >= 0.6)，准备前往！")
                else:
                    print(f"⚠️ 最高分物体 {best_object_id} 得分仅为 {best_score:.2f} < 0.6，置信度不足，拒绝前往，继续探索！")

            # 如果没有找到候选物体，或者最高分都没过 0.6，强制进入探索模式
            if not has_confident_target:
                # =================================================================
                # Phase 2: 高级探索策略 & 放弃机制 (🎯 目标切换 + 物理坐标 + 防止无限重置)
                # =================================================================
                
                # 初始化复查标志位 (如果不存在)
                if not hasattr(self, 'final_check_done'):
                    self.final_check_done = False

                # 1. 目标切换检测：如果换了新目标，必须清空历史黑名单，并重置复查标志！
                if not hasattr(self, 'last_target'):
                    self.last_target = target
                if self.last_target != target:
                    print(f"🔄 目标已从 '{self.last_target}' 切换为 '{target}'，重置探索状态！")
                    if hasattr(self, 'visited_frontiers'):
                        self.visited_frontiers.clear()
                    if hasattr(self, 'visited_frontiers_coords'):
                        self.visited_frontiers_coords.clear()
                    self.final_check_done = False  # 换目标了，允许该目标有一次复查机会
                    self.last_target = target

                # 2. 坐标初始化
                if not hasattr(self, 'visited_frontiers_coords'):
                    self.visited_frontiers_coords = []
                
                all_frontiers = self.sg_sim.frontier_node_ids
                
                # 3. 基于真实物理坐标过滤（距离小于 1.0 米视为已访问）
                unvisited_frontiers = []
                for f_id in all_frontiers:
                    try:
                        f_pos = self.sg_sim.get_position_from_id(f_id)
                        is_visited = False
                        for v_pos in self.visited_frontiers_coords:
                            if np.linalg.norm(f_pos - v_pos) < 1.0: 
                                is_visited = True
                                break
                        if not is_visited:
                            unvisited_frontiers.append(f_id)
                    except:
                        pass
                
                # 4. 防死锁机制 (修复版)：都找遍了？只允许【清空一次】黑名单！
                if not unvisited_frontiers and all_frontiers:
                    if not self.final_check_done:
                        print("⚠️ [仅此一次] 当前目标已知边界探索完毕但未找到！清空黑名单进行最后复查...")
                        if hasattr(self, 'visited_frontiers'):
                            self.visited_frontiers.clear()
                        self.visited_frontiers_coords.clear()
                        unvisited_frontiers = all_frontiers
                        self.final_check_done = True  # 标记已复查，下次绝不再清空
                    else:
                        print("🚨 之前已经复查过了，前沿点仍未消除 (幽灵前沿)。拒绝再次陷入死循环！")
                        unvisited_frontiers = [] # 强行保持为空，让其落入 Step 5

                # 5. 真正的放弃机制 -> 拦截并升级为【最后一搏】
                if not unvisited_frontiers:
                    
                    # ==========================================================
                    # 🚀 [核心重构] 如果全屋探索完了没找到完美匹配，尝试死马当活马医
                    # ==========================================================
                    if global_best_object_id and global_max_score > 0.1:
                        print(f"🚨 [最后一搏触发] 全屋边界探索完毕，未发现高置信度目标！")
                        print(f"🎯 决定盲赌历史相似度最高的物体: {global_best_object_id} (全场最高得分: {global_max_score:.2f})")
                        
                        confidence_level = global_max_score
                        target_pose = self.sg_sim.get_position_from_id(global_best_object_id)
                        target_id = global_best_object_id
                        action_output = 'move_to'
                        is_confident = True  # 🌟 强行设为 True，命令底层的小脑 Runner 走过去并直接触发子任务结算！
                        
                        # 仍然可以作为失败经验记进皮层，供后续子任务参考
                        self.experience_memory.add_experience(
                            query=f"寻找 {target}",
                            thought=f"全屋搜查未找到完美匹配，最终选择前往最相似物体 {global_best_object_id} 进行盲赌。",
                            outcome="Gamble_Fallback",
                            graph_anchor=global_best_object_id
                        )
                    else:
                        # 如果整个场景图里甚至一个物体都没扫描出来，才彻底两手一摊宣告放弃
                        print(f"🚨 警告：全屋边界已彻底探索完毕，且场景图中未识别到任何有效物体！无处可赌，宣告放弃！")
                        action_output = 'give_up'
                        confidence_level = 0.0
                        is_confident = False
                        target_pose = None
                        target_id = None
                        
                        self.experience_memory.add_experience(
                            query=f"寻找 {target}",
                            thought=f"在全屋彻底搜查后未找到 {target}。这说明早期的默认搜寻假设可能错误，需换思路。",
                            outcome="Failure",
                            graph_anchor=current_room_id if current_room_id else "unknown"
                        )

                else:
                    # =========================================================
                    # 🚀 [重构升级] 全新四维探索评估打分公式 (Next-Gen Active Exploration)
                    # =========================================================
                    alpha = 3.0        # 信息增益权重 (IG)
                    beta = 2.0         # 房间剩余价值权重
                    gamma_room = 20.0  # 房间名称语义权重
                    gamma_obj = 35.0   # 周边物体语义权重
                    delta = 1.5        # 距离惩罚权重
                    
                    best_frontier = None
                    best_score = -float('inf')
                    best_frontier_pos = None
                    
                    # 打印表头，方便对齐查看
                    print(f"\n" + "="*90)
                    print(f"🎯 目标: '{target}' | 注入的经验: {'有' if exp_context else '无'} | 当前权重: α={alpha}, β={beta}, γ_room={gamma_room}, γ_obj={gamma_obj}, δ={delta}")
                    print(f"{'ID':<12} | {'Room':<12} | {'Dist':<5} | {'IG':<4} | {'RoomVal':<7} | {'Sim(R)':<6} | {'Sim(O)':<6} | {'SCORE':<7}")
                    print("-" * 90)

                    for f_id in unvisited_frontiers:
                        try:
                            # 1. 基础物理数据
                            f_pos = self.sg_sim.get_position_from_id(f_id)
                            distance = np.linalg.norm(current_pos - f_pos)
                            
                            # 2. 局部信息增益 (IG) 
                            ig_score = self.sg_sim.get_frontier_ig(f_id)
                            
                            # 3. 获取前沿点所属的房间信息
                            room_id, room_name = self.sg_sim.get_room_for_frontier(f_id)
                            
                            # 4. 全局拓扑：房间剩余探索价值
                            raw_room_val = self.sg_sim.get_room_unexplored_ratio(room_id)
                            room_unexplored_val = min(10.0, raw_room_val / 50.0)  #归一化

                            # 5. 上下文语义潜力 (Semantic Potential)
                            sim_room = 0.0
                            if room_name and room_name not in ["unknown region", "room", "room_0"]:
                                # 🌟 传入 exp_context 引导房间评估
                                r_sim, _, _ = self.get_semantic_similarity(target, room_name, exp_context)
                                sim_room = r_sim
                                
                            sim_obj_max = 0.0
                            nearby_objects = self.sg_sim.get_objects_near_frontier(f_id, radius=2.0) 
                            for obj_name in nearby_objects:
                                # 🌟 传入 exp_context 引导周边物体评估
                                obj_sim, _, _ = self.get_semantic_similarity(target, obj_name, exp_context)
                                if obj_sim > sim_obj_max:
                                    sim_obj_max = obj_sim
                                    
                            # ✨ 计算各部分得分与最终总分
                            score_ig = alpha * ig_score
                            score_room_val = beta * room_unexplored_val
                            score_sim_room = gamma_room * sim_room
                            score_sim_obj = gamma_obj * sim_obj_max
                            penalty_dist = delta * distance
                            
                            score = score_ig + score_room_val + score_sim_room + score_sim_obj - penalty_dist
                            
                            # 格式化打印这行数据，形成对齐的表格
                            print(f"{f_id[:10]+'..':<12} | {room_name[:10]:<12} | {distance:>5.1f} | {ig_score:>4.1f} | {room_unexplored_val:>7.1f} | {sim_room:>6.2f} | {sim_obj_max:>6.2f} | {score:>7.2f}")
                            
                            if score > best_score:
                                best_score = score
                                best_frontier = f_id
                                best_frontier_pos = f_pos
                                
                        except Exception as e:
                            print(f"  评估前沿点 {f_id} 失败: {e}")
                            
                    print("=" * 90)
                            
                    if best_frontier:
                        target_pose = best_frontier_pos
                        target_id = best_frontier
                        action_output = 'explore_frontier'
                        
                        self.visited_frontiers_coords.append(target_pose)
                        
                        confidence_level = 0.4
                        is_confident = False
                        print(f"💡 决策完成！选择最优探索前沿: {best_frontier} (最高得分: {best_score:.2f})")

                    else:
                        print("⚠️ 无法打分，回退到原生的外部 frontier_nodes 距离判断")
                        if len(frontier_nodes) > 0:
                            distances = [np.linalg.norm(current_pos - p) for p in frontier_nodes]
                            closest_idx = np.argmin(distances)
                            target_pose = frontier_nodes[closest_idx]
                            target_id = None
                            action_output = 'explore_frontier'
                            self.visited_frontiers_coords.append(target_pose)
                            confidence_level = 0.3
                            is_confident = False

        self._t += 1
        return target_pose, target_id, is_confident, confidence_level, action_output

    def get_current_state_prompt_for_subtask(self, scene_graph, agent_state, current_subtask):
        """为旧接口准备的状态提示"""
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n \
            CURRENT SUBTASK: {current_subtask}. \n "

        if self._add_history:
            prompt += f"HISTORY: {self._history}"
        return prompt

    def get_scene_graph_summary(self):
        """获取场景图摘要，过滤已访问frontiers"""
        rooms = self.sg_sim.room_node_names[:5] if hasattr(self.sg_sim, 'room_node_names') else []
        objects = self.sg_sim.object_node_names[:10] if hasattr(self.sg_sim, 'object_node_names') else []
        frontiers = [f for f in (self.sg_sim.frontier_node_ids[:5] if hasattr(self.sg_sim, 'frontier_node_ids') else []) if f not in self.visited_frontiers]
        
        # 获取智能体状态
        agent_state = self.sg_sim.get_current_semantic_state_str()
        
        summary = {
            'rooms': rooms,
            'visible_objects': objects,
            'frontiers': frontiers,
            'agent_state': agent_state
        }
        
        return json.dumps(summary, indent=2)

    def find_target_position(self, target_name):
        """查找目标对象的位置"""
        if not hasattr(self.sg_sim, 'object_node_ids') or not hasattr(self.sg_sim, 'object_node_names'):
            return None
            
        for obj_id, obj_name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            if target_name.lower() in obj_name.lower():
                try:
                    return self.sg_sim.get_position_from_id(obj_id)
                except:
                    continue
        return None

    def is_task_completed(self):
        """检查任务是否完成"""
        if self.task_manager:
            return self.task_manager.is_task_completed()
        return False