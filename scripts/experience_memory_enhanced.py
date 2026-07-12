import numpy as np
import time
import uuid
from typing import List, Dict
from openai import OpenAI

class ExperienceNode:
    """记忆库的最小单元：一条经验"""
    def __init__(self, query: str, thought: str, outcome: str, graph_anchor: str, embedding: np.ndarray):
        self.id = str(uuid.uuid4())[:8]  
        self.query = query               
        self.thought = thought           
        self.outcome = outcome           
        self.graph_anchor = graph_anchor 
        self.embedding = embedding       
        
        self.creation_time = time.time() 
        self.last_access_time = self.creation_time 
        self.use_count = 0               

    def to_dict(self):
        return {
            "id": self.id,
            "query": self.query,
            "thought": self.thought,
            "outcome": self.outcome,
            "anchor": self.graph_anchor,
            "use_count": self.use_count,
            "last_access_time": self.last_access_time
        }


class VectorizedExperienceMemory:
    """向量化经验记忆模块 (双通道系统的大脑皮层 - 基于 OpenAI Embedding)"""
    def __init__(self, similarity_threshold=0.3, max_capacity=50):
        print(f"🧠 [Memory Engine] 正在初始化 OpenAI 向量引擎 (text-embedding-3-small)...")
        self.client = OpenAI()
        self.memory_bank: List[ExperienceNode] = []
        self.similarity_threshold = similarity_threshold 
        self.max_capacity = max_capacity # 论文 5.5 节承诺的最大硬件稳态容量

    def _get_embedding(self, text: str) -> np.ndarray:
        """调用 OpenAI 获取高精度语义向量"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"⚠️ OpenAI Embedding 获取失败: {e}")
            return np.zeros(1536) 

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def add_experience(self, query: str, thought: str, outcome: str, graph_anchor: str):
        """写入新记忆并自动触发硬件容量维护"""
        # 🌟 核心改进：大盘建立时，仅对目标和核心语境编码，保持检索结构的语义对称
        text_to_encode = f"当前导航目标：{query}"
        embedding = self._get_embedding(text_to_encode)
        
        new_node = ExperienceNode(query, thought, outcome, graph_anchor, embedding)
        self.memory_bank.append(new_node)
        print(f"💾 [Memory Saved] 成功存入经验 ({outcome}): {thought[:30]}... 锚点: {graph_anchor}")
        
        # 🌟 论文 5.5 节落地：存入后自动检查容量，超标则触发主动遗忘剪枝
        if len(self.memory_bank) > self.max_capacity:
            self.prune_memory()

    def retrieve_experience(self, current_query: str, current_pos: np.ndarray, 
                            get_anchor_pos_func, top_k: int = 2, 
                            w_relevance: float = 0.7, w_reachability: float = 0.3) -> List[Dict]:
        """双维度检索：相关性 (Relevance) + 可达性 (Reachability) + 避坑修正"""
        if not self.memory_bank:
            return []

        # 🌟 核心改进：保持查询句式和 add_experience 严格对称，大幅提升匹配率
        search_text = f"当前导航目标：{current_query}"
        query_embedding = self._get_embedding(search_text)
        scored_memories = []

        print(f"\n🔍 [Memory Query] 正在检索: '{current_query}' (当前坐标: {current_pos})")
        print("-" * 80)
        
        for exp in self.memory_bank:
            relevance = self._cosine_similarity(query_embedding, exp.embedding)
            
            if relevance < self.similarity_threshold:
                continue

            # 计算空间可达性
            reachability = 0.0
            anchor_pos = get_anchor_pos_func(exp.graph_anchor)
            distance = -1.0
            
            if anchor_pos is not None:
                distance = np.linalg.norm(current_pos - np.array(anchor_pos))
                reachability = max(0.0, 1.0 - (distance / 20.0))
            
            # 计算基础得分
            total_score = (w_relevance * relevance) + (w_reachability * reachability)
            
            # 🌟 核心改进：引入失败经验拦截加成
            # 如果机器人当前距离之前失败过的锚点在 2.5 米以内，且语义高度相关
            # 说明这块地方是已知的“死胡同/陷阱”，强行将该记忆的推荐排序拉满，作为必选提示词塞给 GPT 避坑！
            if exp.outcome.lower() == "failure" and distance >= 0 and distance < 2.5:
                total_score += 0.4 
                print(f"🚨 [陷阱预警机制触发] 智能体当前极度接近历史失败区域 {exp.graph_anchor}，赋予避坑最高优先级！")

            scored_memories.append((total_score, relevance, reachability, distance, exp))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, rel, reach, dist, exp) in enumerate(scored_memories[:top_k]):
            exp.use_count += 1
            exp.last_access_time = time.time() # 更新访问时间，用于后续的记忆钝化计算
            
            dist_str = f"{dist:.1f}m" if dist >= 0 else "未知"
            print(f" ⭐ Top {rank+1} [总分:{score:.2f} | 语义:{rel:.2f} | 可达性:{reach:.2f} (距 {dist_str})]")
            print(f"    ↳ 状态: [{exp.outcome}] 锚点: {exp.graph_anchor}")
            print(f"    ↳ 经验: {exp.thought}")
            
            results.append(exp.to_dict())

        if not results:
            print(f" ⚠️ 未检索到高于阈值({self.similarity_threshold})的有效历史经验。")

        print("-" * 80)
        return results

    def prune_memory(self):
        """🌟 论文 5.5 节系统具体实现：基于贡献度指数的主动遗忘算法"""
        if not self.memory_bank:
            return
            
        current_time = time.time()
        node_scores = []
        
        print(f"🔄 [Smart Memory Management] 记忆容量超标(>{self.max_capacity})，启动贡献值剪枝程序...")
        
        for idx, node in enumerate(self.memory_bank):
            # 1. 计算时间衰减因子 (Recency): 长期未被访问的记忆会发生自然钝化
            time_elapsed = current_time - node.last_access_time
            # 使用经典的指数时间衰减模型 (半衰期大约设为 600 秒，可根据长视野实际耗时微调)
            recency_factor = np.exp(-time_elapsed / 600.0)
            
            # 2. 结合使用频率 (Frequency) 计算整体贡献值得分
            contribution_score = (node.use_count + 1.0) * recency_factor
            node_scores.append((contribution_score, idx, node))
            
        # 按照贡献度得分从低到高排序，移除得分最低的那条“无价值冗余记忆”
        node_scores.sort(key=lambda x: x[0])
        victim_idx = node_scores[0][1]
        victim_node = node_scores[0][2]
        
        print(f"🗑️ [Active Forgetting] 成功清除低价值长期冷冻记忆节点: ID [{victim_node.id}] | 历史命中次数: {victim_node.use_count} | 经验内容: {victim_node.thought[:20]}...")
        self.memory_bank.pop(victim_idx)


# =========================================================================
# 独立测试模块 (验证修改后的物理表现)
# =========================================================================
if __name__ == "__main__":
    def mock_get_anchor_pos(anchor_id: str):
        mock_map = {
            "room_laundry": [2.0, 0.0, 2.0],   
            "room_bathroom": [1.5, 0.0, 1.0], # 缩短距离，模拟机器人目前正走在卫生间门口 
            "object_microwave": [5.0, 1.0, 0.0]
        }
        return mock_map.get(anchor_id, None)

    # 初始化容量设为 3，测试剪枝溢出
    memory_sys = VectorizedExperienceMemory(max_capacity=3)

    print("\n--- 预先注入历史经验 ---")
    memory_sys.add_experience(
        query="给植物浇水",
        thought="在卫生间寻找水盆失败，最终在洗衣房水槽接到了水。卫生间根本没水源。",
        outcome="Failure",
        graph_anchor="room_bathroom"
    )
    memory_sys.add_experience(
        query="拿备用卫生纸",
        thought="不要在卫生间死磕。备用纸巾通常储存在洗衣房高层的架子上。",
        outcome="Success",
        graph_anchor="room_laundry"
    )
    memory_sys.add_experience(
        query="随便逛逛",
        thought="在厨房岛台区域发现了一台微波炉。",
        outcome="Exploration",
        graph_anchor="object_microwave"
    )
    
    # 模拟注入第四条，强行挤爆记忆库，触发主动遗忘
    time.sleep(1) # 制造时间差
    memory_sys.add_experience(
        query="寻找充电桩",
        thought="充电桩在客厅大电视左侧角落里。",
        outcome="Success",
        graph_anchor="room_living"
    )

    current_robot_pos = np.array([0.0, 0.0, 0.0])
    
    # 验证对称匹配与避坑拦截
    memory_sys.retrieve_experience(
        current_query="寻找水源接水", 
        current_pos=current_robot_pos, 
        get_anchor_pos_func=mock_get_anchor_pos,
        top_k=1
    )