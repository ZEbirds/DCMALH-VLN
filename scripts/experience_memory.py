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
            "use_count": self.use_count
        }


class VectorizedExperienceMemory:
    """向量化经验记忆模块 (双通道系统的大脑皮层 - 基于 OpenAI Embedding)"""
    def __init__(self, similarity_threshold=0.3):
        print(f"🧠 [Memory Engine] 正在初始化 OpenAI 向量引擎 (text-embedding-3-small)...")
        self.client = OpenAI()
        self.memory_bank: List[ExperienceNode] = []
        # OpenAI 的相似度区分度很大，不相关的通常在 0.1~0.2，高度相关的在 0.5+
        self.similarity_threshold = similarity_threshold 

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
            return np.zeros(1536) # 3-small 的维度是 1536

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def add_experience(self, query: str, thought: str, outcome: str, graph_anchor: str):
        """
        写入新记忆
        [修复] 不加任何英文模板前缀，直接拼接核心中文语义，最大化区分度
        """
        text_to_encode = f"目标：{query}。经验与策略：{thought}"
        embedding = self._get_embedding(text_to_encode)
        
        new_node = ExperienceNode(query, thought, outcome, graph_anchor, embedding)
        self.memory_bank.append(new_node)
        print(f"💾 [Memory Saved] 成功存入经验 ({outcome}): {thought[:30]}... 锚点: {graph_anchor}")

    def retrieve_experience(self, current_query: str, current_pos: np.ndarray, 
                            get_anchor_pos_func, top_k: int = 2, 
                            w_relevance: float = 0.7, w_reachability: float = 0.3) -> List[Dict]:
        """双维度检索：相关性 (Relevance) + 可达性 (Reachability)"""
        if not self.memory_bank:
            return []

        # 仅对当前的 Query 进行纯粹的向量化
        query_embedding = self._get_embedding(current_query)
        scored_memories = []

        print(f"\n🔍 [Memory Query] 正在检索: '{current_query}' (当前坐标: {current_pos})")
        print("-" * 80)
        
        for exp in self.memory_bank:
            relevance = self._cosine_similarity(query_embedding, exp.embedding)
            
            if relevance < self.similarity_threshold:
                continue

            reachability = 0.0
            anchor_pos = get_anchor_pos_func(exp.graph_anchor)
            distance = -1.0
            
            if anchor_pos is not None:
                distance = np.linalg.norm(current_pos - np.array(anchor_pos))
                reachability = max(0.0, 1.0 - (distance / 20.0))
            
            total_score = (w_relevance * relevance) + (w_reachability * reachability)
            scored_memories.append((total_score, relevance, reachability, distance, exp))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, rel, reach, dist, exp) in enumerate(scored_memories[:top_k]):
            exp.use_count += 1
            exp.last_access_time = time.time()
            
            dist_str = f"{dist:.1f}m" if dist >= 0 else "未知"
            print(f" ⭐ Top {rank+1} [总分:{score:.2f} | 语义:{rel:.2f} | 可达性:{reach:.2f} (距 {dist_str})]")
            print(f"    ↳ 标签: [{exp.outcome}] 锚点: {exp.graph_anchor}")
            print(f"    ↳ 经验: {exp.thought}")
            
            results.append(exp.to_dict())

        if not results:
            print(" ⚠️ 未检索到高于阈值(0.3)的有效历史经验。")

        print("-" * 80)
        return results

# =========================================================================
# 独立测试模块 (Standalone Tester)
# =========================================================================
if __name__ == "__main__":
    def mock_get_anchor_pos(anchor_id: str):
        mock_map = {
            "room_laundry": [2.0, 0.0, 2.0],   
            "room_bathroom": [12.0, 0.0, 9.0], 
            "object_microwave": [5.0, 1.0, 0.0]
        }
        return mock_map.get(anchor_id, None)

    memory_sys = VectorizedExperienceMemory()

    print("\n--- 预先注入历史经验 ---")
    memory_sys.add_experience(
        query="给植物浇水",
        thought="在卫生间寻找水盆失败，最终在洗衣房水槽接到了水。洗衣房更适合处理杂物水源。",
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

    current_robot_pos = np.array([0.0, 0.0, 0.0])
    
    # 案例 A:
    memory_sys.retrieve_experience(
        current_query="寻找卫生纸", 
        current_pos=current_robot_pos, 
        get_anchor_pos_func=mock_get_anchor_pos,
        top_k=2
    )

    # 案例 B:
    memory_sys.retrieve_experience(
        current_query="找一个能接水的地方", 
        current_pos=current_robot_pos, 
        get_anchor_pos_func=mock_get_anchor_pos,
        top_k=2
    )