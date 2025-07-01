"""
增强记忆进化管理模块
实现多层次记忆结构、动态权重调整、知识图谱等高级功能
"""
import sqlite3
import json
import time
import math
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib

@dataclass
class EnhancedMemory:
    """增强记忆数据结构"""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "conversation"  # conversation, experience, knowledge, pattern
    importance: float = 0.5  # 重要性评分 0-1
    emotional_valence: float = 0.0  # 情感价值 -1到1
    access_frequency: int = 0  # 访问频次
    last_accessed: float = 0.0  # 最后访问时间
    creation_time: float = 0.0  # 创建时间
    decay_rate: float = 0.01  # 遗忘率
    associations: List[int] = None  # 关联记忆ID
    tags: List[str] = None  # 标签
    metadata: Dict[str, Any] = None  # 元数据
    embedding: List[float] = None  # 向量嵌入
    memory_weight: float = 1.0  # 记忆权重
    consolidation_level: int = 0  # 巩固级别 0-3
    
    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.creation_time
        if self.associations is None:
            self.associations = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.embedding is None:
            self.embedding = []

@dataclass
class MemoryCluster:
    """记忆聚类"""
    id: int
    centroid: List[float]
    memories: List[int]  # 记忆ID列表
    coherence: float = 0.0  # 聚类内聚度
    stability: float = 0.0  # 稳定性
    last_updated: float = 0.0
    
    def __post_init__(self):
        if self.last_updated == 0.0:
            self.last_updated = time.time()

class MemoryGraph:
    """记忆知识图谱"""
    
    def __init__(self):
        self.nodes: Dict[int, EnhancedMemory] = {}  # 节点（记忆）
        self.edges: Dict[int, Dict[int, float]] = defaultdict(dict)  # 边（关联强度）
        self.centrality_scores: Dict[int, float] = {}  # 中心性得分
        self.communities: List[Set[int]] = []  # 社区划分
    
    def add_memory(self, memory: EnhancedMemory):
        """添加记忆节点"""
        if memory.id:
            self.nodes[memory.id] = memory
    
    def add_association(self, memory_id1: int, memory_id2: int, strength: float):
        """添加记忆关联"""
        self.edges[memory_id1][memory_id2] = strength
        self.edges[memory_id2][memory_id1] = strength
    
    def calculate_centrality(self):
        """计算节点中心性"""
        for node_id in self.nodes:
            # 度中心性
            degree = len(self.edges.get(node_id, {}))
            self.centrality_scores[node_id] = degree / max(1, len(self.nodes) - 1)
    
    def find_communities(self):
        """社区检测（简化版本）"""
        visited = set()
        self.communities = []
        
        for node_id in self.nodes:
            if node_id not in visited:
                community = self._dfs_community(node_id, visited)
                if len(community) >= 2:
                    self.communities.append(community)
    
    def _dfs_community(self, start_id: int, visited: Set[int], 
                      threshold: float = 0.5) -> Set[int]:
        """深度优先搜索社区"""
        community = {start_id}
        visited.add(start_id)
        
        for neighbor_id, strength in self.edges.get(start_id, {}).items():
            if neighbor_id not in visited and strength >= threshold:
                community.update(self._dfs_community(neighbor_id, visited, threshold))
        
        return community
    
    def get_memory_context(self, memory_id: int, depth: int = 2) -> List[EnhancedMemory]:
        """获取记忆上下文"""
        context_memories = []
        visited = {memory_id}
        current_level = [memory_id]
        
        for _ in range(depth):
            next_level = []
            for mem_id in current_level:
                for neighbor_id, strength in self.edges.get(mem_id, {}).items():
                    if neighbor_id not in visited and strength > 0.3:
                        visited.add(neighbor_id)
                        next_level.append(neighbor_id)
                        if neighbor_id in self.nodes:
                            context_memories.append(self.nodes[neighbor_id])
            current_level = next_level
            if not current_level:
                break
        
        return context_memories

class EnhancedMemoryManager:
    """增强记忆管理器"""
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        self.memory_graph = MemoryGraph()
        self.memory_clusters: List[MemoryCluster] = []
        self.active_patterns: Dict[str, float] = {}  # 活跃模式
        self.forgetting_curve_params = {"alpha": 0.1, "beta": 1.0}  # 遗忘曲线参数
        
        self.init_database()
        self.load_memory_graph()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 增强记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'conversation',
                importance REAL DEFAULT 0.5,
                emotional_valence REAL DEFAULT 0.0,
                access_frequency INTEGER DEFAULT 0,
                last_accessed REAL NOT NULL,
                creation_time REAL NOT NULL,
                decay_rate REAL DEFAULT 0.01,
                associations TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                embedding TEXT DEFAULT '[]',
                memory_weight REAL DEFAULT 1.0,
                consolidation_level INTEGER DEFAULT 0
            )
        """)
        
        # 记忆关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id1 INTEGER,
                memory_id2 INTEGER,
                association_strength REAL DEFAULT 0.5,
                association_type TEXT DEFAULT 'semantic',
                created_time REAL NOT NULL,
                FOREIGN KEY (memory_id1) REFERENCES enhanced_memories (id),
                FOREIGN KEY (memory_id2) REFERENCES enhanced_memories (id)
            )
        """)
        
        # 记忆聚类表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                centroid TEXT NOT NULL,
                memory_ids TEXT NOT NULL,
                coherence REAL DEFAULT 0.0,
                stability REAL DEFAULT 0.0,
                last_updated REAL NOT NULL
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON enhanced_memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON enhanced_memories(importance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON enhanced_memories(last_accessed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consolidation ON enhanced_memories(consolidation_level)")
        
        conn.commit()
        conn.close()
    
    def add_memory(self, memory: EnhancedMemory) -> int:
        """添加增强记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO enhanced_memories (
                content, memory_type, importance, emotional_valence,
                access_frequency, last_accessed, creation_time, decay_rate,
                associations, tags, metadata, embedding, memory_weight,
                consolidation_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.content, memory.memory_type, memory.importance,
            memory.emotional_valence, memory.access_frequency,
            memory.last_accessed, memory.creation_time, memory.decay_rate,
            json.dumps(memory.associations), json.dumps(memory.tags),
            json.dumps(memory.metadata), json.dumps(memory.embedding),
            memory.memory_weight, memory.consolidation_level
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        if memory_id:
            memory.id = memory_id
            self.memory_graph.add_memory(memory)
            self._update_memory_associations(memory)
        
        return memory_id or 0
    
    def get_memory_by_id(self, memory_id: int) -> Optional[EnhancedMemory]:
        """根据ID获取记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM enhanced_memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_enhanced_memory(row)
        return None
    
    def search_memories(self, query: str, limit: int = 10, 
                       memory_types: Optional[List[str]] = None) -> List[EnhancedMemory]:
        """智能记忆搜索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本文本搜索
        base_query = "SELECT * FROM enhanced_memories WHERE content LIKE ?"
        params = [f"%{query}%"]
        
        # 按类型过滤
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            base_query += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        
        # 按重要性和新近性排序
        base_query += """
            ORDER BY (importance * memory_weight * 
                     (1.0 - (? - last_accessed) / 86400.0 * decay_rate)) DESC
            LIMIT ?
        """
        params.extend([time.time(), limit])
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        conn.close()
        
        memories = [self._row_to_enhanced_memory(row) for row in rows]
        
        # 更新访问记录
        for memory in memories:
            self._update_access_record(memory.id)
        
        return memories
    
    def get_contextual_memories(self, memory_id: int, context_size: int = 5) -> List[EnhancedMemory]:
        """获取上下文相关记忆"""
        # 使用知识图谱获取上下文
        context_memories = self.memory_graph.get_memory_context(memory_id, depth=2)
        
        # 按关联强度排序
        scored_memories = []
        for memory in context_memories:
            if memory.id and memory_id in self.memory_graph.edges:
                strength = self.memory_graph.edges[memory_id].get(memory.id, 0.0)
                scored_memories.append((memory, strength))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in scored_memories[:context_size]]
    
    def consolidate_memories(self, threshold_hours: float = 24.0):
        """记忆巩固过程"""
        cutoff_time = time.time() - (threshold_hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取需要巩固的记忆
        cursor.execute("""
            SELECT * FROM enhanced_memories 
            WHERE creation_time < ? AND consolidation_level < 3
            ORDER BY importance DESC, access_frequency DESC
        """, (cutoff_time,))
        
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_enhanced_memory(row)
            if memory.id:
                self._perform_consolidation(memory)
        
        conn.close()
    
    def _perform_consolidation(self, memory: EnhancedMemory):
        """执行记忆巩固"""
        # 计算巩固因子
        age_factor = (time.time() - memory.creation_time) / 86400.0  # 天数
        access_factor = min(1.0, memory.access_frequency / 10.0)
        importance_factor = memory.importance
        
        consolidation_score = (age_factor * 0.3 + access_factor * 0.4 + 
                             importance_factor * 0.3)
        
        # 更新巩固级别
        if consolidation_score > 0.8 and memory.consolidation_level < 3:
            memory.consolidation_level += 1
            memory.memory_weight *= 1.2  # 增加权重
            memory.decay_rate *= 0.8  # 降低遗忘率
            
            self._update_memory_in_db(memory)
    
    def apply_forgetting_curve(self, days_threshold: int = 30):
        """应用遗忘曲线"""
        cutoff_time = time.time() - (days_threshold * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, last_accessed, decay_rate, memory_weight, consolidation_level
            FROM enhanced_memories
            WHERE last_accessed < ?
        """, (cutoff_time,))
        
        rows = cursor.fetchall()
        
        for row in rows:
            memory_id, last_accessed, decay_rate, memory_weight, consolidation_level = row
            
            # 计算时间衰减
            time_diff = (time.time() - last_accessed) / 86400.0  # 天数
            decay_factor = math.exp(-decay_rate * time_diff)
            
            # 巩固级别影响衰减
            consolidation_bonus = 1.0 + (consolidation_level * 0.1)
            new_weight = memory_weight * decay_factor * consolidation_bonus
            
            # 更新权重
            cursor.execute("""
                UPDATE enhanced_memories 
                SET memory_weight = ?
                WHERE id = ?
            """, (max(0.1, new_weight), memory_id))
        
        conn.commit()
        conn.close()
    
    def cluster_memories(self, num_clusters: int = 10):
        """记忆聚类"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取有嵌入向量的记忆
        cursor.execute("""
            SELECT id, embedding FROM enhanced_memories 
            WHERE embedding != '[]' AND json_array_length(embedding) > 0
        """)
        
        rows = cursor.fetchall()
        
        if len(rows) < num_clusters:
            return
        
        # 简化的K-means聚类
        memories_with_embeddings = []
        for row in rows:
            memory_id, embedding_str = row
            try:
                embedding = json.loads(embedding_str)
                if embedding:
                    memories_with_embeddings.append((memory_id, embedding))
            except:
                continue
        
        if len(memories_with_embeddings) < num_clusters:
            return
        
        # 随机初始化聚类中心
        import random
        cluster_centers = random.sample(memories_with_embeddings, num_clusters)
        clusters = [[] for _ in range(num_clusters)]
        
        # 迭代聚类
        for iteration in range(10):  # 最多10次迭代
            # 分配到最近的聚类
            for memory_id, embedding in memories_with_embeddings:
                min_distance = float('inf')
                closest_cluster = 0
                
                for i, (_, center_embedding) in enumerate(cluster_centers):
                    distance = self._calculate_distance(embedding, center_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster = i
                
                clusters[closest_cluster].append(memory_id)
            
            # 更新聚类中心
            new_centers = []
            for i, cluster in enumerate(clusters):
                if cluster:
                    # 计算中心
                    center_embedding = self._calculate_centroid(
                        [embedding for mid, embedding in memories_with_embeddings 
                         if mid in cluster]
                    )
                    new_centers.append((0, center_embedding))
                else:
                    new_centers.append(cluster_centers[i])
            
            cluster_centers = new_centers
            clusters = [[] for _ in range(num_clusters)]
        
        # 保存聚类结果
        self._save_clusters(clusters, cluster_centers)
        conn.close()
    
    def _calculate_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算嵌入向量距离"""
        if len(embedding1) != len(embedding2):
            return float('inf')
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)))
    
    def _calculate_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """计算向量中心"""
        if not embeddings:
            return []
        
        dimension = len(embeddings[0])
        centroid = [0.0] * dimension
        
        for embedding in embeddings:
            for i in range(dimension):
                centroid[i] += embedding[i]
        
        for i in range(dimension):
            centroid[i] /= len(embeddings)
        
        return centroid
    
    def _save_clusters(self, clusters: List[List[int]], centers: List[Tuple[int, List[float]]]):
        """保存聚类结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 清除旧聚类
        cursor.execute("DELETE FROM memory_clusters")
        
        # 保存新聚类
        for i, (cluster, (_, center)) in enumerate(zip(clusters, centers)):
            if cluster:
                coherence = self._calculate_cluster_coherence(cluster)
                cursor.execute("""
                    INSERT INTO memory_clusters (centroid, memory_ids, coherence, stability, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    json.dumps(center),
                    json.dumps(cluster),
                    coherence,
                    0.0,  # 稳定性需要历史数据计算
                    time.time()
                ))
        
        conn.commit()
        conn.close()
    
    def _calculate_cluster_coherence(self, cluster: List[int]) -> float:
        """计算聚类内聚度"""
        if len(cluster) <= 1:
            return 1.0
        
        # 简化计算：基于记忆间的关联强度
        total_strength = 0.0
        pair_count = 0
        
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                mem_id1, mem_id2 = cluster[i], cluster[j]
                strength = self.memory_graph.edges.get(mem_id1, {}).get(mem_id2, 0.0)
                total_strength += strength
                pair_count += 1
        
        return total_strength / max(1, pair_count)
    
    def _update_memory_associations(self, memory: EnhancedMemory):
        """更新记忆关联"""
        if not memory.id:
            return
        
        # 基于内容相似性建立关联
        similar_memories = self.search_memories(
            memory.content[:50], limit=5, 
            memory_types=[memory.memory_type]
        )
        
        for similar_memory in similar_memories:
            if similar_memory.id != memory.id and similar_memory.id:
                # 计算关联强度
                strength = self._calculate_semantic_similarity(memory, similar_memory)
                if strength > 0.3:
                    self.memory_graph.add_association(memory.id, similar_memory.id, strength)
                    self._save_association(memory.id, similar_memory.id, strength)
    
    def _calculate_semantic_similarity(self, memory1: EnhancedMemory, 
                                     memory2: EnhancedMemory) -> float:
        """计算语义相似性"""
        # 简化版本：基于共同词汇
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _save_association(self, memory_id1: int, memory_id2: int, strength: float):
        """保存记忆关联"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memory_associations 
            (memory_id1, memory_id2, association_strength, association_type, created_time)
            VALUES (?, ?, ?, ?, ?)
        """, (memory_id1, memory_id2, strength, "semantic", time.time()))
        
        conn.commit()
        conn.close()
    
    def _update_access_record(self, memory_id: Optional[int]):
        """更新访问记录"""
        if not memory_id:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE enhanced_memories 
            SET access_frequency = access_frequency + 1,
                last_accessed = ?
            WHERE id = ?
        """, (time.time(), memory_id))
        
        conn.commit()
        conn.close()
    
    def _update_memory_in_db(self, memory: EnhancedMemory):
        """更新数据库中的记忆"""
        if not memory.id:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE enhanced_memories 
            SET importance = ?, memory_weight = ?, decay_rate = ?, 
                consolidation_level = ?, last_accessed = ?
            WHERE id = ?
        """, (
            memory.importance, memory.memory_weight, memory.decay_rate,
            memory.consolidation_level, time.time(), memory.id
        ))
        
        conn.commit()
        conn.close()
    
    def _row_to_enhanced_memory(self, row) -> EnhancedMemory:
        """将数据库行转换为EnhancedMemory对象"""
        return EnhancedMemory(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            importance=row[3],
            emotional_valence=row[4],
            access_frequency=row[5],
            last_accessed=row[6],
            creation_time=row[7],
            decay_rate=row[8],
            associations=json.loads(row[9]),
            tags=json.loads(row[10]),
            metadata=json.loads(row[11]),
            embedding=json.loads(row[12]),
            memory_weight=row[13],
            consolidation_level=row[14]
        )
    
    def load_memory_graph(self):
        """加载记忆图谱"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 加载记忆节点
        cursor.execute("SELECT * FROM enhanced_memories")
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_enhanced_memory(row)
            self.memory_graph.add_memory(memory)
        
        # 加载关联边
        cursor.execute("SELECT memory_id1, memory_id2, association_strength FROM memory_associations")
        associations = cursor.fetchall()
        
        for mem_id1, mem_id2, strength in associations:
            self.memory_graph.add_association(mem_id1, mem_id2, strength)
        
        conn.close()
        
        # 计算图谱统计信息
        self.memory_graph.calculate_centrality()
        self.memory_graph.find_communities()
    
    def get_memory_evolution_stats(self) -> Dict[str, Any]:
        """获取记忆进化统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本统计
        cursor.execute("SELECT COUNT(*) FROM enhanced_memories")
        total_memories = cursor.fetchone()[0]
        
        # 按巩固级别统计
        cursor.execute("""
            SELECT consolidation_level, COUNT(*) 
            FROM enhanced_memories 
            GROUP BY consolidation_level
        """)
        consolidation_stats = dict(cursor.fetchall())
        
        # 平均重要性
        cursor.execute("SELECT AVG(importance) FROM enhanced_memories")
        avg_importance = cursor.fetchone()[0] or 0.0
        
        # 记忆类型分布
        cursor.execute("""
            SELECT memory_type, COUNT(*) 
            FROM enhanced_memories 
            GROUP BY memory_type
        """)
        type_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_memories": total_memories,
            "consolidation_distribution": consolidation_stats,
            "average_importance": avg_importance,
            "type_distribution": type_distribution,
            "graph_stats": {
                "nodes": len(self.memory_graph.nodes),
                "edges": sum(len(edges) for edges in self.memory_graph.edges.values()) // 2,
                "communities": len(self.memory_graph.communities),
                "avg_centrality": sum(self.memory_graph.centrality_scores.values()) / 
                                max(1, len(self.memory_graph.centrality_scores))
            },
            "cluster_count": len(self.memory_clusters)
        }