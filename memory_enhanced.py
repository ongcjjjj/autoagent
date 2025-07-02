"""
记忆管理模块 - 深度增强版
在原有基础上增加：智能记忆整理、情感关联分析、记忆压缩算法、相似性检测、自动标签生成
"""
import sqlite3
import json
import time
import logging
import statistics
import hashlib
import asyncio
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor
import pickle
import lz4.frame

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """记忆类型枚举"""
    CONVERSATION = "conversation"
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"
    EMOTION = "emotion"
    SKILL = "skill"
    SYSTEM = "system"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

class ImportanceLevel(Enum):
    """重要性级别"""
    TRIVIAL = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9

@dataclass
class MemoryEnhanced:
    """增强记忆数据结构"""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "conversation"
    importance: float = 0.5
    timestamp: float = 0.0
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # 深度增强字段
    emotional_valence: float = 0.0  # 情感价值 -1到1
    emotional_arousal: float = 0.0  # 情感激活度 0到1
    access_count: int = 0  # 访问次数
    last_access: float = 0.0  # 最后访问时间
    similarity_hash: str = ""  # 相似性哈希
    compression_level: int = 0  # 压缩级别 0-9
    linked_memories: Optional[List[int]] = None  # 关联记忆ID
    
    # 新增高级字段
    context_vector: Optional[List[float]] = None  # 上下文向量
    knowledge_entities: Optional[List[str]] = None  # 知识实体
    temporal_context: Optional[Dict[str, Any]] = None  # 时间上下文
    spatial_context: Optional[Dict[str, Any]] = None  # 空间上下文
    confidence_score: float = 1.0  # 置信度分数
    decay_factor: float = 1.0  # 衰减因子
    consolidation_level: int = 0  # 巩固级别
    retrieval_strength: float = 1.0  # 检索强度
    interference_resistance: float = 1.0  # 干扰抵抗
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.last_access == 0.0:
            self.last_access = self.timestamp
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.linked_memories is None:
            self.linked_memories = []
        if self.knowledge_entities is None:
            self.knowledge_entities = []
        if self.temporal_context is None:
            self.temporal_context = {}
        if self.spatial_context is None:
            self.spatial_context = {}
        if not self.similarity_hash:
            self.similarity_hash = self._generate_similarity_hash()
        if self.context_vector is None:
            self.context_vector = self._generate_context_vector()
    
    def _generate_similarity_hash(self) -> str:
        """生成相似性哈希"""
        # 预处理内容
        content_words = re.findall(r'\w+', self.content.lower())
        important_words = [word for word in content_words if len(word) > 3][:15]
        content_for_hash = " ".join(sorted(important_words))
        
        # 生成哈希
        return hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
    
    def _generate_context_vector(self) -> List[float]:
        """生成上下文向量"""
        # 简化版本：基于词频和位置的向量
        words = re.findall(r'\w+', self.content.lower())
        vector_size = 64
        vector = [0.0] * vector_size
        
        for i, word in enumerate(words[:vector_size]):
            # 基于词汇的ASCII值和位置生成向量分量
            hash_val = hash(word + str(i)) % vector_size
            vector[hash_val] += 1.0 / (i + 1)  # 位置权重
        
        # 归一化
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
    def calculate_age_hours(self) -> float:
        """计算记忆年龄（小时）"""
        return (time.time() - self.timestamp) / 3600
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = time.time()
        
        # 更新检索强度（经常访问的记忆强度增加）
        self.retrieval_strength = min(2.0, self.retrieval_strength + 0.1)
    
    def calculate_forgetting_curve(self) -> float:
        """计算遗忘曲线值"""
        age_days = self.calculate_age_hours() / 24
        
        # Ebbinghaus遗忘曲线：R = e^(-t/S)
        # 其中R是记忆保持率，t是时间，S是记忆强度
        strength = self.importance * self.retrieval_strength * self.interference_resistance
        retention = np.exp(-age_days / max(strength, 0.1))
        
        return retention
    
    def should_be_compressed(self) -> bool:
        """判断是否应该被压缩"""
        age_hours = self.calculate_age_hours()
        retention = self.calculate_forgetting_curve()
        
        # 超过24小时且重要性低于0.7且访问次数少的记忆可以压缩
        return (age_hours > 24 and 
                self.importance < 0.7 and 
                self.access_count < 3 and
                retention < 0.5)

class MemoryIndexManager:
    """记忆索引管理器"""
    
    def __init__(self):
        self.content_index = {}  # 内容索引
        self.tag_index = defaultdict(set)  # 标签索引
        self.time_index = {}  # 时间索引
        self.similarity_index = defaultdict(set)  # 相似性索引
        self.entity_index = defaultdict(set)  # 实体索引
        
    def add_memory_to_index(self, memory: MemoryEnhanced):
        """将记忆添加到索引"""
        if memory.id is None:
            return
            
        # 内容索引（基于关键词）
        words = re.findall(r'\w+', memory.content.lower())
        for word in words:
            if len(word) > 2:
                if word not in self.content_index:
                    self.content_index[word] = set()
                self.content_index[word].add(memory.id)
        
        # 标签索引
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        # 时间索引（按小时分组）
        time_key = int(memory.timestamp // 3600)
        if time_key not in self.time_index:
            self.time_index[time_key] = set()
        self.time_index[time_key].add(memory.id)
        
        # 相似性索引
        sim_key = memory.similarity_hash[:4]  # 使用前4位作为相似性键
        self.similarity_index[sim_key].add(memory.id)
        
        # 实体索引
        for entity in memory.knowledge_entities:
            self.entity_index[entity.lower()].add(memory.id)
    
    def remove_memory_from_index(self, memory: MemoryEnhanced):
        """从索引中移除记忆"""
        if memory.id is None:
            return
            
        # 从各个索引中移除
        for index in [self.content_index, self.tag_index, self.similarity_index, self.entity_index]:
            for key, memory_set in index.items():
                memory_set.discard(memory.id)
    
    def search_by_content(self, query: str, limit: int = 10) -> Set[int]:
        """通过内容搜索"""
        query_words = re.findall(r'\w+', query.lower())
        memory_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.content_index:
                for memory_id in self.content_index[word]:
                    memory_scores[memory_id] += 1.0 / len(query_words)
        
        # 返回评分最高的记忆ID
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return set(mid for mid, score in sorted_memories[:limit])
    
    def search_by_similarity(self, similarity_hash: str, limit: int = 5) -> Set[int]:
        """通过相似性搜索"""
        sim_key = similarity_hash[:4]
        return self.similarity_index.get(sim_key, set())

class MemoryCompressionEngine:
    """记忆压缩引擎"""
    
    def __init__(self):
        self.compression_algorithms = {
            1: self._light_compression,
            2: self._medium_compression,
            3: self._heavy_compression
        }
    
    def compress_memory(self, memory: MemoryEnhanced, level: int = 1) -> MemoryEnhanced:
        """压缩记忆"""
        if level in self.compression_algorithms:
            compressed_content = self.compression_algorithms[level](memory.content)
            memory.content = compressed_content
            memory.compression_level = level
        
        return memory
    
    def _light_compression(self, content: str) -> str:
        """轻度压缩：移除多余空格和标点"""
        # 移除多余空格
        content = re.sub(r'\s+', ' ', content).strip()
        # 移除重复标点
        content = re.sub(r'[,.!?]{2,}', lambda m: m.group()[0], content)
        return content
    
    def _medium_compression(self, content: str) -> str:
        """中度压缩：提取关键句子"""
        sentences = re.split(r'[.!?]+', content)
        
        # 计算句子重要性（基于长度和关键词）
        important_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # 过滤太短的句子
                score = len(sentence) + len(re.findall(r'\b[A-Z]\w+\b', sentence)) * 5
                important_sentences.append((sentence.strip(), score))
        
        # 保留重要性最高的句子
        important_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in important_sentences[:3]]
        
        return '. '.join(top_sentences) + '.'
    
    def _heavy_compression(self, content: str) -> str:
        """重度压缩：提取关键词和短语"""
        # 提取关键词
        words = re.findall(r'\b\w{4,}\b', content.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 获取高频关键词
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, freq in top_words]
        
        # 提取命名实体（简化版本）
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # 组合压缩结果
        result = f"关键词: {', '.join(keywords)}"
        if entities:
            result += f" | 实体: {', '.join(entities[:5])}"
        
        return result

class MemoryManagerEnhanced:
    """增强记忆管理器 - 深度优化版"""
    
    def __init__(self, db_path: str = "agent_memory_enhanced.db"):
        self.db_path = db_path
        self.index_manager = MemoryIndexManager()
        self.compression_engine = MemoryCompressionEngine()
        self.memory_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 初始化数据库
        self.init_database()
        
        # 加载索引
        self._rebuild_indexes()
        
        # 启动后台任务
        self._start_background_tasks()
        
        logger.info("增强记忆管理器初始化完成")
    
    def init_database(self):
        """初始化数据库 - 增强版"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建主记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'conversation',
                importance REAL DEFAULT 0.5,
                timestamp REAL NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                
                -- 增强字段
                emotional_valence REAL DEFAULT 0.0,
                emotional_arousal REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_access REAL DEFAULT 0.0,
                similarity_hash TEXT DEFAULT '',
                compression_level INTEGER DEFAULT 0,
                linked_memories TEXT DEFAULT '[]',
                
                -- 深度增强字段
                context_vector TEXT DEFAULT '[]',
                knowledge_entities TEXT DEFAULT '[]',
                temporal_context TEXT DEFAULT '{}',
                spatial_context TEXT DEFAULT '{}',
                confidence_score REAL DEFAULT 1.0,
                decay_factor REAL DEFAULT 1.0,
                consolidation_level INTEGER DEFAULT 0,
                retrieval_strength REAL DEFAULT 1.0,
                interference_resistance REAL DEFAULT 1.0
            )
        """)
        
        # 创建记忆关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id_1 INTEGER,
                memory_id_2 INTEGER,
                association_type TEXT,
                strength REAL DEFAULT 1.0,
                created_at REAL DEFAULT (julianday('now')),
                FOREIGN KEY (memory_id_1) REFERENCES memories_enhanced (id),
                FOREIGN KEY (memory_id_2) REFERENCES memories_enhanced (id)
            )
        """)
        
        # 创建记忆统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_statistics (
                date TEXT PRIMARY KEY,
                total_memories INTEGER DEFAULT 0,
                memories_added INTEGER DEFAULT 0,
                memories_accessed INTEGER DEFAULT 0,
                memories_compressed INTEGER DEFAULT 0,
                avg_importance REAL DEFAULT 0.0,
                avg_retention REAL DEFAULT 0.0
            )
        """)
        
        # 创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_timestamp_enhanced ON memories_enhanced(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_importance_enhanced ON memories_enhanced(importance)",
            "CREATE INDEX IF NOT EXISTS idx_type_enhanced ON memories_enhanced(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_similarity_hash ON memories_enhanced(similarity_hash)",
            "CREATE INDEX IF NOT EXISTS idx_last_access ON memories_enhanced(last_access)",
            "CREATE INDEX IF NOT EXISTS idx_consolidation ON memories_enhanced(consolidation_level)",
            "CREATE INDEX IF NOT EXISTS idx_associations_m1 ON memory_associations(memory_id_1)",
            "CREATE INDEX IF NOT EXISTS idx_associations_m2 ON memory_associations(memory_id_2)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
    
    def add_memory(self, memory: MemoryEnhanced) -> int:
        """添加记忆 - 增强版"""
        # 自动生成标签
        if not memory.tags:
            memory.tags = self._generate_auto_tags(memory.content)
        
        # 提取知识实体
        if not memory.knowledge_entities:
            memory.knowledge_entities = self._extract_entities(memory.content)
        
        # 分析情感
        if memory.emotional_valence == 0.0:
            memory.emotional_valence, memory.emotional_arousal = self._analyze_emotion(memory.content)
        
        # 设置时间上下文
        memory.temporal_context = {
            "hour": datetime.fromtimestamp(memory.timestamp).hour,
            "day_of_week": datetime.fromtimestamp(memory.timestamp).weekday(),
            "season": self._get_season(datetime.fromtimestamp(memory.timestamp))
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories_enhanced (
                content, memory_type, importance, timestamp, tags, metadata,
                emotional_valence, emotional_arousal, access_count, last_access,
                similarity_hash, compression_level, linked_memories,
                context_vector, knowledge_entities, temporal_context, spatial_context,
                confidence_score, decay_factor, consolidation_level,
                retrieval_strength, interference_resistance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.content, memory.memory_type, memory.importance, memory.timestamp,
            json.dumps(memory.tags), json.dumps(memory.metadata),
            memory.emotional_valence, memory.emotional_arousal, memory.access_count,
            memory.last_access, memory.similarity_hash, memory.compression_level,
            json.dumps(memory.linked_memories), json.dumps(memory.context_vector),
            json.dumps(memory.knowledge_entities), json.dumps(memory.temporal_context),
            json.dumps(memory.spatial_context), memory.confidence_score,
            memory.decay_factor, memory.consolidation_level,
            memory.retrieval_strength, memory.interference_resistance
        ))
        
        memory_id = cursor.lastrowid
        memory.id = memory_id
        
        conn.commit()
        conn.close()
        
        # 添加到索引
        self.index_manager.add_memory_to_index(memory)
        
        # 缓存热门记忆
        if memory.importance > 0.7:
            self.memory_cache[memory_id] = memory
        
        # 寻找相似记忆并建立关联
        self._find_and_create_associations(memory)
        
        logger.info(f"添加记忆: ID={memory_id}, 类型={memory.memory_type}, 重要性={memory.importance:.2f}")
        
        return memory_id or 0
    
    def get_memory_by_id(self, memory_id: int) -> Optional[MemoryEnhanced]:
        """根据ID获取记忆"""
        # 首先检查缓存
        if memory_id in self.memory_cache:
            memory = self.memory_cache[memory_id]
            memory.update_access()
            return memory
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories_enhanced WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            memory = self._row_to_memory_enhanced(row)
            memory.update_access()
            
            # 更新访问信息到数据库
            self._update_memory_access(memory_id)
            
            return memory
        
        return None
    
    def search_memories_intelligent(self, 
                                  query: str, 
                                  limit: int = 10,
                                  memory_types: Optional[List[str]] = None,
                                  min_importance: float = 0.0,
                                  time_range: Optional[Tuple[float, float]] = None,
                                  emotional_filter: Optional[Tuple[float, float]] = None) -> List[MemoryEnhanced]:
        """智能记忆搜索 - 多维度搜索"""
        
        # 使用索引进行初步搜索
        candidate_ids = self.index_manager.search_by_content(query, limit * 3)
        
        if not candidate_ids:
            return []
        
        # 从数据库获取候选记忆
        id_list = list(candidate_ids)
        placeholders = ','.join('?' * len(id_list))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        base_query = f"""
            SELECT * FROM memories_enhanced 
            WHERE id IN ({placeholders})
            AND importance >= ?
        """
        
        params = id_list + [min_importance]
        
        # 添加过滤条件
        if memory_types:
            type_placeholders = ','.join('?' * len(memory_types))
            base_query += f" AND memory_type IN ({type_placeholders})"
            params.extend(memory_types)
        
        if time_range:
            base_query += " AND timestamp BETWEEN ? AND ?"
            params.extend(time_range)
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # 转换为记忆对象并计算相关性分数
        memories_with_scores = []
        query_words = set(re.findall(r'\w+', query.lower()))
        
        for row in rows:
            memory = self._row_to_memory_enhanced(row)
            
            # 情感过滤
            if emotional_filter:
                if not (emotional_filter[0] <= memory.emotional_valence <= emotional_filter[1]):
                    continue
            
            # 计算综合相关性分数
            relevance_score = self._calculate_relevance_score(memory, query_words)
            memories_with_scores.append((memory, relevance_score))
        
        # 按相关性排序
        memories_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in memories_with_scores[:limit]]
    
    def _calculate_relevance_score(self, memory: MemoryEnhanced, query_words: Set[str]) -> float:
        """计算记忆相关性分数"""
        # 内容匹配分数
        memory_words = set(re.findall(r'\w+', memory.content.lower()))
        content_overlap = len(query_words.intersection(memory_words))
        content_score = content_overlap / max(len(query_words), 1)
        
        # 重要性分数
        importance_score = memory.importance
        
        # 时间衰减分数（最近的记忆得分更高）
        age_hours = memory.calculate_age_hours()
        time_score = np.exp(-age_hours / 168)  # 一周的衰减
        
        # 访问频率分数
        access_score = min(1.0, memory.access_count / 10)
        
        # 检索强度分数
        retrieval_score = memory.retrieval_strength / 2.0
        
        # 综合分数
        total_score = (
            content_score * 0.4 +
            importance_score * 0.25 +
            time_score * 0.15 +
            access_score * 0.1 +
            retrieval_score * 0.1
        )
        
        return total_score
    
    def consolidate_memories(self, consolidation_threshold: float = 0.5) -> int:
        """记忆整理和巩固"""
        consolidated_count = 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查找需要巩固的记忆
        cursor.execute("""
            SELECT * FROM memories_enhanced 
            WHERE consolidation_level < 3 
            AND (importance >= ? OR access_count >= 5)
            ORDER BY importance DESC, access_count DESC
        """, (consolidation_threshold,))
        
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_memory_enhanced(row)
            
            # 巩固记忆
            if self._should_consolidate(memory):
                memory.consolidation_level += 1
                memory.interference_resistance = min(2.0, memory.interference_resistance + 0.2)
                
                # 更新数据库
                cursor.execute("""
                    UPDATE memories_enhanced 
                    SET consolidation_level = ?, interference_resistance = ?
                    WHERE id = ?
                """, (memory.consolidation_level, memory.interference_resistance, memory.id))
                
                consolidated_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"巩固了 {consolidated_count} 个记忆")
        return consolidated_count
    
    def _should_consolidate(self, memory: MemoryEnhanced) -> bool:
        """判断记忆是否应该被巩固"""
        # 高重要性、高访问频率、或者与其他记忆强关联的记忆应该被巩固
        return (memory.importance > 0.6 or 
                memory.access_count > 3 or
                len(memory.linked_memories) > 2)
    
    def compress_old_memories(self, age_threshold_hours: int = 168) -> int:
        """压缩旧记忆"""
        compressed_count = 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查找需要压缩的记忆
        threshold_timestamp = time.time() - (age_threshold_hours * 3600)
        
        cursor.execute("""
            SELECT * FROM memories_enhanced 
            WHERE timestamp < ? 
            AND compression_level = 0 
            AND importance < 0.7
            AND access_count < 3
        """, (threshold_timestamp,))
        
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_memory_enhanced(row)
            
            if memory.should_be_compressed():
                # 确定压缩级别
                if memory.importance < 0.3:
                    compression_level = 3  # 重度压缩
                elif memory.importance < 0.5:
                    compression_level = 2  # 中度压缩
                else:
                    compression_level = 1  # 轻度压缩
                
                # 压缩记忆
                compressed_memory = self.compression_engine.compress_memory(memory, compression_level)
                
                # 更新数据库
                cursor.execute("""
                    UPDATE memories_enhanced 
                    SET content = ?, compression_level = ?
                    WHERE id = ?
                """, (compressed_memory.content, compressed_memory.compression_level, memory.id))
                
                compressed_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"压缩了 {compressed_count} 个记忆")
        return compressed_count
    
    def _generate_auto_tags(self, content: str) -> List[str]:
        """自动生成标签"""
        tags = []
        
        # 技术相关标签
        tech_keywords = {
            'python': ['python', 'py', '编程', '代码'],
            'ai': ['ai', 'artificial intelligence', '人工智能', 'machine learning', '机器学习'],
            'data': ['data', '数据', 'database', '数据库'],
            'web': ['web', 'website', '网站', 'html', 'css', 'javascript']
        }
        
        content_lower = content.lower()
        for tag, keywords in tech_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        # 情感标签
        if any(word in content_lower for word in ['开心', '高兴', 'happy', 'good', '棒']):
            tags.append('positive')
        elif any(word in content_lower for word in ['难过', '沮丧', 'sad', 'bad', '差']):
            tags.append('negative')
        
        # 问题类型标签
        if any(word in content_lower for word in ['问题', 'problem', 'issue', '错误', 'error']):
            tags.append('problem')
        elif any(word in content_lower for word in ['学习', 'learn', 'study', '教程']):
            tags.append('learning')
        
        return tags
    
    def _extract_entities(self, content: str) -> List[str]:
        """提取知识实体"""
        entities = []
        
        # 简化的命名实体识别
        # 提取大写开头的词组（可能是人名、地名、产品名等）
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        entities.extend(named_entities)
        
        # 技术术语
        tech_terms = re.findall(r'\b(?:Python|JavaScript|HTML|CSS|SQL|API|JSON|XML)\b', content, re.IGNORECASE)
        entities.extend(tech_terms)
        
        # 去重并限制数量
        entities = list(set(entities))[:10]
        
        return entities
    
    def _analyze_emotion(self, content: str) -> Tuple[float, float]:
        """分析情感价值和激活度"""
        positive_words = ['好', '棒', '优秀', '喜欢', '满意', '开心', 'good', 'great', 'excellent', 'love', 'happy', 'amazing']
        negative_words = ['坏', '差', '糟糕', '讨厌', '不满', '难过', 'bad', 'terrible', 'hate', 'awful', 'sad', 'horrible']
        high_arousal_words = ['兴奋', '激动', '紧张', '焦虑', '愤怒', 'excited', 'angry', 'anxious', 'thrilled', 'furious']
        
        words = re.findall(r'\w+', content.lower())
        
        # 计算情感价值 (-1 到 1)
        positive_count = sum(1 for word in words if any(pos in word for pos in positive_words))
        negative_count = sum(1 for word in words if any(neg in word for neg in negative_words))
        
        if positive_count + negative_count > 0:
            valence = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            valence = 0.0
        
        # 计算激活度 (0 到 1)
        arousal_count = sum(1 for word in words if any(arousal in word for arousal in high_arousal_words))
        arousal = min(1.0, arousal_count / max(len(words), 1) * 10)
        
        return valence, arousal
    
    def _get_season(self, dt: datetime) -> str:
        """获取季节"""
        month = dt.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"
    
    def _find_and_create_associations(self, memory: MemoryEnhanced):
        """寻找并创建记忆关联"""
        # 基于相似性哈希寻找相似记忆
        similar_ids = self.index_manager.search_by_similarity(memory.similarity_hash, 5)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for similar_id in similar_ids:
            if similar_id != memory.id:
                # 计算关联强度
                strength = self._calculate_association_strength(memory.id, similar_id)
                
                if strength > 0.3:  # 只保存强关联
                    cursor.execute("""
                        INSERT OR IGNORE INTO memory_associations 
                        (memory_id_1, memory_id_2, association_type, strength, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (memory.id, similar_id, "similarity", strength, time.time()))
        
        conn.commit()
        conn.close()
    
    def _calculate_association_strength(self, id1: int, id2: int) -> float:
        """计算记忆关联强度"""
        # 简化版本：基于内容相似性
        mem1 = self.get_memory_by_id(id1)
        mem2 = self.get_memory_by_id(id2)
        
        if not mem1 or not mem2:
            return 0.0
        
        # 计算向量相似性
        if mem1.context_vector and mem2.context_vector:
            dot_product = sum(a * b for a, b in zip(mem1.context_vector, mem2.context_vector))
            return max(0.0, dot_product)
        
        return 0.0
    
    def _row_to_memory_enhanced(self, row) -> MemoryEnhanced:
        """将数据库行转换为增强记忆对象"""
        return MemoryEnhanced(
            id=row[0], content=row[1], memory_type=row[2], importance=row[3],
            timestamp=row[4], tags=json.loads(row[5]), metadata=json.loads(row[6]),
            emotional_valence=row[7], emotional_arousal=row[8], access_count=row[9],
            last_access=row[10], similarity_hash=row[11], compression_level=row[12],
            linked_memories=json.loads(row[13]), context_vector=json.loads(row[14]),
            knowledge_entities=json.loads(row[15]), temporal_context=json.loads(row[16]),
            spatial_context=json.loads(row[17]), confidence_score=row[18],
            decay_factor=row[19], consolidation_level=row[20],
            retrieval_strength=row[21], interference_resistance=row[22]
        )
    
    def _update_memory_access(self, memory_id: int):
        """更新记忆访问信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories_enhanced 
            SET access_count = access_count + 1, last_access = ?
            WHERE id = ?
        """, (time.time(), memory_id))
        
        conn.commit()
        conn.close()
    
    def _rebuild_indexes(self):
        """重建索引"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories_enhanced")
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_memory_enhanced(row)
            self.index_manager.add_memory_to_index(memory)
        
        conn.close()
        logger.info(f"重建索引完成，索引了 {len(rows)} 个记忆")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 这里可以启动定期的记忆整理和压缩任务
        # 在实际应用中，可以使用 APScheduler 或类似的任务调度器
        pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取全面的记忆统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基础统计
        cursor.execute("SELECT COUNT(*) FROM memories_enhanced")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories_enhanced GROUP BY memory_type")
        type_stats = dict(cursor.fetchall())
        
        cursor.execute("SELECT AVG(importance) FROM memories_enhanced")
        avg_importance = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(access_count) FROM memories_enhanced")
        avg_access = cursor.fetchone()[0] or 0.0
        
        # 压缩统计
        cursor.execute("SELECT compression_level, COUNT(*) FROM memories_enhanced GROUP BY compression_level")
        compression_stats = dict(cursor.fetchall())
        
        # 情感统计
        cursor.execute("SELECT AVG(emotional_valence), AVG(emotional_arousal) FROM memories_enhanced")
        emotion_stats = cursor.fetchone()
        
        # 关联统计
        cursor.execute("SELECT COUNT(*) FROM memory_associations")
        association_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_memories": total_count,
            "type_distribution": type_stats,
            "average_importance": round(avg_importance, 3),
            "average_access_count": round(avg_access, 2),
            "compression_distribution": compression_stats,
            "emotion_statistics": {
                "average_valence": round(emotion_stats[0] or 0.0, 3),
                "average_arousal": round(emotion_stats[1] or 0.0, 3)
            },
            "total_associations": association_count,
            "cache_size": len(self.memory_cache),
            "database_path": self.db_path
        }
    
    def cleanup(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("记忆管理器已清理")

# 全局增强记忆管理器实例
memory_manager_enhanced = MemoryManagerEnhanced()