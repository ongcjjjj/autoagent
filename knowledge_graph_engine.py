"""
智能知识图谱与推理引擎
实现语义推理、知识发现、智能问答、动态知识构建
"""
import json
import time
import math
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class RelationType(Enum):
    """关系类型"""
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"

class NodeType(Enum):
    """节点类型"""
    CONCEPT = "concept"
    ENTITY = "entity"
    PROPERTY = "property"
    EVENT = "event"
    RULE = "rule"
    FACT = "fact"

@dataclass
class KnowledgeNode:
    """知识节点"""
    node_id: str
    label: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 0.5
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeRelation:
    """知识关系"""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceRule:
    """推理规则"""
    rule_id: str
    name: str
    premise_pattern: Dict[str, Any]
    conclusion_pattern: Dict[str, Any]
    confidence_factor: float = 0.8
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryResult:
    """查询结果"""
    query: str
    results: List[Dict[str, Any]]
    confidence: float
    reasoning_path: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    total_explored_nodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticEmbedder:
    """语义嵌入器"""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.word_vectors = {}
        self.trained_embeddings = {}
        
    def generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 简化的文本嵌入实现
        words = text.lower().split()
        
        if not words:
            return [0.0] * self.dimension
        
        # 使用哈希函数生成伪随机但一致的向量
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # 转换哈希为向量
        vector = []
        for i in range(self.dimension):
            hash_part = text_hash[(i * 2) % len(text_hash):(i * 2 + 2) % len(text_hash) + 1]
            if len(hash_part) == 2:
                value = int(hash_part, 16) / 255.0 * 2 - 1  # 归一化到 [-1, 1]
            else:
                value = 0.0
            vector.append(value)
        
        # 归一化向量
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算嵌入向量相似度"""
        if len(embedding1) != len(embedding2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        return max(0.0, dot_product)  # 余弦相似度的简化版本

class KnowledgeGraphStore:
    """知识图谱存储"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.node_relations: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[RelationType, Set[str]] = defaultdict(set)
        self.embedder = SemanticEmbedder()
        
    def add_node(self, node: KnowledgeNode) -> bool:
        """添加节点"""
        if node.node_id in self.nodes:
            # 更新现有节点
            existing = self.nodes[node.node_id]
            existing.properties.update(node.properties)
            existing.last_updated = time.time()
            existing.confidence = max(existing.confidence, node.confidence)
            return True
        
        # 生成嵌入向量
        if not node.embeddings:
            text_for_embedding = f"{node.label} {' '.join(str(v) for v in node.properties.values())}"
            node.embeddings = self.embedder.generate_embedding(text_for_embedding)
        
        self.nodes[node.node_id] = node
        logger.info(f"Added knowledge node: {node.node_id} ({node.label})")
        return True
    
    def add_relation(self, relation: KnowledgeRelation) -> bool:
        """添加关系"""
        if relation.source_id not in self.nodes or relation.target_id not in self.nodes:
            logger.warning(f"Cannot add relation {relation.relation_id}: missing nodes")
            return False
        
        self.relations[relation.relation_id] = relation
        self.node_relations[relation.source_id].add(relation.relation_id)
        self.node_relations[relation.target_id].add(relation.relation_id)
        self.relation_index[relation.relation_type].add(relation.relation_id)
        
        logger.info(f"Added relation: {relation.source_id} -{relation.relation_type.value}-> {relation.target_id}")
        return True
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """按类型查找节点"""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def find_similar_nodes(self, query_node: KnowledgeNode, threshold: float = 0.7) -> List[Tuple[KnowledgeNode, float]]:
        """查找相似节点"""
        if not query_node.embeddings:
            return []
        
        similar_nodes = []
        for node in self.nodes.values():
            if node.node_id != query_node.node_id and node.embeddings:
                similarity = self.embedder.calculate_similarity(query_node.embeddings, node.embeddings)
                if similarity >= threshold:
                    similar_nodes.append((node, similarity))
        
        return sorted(similar_nodes, key=lambda x: x[1], reverse=True)
    
    def get_node_neighbors(self, node_id: str, relation_types: Optional[List[RelationType]] = None) -> List[KnowledgeNode]:
        """获取节点的邻居"""
        neighbors = []
        
        if node_id not in self.node_relations:
            return neighbors
        
        for relation_id in self.node_relations[node_id]:
            relation = self.relations[relation_id]
            
            if relation_types and relation.relation_type not in relation_types:
                continue
            
            # 找到邻居节点
            neighbor_id = relation.target_id if relation.source_id == node_id else relation.source_id
            if neighbor_id in self.nodes:
                neighbors.append(self.nodes[neighbor_id])
        
        return neighbors
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> Optional[List[str]]:
        """查找两个节点之间的路径"""
        if source_id == target_id:
            return [source_id]
        
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        # 广度优先搜索
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            # 获取邻居
            neighbors = self.get_node_neighbors(current_id)
            
            for neighbor in neighbors:
                if neighbor.node_id == target_id:
                    return path + [neighbor.node_id]
                
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    queue.append((neighbor.node_id, path + [neighbor.node_id]))
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        node_type_counts = defaultdict(int)
        relation_type_counts = defaultdict(int)
        
        for node in self.nodes.values():
            node_type_counts[node.node_type.value] += 1
        
        for relation in self.relations.values():
            relation_type_counts[relation.relation_type.value] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_relations": len(self.relations),
            "node_type_distribution": dict(node_type_counts),
            "relation_type_distribution": dict(relation_type_counts),
            "average_node_degree": sum(len(relations) for relations in self.node_relations.values()) / max(len(self.nodes), 1)
        }

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        self.knowledge_store = knowledge_store
        self.inference_rules: Dict[str, InferenceRule] = {}
        self.inference_cache = {}
        self.reasoning_history = deque(maxlen=1000)
        
        # 初始化基础推理规则
        self._initialize_basic_rules()
    
    def _initialize_basic_rules(self):
        """初始化基础推理规则"""
        # 传递性规则
        transitivity_rule = InferenceRule(
            rule_id="transitivity_is_a",
            name="Is-A传递性",
            premise_pattern={"relation_type": RelationType.IS_A, "chain_length": 2},
            conclusion_pattern={"relation_type": RelationType.IS_A, "confidence_factor": 0.8}
        )
        self.add_inference_rule(transitivity_rule)
        
        # 相似性推理
        similarity_rule = InferenceRule(
            rule_id="similarity_property_inheritance",
            name="相似性属性继承",
            premise_pattern={"relation_type": RelationType.SIMILAR_TO, "property_exists": True},
            conclusion_pattern={"relation_type": RelationType.HAS_PROPERTY, "confidence_factor": 0.6}
        )
        self.add_inference_rule(similarity_rule)
        
        # 因果推理
        causal_rule = InferenceRule(
            rule_id="causal_chain",
            name="因果链推理",
            premise_pattern={"relation_type": RelationType.CAUSES, "chain_length": 2},
            conclusion_pattern={"relation_type": RelationType.CAUSES, "confidence_factor": 0.7}
        )
        self.add_inference_rule(causal_rule)
    
    def add_inference_rule(self, rule: InferenceRule):
        """添加推理规则"""
        self.inference_rules[rule.rule_id] = rule
        logger.info(f"Added inference rule: {rule.name}")
    
    def infer_new_knowledge(self, max_iterations: int = 5) -> List[KnowledgeRelation]:
        """推理新知识"""
        new_relations = []
        
        for iteration in range(max_iterations):
            iteration_relations = []
            
            for rule in self.inference_rules.values():
                rule_relations = self._apply_inference_rule(rule)
                iteration_relations.extend(rule_relations)
            
            if not iteration_relations:
                break  # 没有新推理出的关系
            
            # 添加新关系到知识图谱
            for relation in iteration_relations:
                if self.knowledge_store.add_relation(relation):
                    new_relations.append(relation)
        
        logger.info(f"Inferred {len(new_relations)} new relations")
        return new_relations
    
    def _apply_inference_rule(self, rule: InferenceRule) -> List[KnowledgeRelation]:
        """应用推理规则"""
        new_relations = []
        
        try:
            if rule.rule_id == "transitivity_is_a":
                new_relations = self._apply_transitivity_rule(RelationType.IS_A)
            elif rule.rule_id == "similarity_property_inheritance":
                new_relations = self._apply_similarity_inheritance_rule()
            elif rule.rule_id == "causal_chain":
                new_relations = self._apply_transitivity_rule(RelationType.CAUSES)
            
            # 更新规则使用统计
            rule.usage_count += len(new_relations)
            rule.last_used = time.time()
            
        except Exception as e:
            logger.error(f"Error applying inference rule {rule.rule_id}: {e}")
        
        return new_relations
    
    def _apply_transitivity_rule(self, relation_type: RelationType) -> List[KnowledgeRelation]:
        """应用传递性规则"""
        new_relations = []
        
        # 找到所有指定类型的关系
        target_relations = [
            rel for rel in self.knowledge_store.relations.values()
            if rel.relation_type == relation_type
        ]
        
        # 查找传递性路径
        for rel1 in target_relations:
            for rel2 in target_relations:
                if rel1.target_id == rel2.source_id and rel1.source_id != rel2.target_id:
                    # 检查是否已存在直接关系
                    direct_relation_exists = any(
                        rel.source_id == rel1.source_id and rel.target_id == rel2.target_id
                        for rel in target_relations
                    )
                    
                    if not direct_relation_exists:
                        # 创建新的推理关系
                        new_confidence = min(rel1.confidence, rel2.confidence) * 0.8
                        
                        new_relation = KnowledgeRelation(
                            relation_id=f"inferred_{relation_type.value}_{rel1.source_id}_{rel2.target_id}",
                            source_id=rel1.source_id,
                            target_id=rel2.target_id,
                            relation_type=relation_type,
                            confidence=new_confidence,
                            evidence=[rel1.relation_id, rel2.relation_id],
                            metadata={"inference_type": "transitivity", "rule_id": "transitivity_is_a"}
                        )
                        
                        new_relations.append(new_relation)
        
        return new_relations
    
    def _apply_similarity_inheritance_rule(self) -> List[KnowledgeRelation]:
        """应用相似性属性继承规则"""
        new_relations = []
        
        # 找到相似关系
        similarity_relations = [
            rel for rel in self.knowledge_store.relations.values()
            if rel.relation_type == RelationType.SIMILAR_TO
        ]
        
        # 找到属性关系
        property_relations = [
            rel for rel in self.knowledge_store.relations.values()
            if rel.relation_type == RelationType.HAS_PROPERTY
        ]
        
        for sim_rel in similarity_relations:
            for prop_rel in property_relations:
                # 如果A相似B，A有属性P，则推理B可能也有属性P
                if sim_rel.source_id == prop_rel.source_id:
                    # 检查B是否已经有这个属性
                    existing_property = any(
                        rel.source_id == sim_rel.target_id and rel.target_id == prop_rel.target_id
                        for rel in property_relations
                    )
                    
                    if not existing_property:
                        new_confidence = min(sim_rel.confidence, prop_rel.confidence) * 0.6
                        
                        new_relation = KnowledgeRelation(
                            relation_id=f"inferred_property_{sim_rel.target_id}_{prop_rel.target_id}",
                            source_id=sim_rel.target_id,
                            target_id=prop_rel.target_id,
                            relation_type=RelationType.HAS_PROPERTY,
                            confidence=new_confidence,
                            evidence=[sim_rel.relation_id, prop_rel.relation_id],
                            metadata={"inference_type": "similarity_inheritance"}
                        )
                        
                        new_relations.append(new_relation)
        
        return new_relations
    
    def answer_query(self, query: str, max_results: int = 10) -> QueryResult:
        """回答查询"""
        start_time = time.time()
        
        # 解析查询
        parsed_query = self._parse_query(query)
        
        # 执行查询
        results = []
        reasoning_path = []
        explored_nodes = 0
        
        if parsed_query["type"] == "find_entity":
            results, path, explored = self._find_entity_query(parsed_query, max_results)
            reasoning_path = path
            explored_nodes = explored
        elif parsed_query["type"] == "relationship_query":
            results, path, explored = self._relationship_query(parsed_query, max_results)
            reasoning_path = path
            explored_nodes = explored
        elif parsed_query["type"] == "inference_query":
            results, path, explored = self._inference_query(parsed_query, max_results)
            reasoning_path = path
            explored_nodes = explored
        
        # 计算置信度
        confidence = self._calculate_query_confidence(results, reasoning_path)
        
        execution_time = time.time() - start_time
        
        query_result = QueryResult(
            query=query,
            results=results,
            confidence=confidence,
            reasoning_path=reasoning_path,
            execution_time=execution_time,
            total_explored_nodes=explored_nodes
        )
        
        # 记录推理历史
        self.reasoning_history.append({
            "query": query,
            "results_count": len(results),
            "confidence": confidence,
            "execution_time": execution_time,
            "timestamp": time.time()
        })
        
        return query_result
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """解析查询"""
        query_lower = query.lower().strip()
        
        # 简化的查询解析
        if "what is" in query_lower or "define" in query_lower:
            entity = query_lower.replace("what is", "").replace("define", "").strip()
            return {"type": "find_entity", "entity": entity}
        
        elif "relationship" in query_lower or "related to" in query_lower:
            parts = query_lower.split()
            if len(parts) >= 3:
                return {"type": "relationship_query", "entities": parts[1:3]}
        
        elif "why" in query_lower or "how" in query_lower or "infer" in query_lower:
            return {"type": "inference_query", "question": query}
        
        # 默认实体查找
        return {"type": "find_entity", "entity": query_lower}
    
    def _find_entity_query(self, parsed_query: Dict[str, Any], max_results: int) -> Tuple[List[Dict[str, Any]], List[str], int]:
        """执行实体查找查询"""
        entity_name = parsed_query["entity"]
        results = []
        reasoning_path = [f"搜索实体: {entity_name}"]
        explored_nodes = 0
        
        # 精确匹配
        for node in self.knowledge_store.nodes.values():
            if entity_name in node.label.lower():
                node.access_count += 1
                results.append({
                    "node_id": node.node_id,
                    "label": node.label,
                    "type": node.node_type.value,
                    "properties": node.properties,
                    "confidence": node.confidence,
                    "match_type": "exact"
                })
                explored_nodes += 1
        
        # 如果精确匹配结果不足，使用语义相似度
        if len(results) < max_results:
            query_embedding = self.knowledge_store.embedder.generate_embedding(entity_name)
            
            for node in self.knowledge_store.nodes.values():
                if node.embeddings and entity_name not in node.label.lower():
                    similarity = self.knowledge_store.embedder.calculate_similarity(
                        query_embedding, node.embeddings
                    )
                    
                    if similarity > 0.6:  # 相似度阈值
                        results.append({
                            "node_id": node.node_id,
                            "label": node.label,
                            "type": node.node_type.value,
                            "properties": node.properties,
                            "confidence": node.confidence * similarity,
                            "match_type": "semantic",
                            "similarity": similarity
                        })
                        explored_nodes += 1
        
        reasoning_path.append(f"找到 {len(results)} 个匹配结果")
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:max_results], reasoning_path, explored_nodes
    
    def _relationship_query(self, parsed_query: Dict[str, Any], max_results: int) -> Tuple[List[Dict[str, Any]], List[str], int]:
        """执行关系查询"""
        entities = parsed_query["entities"]
        results = []
        reasoning_path = [f"查找关系: {' 和 '.join(entities)}"]
        explored_nodes = 0
        
        # 找到相关实体
        entity_nodes = []
        for entity_name in entities:
            for node in self.knowledge_store.nodes.values():
                if entity_name in node.label.lower():
                    entity_nodes.append(node)
                    explored_nodes += 1
                    break
        
        if len(entity_nodes) >= 2:
            # 查找实体间的关系
            for i, node1 in enumerate(entity_nodes):
                for node2 in entity_nodes[i+1:]:
                    # 直接关系
                    for relation_id in self.knowledge_store.node_relations[node1.node_id]:
                        relation = self.knowledge_store.relations[relation_id]
                        if relation.target_id == node2.node_id or relation.source_id == node2.node_id:
                            results.append({
                                "source": node1.label,
                                "target": node2.label,
                                "relation": relation.relation_type.value,
                                "confidence": relation.confidence,
                                "evidence": relation.evidence,
                                "type": "direct"
                            })
                    
                    # 间接关系（通过路径）
                    path = self.knowledge_store.find_path(node1.node_id, node2.node_id, max_depth=3)
                    if path and len(path) > 2:
                        results.append({
                            "source": node1.label,
                            "target": node2.label,
                            "relation": "间接关联",
                            "confidence": 0.7,
                            "path_length": len(path) - 1,
                            "type": "indirect"
                        })
        
        reasoning_path.append(f"分析了 {len(entity_nodes)} 个实体，找到 {len(results)} 个关系")
        
        return results[:max_results], reasoning_path, explored_nodes
    
    def _inference_query(self, parsed_query: Dict[str, Any], max_results: int) -> Tuple[List[Dict[str, Any]], List[str], int]:
        """执行推理查询"""
        question = parsed_query["question"]
        results = []
        reasoning_path = [f"推理分析: {question}"]
        explored_nodes = 0
        
        # 简化的推理实现
        # 基于问题关键词找到相关节点
        keywords = question.lower().split()
        relevant_nodes = []
        
        for node in self.knowledge_store.nodes.values():
            node_text = f"{node.label} {' '.join(str(v) for v in node.properties.values())}".lower()
            
            keyword_matches = sum(1 for keyword in keywords if keyword in node_text)
            if keyword_matches > 0:
                relevant_nodes.append((node, keyword_matches))
                explored_nodes += 1
        
        # 按匹配度排序
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # 基于相关节点生成推理结果
        for node, match_count in relevant_nodes[:max_results]:
            # 获取节点的关系
            related_info = []
            for relation_id in self.knowledge_store.node_relations[node.node_id]:
                relation = self.knowledge_store.relations[relation_id]
                target_node = self.knowledge_store.nodes.get(relation.target_id)
                if target_node:
                    related_info.append(f"{relation.relation_type.value}: {target_node.label}")
            
            results.append({
                "node": node.label,
                "reasoning": f"基于 {match_count} 个关键词匹配",
                "related_concepts": related_info[:5],
                "confidence": min(1.0, match_count / len(keywords)),
                "node_type": node.node_type.value
            })
        
        reasoning_path.append(f"分析了 {explored_nodes} 个节点，生成 {len(results)} 个推理结果")
        
        return results, reasoning_path, explored_nodes
    
    def _calculate_query_confidence(self, results: List[Dict[str, Any]], reasoning_path: List[str]) -> float:
        """计算查询置信度"""
        if not results:
            return 0.0
        
        # 基于结果的平均置信度
        confidences = [r.get("confidence", 0.5) for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 基于推理路径长度的调整
        path_factor = max(0.5, 1.0 - (len(reasoning_path) - 2) * 0.1)
        
        return min(1.0, avg_confidence * path_factor)

class KnowledgeDiscoveryEngine:
    """知识发现引擎"""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        self.knowledge_store = knowledge_store
        self.discovery_patterns = []
        self.discovered_knowledge = deque(maxlen=1000)
        
    def discover_patterns(self) -> List[Dict[str, Any]]:
        """发现知识模式"""
        patterns = []
        
        # 发现频繁共现模式
        cooccurrence_patterns = self._discover_cooccurrence_patterns()
        patterns.extend(cooccurrence_patterns)
        
        # 发现异常模式
        anomaly_patterns = self._discover_anomaly_patterns()
        patterns.extend(anomaly_patterns)
        
        # 发现层次结构模式
        hierarchy_patterns = self._discover_hierarchy_patterns()
        patterns.extend(hierarchy_patterns)
        
        return patterns
    
    def _discover_cooccurrence_patterns(self) -> List[Dict[str, Any]]:
        """发现共现模式"""
        patterns = []
        
        # 统计关系类型的共现
        relation_pairs = defaultdict(int)
        
        for node_id in self.knowledge_store.nodes:
            relations = [
                self.knowledge_store.relations[rel_id].relation_type
                for rel_id in self.knowledge_store.node_relations[node_id]
            ]
            
            # 计算关系对的共现
            for i, rel1 in enumerate(relations):
                for rel2 in relations[i+1:]:
                    pair = tuple(sorted([rel1.value, rel2.value]))
                    relation_pairs[pair] += 1
        
        # 找到频繁共现的关系对
        for (rel1, rel2), count in relation_pairs.items():
            if count >= 3:  # 阈值
                patterns.append({
                    "type": "cooccurrence",
                    "pattern": f"{rel1} 与 {rel2} 共现",
                    "frequency": count,
                    "confidence": min(1.0, count / 10.0)
                })
        
        return patterns
    
    def _discover_anomaly_patterns(self) -> List[Dict[str, Any]]:
        """发现异常模式"""
        patterns = []
        
        # 发现度数异常的节点
        node_degrees = {
            node_id: len(relations)
            for node_id, relations in self.knowledge_store.node_relations.items()
        }
        
        if node_degrees:
            avg_degree = sum(node_degrees.values()) / len(node_degrees)
            
            for node_id, degree in node_degrees.items():
                if degree > avg_degree * 3:  # 度数异常高
                    node = self.knowledge_store.nodes[node_id]
                    patterns.append({
                        "type": "anomaly",
                        "pattern": f"高连接度节点: {node.label}",
                        "degree": degree,
                        "avg_degree": avg_degree,
                        "anomaly_score": degree / avg_degree
                    })
        
        return patterns
    
    def _discover_hierarchy_patterns(self) -> List[Dict[str, Any]]:
        """发现层次结构模式"""
        patterns = []
        
        # 分析IS_A关系形成的层次
        is_a_relations = [
            rel for rel in self.knowledge_store.relations.values()
            if rel.relation_type == RelationType.IS_A
        ]
        
        # 构建层次树
        hierarchy_levels = defaultdict(set)
        
        for relation in is_a_relations:
            source_node = self.knowledge_store.nodes[relation.source_id]
            target_node = self.knowledge_store.nodes[relation.target_id]
            
            # 简化的层次级别分配
            hierarchy_levels[0].add(source_node.node_id)
            hierarchy_levels[1].add(target_node.node_id)
        
        if len(hierarchy_levels) > 1:
            patterns.append({
                "type": "hierarchy",
                "pattern": "IS_A层次结构",
                "levels": len(hierarchy_levels),
                "nodes_per_level": {
                    level: len(nodes) for level, nodes in hierarchy_levels.items()
                }
            })
        
        return patterns

class KnowledgeGraphEngine:
    """知识图谱引擎主类"""
    
    def __init__(self):
        self.knowledge_store = KnowledgeGraphStore()
        self.inference_engine = InferenceEngine(self.knowledge_store)
        self.discovery_engine = KnowledgeDiscoveryEngine(self.knowledge_store)
        
        self.auto_inference_enabled = True
        self.last_inference_time = 0
        self.inference_interval = 300  # 5分钟
        
    def add_knowledge(self, text: str, source: str = "user_input") -> Dict[str, Any]:
        """从文本添加知识"""
        # 简化的知识提取
        extracted_knowledge = self._extract_knowledge_from_text(text)
        
        added_nodes = 0
        added_relations = 0
        
        # 添加节点
        for node_data in extracted_knowledge.get("nodes", []):
            node = KnowledgeNode(
                node_id=node_data["id"],
                label=node_data["label"],
                node_type=NodeType(node_data.get("type", "concept")),
                properties=node_data.get("properties", {}),
                metadata={"source": source}
            )
            
            if self.knowledge_store.add_node(node):
                added_nodes += 1
        
        # 添加关系
        for relation_data in extracted_knowledge.get("relations", []):
            relation = KnowledgeRelation(
                relation_id=relation_data["id"],
                source_id=relation_data["source"],
                target_id=relation_data["target"],
                relation_type=RelationType(relation_data.get("type", "related_to")),
                confidence=relation_data.get("confidence", 0.8),
                metadata={"source": source}
            )
            
            if self.knowledge_store.add_relation(relation):
                added_relations += 1
        
        # 触发自动推理
        if self.auto_inference_enabled:
            self._trigger_auto_inference()
        
        return {
            "added_nodes": added_nodes,
            "added_relations": added_relations,
            "total_nodes": len(self.knowledge_store.nodes),
            "total_relations": len(self.knowledge_store.relations)
        }
    
    def _extract_knowledge_from_text(self, text: str) -> Dict[str, Any]:
        """从文本提取知识"""
        # 简化的知识提取实现
        words = text.lower().split()
        
        nodes = []
        relations = []
        
        # 提取名词作为概念
        important_words = [word for word in words if len(word) > 3]
        
        for i, word in enumerate(important_words):
            # 创建概念节点
            node_id = f"concept_{word}"
            nodes.append({
                "id": node_id,
                "label": word,
                "type": "concept",
                "properties": {"extracted_from": text[:50]}
            })
            
            # 创建相邻词的关系
            if i > 0:
                prev_word = important_words[i-1]
                relation_id = f"rel_{prev_word}_{word}"
                relations.append({
                    "id": relation_id,
                    "source": f"concept_{prev_word}",
                    "target": node_id,
                    "type": "related_to",
                    "confidence": 0.6
                })
        
        return {"nodes": nodes, "relations": relations}
    
    def _trigger_auto_inference(self):
        """触发自动推理"""
        current_time = time.time()
        
        if current_time - self.last_inference_time > self.inference_interval:
            new_relations = self.inference_engine.infer_new_knowledge(max_iterations=2)
            
            if new_relations:
                logger.info(f"Auto-inference generated {len(new_relations)} new relations")
            
            self.last_inference_time = current_time
    
    def query_knowledge(self, query: str) -> QueryResult:
        """查询知识"""
        return self.inference_engine.answer_query(query)
    
    def discover_knowledge_patterns(self) -> List[Dict[str, Any]]:
        """发现知识模式"""
        return self.discovery_engine.discover_patterns()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        store_stats = self.knowledge_store.get_statistics()
        
        return {
            "knowledge_store": store_stats,
            "inference_rules": len(self.inference_engine.inference_rules),
            "auto_inference_enabled": self.auto_inference_enabled,
            "last_inference": self.last_inference_time,
            "reasoning_history_size": len(self.inference_engine.reasoning_history),
            "discovered_patterns": len(self.discovery_engine.discovered_knowledge)
        }
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        """导出知识图谱"""
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "label": node.label,
                    "type": node.node_type.value,
                    "properties": node.properties,
                    "confidence": node.confidence
                }
                for node in self.knowledge_store.nodes.values()
            ],
            "relations": [
                {
                    "id": rel.relation_id,
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.relation_type.value,
                    "weight": rel.weight,
                    "confidence": rel.confidence
                }
                for rel in self.knowledge_store.relations.values()
            ],
            "metadata": {
                "export_time": time.time(),
                "total_nodes": len(self.knowledge_store.nodes),
                "total_relations": len(self.knowledge_store.relations)
            }
        }