# 自主进化Agent - 第5轮提升：创造力引擎
# Creativity Engine - 创意生成与跨域创新系统

import asyncio
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreativityType(Enum):
    """创造力类型枚举"""
    COMBINATORIAL = "combinatorial"    # 组合式创造
    EXPLORATORY = "exploratory"        # 探索式创造
    TRANSFORMATIONAL = "transformational"  # 转换式创造
    ANALOGICAL = "analogical"          # 类比式创造
    EMERGENT = "emergent"              # 涌现式创造

class CreativeField(Enum):
    """创意领域枚举"""
    ART = "art"
    LITERATURE = "literature"
    MUSIC = "music"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    DESIGN = "design"
    BUSINESS = "business"
    EDUCATION = "education"
    PHILOSOPHY = "philosophy"

class OriginalityLevel(Enum):
    """原创性水平枚举"""
    INCREMENTAL = 0.3      # 渐进性创新
    MODERATE = 0.6         # 中等创新
    RADICAL = 0.9          # 根本性创新

@dataclass
class CreativeIdea:
    """创意想法数据结构"""
    idea_id: str
    content: str
    field: CreativeField
    creativity_type: CreativityType
    originality: float
    feasibility: float
    value: float
    components: List[str] = field(default_factory=list)
    inspirations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class KnowledgeNode:
    """知识节点数据结构"""
    node_id: str
    content: str
    domain: str
    concepts: List[str]
    connections: List[str] = field(default_factory=list)
    usage_frequency: int = 0

class ConceptualBlendingEngine:
    """概念混合引擎"""
    
    def __init__(self):
        self.concept_spaces = defaultdict(list)
        self.blending_patterns = self._initialize_blending_patterns()
        
    def _initialize_blending_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化混合模式"""
        return {
            "metaphorical": {
                "description": "隐喻式混合",
                "strength": 0.8,
                "examples": ["生活如戏", "时间如流水"]
            },
            "functional": {
                "description": "功能式混合",
                "strength": 0.7,
                "examples": ["智能手机", "电子书"]
            },
            "structural": {
                "description": "结构式混合",
                "strength": 0.6,
                "examples": ["网络社交", "虚拟现实"]
            },
            "causal": {
                "description": "因果式混合",
                "strength": 0.75,
                "examples": ["环保能源", "预防医学"]
            }
        }
        
    def add_concept(self, concept: str, domain: str, properties: List[str]):
        """添加概念到概念空间"""
        concept_data = {
            'concept': concept,
            'domain': domain,
            'properties': properties,
            'timestamp': time.time()
        }
        self.concept_spaces[domain].append(concept_data)
        
    def blend_concepts(self, concept1: Dict[str, Any], concept2: Dict[str, Any], 
                      blend_type: str = "metaphorical") -> Dict[str, Any]:
        """混合两个概念"""
        if blend_type not in self.blending_patterns:
            blend_type = "metaphorical"
            
        pattern = self.blending_patterns[blend_type]
        
        # 找到共同属性和互补属性
        properties1 = set(concept1.get('properties', []))
        properties2 = set(concept2.get('properties', []))
        
        common_properties = properties1 & properties2
        unique_properties1 = properties1 - properties2
        unique_properties2 = properties2 - properties1
        
        # 生成混合概念
        blended_concept = {
            'source_concepts': [concept1['concept'], concept2['concept']],
            'source_domains': [concept1['domain'], concept2['domain']],
            'blend_type': blend_type,
            'common_properties': list(common_properties),
            'inherited_properties': list(unique_properties1 | unique_properties2),
            'emergent_properties': self._generate_emergent_properties(
                concept1, concept2, blend_type
            ),
            'strength': pattern['strength'],
            'description': self._generate_blend_description(concept1, concept2, blend_type)
        }
        
        return blended_concept
        
    def _generate_emergent_properties(self, concept1: Dict[str, Any], 
                                    concept2: Dict[str, Any], blend_type: str) -> List[str]:
        """生成涌现属性"""
        emergent_props = []
        
        if blend_type == "functional":
            # 功能混合可能产生新功能
            emergent_props = ["多功能性", "集成化", "智能化"]
        elif blend_type == "metaphorical":
            # 隐喻混合可能产生新的表达方式
            emergent_props = ["象征性", "表达力", "情感共鸣"]
        elif blend_type == "structural":
            # 结构混合可能产生新的组织形式
            emergent_props = ["层次性", "网络化", "模块化"]
        elif blend_type == "causal":
            # 因果混合可能产生新的因果关系
            emergent_props = ["系统性", "连锁反应", "预测性"]
            
        return emergent_props
        
    def _generate_blend_description(self, concept1: Dict[str, Any], 
                                  concept2: Dict[str, Any], blend_type: str) -> str:
        """生成混合描述"""
        name1 = concept1['concept']
        name2 = concept2['concept']
        domain1 = concept1['domain']
        domain2 = concept2['domain']
        
        if blend_type == "metaphorical":
            return f"将{domain1}中的{name1}与{domain2}中的{name2}进行隐喻性结合"
        elif blend_type == "functional":
            return f"融合{name1}和{name2}的功能特性"
        elif blend_type == "structural":
            return f"结合{name1}和{name2}的结构特点"
        else:
            return f"{name1}与{name2}的创新性结合"

class AnalogicalReasoningEngine:
    """类比推理引擎"""
    
    def __init__(self):
        self.analogy_database = []
        self.similarity_threshold = 0.6
        
    def find_analogies(self, source_problem: Dict[str, Any], 
                      target_domains: List[str]) -> List[Dict[str, Any]]:
        """寻找类比"""
        analogies = []
        
        # 从目标领域中寻找结构相似的问题
        for domain in target_domains:
            domain_analogies = self._search_domain_analogies(source_problem, domain)
            analogies.extend(domain_analogies)
            
        # 按相似度排序
        analogies.sort(key=lambda x: x['similarity'], reverse=True)
        
        return analogies[:5]  # 返回最相似的5个类比
        
    def _search_domain_analogies(self, source_problem: Dict[str, Any], 
                                target_domain: str) -> List[Dict[str, Any]]:
        """在特定领域中搜索类比"""
        # 这里使用简化的类比搜索
        domain_problems = self._get_domain_problems(target_domain)
        analogies = []
        
        for target_problem in domain_problems:
            similarity = self._compute_structural_similarity(source_problem, target_problem)
            
            if similarity >= self.similarity_threshold:
                analogy = {
                    'source': source_problem,
                    'target': target_problem,
                    'similarity': similarity,
                    'mapping': self._create_analogy_mapping(source_problem, target_problem),
                    'insights': self._extract_analogical_insights(source_problem, target_problem)
                }
                analogies.append(analogy)
                
        return analogies
        
    def _get_domain_problems(self, domain: str) -> List[Dict[str, Any]]:
        """获取特定领域的问题集合"""
        # 预定义的领域问题库
        domain_problems = {
            "biology": [
                {"problem": "鸟类飞行", "structure": ["轻量", "动力", "控制"], "solution": "翅膀设计"},
                {"problem": "鱼类游泳", "structure": ["流线型", "推进", "平衡"], "solution": "鳍和尾巴"},
                {"problem": "植物光合作用", "structure": ["能量转换", "材料合成", "废物处理"], "solution": "叶绿体"}
            ],
            "physics": [
                {"problem": "波的传播", "structure": ["介质", "频率", "振幅"], "solution": "波动方程"},
                {"problem": "电路设计", "structure": ["电源", "阻抗", "控制"], "solution": "电路理论"},
                {"problem": "热传导", "structure": ["温差", "材料", "时间"], "solution": "传导定律"}
            ],
            "engineering": [
                {"problem": "桥梁设计", "structure": ["载荷", "材料", "稳定性"], "solution": "结构工程"},
                {"problem": "机械传动", "structure": ["动力", "传递", "控制"], "solution": "齿轮系统"},
                {"problem": "通信系统", "structure": ["发送", "传输", "接收"], "solution": "信号处理"}
            ]
        }
        
        return domain_problems.get(domain, [])
        
    def _compute_structural_similarity(self, problem1: Dict[str, Any], 
                                     problem2: Dict[str, Any]) -> float:
        """计算结构相似度"""
        struct1 = set(problem1.get('structure', []))
        struct2 = set(problem2.get('structure', []))
        
        if not struct1 or not struct2:
            return 0.0
            
        intersection = struct1 & struct2
        union = struct1 | struct2
        
        return len(intersection) / len(union)
        
    def _create_analogy_mapping(self, source: Dict[str, Any], 
                              target: Dict[str, Any]) -> Dict[str, str]:
        """创建类比映射"""
        mapping = {}
        
        source_struct = source.get('structure', [])
        target_struct = target.get('structure', [])
        
        # 简单的一对一映射
        for i, (s_elem, t_elem) in enumerate(zip(source_struct, target_struct)):
            mapping[s_elem] = t_elem
            
        return mapping
        
    def _extract_analogical_insights(self, source: Dict[str, Any], 
                                   target: Dict[str, Any]) -> List[str]:
        """提取类比洞察"""
        insights = []
        
        target_solution = target.get('solution', '')
        if target_solution:
            insights.append(f"可以借鉴{target['problem']}中的{target_solution}方法")
            
        # 基于结构映射生成洞察
        mapping = self._create_analogy_mapping(source, target)
        for source_elem, target_elem in mapping.items():
            insights.append(f"{source_elem}可以类比为{target_elem}")
            
        return insights

class DivergentThinkingEngine:
    """发散思维引擎"""
    
    def __init__(self):
        self.thinking_techniques = self._initialize_techniques()
        
    def _initialize_techniques(self) -> Dict[str, Dict[str, Any]]:
        """初始化思维技术"""
        return {
            "brainstorming": {
                "name": "头脑风暴",
                "description": "无约束地生成大量想法",
                "parameters": {"time_limit": 300, "idea_target": 50}
            },
            "scamper": {
                "name": "SCAMPER技法",
                "description": "系统性地变换和组合想法",
                "parameters": {
                    "operations": ["substitute", "combine", "adapt", "modify", "put_to_other_use", "eliminate", "reverse"]
                }
            },
            "random_word": {
                "name": "随机词汇刺激",
                "description": "使用随机词汇激发创意",
                "parameters": {"word_count": 10, "association_depth": 3}
            },
            "morphological_analysis": {
                "name": "形态分析法",
                "description": "系统分析问题的各个维度",
                "parameters": {"dimensions": 3, "options_per_dimension": 5}
            }
        }
        
    def generate_ideas(self, problem: str, technique: str = "brainstorming", 
                      count: int = 10) -> List[Dict[str, Any]]:
        """生成创意想法"""
        if technique not in self.thinking_techniques:
            technique = "brainstorming"
            
        method = getattr(self, f"_{technique}", self._brainstorming)
        return method(problem, count)
        
    def _brainstorming(self, problem: str, count: int) -> List[Dict[str, Any]]:
        """头脑风暴法"""
        ideas = []
        
        # 关键词提取
        keywords = self._extract_keywords(problem)
        
        # 生成联想词汇
        associations = []
        for keyword in keywords:
            associations.extend(self._generate_associations(keyword))
            
        # 基于联想生成想法
        for i in range(count):
            # 随机选择联想词组合
            selected_words = random.sample(associations, min(3, len(associations)))
            
            idea = {
                'id': f"idea_{i}",
                'content': self._combine_words_to_idea(selected_words, problem),
                'trigger_words': selected_words,
                'technique': 'brainstorming',
                'timestamp': time.time()
            }
            ideas.append(idea)
            
        return ideas
        
    def _scamper(self, problem: str, count: int) -> List[Dict[str, Any]]:
        """SCAMPER技法"""
        operations = self.thinking_techniques["scamper"]["parameters"]["operations"]
        ideas = []
        
        for i in range(count):
            operation = random.choice(operations)
            idea_content = self._apply_scamper_operation(problem, operation)
            
            idea = {
                'id': f"scamper_{i}",
                'content': idea_content,
                'operation': operation,
                'technique': 'scamper',
                'timestamp': time.time()
            }
            ideas.append(idea)
            
        return ideas
        
    def _random_word(self, problem: str, count: int) -> List[Dict[str, Any]]:
        """随机词汇刺激法"""
        random_words = [
            "苹果", "飞机", "音乐", "海洋", "阳光", "书籍", "游戏", "花朵",
            "mountain", "robot", "painting", "dance", "crystal", "bridge",
            "thunder", "whisper", "maze", "mirror", "forest", "star"
        ]
        
        ideas = []
        for i in range(count):
            trigger_word = random.choice(random_words)
            idea_content = self._connect_random_word(problem, trigger_word)
            
            idea = {
                'id': f"random_{i}",
                'content': idea_content,
                'trigger_word': trigger_word,
                'technique': 'random_word',
                'timestamp': time.time()
            }
            ideas.append(idea)
            
        return ideas
        
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        stop_words = {'的', '是', '在', '有', '和', '了', '与', '或', '但', '如果', '因为'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 1]
        return words[:5]  # 返回前5个关键词
        
    def _generate_associations(self, word: str) -> List[str]:
        """生成联想词汇"""
        # 预定义的联想词典（简化版）
        associations_dict = {
            "技术": ["创新", "效率", "自动化", "智能", "数字化"],
            "教育": ["学习", "知识", "培训", "启发", "成长"],
            "艺术": ["创作", "美感", "表达", "色彩", "形式"],
            "商业": ["价值", "市场", "客户", "利润", "服务"],
            "科学": ["实验", "发现", "理论", "研究", "证据"]
        }
        
        # 返回相关联想或随机联想
        return associations_dict.get(word, ["创新", "连接", "变化", "可能", "机会"])
        
    def _combine_words_to_idea(self, words: List[str], problem: str) -> str:
        """将词汇组合成想法"""
        combined = "、".join(words)
        return f"结合{combined}的概念来解决{problem[:20]}..."
        
    def _apply_scamper_operation(self, problem: str, operation: str) -> str:
        """应用SCAMPER操作"""
        operation_templates = {
            "substitute": f"用什么可以替代{problem}中的关键要素？",
            "combine": f"可以将{problem}与什么结合起来？",
            "adapt": f"从其他领域可以学习什么来改进{problem}？",
            "modify": f"如何放大或缩小{problem}的某些方面？",
            "put_to_other_use": f"{problem}还可以用于什么其他目的？",
            "eliminate": f"可以从{problem}中移除什么？",
            "reverse": f"如何颠倒或重新排列{problem}？"
        }
        
        return operation_templates.get(operation, f"对{problem}进行{operation}操作")
        
    def _connect_random_word(self, problem: str, random_word: str) -> str:
        """连接随机词汇和问题"""
        return f"如何将{random_word}的特性应用到{problem}中？"

class CreativityEvaluationEngine:
    """创造力评估引擎"""
    
    def __init__(self):
        self.evaluation_criteria = self._initialize_criteria()
        
    def _initialize_criteria(self) -> Dict[str, Dict[str, Any]]:
        """初始化评估标准"""
        return {
            "originality": {
                "weight": 0.3,
                "description": "想法的新颖性和独特性",
                "metrics": ["novelty", "uniqueness", "surprise"]
            },
            "feasibility": {
                "weight": 0.25,
                "description": "想法的可实现性",
                "metrics": ["technical_viability", "resource_requirements", "time_constraints"]
            },
            "value": {
                "weight": 0.25,
                "description": "想法的实用价值",
                "metrics": ["problem_solving", "user_benefit", "market_potential"]
            },
            "elegance": {
                "weight": 0.2,
                "description": "想法的优雅性和简洁性",
                "metrics": ["simplicity", "aesthetic_appeal", "coherence"]
            }
        }
        
    def evaluate_idea(self, idea: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, float]:
        """评估创意想法"""
        evaluation_result = {}
        
        for criterion, config in self.evaluation_criteria.items():
            score = self._evaluate_criterion(idea, criterion, config, context)
            evaluation_result[criterion] = score
            
        # 计算综合分数
        total_score = sum(
            score * self.evaluation_criteria[criterion]["weight"]
            for criterion, score in evaluation_result.items()
        )
        evaluation_result["total_score"] = total_score
        
        return evaluation_result
        
    def _evaluate_criterion(self, idea: Dict[str, Any], criterion: str, 
                          config: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """评估单个标准"""
        if criterion == "originality":
            return self._evaluate_originality(idea, context)
        elif criterion == "feasibility":
            return self._evaluate_feasibility(idea, context)
        elif criterion == "value":
            return self._evaluate_value(idea, context)
        elif criterion == "elegance":
            return self._evaluate_elegance(idea, context)
        else:
            return 0.5  # 默认分数
            
    def _evaluate_originality(self, idea: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """评估原创性"""
        content = idea.get('content', '')
        
        # 检查常见词汇的使用
        common_words = ['创新', '改进', '优化', '结合', '应用']
        common_count = sum(1 for word in common_words if word in content)
        
        # 检查技术词汇的使用
        technique = idea.get('technique', '')
        technique_bonus = 0.1 if technique in ['scamper', 'random_word'] else 0
        
        # 基础原创性分数
        base_score = 0.6
        
        # 根据常见词汇调整
        if common_count > 2:
            base_score -= 0.2
        elif common_count == 0:
            base_score += 0.2
            
        # 添加技术奖励
        final_score = min(1.0, base_score + technique_bonus + random.uniform(-0.1, 0.1))
        
        return max(0.0, final_score)
        
    def _evaluate_feasibility(self, idea: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """评估可行性"""
        content = idea.get('content', '')
        
        # 检查实现难度相关词汇
        easy_indicators = ['简单', '直接', '现有', 'simple', 'direct', 'existing']
        hard_indicators = ['复杂', '困难', '需要研发', 'complex', 'difficult', 'research']
        
        easy_count = sum(1 for word in easy_indicators if word in content.lower())
        hard_count = sum(1 for word in hard_indicators if word in content.lower())
        
        # 基础可行性分数
        base_score = 0.7
        
        # 根据难度指标调整
        if easy_count > hard_count:
            base_score += 0.2
        elif hard_count > easy_count:
            base_score -= 0.3
            
        # 添加随机波动
        final_score = base_score + random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, final_score))
        
    def _evaluate_value(self, idea: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """评估价值"""
        content = idea.get('content', '')
        
        # 检查价值相关词汇
        value_indicators = ['解决', '改善', '提高', '节省', '增加', 'solve', 'improve', 'increase']
        value_count = sum(1 for word in value_indicators if word in content.lower())
        
        # 基础价值分数
        base_score = 0.6
        
        # 根据价值指标调整
        base_score += value_count * 0.1
        
        # 添加随机波动
        final_score = base_score + random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, final_score))
        
    def _evaluate_elegance(self, idea: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """评估优雅性"""
        content = idea.get('content', '')
        
        # 基于内容长度和复杂性评估
        length = len(content)
        
        # 适中长度的想法通常更优雅
        if 20 <= length <= 100:
            length_score = 0.8
        elif 10 <= length <= 150:
            length_score = 0.6
        else:
            length_score = 0.4
            
        # 检查优雅性指标
        elegance_indicators = ['简洁', '清晰', '直观', 'elegant', 'simple', 'clear']
        elegance_count = sum(1 for word in elegance_indicators if word in content.lower())
        
        elegance_bonus = elegance_count * 0.1
        
        final_score = length_score + elegance_bonus + random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, final_score))

class CreativityEngine:
    """创造力引擎主控制器"""
    
    def __init__(self):
        self.conceptual_blending = ConceptualBlendingEngine()
        self.analogical_reasoning = AnalogicalReasoningEngine()
        self.divergent_thinking = DivergentThinkingEngine()
        self.evaluation_engine = CreativityEvaluationEngine()
        
        # 创意库
        self.idea_repository = []
        self.concept_database = []
        
        # 性能指标
        self.performance_metrics = {
            'total_ideas_generated': 0,
            'high_quality_ideas': 0,
            'average_originality': 0.0,
            'average_feasibility': 0.0,
            'cross_domain_combinations': 0
        }
        
    def generate_creative_solution(self, problem: str, field: CreativeField = CreativeField.TECHNOLOGY,
                                 approach: str = "multi_method") -> Dict[str, Any]:
        """生成创意解决方案"""
        logger.info(f"开始为问题生成创意解决方案: {problem[:50]}...")
        
        start_time = time.time()
        all_ideas = []
        
        if approach == "multi_method" or approach == "divergent":
            # 发散思维生成想法
            divergent_ideas = self.divergent_thinking.generate_ideas(problem, "brainstorming", 5)
            scamper_ideas = self.divergent_thinking.generate_ideas(problem, "scamper", 3)
            random_ideas = self.divergent_thinking.generate_ideas(problem, "random_word", 3)
            
            all_ideas.extend(divergent_ideas + scamper_ideas + random_ideas)
            
        if approach == "multi_method" or approach == "analogical":
            # 类比推理生成想法
            target_domains = ["biology", "physics", "engineering"]
            source_problem = {"problem": problem, "structure": ["输入", "处理", "输出"]}
            analogies = self.analogical_reasoning.find_analogies(source_problem, target_domains)
            
            for analogy in analogies:
                analogy_idea = {
                    'id': f"analogy_{len(all_ideas)}",
                    'content': f"基于{analogy['target']['problem']}的类比: " + 
                              "; ".join(analogy['insights']),
                    'technique': 'analogical',
                    'source_analogy': analogy,
                    'timestamp': time.time()
                }
                all_ideas.append(analogy_idea)
                
        if approach == "multi_method" or approach == "blending":
            # 概念混合生成想法
            related_concepts = self._find_related_concepts(problem, field)
            if len(related_concepts) >= 2:
                for i in range(min(3, len(related_concepts) - 1)):
                    blend = self.conceptual_blending.blend_concepts(
                        related_concepts[i], related_concepts[i + 1]
                    )
                    
                    blend_idea = {
                        'id': f"blend_{len(all_ideas)}",
                        'content': f"概念混合方案: {blend['description']}",
                        'technique': 'blending',
                        'blend_details': blend,
                        'timestamp': time.time()
                    }
                    all_ideas.append(blend_idea)
                    
        # 评估所有想法
        evaluated_ideas = []
        for idea in all_ideas:
            evaluation = self.evaluation_engine.evaluate_idea(idea)
            idea['evaluation'] = evaluation
            evaluated_ideas.append(idea)
            
        # 按总分排序
        evaluated_ideas.sort(key=lambda x: x['evaluation']['total_score'], reverse=True)
        
        # 选择最佳想法
        best_ideas = evaluated_ideas[:5]
        
        # 更新性能指标
        self._update_performance_metrics(evaluated_ideas)
        
        # 存储想法到仓库
        self.idea_repository.extend(evaluated_ideas)
        
        solution_result = {
            'problem': problem,
            'field': field.value,
            'approach': approach,
            'total_ideas_generated': len(all_ideas),
            'best_ideas': best_ideas,
            'all_ideas': evaluated_ideas,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        
        logger.info(f"创意生成完成，共生成{len(all_ideas)}个想法")
        
        return solution_result
        
    def _find_related_concepts(self, problem: str, field: CreativeField) -> List[Dict[str, Any]]:
        """寻找相关概念"""
        # 预定义的概念库
        concept_library = {
            CreativeField.TECHNOLOGY: [
                {'concept': '人工智能', 'domain': '计算机科学', 'properties': ['自动化', '学习', '决策']},
                {'concept': '物联网', 'domain': '网络技术', 'properties': ['连接', '数据', '远程控制']},
                {'concept': '区块链', 'domain': '分布式系统', 'properties': ['去中心化', '安全', '透明']}
            ],
            CreativeField.ART: [
                {'concept': '抽象艺术', 'domain': '视觉艺术', 'properties': ['表达', '情感', '形式']},
                {'concept': '互动装置', 'domain': '新媒体艺术', 'properties': ['参与', '体验', '技术']},
                {'concept': '数字艺术', 'domain': '计算机艺术', 'properties': ['数字化', '创新', '虚拟']}
            ],
            CreativeField.BUSINESS: [
                {'concept': '共享经济', 'domain': '商业模式', 'properties': ['共享', '平台', '效率']},
                {'concept': '订阅服务', 'domain': '服务模式', 'properties': ['持续', '个性化', '便利']},
                {'concept': '众包', 'domain': '组织模式', 'properties': ['集体智慧', '分布式', '协作']}
            ]
        }
        
        return concept_library.get(field, [])
        
    def _update_performance_metrics(self, ideas: List[Dict[str, Any]]):
        """更新性能指标"""
        self.performance_metrics['total_ideas_generated'] += len(ideas)
        
        # 统计高质量想法（总分>0.7）
        high_quality_count = sum(1 for idea in ideas if idea['evaluation']['total_score'] > 0.7)
        self.performance_metrics['high_quality_ideas'] += high_quality_count
        
        # 更新平均分数
        if ideas:
            avg_originality = sum(idea['evaluation']['originality'] for idea in ideas) / len(ideas)
            avg_feasibility = sum(idea['evaluation']['feasibility'] for idea in ideas) / len(ideas)
            
            total_ideas = self.performance_metrics['total_ideas_generated']
            prev_total = total_ideas - len(ideas)
            
            if prev_total > 0:
                self.performance_metrics['average_originality'] = (
                    self.performance_metrics['average_originality'] * prev_total + 
                    avg_originality * len(ideas)
                ) / total_ideas
                
                self.performance_metrics['average_feasibility'] = (
                    self.performance_metrics['average_feasibility'] * prev_total + 
                    avg_feasibility * len(ideas)
                ) / total_ideas
            else:
                self.performance_metrics['average_originality'] = avg_originality
                self.performance_metrics['average_feasibility'] = avg_feasibility
                
        # 统计跨域组合
        cross_domain_count = sum(1 for idea in ideas if idea.get('technique') == 'blending')
        self.performance_metrics['cross_domain_combinations'] += cross_domain_count
        
    def get_creativity_summary(self, solution_result: Dict[str, Any]) -> str:
        """生成创造力分析摘要"""
        problem = solution_result['problem']
        best_ideas = solution_result['best_ideas']
        total_ideas = solution_result['total_ideas_generated']
        
        summary = f"""
🎨 创造力引擎分析报告
{"="*40}

🎯 问题: {problem}
📊 生成统计:
  总想法数: {total_ideas}
  最佳想法数: {len(best_ideas)}
  处理时间: {solution_result['processing_time']:.2f}秒

💡 顶级创意:
"""
        
        for i, idea in enumerate(best_ideas[:3], 1):
            eval_data = idea['evaluation']
            summary += f"""
  {i}. {idea['content'][:80]}...
     原创性: {eval_data['originality']:.2f} | 可行性: {eval_data['feasibility']:.2f}
     价值: {eval_data['value']:.2f} | 优雅性: {eval_data['elegance']:.2f}
     总分: {eval_data['total_score']:.2f} | 方法: {idea['technique']}
"""
        
        return summary
        
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        high_quality_rate = (
            self.performance_metrics['high_quality_ideas'] / 
            max(self.performance_metrics['total_ideas_generated'], 1)
        )
        
        return {
            "总想法生成数": self.performance_metrics['total_ideas_generated'],
            "高质量想法数": self.performance_metrics['high_quality_ideas'],
            "高质量想法率": f"{high_quality_rate:.2%}",
            "平均原创性": f"{self.performance_metrics['average_originality']:.3f}",
            "平均可行性": f"{self.performance_metrics['average_feasibility']:.3f}",
            "跨域组合数": self.performance_metrics['cross_domain_combinations'],
            "想法库存储数": len(self.idea_repository)
        }

# 示例使用和测试
async def demonstrate_creativity_engine():
    """演示创造力引擎功能"""
    print("🎨 自主进化Agent - 第5轮提升：创造力引擎")
    print("=" * 60)
    
    # 创建创造力引擎
    creativity_engine = CreativityEngine()
    
    # 测试创意生成场景
    test_problems = [
        {
            "problem": "如何提高在线教育的学习效果和学生参与度",
            "field": CreativeField.EDUCATION
        },
        {
            "problem": "设计一个环保且实用的城市交通解决方案",
            "field": CreativeField.TECHNOLOGY
        },
        {
            "problem": "创造一种新的艺术表达形式来反映数字时代的特征",
            "field": CreativeField.ART
        }
    ]
    
    for i, scenario in enumerate(test_problems, 1):
        print(f"\n🧠 创意挑战 {i}: {scenario['problem'][:40]}...")
        
        # 生成创意解决方案
        solution = creativity_engine.generate_creative_solution(
            scenario['problem'], 
            scenario['field']
        )
        
        # 显示分析结果
        summary = creativity_engine.get_creativity_summary(solution)
        print(summary)
        
    # 测试概念混合
    print("\n🔀 概念混合示例")
    concept1 = {'concept': '音乐', 'domain': '艺术', 'properties': ['节奏', '旋律', '情感']}
    concept2 = {'concept': '编程', 'domain': '技术', 'properties': ['逻辑', '结构', '创造']}
    
    blend = creativity_engine.conceptual_blending.blend_concepts(concept1, concept2)
    print(f"混合结果: {blend['description']}")
    print(f"涌现属性: {', '.join(blend['emergent_properties'])}")
    
    # 性能报告
    print("\n📊 创造力引擎性能报告")
    report = creativity_engine.get_performance_report()
    for key, value in report.items():
        print(f"{key}: {value}")
        
    print("\n✅ 第5轮提升完成！创造力引擎已成功部署")

if __name__ == "__main__":
    asyncio.run(demonstrate_creativity_engine())