# 自主进化Agent - 第2轮提升：多模态感知系统
# Multimodal Perception System - 统一处理文本、图像、音频

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import base64
import hashlib
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"

class PerceptionConfidence(Enum):
    """感知置信度等级"""
    VERY_HIGH = 0.95
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2

@dataclass
class PerceptionResult:
    """感知结果数据结构"""
    modality: ModalityType
    content: Any
    confidence: float
    features: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None

@dataclass
class CrossModalMapping:
    """跨模态映射"""
    source_modality: ModalityType
    target_modality: ModalityType
    mapping_function: str
    confidence: float
    semantic_similarity: float

class TextPerceptionEngine:
    """文本感知引擎"""
    
    def __init__(self):
        self.vocabulary = set()
        self.semantic_cache = {}
        self.language_models = {}
        self.entity_extractors = {}
        
    def extract_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'language': self.detect_language(text),
            'sentiment': self.analyze_sentiment(text),
            'entities': self.extract_entities(text),
            'keywords': self.extract_keywords(text),
            'topics': self.extract_topics(text),
            'complexity': self.calculate_complexity(text)
        }
        return features
        
    def detect_language(self, text: str) -> str:
        """语言检测（简化版）"""
        # 基于字符模式的简单语言检测
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > 0:
            return "en"
        else:
            return "unknown"
            
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """情感分析（简化版）"""
        positive_words = ['好', '优秀', '棒', 'excellent', 'good', 'great', 'amazing']
        negative_words = ['坏', '差', '糟糕', 'bad', 'terrible', 'awful', 'poor']
        
        words = text.lower().split()
        positive_score = sum(1 for word in words if word in positive_words)
        negative_score = sum(1 for word in words if word in negative_words)
        
        total_score = positive_score + negative_score
        if total_score == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 1.0}
            
        return {
            "positive": positive_score / total_score,
            "negative": negative_score / total_score,
            "neutral": 1.0 - (positive_score + negative_score) / len(words)
        }
        
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """实体提取（简化版）"""
        entities = []
        words = text.split()
        
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entities.append({
                    "text": word,
                    "type": "PERSON" if any(c.isupper() for c in word) else "ORGANIZATION",
                    "position": i
                })
                
        return entities
        
    def extract_keywords(self, text: str) -> List[str]:
        """关键词提取（简化版）"""
        stop_words = {'的', '是', '在', '有', '和', 'the', 'is', 'in', 'and', 'of', 'to', 'a'}
        words = [word.lower() for word in text.split() if word.lower() not in stop_words and len(word) > 2]
        
        # 简单的频率统计
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
            
        # 返回频率最高的前10个词
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
        
    def extract_topics(self, text: str) -> List[str]:
        """主题提取（简化版）"""
        topic_keywords = {
            "technology": ["AI", "机器学习", "人工智能", "算法", "technology", "computer"],
            "business": ["公司", "商业", "营销", "business", "company", "market"],
            "science": ["科学", "研究", "实验", "science", "research", "experiment"],
            "education": ["教育", "学习", "课程", "education", "learning", "course"]
        }
        
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                topics.append(topic)
                
        return topics if topics else ["general"]
        
    def calculate_complexity(self, text: str) -> float:
        """计算文本复杂度"""
        words = text.split()
        if not words:
            return 0.0
            
        avg_word_length = np.mean([len(word) for word in words])
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        words_per_sentence = len(words) / sentence_count
        
        # 复杂度综合评分
        complexity = (avg_word_length * 0.3 + words_per_sentence * 0.7) / 10
        return min(complexity, 1.0)
        
    def perceive(self, text: str) -> PerceptionResult:
        """文本感知主函数"""
        features = self.extract_features(text)
        
        # 计算置信度
        confidence = PerceptionConfidence.HIGH.value
        if features['length'] < 10:
            confidence = PerceptionConfidence.LOW.value
        elif features['word_count'] > 100:
            confidence = PerceptionConfidence.VERY_HIGH.value
            
        return PerceptionResult(
            modality=ModalityType.TEXT,
            content=text,
            confidence=confidence,
            features=features,
            timestamp=time.time(),
            metadata={
                "processing_method": "text_perception_engine",
                "version": "2.0"
            }
        )

class ImagePerceptionEngine:
    """图像感知引擎"""
    
    def __init__(self):
        self.color_extractors = {}
        self.shape_detectors = {}
        self.object_classifiers = {}
        
    def extract_features(self, image_data: Any) -> Dict[str, Any]:
        """提取图像特征（模拟）"""
        # 这里模拟图像特征提取
        features = {
            'dimensions': (224, 224, 3),  # 假设的图像尺寸
            'color_histogram': self.extract_color_histogram(),
            'edges': self.detect_edges(),
            'textures': self.analyze_textures(),
            'objects': self.detect_objects(),
            'faces': self.detect_faces(),
            'scene_type': self.classify_scene(),
            'aesthetic_score': self.calculate_aesthetic_score(),
            'complexity': self.calculate_visual_complexity()
        }
        return features
        
    def extract_color_histogram(self) -> Dict[str, int]:
        """颜色直方图（模拟）"""
        return {
            'red': np.random.randint(0, 255),
            'green': np.random.randint(0, 255),
            'blue': np.random.randint(0, 255),
            'dominant_color': ['red', 'green', 'blue'][np.random.randint(0, 3)]
        }
        
    def detect_edges(self) -> Dict[str, Any]:
        """边缘检测（模拟）"""
        return {
            'edge_density': np.random.uniform(0.1, 0.9),
            'dominant_orientation': np.random.choice(['horizontal', 'vertical', 'diagonal']),
            'sharpness': np.random.uniform(0.0, 1.0)
        }
        
    def analyze_textures(self) -> Dict[str, float]:
        """纹理分析（模拟）"""
        return {
            'smoothness': np.random.uniform(0.0, 1.0),
            'roughness': np.random.uniform(0.0, 1.0),
            'regularity': np.random.uniform(0.0, 1.0),
            'contrast': np.random.uniform(0.0, 1.0)
        }
        
    def detect_objects(self) -> List[Dict[str, Any]]:
        """物体检测（模拟）"""
        possible_objects = ['person', 'car', 'building', 'tree', 'animal', 'book', 'computer']
        num_objects = np.random.randint(1, 4)
        
        objects = []
        for _ in range(num_objects):
            objects.append({
                'class': np.random.choice(possible_objects),
                'confidence': np.random.uniform(0.5, 0.95),
                'bbox': [np.random.randint(0, 200) for _ in range(4)]
            })
        return objects
        
    def detect_faces(self) -> List[Dict[str, Any]]:
        """人脸检测（模拟）"""
        num_faces = np.random.randint(0, 3)
        faces = []
        
        for _ in range(num_faces):
            faces.append({
                'confidence': np.random.uniform(0.7, 0.98),
                'age_estimate': np.random.randint(18, 80),
                'gender_estimate': np.random.choice(['male', 'female']),
                'emotion': np.random.choice(['happy', 'sad', 'neutral', 'surprised'])
            })
        return faces
        
    def classify_scene(self) -> str:
        """场景分类（模拟）"""
        scenes = ['indoor', 'outdoor', 'urban', 'nature', 'office', 'home', 'street']
        return np.random.choice(scenes)
        
    def calculate_aesthetic_score(self) -> float:
        """美学评分（模拟）"""
        return np.random.uniform(0.3, 0.9)
        
    def calculate_visual_complexity(self) -> float:
        """视觉复杂度（模拟）"""
        return np.random.uniform(0.2, 0.8)
        
    def perceive(self, image_data: Any) -> PerceptionResult:
        """图像感知主函数"""
        features = self.extract_features(image_data)
        
        # 基于特征计算置信度
        confidence = PerceptionConfidence.HIGH.value
        if features['objects']:
            avg_object_confidence = np.mean([obj['confidence'] for obj in features['objects']])
            confidence = avg_object_confidence
            
        return PerceptionResult(
            modality=ModalityType.IMAGE,
            content=image_data,
            confidence=confidence,
            features=features,
            timestamp=time.time(),
            metadata={
                "processing_method": "image_perception_engine",
                "version": "2.0"
            }
        )

class AudioPerceptionEngine:
    """音频感知引擎"""
    
    def __init__(self):
        self.speech_recognizers = {}
        self.audio_classifiers = {}
        self.emotion_analyzers = {}
        
    def extract_features(self, audio_data: Any) -> Dict[str, Any]:
        """提取音频特征（模拟）"""
        features = {
            'duration': np.random.uniform(1.0, 30.0),
            'sample_rate': 44100,
            'channels': np.random.choice([1, 2]),
            'spectral_features': self.extract_spectral_features(),
            'temporal_features': self.extract_temporal_features(),
            'speech_detection': self.detect_speech(),
            'music_detection': self.detect_music(),
            'emotion_analysis': self.analyze_audio_emotion(),
            'energy_level': np.random.uniform(0.1, 0.9),
            'pitch_analysis': self.analyze_pitch()
        }
        return features
        
    def extract_spectral_features(self) -> Dict[str, float]:
        """频谱特征（模拟）"""
        return {
            'spectral_centroid': np.random.uniform(1000, 4000),
            'spectral_bandwidth': np.random.uniform(500, 2000),
            'spectral_rolloff': np.random.uniform(2000, 8000),
            'zero_crossing_rate': np.random.uniform(0.05, 0.3),
            'mfcc_coefficients': np.random.uniform(-10, 10, 13).tolist()
        }
        
    def extract_temporal_features(self) -> Dict[str, float]:
        """时域特征（模拟）"""
        return {
            'rms_energy': np.random.uniform(0.01, 0.5),
            'tempo': np.random.uniform(60, 180),
            'rhythm_regularity': np.random.uniform(0.3, 0.9),
            'silence_ratio': np.random.uniform(0.0, 0.4)
        }
        
    def detect_speech(self) -> Dict[str, Any]:
        """语音检测（模拟）"""
        has_speech = np.random.choice([True, False], p=[0.7, 0.3])
        
        if has_speech:
            return {
                'has_speech': True,
                'speech_confidence': np.random.uniform(0.7, 0.95),
                'language': np.random.choice(['zh', 'en', 'unknown']),
                'speaker_count': np.random.randint(1, 4),
                'transcription': "这是模拟的语音识别结果"
            }
        else:
            return {'has_speech': False}
            
    def detect_music(self) -> Dict[str, Any]:
        """音乐检测（模拟）"""
        has_music = np.random.choice([True, False], p=[0.5, 0.5])
        
        if has_music:
            return {
                'has_music': True,
                'music_confidence': np.random.uniform(0.6, 0.9),
                'genre': np.random.choice(['classical', 'rock', 'pop', 'jazz', 'electronic']),
                'instruments': np.random.choice(['piano', 'guitar', 'violin', 'drums'], size=np.random.randint(1, 3)).tolist()
            }
        else:
            return {'has_music': False}
            
    def analyze_audio_emotion(self) -> Dict[str, float]:
        """音频情感分析（模拟）"""
        emotions = ['happy', 'sad', 'angry', 'neutral', 'excited', 'calm']
        emotion_scores = np.random.dirichlet(np.ones(len(emotions)))
        
        return {emotion: float(score) for emotion, score in zip(emotions, emotion_scores)}
        
    def analyze_pitch(self) -> Dict[str, float]:
        """音调分析（模拟）"""
        return {
            'fundamental_frequency': np.random.uniform(80, 400),
            'pitch_stability': np.random.uniform(0.3, 0.9),
            'pitch_range': np.random.uniform(50, 300),
            'intonation_pattern': np.random.choice(['rising', 'falling', 'flat', 'varying'])
        }
        
    def perceive(self, audio_data: Any) -> PerceptionResult:
        """音频感知主函数"""
        features = self.extract_features(audio_data)
        
        # 计算置信度
        confidence = PerceptionConfidence.MEDIUM.value
        if features['speech_detection'].get('has_speech'):
            confidence = features['speech_detection']['speech_confidence']
        elif features['music_detection'].get('has_music'):
            confidence = features['music_detection']['music_confidence']
            
        return PerceptionResult(
            modality=ModalityType.AUDIO,
            content=audio_data,
            confidence=confidence,
            features=features,
            timestamp=time.time(),
            metadata={
                "processing_method": "audio_perception_engine",
                "version": "2.0"
            }
        )

class CrossModalFusionEngine:
    """跨模态融合引擎"""
    
    def __init__(self):
        self.fusion_strategies = {}
        self.semantic_alignments = {}
        self.cross_modal_mappings = []
        
    def add_cross_modal_mapping(self, mapping: CrossModalMapping):
        """添加跨模态映射"""
        self.cross_modal_mappings.append(mapping)
        logger.info(f"添加跨模态映射: {mapping.source_modality.value} -> {mapping.target_modality.value}")
        
    def semantic_alignment(self, results: List[PerceptionResult]) -> Dict[str, Any]:
        """语义对齐"""
        aligned_semantics = {
            'common_concepts': [],
            'cross_modal_consistency': 0.0,
            'semantic_richness': 0.0,
            'unified_representation': {}
        }
        
        # 提取共同概念
        text_concepts = set()
        image_concepts = set()
        audio_concepts = set()
        
        for result in results:
            if result.modality == ModalityType.TEXT:
                text_concepts.update(result.features.get('keywords', []))
                text_concepts.update(result.features.get('topics', []))
            elif result.modality == ModalityType.IMAGE:
                image_concepts.update([obj['class'] for obj in result.features.get('objects', [])])
                image_concepts.add(result.features.get('scene_type', ''))
            elif result.modality == ModalityType.AUDIO:
                if result.features.get('music_detection', {}).get('has_music'):
                    audio_concepts.add('music')
                    audio_concepts.update(result.features['music_detection'].get('instruments', []))
                if result.features.get('speech_detection', {}).get('has_speech'):
                    audio_concepts.add('speech')
                    
        # 寻找共同概念
        all_concepts = text_concepts | image_concepts | audio_concepts
        common_concepts = []
        
        for concept in all_concepts:
            modality_count = 0
            if concept in text_concepts:
                modality_count += 1
            if concept in image_concepts:
                modality_count += 1
            if concept in audio_concepts:
                modality_count += 1
                
            if modality_count > 1:
                common_concepts.append(concept)
                
        aligned_semantics['common_concepts'] = common_concepts
        aligned_semantics['cross_modal_consistency'] = len(common_concepts) / max(len(all_concepts), 1)
        aligned_semantics['semantic_richness'] = len(all_concepts) / len(results) if results else 0
        
        return aligned_semantics
        
    def temporal_synchronization(self, results: List[PerceptionResult]) -> Dict[str, Any]:
        """时序同步"""
        if not results:
            return {'synchronized': False}
            
        timestamps = [result.timestamp for result in results]
        time_window = max(timestamps) - min(timestamps)
        
        return {
            'synchronized': time_window < 1.0,  # 1秒内认为是同步的
            'time_window': time_window,
            'temporal_order': sorted(results, key=lambda x: x.timestamp)
        }
        
    def confidence_weighting(self, results: List[PerceptionResult]) -> Dict[str, float]:
        """置信度加权"""
        if not results:
            return {}
            
        total_confidence = sum(result.confidence for result in results)
        weights = {}
        
        for result in results:
            weights[result.modality.value] = result.confidence / total_confidence if total_confidence > 0 else 0
            
        return weights
        
    def fuse_multimodal(self, results: List[PerceptionResult]) -> Dict[str, Any]:
        """多模态融合"""
        if not results:
            return {}
            
        fusion_result = {
            'input_modalities': [result.modality.value for result in results],
            'semantic_alignment': self.semantic_alignment(results),
            'temporal_sync': self.temporal_synchronization(results),
            'confidence_weights': self.confidence_weighting(results),
            'overall_confidence': np.mean([result.confidence for result in results]),
            'fusion_timestamp': time.time(),
            'cross_modal_features': self.extract_cross_modal_features(results)
        }
        
        return fusion_result
        
    def extract_cross_modal_features(self, results: List[PerceptionResult]) -> Dict[str, Any]:
        """提取跨模态特征"""
        cross_features = {
            'modality_diversity': len(set(result.modality for result in results)),
            'feature_complexity': 0.0,
            'information_density': 0.0,
            'semantic_coherence': 0.0
        }
        
        # 计算特征复杂度
        total_features = 0
        for result in results:
            total_features += len(result.features)
            
        cross_features['feature_complexity'] = total_features / len(results) if results else 0
        
        # 计算信息密度
        content_lengths = []
        for result in results:
            if result.modality == ModalityType.TEXT:
                content_lengths.append(len(str(result.content)))
            else:
                content_lengths.append(100)  # 默认值for非文本模态
                
        cross_features['information_density'] = np.mean(content_lengths) if content_lengths else 0
        
        # 计算语义一致性
        semantic_alignment = self.semantic_alignment(results)
        cross_features['semantic_coherence'] = semantic_alignment['cross_modal_consistency']
        
        return cross_features

class MultimodalPerceptionSystem:
    """多模态感知系统主控制器"""
    
    def __init__(self):
        self.text_engine = TextPerceptionEngine()
        self.image_engine = ImagePerceptionEngine()
        self.audio_engine = AudioPerceptionEngine()
        self.fusion_engine = CrossModalFusionEngine()
        
        self.perception_history = []
        self.performance_metrics = {
            'total_perceptions': 0,
            'successful_fusions': 0,
            'modality_distribution': defaultdict(int),
            'avg_confidence': 0.0,
            'processing_times': []
        }
        
        # 初始化跨模态映射
        self._initialize_cross_modal_mappings()
        
    def _initialize_cross_modal_mappings(self):
        """初始化跨模态映射"""
        mappings = [
            CrossModalMapping(
                source_modality=ModalityType.TEXT,
                target_modality=ModalityType.IMAGE,
                mapping_function="text_to_image_generation",
                confidence=0.7,
                semantic_similarity=0.8
            ),
            CrossModalMapping(
                source_modality=ModalityType.AUDIO,
                target_modality=ModalityType.TEXT,
                mapping_function="speech_to_text",
                confidence=0.9,
                semantic_similarity=0.95
            ),
            CrossModalMapping(
                source_modality=ModalityType.IMAGE,
                target_modality=ModalityType.TEXT,
                mapping_function="image_captioning",
                confidence=0.8,
                semantic_similarity=0.85
            )
        ]
        
        for mapping in mappings:
            self.fusion_engine.add_cross_modal_mapping(mapping)
            
    def perceive_multimodal(self, inputs: Dict[ModalityType, Any]) -> Dict[str, Any]:
        """多模态感知主函数"""
        start_time = time.time()
        perception_results = []
        
        # 各模态独立感知
        for modality, data in inputs.items():
            try:
                if modality == ModalityType.TEXT:
                    result = self.text_engine.perceive(data)
                elif modality == ModalityType.IMAGE:
                    result = self.image_engine.perceive(data)
                elif modality == ModalityType.AUDIO:
                    result = self.audio_engine.perceive(data)
                else:
                    logger.warning(f"未支持的模态类型: {modality}")
                    continue
                    
                perception_results.append(result)
                self.performance_metrics['modality_distribution'][modality.value] += 1
                
            except Exception as e:
                logger.error(f"感知错误 - {modality.value}: {str(e)}")
                continue
                
        # 跨模态融合
        fusion_result = self.fusion_engine.fuse_multimodal(perception_results)
        
        # 生成综合感知结果
        comprehensive_result = {
            'individual_perceptions': perception_results,
            'fusion_result': fusion_result,
            'processing_time': time.time() - start_time,
            'timestamp': time.time(),
            'metadata': {
                'version': '2.0',
                'modalities_processed': len(inputs),
                'fusion_strategy': 'weighted_semantic_alignment'
            }
        }
        
        # 更新性能指标
        self._update_metrics(comprehensive_result)
        
        # 保存到历史记录
        self.perception_history.append(comprehensive_result)
        
        return comprehensive_result
        
    def _update_metrics(self, result: Dict[str, Any]):
        """更新性能指标"""
        self.performance_metrics['total_perceptions'] += 1
        
        if result['fusion_result']:
            self.performance_metrics['successful_fusions'] += 1
            
        # 更新平均置信度
        if result['individual_perceptions']:
            avg_confidence = np.mean([p.confidence for p in result['individual_perceptions']])
            self.performance_metrics['avg_confidence'] = (
                self.performance_metrics['avg_confidence'] * (self.performance_metrics['total_perceptions'] - 1) +
                avg_confidence
            ) / self.performance_metrics['total_perceptions']
            
        # 记录处理时间
        self.performance_metrics['processing_times'].append(result['processing_time'])
        
    def get_perception_summary(self, result: Dict[str, Any]) -> str:
        """生成感知结果摘要"""
        summary = "🔍 多模态感知结果摘要\n"
        summary += "=" * 40 + "\n"
        
        # 个体感知结果
        summary += "📊 各模态感知结果:\n"
        for perception in result['individual_perceptions']:
            summary += f"  {perception.modality.value}: 置信度 {perception.confidence:.3f}\n"
            
        # 融合结果
        fusion = result['fusion_result']
        if fusion:
            summary += "\n🔗 跨模态融合:\n"
            summary += f"  输入模态: {', '.join(fusion['input_modalities'])}\n"
            summary += f"  整体置信度: {fusion['overall_confidence']:.3f}\n"
            summary += f"  语义一致性: {fusion['semantic_alignment']['cross_modal_consistency']:.3f}\n"
            summary += f"  共同概念: {', '.join(fusion['semantic_alignment']['common_concepts'])}\n"
            
        # 处理性能
        summary += f"\n⏱️  处理时间: {result['processing_time']:.3f}秒\n"
        
        return summary
        
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        success_rate = (
            self.performance_metrics['successful_fusions'] / 
            max(self.performance_metrics['total_perceptions'], 1)
        )
        
        avg_processing_time = (
            np.mean(self.performance_metrics['processing_times']) 
            if self.performance_metrics['processing_times'] else 0
        )
        
        return {
            "总感知次数": self.performance_metrics['total_perceptions'],
            "成功融合次数": self.performance_metrics['successful_fusions'],
            "融合成功率": f"{success_rate:.2%}",
            "平均置信度": f"{self.performance_metrics['avg_confidence']:.3f}",
            "平均处理时间": f"{avg_processing_time:.3f}秒",
            "模态分布": dict(self.performance_metrics['modality_distribution']),
            "历史记录数": len(self.perception_history)
        }

# 示例使用和测试
async def demonstrate_multimodal_perception():
    """演示多模态感知系统功能"""
    print("🌟 自主进化Agent - 第2轮提升：多模态感知系统")
    print("=" * 60)
    
    # 创建多模态感知系统
    perception_system = MultimodalPerceptionSystem()
    
    # 示例1: 文本+图像感知
    print("\n📖🖼️ 示例1: 文本+图像多模态感知")
    multimodal_input1 = {
        ModalityType.TEXT: "一只可爱的小猫在阳光下睡觉",
        ModalityType.IMAGE: "cat_image_data"  # 模拟图像数据
    }
    
    result1 = perception_system.perceive_multimodal(multimodal_input1)
    print(perception_system.get_perception_summary(result1))
    
    # 示例2: 文本+音频感知
    print("\n📖🎵 示例2: 文本+音频多模态感知")
    multimodal_input2 = {
        ModalityType.TEXT: "这是一段优美的古典音乐",
        ModalityType.AUDIO: "classical_music_data"  # 模拟音频数据
    }
    
    result2 = perception_system.perceive_multimodal(multimodal_input2)
    print(perception_system.get_perception_summary(result2))
    
    # 示例3: 三模态融合感知
    print("\n📖🖼️🎵 示例3: 文本+图像+音频三模态感知")
    multimodal_input3 = {
        ModalityType.TEXT: "音乐会上的钢琴演奏",
        ModalityType.IMAGE: "concert_image_data",
        ModalityType.AUDIO: "piano_performance_data"
    }
    
    result3 = perception_system.perceive_multimodal(multimodal_input3)
    print(perception_system.get_perception_summary(result3))
    
    # 性能报告
    print("\n📊 系统性能报告")
    report = perception_system.get_performance_report()
    for key, value in report.items():
        print(f"{key}: {value}")
        
    print("\n✅ 第2轮提升完成！多模态感知系统已成功部署")

if __name__ == "__main__":
    asyncio.run(demonstrate_multimodal_perception())