"""
智能感知与环境理解系统
实现多模态感知、环境建模、预测分析、态势感知
"""
import json
import time
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """传感器类型"""
    TEXT = "text"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    STRUCTURED = "structured"
    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"

class AlertLevel(Enum):
    """告警级别"""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4

@dataclass
class SensorReading:
    """传感器读数"""
    sensor_id: str
    sensor_type: SensorType
    value: Any
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvironmentState:
    """环境状态"""
    timestamp: float
    features: Dict[str, Any]
    derived_metrics: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    trend_indicators: Dict[str, str] = field(default_factory=dict)
    confidence_level: float = 1.0

@dataclass
class Prediction:
    """预测结果"""
    target_feature: str
    predicted_value: Any
    confidence: float
    time_horizon: float
    prediction_interval: Optional[Tuple[float, float]] = None
    contributing_factors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    level: AlertLevel
    message: str
    source_sensor: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)

class SensorManager:
    """传感器管理器"""
    
    def __init__(self):
        self.sensors = {}
        self.sensor_data = defaultdict(deque)
        self.sensor_status = {}
        self.data_quality_monitors = {}
        
    def register_sensor(self, sensor_id: str, sensor_type: SensorType, config: Dict[str, Any] = None):
        """注册传感器"""
        if config is None:
            config = {}
        
        self.sensors[sensor_id] = {
            "type": sensor_type,
            "config": config,
            "registered_at": time.time(),
            "last_reading": None,
            "reading_count": 0
        }
        
        self.sensor_data[sensor_id] = deque(maxlen=config.get("history_size", 1000))
        self.sensor_status[sensor_id] = "active"
        
        logger.info(f"Registered sensor {sensor_id} of type {sensor_type.value}")
    
    def add_reading(self, sensor_id: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """添加传感器读数"""
        if sensor_id not in self.sensors:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return False
        
        if metadata is None:
            metadata = {}
        
        sensor_type = self.sensors[sensor_id]["type"]
        
        # 数据质量检查
        quality_score, confidence = self._assess_data_quality(sensor_id, value, sensor_type)
        
        reading = SensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            value=value,
            confidence=confidence,
            quality_score=quality_score,
            metadata=metadata
        )
        
        self.sensor_data[sensor_id].append(reading)
        self.sensors[sensor_id]["last_reading"] = reading
        self.sensors[sensor_id]["reading_count"] += 1
        
        return True
    
    def _assess_data_quality(self, sensor_id: str, value: Any, sensor_type: SensorType) -> Tuple[float, float]:
        """评估数据质量"""
        quality_score = 1.0
        confidence = 1.0
        
        # 基于传感器类型的质量检查
        if sensor_type == SensorType.NUMERICAL:
            if isinstance(value, (int, float)):
                # 检查是否在合理范围内
                if math.isnan(value) or math.isinf(value):
                    quality_score = 0.0
                    confidence = 0.0
            else:
                quality_score = 0.5
                confidence = 0.5
        
        elif sensor_type == SensorType.TEXT:
            if isinstance(value, str):
                if len(value.strip()) == 0:
                    quality_score = 0.1
                elif len(value) < 3:
                    quality_score = 0.5
            else:
                quality_score = 0.3
        
        # 基于历史数据的一致性检查
        if sensor_id in self.sensor_data and len(self.sensor_data[sensor_id]) > 5:
            recent_readings = list(self.sensor_data[sensor_id])[-5:]
            
            if sensor_type == SensorType.NUMERICAL:
                recent_values = [r.value for r in recent_readings if isinstance(r.value, (int, float))]
                if recent_values:
                    mean_val = sum(recent_values) / len(recent_values)
                    variance = sum((x - mean_val) ** 2 for x in recent_values) / len(recent_values)
                    
                    # 如果偏差过大，降低置信度
                    if isinstance(value, (int, float)) and variance > 0:
                        deviation = abs(value - mean_val) / (math.sqrt(variance) + 1e-6)
                        if deviation > 3:  # 3-sigma规则
                            confidence *= 0.7
                            quality_score *= 0.8
        
        return quality_score, confidence
    
    def get_recent_readings(self, sensor_id: str, count: int = 10) -> List[SensorReading]:
        """获取最近的传感器读数"""
        if sensor_id not in self.sensor_data:
            return []
        
        readings = list(self.sensor_data[sensor_id])
        return readings[-count:] if len(readings) >= count else readings
    
    def get_sensor_statistics(self, sensor_id: str) -> Dict[str, Any]:
        """获取传感器统计信息"""
        if sensor_id not in self.sensors:
            return {}
        
        readings = list(self.sensor_data[sensor_id])
        if not readings:
            return {"status": "no_data"}
        
        numerical_values = []
        for reading in readings:
            if isinstance(reading.value, (int, float)):
                numerical_values.append(reading.value)
        
        stats = {
            "total_readings": len(readings),
            "latest_reading": readings[-1].value if readings else None,
            "avg_quality_score": sum(r.quality_score for r in readings) / len(readings),
            "avg_confidence": sum(r.confidence for r in readings) / len(readings),
            "data_rate": len(readings) / ((time.time() - readings[0].timestamp) / 60) if len(readings) > 1 else 0
        }
        
        if numerical_values:
            stats.update({
                "mean": sum(numerical_values) / len(numerical_values),
                "min": min(numerical_values),
                "max": max(numerical_values),
                "variance": sum((x - stats["mean"]) ** 2 for x in numerical_values) / len(numerical_values)
            })
        
        return stats

class PatternDetector:
    """模式检测器"""
    
    def __init__(self):
        self.detected_patterns = {}
        self.pattern_history = deque(maxlen=1000)
        self.anomaly_threshold = 0.8
        
    def detect_temporal_patterns(self, readings: List[SensorReading]) -> Dict[str, Any]:
        """检测时间模式"""
        if len(readings) < 10:
            return {"patterns": [], "confidence": 0.0}
        
        patterns = []
        
        # 检测周期性
        if self._detect_periodicity(readings):
            patterns.append("periodic")
        
        # 检测趋势
        trend = self._detect_trend(readings)
        if trend != "stable":
            patterns.append(f"trend_{trend}")
        
        # 检测异常值
        anomalies = self._detect_anomalies(readings)
        if anomalies:
            patterns.append("anomalous")
        
        return {
            "patterns": patterns,
            "confidence": len(patterns) / 3.0,  # 简化的置信度计算
            "trend": trend,
            "anomaly_count": len(anomalies)
        }
    
    def _detect_periodicity(self, readings: List[SensorReading]) -> bool:
        """检测周期性"""
        # 简化的周期性检测
        numerical_values = [r.value for r in readings if isinstance(r.value, (int, float))]
        
        if len(numerical_values) < 20:
            return False
        
        # 使用简单的自相关检测
        n = len(numerical_values)
        for lag in range(2, min(n // 2, 50)):
            correlation = self._calculate_autocorrelation(numerical_values, lag)
            if correlation > 0.7:  # 高相关性阈值
                return True
        
        return False
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """计算自相关系数"""
        if len(values) <= lag:
            return 0.0
        
        n = len(values) - lag
        mean_val = sum(values) / len(values)
        
        numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n))
        denominator = sum((x - mean_val) ** 2 for x in values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_trend(self, readings: List[SensorReading]) -> str:
        """检测趋势"""
        numerical_values = [r.value for r in readings if isinstance(r.value, (int, float))]
        
        if len(numerical_values) < 5:
            return "stable"
        
        # 简单的线性回归斜率
        n = len(numerical_values)
        x_mean = (n - 1) / 2
        y_mean = sum(numerical_values) / n
        
        numerator = sum((i - x_mean) * (numerical_values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_anomalies(self, readings: List[SensorReading]) -> List[int]:
        """检测异常值"""
        numerical_values = [r.value for r in readings if isinstance(r.value, (int, float))]
        
        if len(numerical_values) < 10:
            return []
        
        mean_val = sum(numerical_values) / len(numerical_values)
        variance = sum((x - mean_val) ** 2 for x in numerical_values) / len(numerical_values)
        std_dev = math.sqrt(variance)
        
        anomalies = []
        for i, value in enumerate(numerical_values):
            z_score = abs(value - mean_val) / (std_dev + 1e-6)
            if z_score > 2.5:  # 2.5-sigma阈值
                anomalies.append(i)
        
        return anomalies

class EnvironmentModeler:
    """环境建模器"""
    
    def __init__(self):
        self.environment_history = deque(maxlen=1000)
        self.feature_correlations = defaultdict(dict)
        self.baseline_metrics = {}
        
    def update_environment_state(self, sensor_readings: Dict[str, SensorReading]) -> EnvironmentState:
        """更新环境状态"""
        current_time = time.time()
        
        # 提取特征
        features = {}
        for sensor_id, reading in sensor_readings.items():
            features[sensor_id] = reading.value
            features[f"{sensor_id}_quality"] = reading.quality_score
            features[f"{sensor_id}_confidence"] = reading.confidence
        
        # 计算派生指标
        derived_metrics = self._calculate_derived_metrics(features)
        
        # 计算异常分数
        anomaly_score = self._calculate_anomaly_score(features)
        
        # 识别趋势指标
        trend_indicators = self._identify_trend_indicators(features)
        
        # 计算整体置信度
        confidence_level = self._calculate_overall_confidence(sensor_readings)
        
        state = EnvironmentState(
            timestamp=current_time,
            features=features,
            derived_metrics=derived_metrics,
            anomaly_score=anomaly_score,
            trend_indicators=trend_indicators,
            confidence_level=confidence_level
        )
        
        self.environment_history.append(state)
        self._update_correlations(features)
        
        return state
    
    def _calculate_derived_metrics(self, features: Dict[str, Any]) -> Dict[str, float]:
        """计算派生指标"""
        derived = {}
        
        # 数值特征的统计指标
        numerical_features = {k: v for k, v in features.items() 
                            if isinstance(v, (int, float)) and not k.endswith(('_quality', '_confidence'))}
        
        if numerical_features:
            values = list(numerical_features.values())
            derived["feature_mean"] = sum(values) / len(values)
            derived["feature_variance"] = sum((x - derived["feature_mean"]) ** 2 for x in values) / len(values)
            derived["feature_range"] = max(values) - min(values)
            derived["feature_count"] = len(values)
        
        # 质量指标
        quality_features = {k: v for k, v in features.items() if k.endswith('_quality')}
        if quality_features:
            derived["avg_quality"] = sum(quality_features.values()) / len(quality_features)
            derived["min_quality"] = min(quality_features.values())
        
        return derived
    
    def _calculate_anomaly_score(self, features: Dict[str, Any]) -> float:
        """计算异常分数"""
        if not self.baseline_metrics:
            # 建立基线
            self._establish_baseline(features)
            return 0.0
        
        anomaly_score = 0.0
        feature_count = 0
        
        for feature_name, value in features.items():
            if isinstance(value, (int, float)) and feature_name in self.baseline_metrics:
                baseline = self.baseline_metrics[feature_name]
                if baseline["std"] > 0:
                    z_score = abs(value - baseline["mean"]) / baseline["std"]
                    anomaly_score += min(z_score / 3.0, 1.0)  # 归一化到0-1
                    feature_count += 1
        
        return anomaly_score / max(feature_count, 1)
    
    def _establish_baseline(self, features: Dict[str, Any]):
        """建立基线指标"""
        if len(self.environment_history) < 50:
            return
        
        # 从历史数据计算基线
        historical_features = defaultdict(list)
        for state in list(self.environment_history)[-50:]:
            for feature_name, value in state.features.items():
                if isinstance(value, (int, float)):
                    historical_features[feature_name].append(value)
        
        for feature_name, values in historical_features.items():
            if len(values) > 10:
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                self.baseline_metrics[feature_name] = {
                    "mean": mean_val,
                    "std": math.sqrt(variance),
                    "min": min(values),
                    "max": max(values)
                }
    
    def _identify_trend_indicators(self, features: Dict[str, Any]) -> Dict[str, str]:
        """识别趋势指标"""
        trends = {}
        
        if len(self.environment_history) < 10:
            return trends
        
        # 分析最近的趋势
        recent_states = list(self.environment_history)[-10:]
        
        for feature_name in features:
            if isinstance(features[feature_name], (int, float)):
                values = []
                for state in recent_states:
                    if feature_name in state.features and isinstance(state.features[feature_name], (int, float)):
                        values.append(state.features[feature_name])
                
                if len(values) >= 5:
                    trend = self._calculate_trend(values)
                    trends[feature_name] = trend
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 3:
            return "insufficient_data"
        
        # 简单的趋势检测
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_ratio = (second_avg - first_avg) / (abs(first_avg) + 1e-6)
        
        if change_ratio > 0.05:
            return "increasing"
        elif change_ratio < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_overall_confidence(self, sensor_readings: Dict[str, SensorReading]) -> float:
        """计算整体置信度"""
        if not sensor_readings:
            return 0.0
        
        confidences = [reading.confidence for reading in sensor_readings.values()]
        qualities = [reading.quality_score for reading in sensor_readings.values()]
        
        avg_confidence = sum(confidences) / len(confidences)
        avg_quality = sum(qualities) / len(qualities)
        
        return (avg_confidence + avg_quality) / 2.0
    
    def _update_correlations(self, features: Dict[str, Any]):
        """更新特征相关性"""
        numerical_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        
        for feature1 in numerical_features:
            for feature2 in numerical_features:
                if feature1 != feature2:
                    # 简化的相关性计算（需要历史数据）
                    correlation = self._calculate_feature_correlation(feature1, feature2)
                    self.feature_correlations[feature1][feature2] = correlation
    
    def _calculate_feature_correlation(self, feature1: str, feature2: str) -> float:
        """计算特征相关性"""
        if len(self.environment_history) < 10:
            return 0.0
        
        values1 = []
        values2 = []
        
        for state in list(self.environment_history)[-20:]:
            if (feature1 in state.features and feature2 in state.features and
                isinstance(state.features[feature1], (int, float)) and
                isinstance(state.features[feature2], (int, float))):
                values1.append(state.features[feature1])
                values2.append(state.features[feature2])
        
        if len(values1) < 5:
            return 0.0
        
        # 皮尔逊相关系数
        n = len(values1)
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n
        
        numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))
        denominator1 = sum((x - mean1) ** 2 for x in values1)
        denominator2 = sum((x - mean2) ** 2 for x in values2)
        
        if denominator1 * denominator2 == 0:
            return 0.0
        
        return numerator / math.sqrt(denominator1 * denominator2)

class PredictiveAnalyzer:
    """预测分析器"""
    
    def __init__(self):
        self.prediction_models = {}
        self.prediction_accuracy = defaultdict(list)
        
    def generate_predictions(self, environment_states: List[EnvironmentState]) -> List[Prediction]:
        """生成预测"""
        if len(environment_states) < 10:
            return []
        
        predictions = []
        
        # 预测主要数值特征
        latest_state = environment_states[-1]
        
        for feature_name, current_value in latest_state.features.items():
            if isinstance(current_value, (int, float)) and not feature_name.endswith(('_quality', '_confidence')):
                prediction = self._predict_feature_value(feature_name, environment_states)
                if prediction:
                    predictions.append(prediction)
        
        return predictions
    
    def _predict_feature_value(self, feature_name: str, states: List[EnvironmentState]) -> Optional[Prediction]:
        """预测特征值"""
        # 提取历史值
        values = []
        timestamps = []
        
        for state in states:
            if feature_name in state.features and isinstance(state.features[feature_name], (int, float)):
                values.append(state.features[feature_name])
                timestamps.append(state.timestamp)
        
        if len(values) < 5:
            return None
        
        # 简单的趋势预测
        time_horizon = 300  # 5分钟后的预测
        
        # 线性趋势预测
        predicted_value, confidence = self._linear_trend_prediction(values, timestamps, time_horizon)
        
        # 识别影响因素
        contributing_factors = self._identify_contributing_factors(feature_name, states)
        
        return Prediction(
            target_feature=feature_name,
            predicted_value=predicted_value,
            confidence=confidence,
            time_horizon=time_horizon,
            contributing_factors=contributing_factors
        )
    
    def _linear_trend_prediction(self, values: List[float], timestamps: List[float], time_horizon: float) -> Tuple[float, float]:
        """线性趋势预测"""
        if len(values) < 2:
            return values[-1] if values else 0.0, 0.1
        
        # 计算线性回归
        n = len(values)
        x = list(range(n))  # 简化的时间索引
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return values[-1], 0.5
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 预测未来值
        future_x = n + (time_horizon / 60)  # 假设每分钟一个数据点
        predicted_value = slope * future_x + intercept
        
        # 基于历史拟合度计算置信度
        residuals = [values[i] - (slope * x[i] + intercept) for i in range(n)]
        rmse = math.sqrt(sum(r ** 2 for r in residuals) / n)
        
        # 简化的置信度计算
        confidence = max(0.1, 1.0 - (rmse / (abs(y_mean) + 1e-6)))
        
        return predicted_value, min(confidence, 0.95)
    
    def _identify_contributing_factors(self, target_feature: str, states: List[EnvironmentState]) -> List[str]:
        """识别影响因素"""
        # 简化的因素识别
        correlations = {}
        
        if len(states) < 10:
            return []
        
        target_values = []
        other_features = defaultdict(list)
        
        for state in states[-20:]:
            if target_feature in state.features and isinstance(state.features[target_feature], (int, float)):
                target_values.append(state.features[target_feature])
                
                for feature_name, value in state.features.items():
                    if feature_name != target_feature and isinstance(value, (int, float)):
                        other_features[feature_name].append(value)
        
        # 计算相关性
        for feature_name, feature_values in other_features.items():
            if len(feature_values) == len(target_values) and len(target_values) > 5:
                correlation = self._calculate_correlation(target_values, feature_values)
                if abs(correlation) > 0.5:  # 强相关性阈值
                    correlations[feature_name] = abs(correlation)
        
        # 返回最相关的前3个因素
        sorted_factors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [factor[0] for factor in sorted_factors[:3]]
    
    def _calculate_correlation(self, values1: List[float], values2: List[float]) -> float:
        """计算相关系数"""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        n = len(values1)
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n
        
        numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))
        denominator1 = sum((x - mean1) ** 2 for x in values1)
        denominator2 = sum((x - mean2) ** 2 for x in values2)
        
        if denominator1 * denominator2 == 0:
            return 0.0
        
        return numerator / math.sqrt(denominator1 * denominator2)

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.suppression_rules = {}
        
    def add_alert_rule(self, rule_id: str, condition: callable, level: AlertLevel, message_template: str):
        """添加告警规则"""
        self.alert_rules.append({
            "rule_id": rule_id,
            "condition": condition,
            "level": level,
            "message_template": message_template,
            "last_triggered": 0,
            "trigger_count": 0
        })
    
    def evaluate_alerts(self, environment_state: EnvironmentState, predictions: List[Prediction]) -> List[Alert]:
        """评估告警"""
        new_alerts = []
        current_time = time.time()
        
        context = {
            "environment_state": environment_state,
            "predictions": predictions,
            "current_time": current_time
        }
        
        for rule in self.alert_rules:
            try:
                # 检查抑制规则
                if self._is_suppressed(rule["rule_id"], current_time):
                    continue
                
                # 评估条件
                if rule["condition"](context):
                    alert_id = f"{rule['rule_id']}_{int(current_time)}"
                    
                    # 检查是否已存在相同告警
                    if not self._has_duplicate_alert(rule["rule_id"]):
                        alert = Alert(
                            alert_id=alert_id,
                            level=rule["level"],
                            message=rule["message_template"].format(**context),
                            source_sensor=rule["rule_id"],
                            context=context,
                            suggested_actions=self._generate_suggested_actions(rule["level"], context)
                        )
                        
                        new_alerts.append(alert)
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        
                        rule["last_triggered"] = current_time
                        rule["trigger_count"] += 1
                        
                        logger.warning(f"Alert triggered: {alert.message}")
            
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['rule_id']}: {e}")
        
        return new_alerts
    
    def _is_suppressed(self, rule_id: str, current_time: float) -> bool:
        """检查告警是否被抑制"""
        if rule_id in self.suppression_rules:
            suppression = self.suppression_rules[rule_id]
            if current_time < suppression["until"]:
                return True
        
        return False
    
    def _has_duplicate_alert(self, rule_id: str) -> bool:
        """检查是否有重复告警"""
        current_time = time.time()
        
        for alert in self.active_alerts.values():
            if (alert.source_sensor == rule_id and 
                current_time - alert.timestamp < 300):  # 5分钟内不重复
                return True
        
        return False
    
    def _generate_suggested_actions(self, level: AlertLevel, context: Dict[str, Any]) -> List[str]:
        """生成建议的行动"""
        actions = []
        
        if level == AlertLevel.CRITICAL or level == AlertLevel.EMERGENCY:
            actions.append("立即检查系统状态")
            actions.append("通知相关人员")
        
        if level == AlertLevel.WARNING:
            actions.append("监控相关指标")
            actions.append("准备应对措施")
        
        # 基于环境状态添加具体建议
        env_state = context.get("environment_state")
        if env_state and env_state.anomaly_score > 0.7:
            actions.append("调查异常数据源")
        
        return actions

class PerceptionSystem:
    """感知系统主类"""
    
    def __init__(self):
        self.sensor_manager = SensorManager()
        self.pattern_detector = PatternDetector()
        self.environment_modeler = EnvironmentModeler()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.alert_manager = AlertManager()
        
        self.processing_loop_running = False
        self.processing_interval = 10  # 10秒处理一次
        
        # 初始化基础告警规则
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self):
        """初始化默认告警规则"""
        # 高异常分数告警
        self.alert_manager.add_alert_rule(
            "high_anomaly",
            lambda ctx: ctx["environment_state"].anomaly_score > 0.8,
            AlertLevel.WARNING,
            "检测到高异常分数: {environment_state.anomaly_score:.2f}"
        )
        
        # 低置信度告警
        self.alert_manager.add_alert_rule(
            "low_confidence",
            lambda ctx: ctx["environment_state"].confidence_level < 0.5,
            AlertLevel.WARNING,
            "系统置信度较低: {environment_state.confidence_level:.2f}"
        )
        
        # 预测偏差告警
        self.alert_manager.add_alert_rule(
            "prediction_deviation",
            lambda ctx: any(p.confidence < 0.3 for p in ctx["predictions"]),
            AlertLevel.INFO,
            "预测置信度降低"
        )
    
    async def start_processing(self):
        """启动感知处理循环"""
        self.processing_loop_running = True
        
        while self.processing_loop_running:
            try:
                await self._process_perception_cycle()
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in perception processing cycle: {e}")
                await asyncio.sleep(5)
    
    async def _process_perception_cycle(self):
        """处理感知周期"""
        # 收集当前传感器读数
        current_readings = {}
        for sensor_id in self.sensor_manager.sensors:
            latest = self.sensor_manager.sensors[sensor_id]["last_reading"]
            if latest:
                current_readings[sensor_id] = latest
        
        if not current_readings:
            return
        
        # 更新环境状态
        environment_state = self.environment_modeler.update_environment_state(current_readings)
        
        # 检测模式
        for sensor_id, reading in current_readings.items():
            recent_readings = self.sensor_manager.get_recent_readings(sensor_id, 20)
            patterns = self.pattern_detector.detect_temporal_patterns(recent_readings)
            
            if patterns["patterns"]:
                logger.info(f"Detected patterns in {sensor_id}: {patterns['patterns']}")
        
        # 生成预测
        historical_states = list(self.environment_modeler.environment_history)
        predictions = self.predictive_analyzer.generate_predictions(historical_states)
        
        # 评估告警
        alerts = self.alert_manager.evaluate_alerts(environment_state, predictions)
        
        if alerts:
            for alert in alerts:
                logger.warning(f"New alert: {alert.level.name} - {alert.message}")
    
    def stop_processing(self):
        """停止感知处理"""
        self.processing_loop_running = False
    
    def add_sensor_data(self, sensor_id: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """添加传感器数据"""
        return self.sensor_manager.add_reading(sensor_id, value, metadata)
    
    def register_sensor(self, sensor_id: str, sensor_type: SensorType, config: Dict[str, Any] = None):
        """注册传感器"""
        self.sensor_manager.register_sensor(sensor_id, sensor_type, config)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        current_time = time.time()
        
        # 传感器状态
        sensor_stats = {}
        for sensor_id in self.sensor_manager.sensors:
            sensor_stats[sensor_id] = self.sensor_manager.get_sensor_statistics(sensor_id)
        
        # 最新环境状态
        latest_state = None
        if self.environment_modeler.environment_history:
            latest_state = list(self.environment_modeler.environment_history)[-1]
        
        # 活跃告警
        active_alerts = [
            alert for alert in self.alert_manager.active_alerts.values()
            if current_time - alert.timestamp < 3600  # 1小时内的告警
        ]
        
        return {
            "processing_active": self.processing_loop_running,
            "sensor_count": len(self.sensor_manager.sensors),
            "sensor_statistics": sensor_stats,
            "environment_state": {
                "anomaly_score": latest_state.anomaly_score if latest_state else 0,
                "confidence_level": latest_state.confidence_level if latest_state else 0,
                "feature_count": len(latest_state.features) if latest_state else 0
            } if latest_state else None,
            "active_alerts": len(active_alerts),
            "alert_levels": {
                level.name: len([a for a in active_alerts if a.level == level])
                for level in AlertLevel
            }
        }