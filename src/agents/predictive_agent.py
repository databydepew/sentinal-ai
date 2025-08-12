"""
Predictive Agent for Sentinel AI - Proactive Incident Prevention

This agent uses machine learning and statistical analysis to predict potential
incidents before they occur, enabling proactive remediation.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import uuid
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from models.incident import Incident, AgentAction, ActionType, IncidentType, IncidentSeverity
from services.gemini_client import GeminiClient
from agents.base_agent import BaseAgent, AgentCapability


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PredictiveAlert:
    """Represents a predictive alert for potential future incident"""
    alert_id: str
    predicted_incident_type: IncidentType
    predicted_severity: IncidentSeverity
    confidence: PredictionConfidence
    time_to_incident: timedelta  # Estimated time until incident occurs
    contributing_factors: List[str]
    recommended_actions: List[str]
    prevention_probability: float  # 0-1, likelihood prevention will work
    created_at: datetime


class PredictiveAgent(BaseAgent):
    """
    Predictive Agent that analyzes patterns and trends to predict potential incidents
    before they occur, enabling proactive prevention rather than reactive response.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config, agent_type="predictive")
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.gemini_client = GeminiClient(config.get("gemini", {}))
        self.prediction_models = {}
        self.historical_window = timedelta(days=30)  # Look back period
        self.prediction_horizon = timedelta(hours=24)  # Look ahead period
        
        # Thresholds for different prediction types
        self.drift_threshold = 0.15
        self.performance_threshold = 0.10
        self.anomaly_threshold = 2.5  # Standard deviations
    
    async def initialize(self) -> None:
        """Initialize the predictive agent"""
        self.logger.info("Initializing PredictiveAgent...")
        await super().initialize()
        self.logger.info("PredictiveAgent initialized successfully")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability(name="analyze", description="Analyze patterns and trends"),
            AgentCapability(name="predict", description="Predict future incidents"),
            AgentCapability(name="alert", description="Generate predictive alerts")
        ]
    
    async def process_incident(self, incident) -> Dict[str, Any]:
        """Process an incident (not used for predictive agent, but required by base class)"""
        return {"status": "predictive_agent_not_used_for_incident_processing"}
        
    async def analyze_predictive_patterns(self, metrics_data: Dict[str, Any]) -> List[PredictiveAlert]:
        """
        Analyze historical patterns to predict potential future incidents
        
        Args:
            metrics_data: Historical metrics and performance data
            
        Returns:
            List of predictive alerts for potential incidents
        """
        alerts = []
        
        try:
            # 1. Data Drift Prediction
            drift_alerts = await self._predict_data_drift(metrics_data)
            alerts.extend(drift_alerts)
            
            # 2. Performance Degradation Prediction  
            performance_alerts = await self._predict_performance_issues(metrics_data)
            alerts.extend(performance_alerts)
            
            # 3. Resource Exhaustion Prediction
            resource_alerts = await self._predict_resource_issues(metrics_data)
            alerts.extend(resource_alerts)
            
            # 4. Model Staleness Prediction
            staleness_alerts = await self._predict_model_staleness(metrics_data)
            alerts.extend(staleness_alerts)
            
            # Sort by urgency (time to incident + confidence)
            alerts.sort(key=lambda x: (x.time_to_incident.total_seconds(), -x.confidence.value))
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error in predictive analysis: {e}")
            return []
    
    async def _predict_data_drift(self, metrics_data: Dict[str, Any]) -> List[PredictiveAlert]:
        """Predict potential data drift incidents"""
        alerts = []
        
        try:
            # Analyze feature distributions over time
            feature_stats = metrics_data.get("feature_statistics", {})
            
            for feature_name, stats in feature_stats.items():
                # Calculate drift velocity (rate of change)
                drift_velocity = self._calculate_drift_velocity(stats)
                
                if drift_velocity > self.drift_threshold:
                    # Predict when drift will become critical
                    time_to_critical = self._estimate_time_to_critical_drift(
                        current_drift=stats.get("current_drift", 0),
                        velocity=drift_velocity
                    )
                    
                    confidence = self._calculate_drift_confidence(drift_velocity, stats)
                    
                    alert = PredictiveAlert(
                        alert_id=f"drift_prediction_{feature_name}_{datetime.now().isoformat()}",
                        predicted_incident_type=IncidentType.DATA_DRIFT,
                        predicted_severity=self._estimate_drift_severity(drift_velocity),
                        confidence=confidence,
                        time_to_incident=time_to_critical,
                        contributing_factors=[
                            f"Feature {feature_name} showing drift velocity: {drift_velocity:.3f}",
                            f"Current drift score: {stats.get('current_drift', 0):.3f}",
                            "Trend analysis indicates accelerating drift pattern"
                        ],
                        recommended_actions=[
                            "Review data pipeline for upstream changes",
                            "Implement feature monitoring alerts",
                            "Consider model retraining with recent data",
                            "Validate data source integrity"
                        ],
                        prevention_probability=0.85,
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
                    
        except Exception as e:
            self.logger.error(f"Error predicting data drift: {e}")
            
        return alerts
    
    async def _predict_performance_issues(self, metrics_data: Dict[str, Any]) -> List[PredictiveAlert]:
        """Predict potential performance degradation incidents"""
        alerts = []
        
        try:
            performance_metrics = metrics_data.get("performance_metrics", {})
            
            # Analyze performance trends
            for metric_name, values in performance_metrics.items():
                if len(values) < 10:  # Need sufficient history
                    continue
                    
                # Calculate performance degradation rate
                degradation_rate = self._calculate_performance_trend(values)
                
                if abs(degradation_rate) > self.performance_threshold:
                    time_to_critical = self._estimate_time_to_performance_failure(
                        current_value=values[-1],
                        degradation_rate=degradation_rate,
                        threshold=performance_metrics.get(f"{metric_name}_threshold", 0.8)
                    )
                    
                    if time_to_critical < self.prediction_horizon:
                        confidence = self._calculate_performance_confidence(degradation_rate, values)
                        
                        alert = PredictiveAlert(
                            alert_id=f"performance_prediction_{metric_name}_{datetime.now().isoformat()}",
                            predicted_incident_type=IncidentType.PERFORMANCE_DEGRADATION,
                            predicted_severity=self._estimate_performance_severity(degradation_rate),
                            confidence=confidence,
                            time_to_incident=time_to_critical,
                            contributing_factors=[
                                f"Metric {metric_name} degrading at rate: {degradation_rate:.4f}/hour",
                                f"Current value: {values[-1]:.3f}",
                                "Trend analysis shows consistent degradation pattern"
                            ],
                            recommended_actions=[
                                "Scale up computational resources",
                                "Optimize model inference pipeline", 
                                "Review recent code deployments",
                                "Check for resource contention"
                            ],
                            prevention_probability=0.78,
                            created_at=datetime.now()
                        )
                        alerts.append(alert)
                        
        except Exception as e:
            self.logger.error(f"Error predicting performance issues: {e}")
            
        return alerts
    
    async def _predict_resource_issues(self, metrics_data: Dict[str, Any]) -> List[PredictiveAlert]:
        """Predict potential resource exhaustion incidents"""
        alerts = []
        
        try:
            resource_metrics = metrics_data.get("resource_usage", {})
            
            for resource_type, usage_data in resource_metrics.items():
                if len(usage_data) < 5:
                    continue
                    
                # Predict resource exhaustion
                usage_trend = self._calculate_resource_trend(usage_data)
                current_usage = usage_data[-1]
                
                if usage_trend > 0 and current_usage > 0.7:  # Growing usage above 70%
                    time_to_exhaustion = self._estimate_time_to_resource_exhaustion(
                        current_usage, usage_trend
                    )
                    
                    if time_to_exhaustion < self.prediction_horizon:
                        alert = PredictiveAlert(
                            alert_id=f"resource_prediction_{resource_type}_{datetime.now().isoformat()}",
                            predicted_incident_type=IncidentType.PERFORMANCE_DEGRADATION,
                            predicted_severity=IncidentSeverity.HIGH,
                            confidence=PredictionConfidence.HIGH,
                            time_to_incident=time_to_exhaustion,
                            contributing_factors=[
                                f"{resource_type} usage at {current_usage*100:.1f}%",
                                f"Growth rate: {usage_trend*100:.2f}%/hour",
                                "Resource consumption trending toward exhaustion"
                            ],
                            recommended_actions=[
                                f"Scale up {resource_type} capacity",
                                "Implement resource usage alerts",
                                "Review resource-intensive processes",
                                "Consider load balancing optimization"
                            ],
                            prevention_probability=0.92,
                            created_at=datetime.now()
                        )
                        alerts.append(alert)
                        
        except Exception as e:
            self.logger.error(f"Error predicting resource issues: {e}")
            
        return alerts
    
    async def _predict_model_staleness(self, metrics_data: Dict[str, Any]) -> List[PredictiveAlert]:
        """Predict when models will become stale and need retraining"""
        alerts = []
        
        try:
            model_metrics = metrics_data.get("model_performance", {})
            
            for model_id, performance_data in model_metrics.items():
                last_training = performance_data.get("last_training_date")
                if not last_training:
                    continue
                    
                days_since_training = (datetime.now() - last_training).days
                performance_decay = self._calculate_performance_decay(performance_data)
                
                # Predict when model will need retraining
                if performance_decay > 0.02:  # 2% decay threshold
                    time_to_retrain = self._estimate_time_to_retraining(
                        days_since_training, performance_decay
                    )
                    
                    if time_to_retrain < timedelta(days=7):  # Within a week
                        alert = PredictiveAlert(
                            alert_id=f"staleness_prediction_{model_id}_{datetime.now().isoformat()}",
                            predicted_incident_type=IncidentType.MODEL_DRIFT,
                            predicted_severity=IncidentSeverity.MEDIUM,
                            confidence=PredictionConfidence.MEDIUM,
                            time_to_incident=time_to_retrain,
                            contributing_factors=[
                                f"Model {model_id} trained {days_since_training} days ago",
                                f"Performance decay rate: {performance_decay:.4f}/day",
                                "Model showing signs of staleness"
                            ],
                            recommended_actions=[
                                "Schedule model retraining",
                                "Prepare fresh training dataset",
                                "Review model performance metrics",
                                "Plan gradual model rollout"
                            ],
                            prevention_probability=0.88,
                            created_at=datetime.now()
                        )
                        alerts.append(alert)
                        
        except Exception as e:
            self.logger.error(f"Error predicting model staleness: {e}")
            
        return alerts
    
    # Helper methods for calculations
    def _calculate_drift_velocity(self, stats: Dict) -> float:
        """Calculate rate of drift change"""
        drift_history = stats.get("drift_history", [])
        if len(drift_history) < 2:
            return 0.0
        return (drift_history[-1] - drift_history[0]) / len(drift_history)
    
    def _estimate_time_to_critical_drift(self, current_drift: float, velocity: float) -> timedelta:
        """Estimate when drift will become critical"""
        critical_threshold = 0.3
        if velocity <= 0:
            return timedelta(days=30)  # Default if no trend
        hours_to_critical = max(1, (critical_threshold - current_drift) / velocity)
        return timedelta(hours=hours_to_critical)
    
    def _calculate_drift_confidence(self, velocity: float, stats: Dict) -> PredictionConfidence:
        """Calculate confidence in drift prediction"""
        if velocity > 0.25:
            return PredictionConfidence.CRITICAL
        elif velocity > 0.20:
            return PredictionConfidence.HIGH
        elif velocity > 0.15:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _estimate_drift_severity(self, velocity: float) -> IncidentSeverity:
        """Estimate severity of predicted drift incident"""
        if velocity > 0.25:
            return IncidentSeverity.CRITICAL
        elif velocity > 0.20:
            return IncidentSeverity.HIGH
        elif velocity > 0.15:
            return IncidentSeverity.MEDIUM
        else:
            return IncidentSeverity.LOW
    
    def _calculate_performance_trend(self, values: List[float]) -> float:
        """Calculate performance degradation rate"""
        if len(values) < 2:
            return 0.0
        # Simple linear regression slope
        x = list(range(len(values)))
        n = len(values)
        slope = (n * sum(i * v for i, v in enumerate(values)) - sum(x) * sum(values)) / \
                (n * sum(i * i for i in x) - sum(x) ** 2)
        return slope
    
    def _estimate_time_to_performance_failure(self, current_value: float, 
                                            degradation_rate: float, threshold: float) -> timedelta:
        """Estimate when performance will hit failure threshold"""
        if degradation_rate >= 0:
            return timedelta(days=30)  # No degradation
        hours_to_failure = max(1, (threshold - current_value) / degradation_rate)
        return timedelta(hours=hours_to_failure)
    
    def _calculate_performance_confidence(self, rate: float, values: List[float]) -> PredictionConfidence:
        """Calculate confidence in performance prediction"""
        variance = np.var(values) if len(values) > 1 else 0
        if abs(rate) > 0.15 and variance < 0.01:
            return PredictionConfidence.HIGH
        elif abs(rate) > 0.10:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _estimate_performance_severity(self, rate: float) -> IncidentSeverity:
        """Estimate severity of performance incident"""
        if abs(rate) > 0.20:
            return IncidentSeverity.CRITICAL
        elif abs(rate) > 0.15:
            return IncidentSeverity.HIGH
        elif abs(rate) > 0.10:
            return IncidentSeverity.MEDIUM
        else:
            return IncidentSeverity.LOW
    
    def _calculate_resource_trend(self, usage_data: List[float]) -> float:
        """Calculate resource usage growth rate"""
        if len(usage_data) < 2:
            return 0.0
        return (usage_data[-1] - usage_data[0]) / len(usage_data)
    
    def _estimate_time_to_resource_exhaustion(self, current_usage: float, trend: float) -> timedelta:
        """Estimate when resource will be exhausted"""
        if trend <= 0:
            return timedelta(days=30)
        hours_to_exhaustion = max(1, (1.0 - current_usage) / trend)
        return timedelta(hours=hours_to_exhaustion)
    
    def _calculate_performance_decay(self, performance_data: Dict) -> float:
        """Calculate model performance decay rate"""
        performance_history = performance_data.get("accuracy_history", [])
        if len(performance_history) < 2:
            return 0.0
        return max(0, (performance_history[0] - performance_history[-1]) / len(performance_history))
    
    def _estimate_time_to_retraining(self, days_since_training: int, decay_rate: float) -> timedelta:
        """Estimate when model will need retraining"""
        # Assume model needs retraining when performance drops 5%
        retraining_threshold = 0.05
        if decay_rate <= 0:
            return timedelta(days=30)
        days_to_retrain = max(1, retraining_threshold / decay_rate)
        return timedelta(days=days_to_retrain)
    
    async def generate_prevention_plan(self, alert: PredictiveAlert) -> Dict[str, Any]:
        """
        Generate a detailed prevention plan for a predictive alert
        """
        prevention_plan = {
            "alert_id": alert.alert_id,
            "prevention_strategy": "proactive",
            "urgency": alert.confidence.value,
            "time_available": alert.time_to_incident.total_seconds() / 3600,  # hours
            "actions": []
        }
        
        # Generate specific actions based on incident type
        if alert.predicted_incident_type == IncidentType.DATA_DRIFT:
            prevention_plan["actions"] = [
                {"action": "data_validation", "priority": 1, "estimated_time": "30min"},
                {"action": "pipeline_review", "priority": 2, "estimated_time": "1hr"},
                {"action": "monitoring_setup", "priority": 3, "estimated_time": "45min"}
            ]
        elif alert.predicted_incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            prevention_plan["actions"] = [
                {"action": "resource_scaling", "priority": 1, "estimated_time": "15min"},
                {"action": "code_optimization", "priority": 2, "estimated_time": "2hr"},
                {"action": "load_testing", "priority": 3, "estimated_time": "1hr"}
            ]
        
        return prevention_plan
