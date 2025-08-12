from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import json
import uuid
import logging
import asyncio

from models.incident import Incident, AgentAction, ActionType
from src.services.gemini_client import GeminiClient
from src.agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class DriftDetectionAgent(BaseAgent):
    """Agent responsible for detecting data drift, concept drift, and anomalies in ML models"""

    def __init__(
        self,
        gemini_client: GeminiClient,
        monitoring_interval: int = 60,
        drift_threshold: float = 0.15,
        anomaly_threshold: float = 0.8,
    ):
        """Initialize the drift detection agent

        Args:
            gemini_client: Client for Gemini API
            monitoring_interval: Interval in minutes between monitoring checks
            drift_threshold: Threshold for drift detection (0.0-1.0)
            anomaly_threshold: Threshold for anomaly detection (0.0-1.0)
        """
        super().__init__({"max_concurrent_tasks": 10}, "drift_detection")
        self.gemini_client = gemini_client
        self.monitoring_interval = monitoring_interval
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold
        self.monitored_models = {}  # Dict to track monitored models
        self.monitoring_tasks = {}  # Dict to track monitoring tasks

    async def initialize(self) -> None:
        """Initialize the agent with required services"""
        # No additional initialization required
        pass

    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent provides"""
        return [
            AgentCapability(
                name="detect_data_drift", description="Detect data distribution drift"
            ),
            AgentCapability(
                name="detect_concept_drift", description="Detect concept drift"
            ),
            AgentCapability(
                name="detect_anomalies",
                description="Detect anomalies in model behavior",
            ),
            AgentCapability(
                name="monitor_model", description="Continuously monitor model for drift"
            ),
        ]

    async def process_incident(self, incident: Incident) -> Dict[str, Any]:
        """Process an incident by analyzing drift patterns

        Args:
            incident: The incident to process

        Returns:
            Dict containing the drift analysis results
        """
        # Create drift analysis action
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.ANALYZE,
            incident_id=incident.id,
            details=f"Analyzing drift for {incident.type.value} incident"
        )

        try:
            # Perform drift analysis
            analysis = await self._analyze_drift(incident)

            # Generate drift report
            report = await self._generate_drift_report(analysis)

            # Update action with success
            action.success = True
            action.output_data = {
                "risk_score": analysis["overall_risk_score"],
                "recommendations": len(analysis["recommended_actions"]),
            }

            # Add action to incident
            incident.add_action(action)

            return {"drift_analysis": analysis, "drift_report": report}

        except Exception as e:
            action.output_data = {"error": str(e)}
            incident.add_action(action)
            raise

    async def start_monitoring(self, model_name: str) -> Dict[str, Any]:
        """Start continuous monitoring for a model

        Args:
            model_name: Name of the model to monitor

        Returns:
            Dict with monitoring status
        """
        if model_name in self.monitoring_tasks:
            return {"status": "already_monitoring", "model_name": model_name}

        # Register model for monitoring
        self.monitored_models[model_name] = {
            "started_at": datetime.now(),
            "last_check": None,
            "incidents_detected": 0,
        }

        # Start monitoring task
        task = asyncio.create_task(self._monitor_model_loop(model_name))
        self.monitoring_tasks[model_name] = task

        return {
            "status": "monitoring_started",
            "model_name": model_name,
            "monitoring_interval_minutes": self.monitoring_interval,
        }

    async def stop_monitoring(self, model_name: str) -> Dict[str, Any]:
        """Stop monitoring for a model

        Args:
            model_name: Name of the model to stop monitoring

        Returns:
            Dict with operation status
        """
        if model_name not in self.monitoring_tasks:
            return {"status": "not_monitoring", "model_name": model_name}

        # Cancel monitoring task
        task = self.monitoring_tasks.pop(model_name)
        task.cancel()

        # Keep model info for reference
        model_info = self.monitored_models.get(model_name, {})

        return {
            "status": "monitoring_stopped",
            "model_name": model_name,
            "monitoring_duration_hours": (
                datetime.now() - model_info.get("started_at", datetime.now())
            ).total_seconds()
            / 3600,
            "incidents_detected": model_info.get("incidents_detected", 0),
        }

    async def _monitor_model_loop(self, model_name: str) -> None:
        """Continuous monitoring loop for a model"""
        try:
            while True:
                # Perform monitoring check
                drift_detected = await self._check_model_drift(model_name)

                # Update last check time
                self.monitored_models[model_name]["last_check"] = datetime.now()

                # If drift detected, create incident
                if drift_detected:
                    await self._create_drift_incident(model_name)
                    self.monitored_models[model_name]["incidents_detected"] += 1

                # Wait for next check
                await asyncio.sleep(self.monitoring_interval * 60)

        except asyncio.CancelledError:
            logger.info(f"Monitoring stopped for model {model_name}")
        except Exception as e:
            logger.error(f"Error in monitoring loop for {model_name}: {e}")

    async def _check_model_drift(self, model_name: str) -> bool:
        """Check if model has drift

        Args:
            model_name: Name of the model to check

        Returns:
            True if drift detected, False otherwise
        """
        # Check data drift
        data_drift = await self._check_data_drift(model_name)

        # Check concept drift
        concept_drift = await self._check_concept_drift(model_name)

        # Check anomalies
        anomalies = await self._check_anomalies(model_name)

        # Calculate overall risk score
        risk_score = self._calculate_risk_score(data_drift, concept_drift, anomalies)

        # Determine if drift is significant enough to create incident
        return risk_score > self.drift_threshold

    async def _create_drift_incident(self, model_name: str) -> str:
        """Create a drift incident

        Args:
            model_name: Name of the model with drift

        Returns:
            Incident ID
        """
        # This would integrate with incident creation system
        # For now, just log the incident
        incident_id = str(uuid.uuid4())
        logger.info(f"Created drift incident {incident_id} for model {model_name}")
        return incident_id

    async def _check_data_drift(self, model_name: str) -> Dict[str, Any]:
        """Check for data drift in a model

        Args:
            model_name: Name of the model to check

        Returns:
            Dict with data drift information
        """
        # This would integrate with actual drift detection systems
        # For now, return mock data
        return {
            "drift_detected": False,
            "drift_magnitude": 0.05,
            "affected_features": [],
            "p_value": 0.72,
            "distribution_distance": 0.08,
        }

    async def _check_concept_drift(self, model_name: str) -> Dict[str, Any]:
        """Check for concept drift in a model

        Args:
            model_name: Name of the model to check

        Returns:
            Dict with concept drift information
        """
        # This would integrate with actual drift detection systems
        # For now, return mock data
        return {
            "concept_drift_detected": False,
            "performance_degradation": 0.03,
            "current_accuracy": 0.91,
            "baseline_accuracy": 0.94,
            "statistical_significance": 0.12,
        }

    async def _check_anomalies(self, model_name: str) -> Dict[str, Any]:
        """Check for anomalies in model predictions

        Args:
            model_name: Name of the model to check

        Returns:
            Dict with anomaly information
        """
        # This would integrate with actual anomaly detection systems
        # For now, return mock data
        return {
            "anomalies_detected": False,
            "anomaly_score": 0.2,
            "anomaly_count": 0,
            "anomaly_pattern": "none",
            "affected_segments": [],
        }

    def _generate_incident_description(
        self,
        data_drift: Dict[str, Any],
        concept_drift: Dict[str, Any],
        anomalies: Dict[str, Any],
    ) -> str:
        """Generate a human-readable incident description"""

        descriptions = []

        if data_drift.get("drift_detected", False):
            magnitude = data_drift.get("drift_magnitude", 0)
            affected_features = data_drift.get("affected_features", [])
            descriptions.append(f"Data drift detected with magnitude {magnitude:.3f}")
            if affected_features:
                descriptions.append(
                    f"Affected features: {', '.join(affected_features)}"
                )

        if concept_drift.get("concept_drift_detected", False):
            degradation = concept_drift.get("performance_degradation", 0)
            current_acc = concept_drift.get("current_accuracy", 0)
            baseline_acc = concept_drift.get("baseline_accuracy", 0)
            descriptions.append(
                f"Concept drift detected with {degradation:.1%} performance degradation"
            )
            descriptions.append(
                f"Accuracy dropped from {baseline_acc:.1%} to {current_acc:.1%}"
            )

        if anomalies.get("anomalies_detected", False):
            score = anomalies.get("anomaly_score", 0)
            count = anomalies.get("anomaly_count", 0)
            descriptions.append(f"Anomalies detected with score {score:.2f}")
            if count > 0:
                descriptions.append(f"Total anomalies: {count}")

        return (
            ". ".join(descriptions)
            if descriptions
            else "Model monitoring detected potential issues"
        )

    async def _analyze_drift(self, incident: Incident) -> Dict[str, Any]:
        """Perform detailed drift analysis for an incident"""
        model_name = incident.source_model_name

        # Extract existing monitoring data from incident metadata
        data_drift = (
            incident.metadata.get("data_drift", {}) if incident.metadata else {}
        )
        concept_drift = (
            incident.metadata.get("concept_drift", {}) if incident.metadata else {}
        )
        anomalies = incident.metadata.get("anomalies", {}) if incident.metadata else {}

        # Perform additional analysis if needed
        if not data_drift:
            data_drift = await self._check_data_drift(model_name)

        if not concept_drift:
            concept_drift = await self._check_concept_drift(model_name)

        if not anomalies:
            anomalies = await self._check_anomalies(model_name)

        # Compile comprehensive analysis
        analysis = {
            "model_name": model_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "anomalies": anomalies,
            "overall_risk_score": self._calculate_risk_score(
                data_drift, concept_drift, anomalies
            ),
            "recommended_actions": self._generate_recommendations(
                data_drift, concept_drift, anomalies
            ),
        }

        return analysis

    def _calculate_risk_score(
        self,
        data_drift: Dict[str, Any],
        concept_drift: Dict[str, Any],
        anomalies: Dict[str, Any],
    ) -> float:
        """Calculate overall risk score based on drift analysis"""
        risk_score = 0.0

        # Data drift contribution
        if data_drift.get("drift_detected", False):
            magnitude = data_drift.get("drift_magnitude", 0)
            risk_score += magnitude * 0.4

        # Concept drift contribution
        if concept_drift.get("concept_drift_detected", False):
            degradation = concept_drift.get("performance_degradation", 0)
            risk_score += degradation * 0.5

        # Anomaly contribution
        if anomalies.get("anomalies_detected", False):
            anomaly_score = anomalies.get("anomaly_score", 0)
            normalized_anomaly = min(anomaly_score / 5.0, 1.0)  # Normalize to 0-1
            risk_score += normalized_anomaly * 0.1

        return min(risk_score, 1.0)  # Cap at 1.0

    def _generate_recommendations(
        self,
        data_drift: Dict[str, Any],
        concept_drift: Dict[str, Any],
        anomalies: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []

        if data_drift.get("drift_detected", False):
            recommendations.append(
                "Retrain model with recent data to address data drift"
            )
            if data_drift.get("affected_features"):
                recommendations.append(
                    "Review feature engineering for affected features"
                )

        if concept_drift.get("concept_drift_detected", False):
            recommendations.append(
                "Investigate changes in target variable distribution"
            )
            recommendations.append(
                "Consider updating model architecture or hyperparameters"
            )

        if anomalies.get("anomalies_detected", False):
            recommendations.append("Investigate anomalous predictions and input data")
            recommendations.append("Review data quality and preprocessing pipeline")

        if not recommendations:
            recommendations.append("Continue monitoring for emerging patterns")

        return recommendations

    async def _generate_drift_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive drift report"""
        return {
            "report_id": str(uuid.uuid4()),
            "model_name": analysis["model_name"],
            "generated_at": datetime.now().isoformat(),
            "risk_score": analysis["overall_risk_score"],
            "summary": self._generate_report_summary(analysis),
            "detailed_findings": analysis,
            "recommendations": analysis["recommended_actions"],
            "next_monitoring_check": (
                datetime.now() + timedelta(minutes=self.monitoring_interval)
            ).isoformat(),
        }

    def _generate_report_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a summary for the drift report"""
        model_name = analysis["model_name"]
        risk_score = analysis["overall_risk_score"]

        risk_level = "LOW"
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.3:
            risk_level = "MEDIUM"

        return f"Drift analysis for {model_name} completed with {risk_level} risk level (score: {risk_score:.2f})"

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "monitoring_interval_minutes": self.monitoring_interval,
            "drift_threshold": self.drift_threshold,
            "anomaly_threshold": self.anomaly_threshold,
            "monitored_models": {
                model_name: {
                    "started_at": info["started_at"].isoformat(),
                    "last_check": info["last_check"].isoformat()
                    if info["last_check"]
                    else None,
                    "incidents_detected": info["incidents_detected"],
                }
                for model_name, info in self.monitored_models.items()
            },
            "active_monitoring_tasks": len(self.monitoring_tasks),
            "total_models_monitored": len(self.monitored_models),
        }
