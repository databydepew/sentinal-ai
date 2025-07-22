# # Add agent actions if requested
# if include_actions:
#     incident.actions = create_mock_actions(incident_type)

# # Mark as resolved if we have full history
# if include_diagnosis and include_remediation and include_actions:
#     incident.status = IncidentStatus.RESOLVED
#     incident.resolved_at = datetime.now()
#     incident.resolution_summary = f"Mock {incident_type.lower()} incident resolved successfully"

# return incident
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from models.incident import (
    Incident,
    IncidentStatus,
    IncidentType,
    IncidentSeverity,
    DiagnosisResult as Diagnosis,
    RemediationPlan,
    AgentAction,
)


def create_mock_incident(
    incident_type: str,
    severity: str,
    model_name: str,
    description: str = ""
) -> Incident:
    """Create a mock incident for testing purposes"""
    
    # Convert string types to enums
    incident_type_enum = getattr(IncidentType, incident_type, IncidentType.DATA_DRIFT)
    severity_enum = getattr(IncidentSeverity, severity, IncidentSeverity.MEDIUM)
    
    # Generate metadata
    metadata = _generate_mock_metadata(incident_type, model_name)
    
    # Add model name to metadata
    metadata["source_model_name"] = model_name
    
    # Create the incident
    incident = Incident(
        incident_id=str(uuid.uuid4()),
        title=f"{incident_type.replace('_', ' ').title()} in {model_name}",
        description=description or f"Mock {incident_type.lower()} incident for testing",
        status=IncidentStatus.OPEN,
        severity=severity_enum,
        incident_type=incident_type_enum,
        source="mock_generator",
        category="ml_model",
        affected_systems=[model_name],
        affected_services=["ml_inference", "model_serving"],
        created_by="mock_system",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata=metadata
    )
    
    return incident


def create_mock_diagnosis(incident_type: str) -> Diagnosis:
    """Create a mock diagnosis for an incident"""

    diagnosis_data = {
        "DATA_DRIFT": {
            "root_cause": "Significant shift in customer demographics data distribution",
            "confidence_score": 0.85,
            "contributing_factors": [
                "New customer acquisition campaign targeting different demographics",
                "Seasonal changes in customer behavior",
                "Data collection methodology changes",
            ],
            "recommended_actions": [
                "Retrain model with recent data",
                "Update feature engineering pipeline",
                "Implement drift monitoring alerts",
            ],
        },
        "PERFORMANCE_DEGRADATION": {
            "root_cause": "Model accuracy degraded due to concept drift in target variable",
            "confidence_score": 0.92,
            "contributing_factors": [
                "Market conditions changed affecting fraud patterns",
                "New fraud techniques not seen in training data",
                "Feature importance shifted over time",
            ],
            "recommended_actions": [
                "Retrain with recent labeled data",
                "Add new features to capture emerging patterns",
                "Implement online learning capabilities",
            ],
        },
        "MODEL_ERROR": {
            "root_cause": "Memory leak in model serving infrastructure causing timeouts",
            "confidence_score": 0.78,
            "contributing_factors": [
                "Increased traffic volume",
                "Memory management issues in serving container",
                "Resource allocation insufficient for current load",
            ],
            "recommended_actions": [
                "Restart model serving containers",
                "Increase memory allocation",
                "Implement better resource monitoring",
            ],
        },
    }

    data = diagnosis_data.get(incident_type, diagnosis_data["DATA_DRIFT"])

    return Diagnosis(
        diagnosis_id=str(uuid.uuid4()),
        root_cause=data["root_cause"],
        confidence_score=data["confidence_score"],
        contributing_factors=data["contributing_factors"],
        recommended_actions=data["recommended_actions"],
        technical_details={
            "analysis_method": "AI-powered diagnosis",
            "data_sources": ["monitoring_logs", "performance_metrics", "error_logs"],
            "analysis_duration_minutes": random.randint(5, 15),
        },
        created_at=datetime.now(),
    )


def create_mock_remediation_plan(incident_type: str) -> RemediationPlan:
    """Create a mock remediation plan for an incident"""

    plan_data = {
        "DATA_DRIFT": {
            "description": "Comprehensive data drift remediation including model retraining and monitoring updates",
            "steps": [
                {
                    "action": "Collect recent training data",
                    "duration_minutes": 30,
                    "resources": {"compute_units": 2, "storage_gb": 50},
                },
                {
                    "action": "Retrain model with updated data",
                    "duration_minutes": 120,
                    "resources": {
                        "compute_units": 8,
                        "storage_gb": 100,
                        "gpu_units": 2,
                    },
                },
                {
                    "action": "Validate model performance",
                    "duration_minutes": 45,
                    "resources": {"compute_units": 4, "storage_gb": 20},
                },
                {
                    "action": "Deploy updated model",
                    "duration_minutes": 30,
                    "resources": {"compute_units": 2, "storage_gb": 10},
                },
            ],
            "risk_level": "MEDIUM",
            "estimated_duration": 225,
        },
        "PERFORMANCE_DEGRADATION": {
            "description": "Performance restoration through model optimization and retraining",
            "steps": [
                {
                    "action": "Analyze performance degradation patterns",
                    "duration_minutes": 20,
                    "resources": {"compute_units": 2, "storage_gb": 10},
                },
                {
                    "action": "Optimize hyperparameters",
                    "duration_minutes": 90,
                    "resources": {"compute_units": 6, "storage_gb": 30},
                },
                {
                    "action": "Retrain with optimized parameters",
                    "duration_minutes": 150,
                    "resources": {
                        "compute_units": 10,
                        "storage_gb": 150,
                        "gpu_units": 4,
                    },
                },
                {
                    "action": "A/B test new model",
                    "duration_minutes": 60,
                    "resources": {"compute_units": 4, "storage_gb": 20},
                },
            ],
            "risk_level": "HIGH",
            "estimated_duration": 320,
        },
        "MODEL_ERROR": {
            "description": "Infrastructure remediation and model redeployment",
            "steps": [
                {
                    "action": "Restart model serving containers",
                    "duration_minutes": 5,
                    "resources": {"compute_units": 1, "storage_gb": 5},
                },
                {
                    "action": "Increase resource allocation",
                    "duration_minutes": 15,
                    "resources": {"compute_units": 2, "storage_gb": 10},
                },
                {
                    "action": "Deploy with updated configuration",
                    "duration_minutes": 20,
                    "resources": {"compute_units": 4, "storage_gb": 15},
                },
                {
                    "action": "Monitor for stability",
                    "duration_minutes": 30,
                    "resources": {"compute_units": 1, "storage_gb": 5},
                },
            ],
            "risk_level": "LOW",
            "estimated_duration": 70,
        },
    }

    data = plan_data.get(incident_type, plan_data["DATA_DRIFT"])

    # Calculate total resources
    total_resources = {"compute_units": 0, "storage_gb": 0, "gpu_units": 0}
    for step in data["steps"]:
        for resource, amount in step["resources"].items():
            total_resources[resource] = total_resources.get(resource, 0) + amount

    return RemediationPlan(
        plan_id=str(uuid.uuid4()),
        description=data["description"],
        steps=data["steps"],
        estimated_duration_minutes=data["estimated_duration"],
        required_resources=total_resources,
        risk_level=data["risk_level"],
        rollback_plan={
            "description": "Rollback to previous model version if issues arise",
            "steps": [
                "Stop new model deployment",
                "Restore previous model version",
                "Verify system stability",
            ],
        },
        created_at=datetime.now(),
    )


def create_mock_actions(incident_type: str) -> List[AgentAction]:
    """Create mock agent actions for an incident"""

    actions = []
    base_time = datetime.now() - timedelta(minutes=30)

    # Detection action
    actions.append(
        AgentAction(
            action_id=str(uuid.uuid4()),
            agent_name="DriftDetectionAgent",
            action_type="detect_incident",
            timestamp=base_time,
            success=True,
            result_data={
                "detection_method": "statistical_analysis",
                "confidence": 0.89,
                "metrics": {"drift_score": 0.15, "threshold": 0.1},
            },
        )
    )

    # Diagnosis action
    actions.append(
        AgentAction(
            action_id=str(uuid.uuid4()),
            agent_name="DiagnosticAgent",
            action_type="diagnose_incident",
            timestamp=base_time + timedelta(minutes=5),
            success=True,
            result_data={
                "diagnosis_method": "ai_analysis",
                "root_cause_identified": True,
                "confidence_score": 0.85,
            },
        )
    )

    # Remediation planning action
    actions.append(
        AgentAction(
            action_id=str(uuid.uuid4()),
            agent_name="RemediationAgent",
            action_type="create_remediation_plan",
            timestamp=base_time + timedelta(minutes=10),
            success=True,
            result_data={
                "plan_created": True,
                "estimated_duration_minutes": 225,
                "risk_level": "MEDIUM",
            },
        )
    )

    # Cost-benefit analysis action
    actions.append(
        AgentAction(
            action_id=str(uuid.uuid4()),
            agent_name="EconomistAgent",
            action_type="cost_benefit_analysis",
            timestamp=base_time + timedelta(minutes=15),
            success=True,
            result_data={
                "estimated_cost_usd": 150.0,
                "business_impact_usd": 2500.0,
                "cost_benefit_ratio": 16.67,
                "recommendation": "APPROVE",
            },
        )
    )

    # Execution action
    actions.append(
        AgentAction(
            agent_name="VerificationAgent",
            action_type="execute_remediation",
            timestamp=base_time + timedelta(minutes=20),
            input_data={"plan_id": str(uuid.uuid4())},
            output_data={
                "execution_completed": True,
                "validation_passed": True,
                "deployment_successful": True,
            },
            success=True,
            duration_seconds=random.uniform(60, 300),
        )
    )

    return actions


def _generate_mock_metadata(incident_type: str, model_name: str) -> Dict[str, Any]:
    """Generate mock metadata for an incident"""

    base_metadata = {
        "detection_timestamp": datetime.now().isoformat(),
        "model_version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
        "environment": random.choice(["production", "staging"]),
        "affected_requests": random.randint(100, 10000),
        "monitoring_service": "sentinel-monitoring",
    }

    type_specific_metadata = {
        "DATA_DRIFT": {
            "drift_score": round(random.uniform(0.6, 0.95), 2),
            "feature_importance": {"age": 0.35, "income": 0.25, "tenure": 0.15},
            "drift_type": "covariate_shift",
            "monitoring_window_hours": 24,
        },
        "PERFORMANCE_DEGRADATION": {
            "accuracy_drop": round(random.uniform(0.05, 0.15), 3),
            "previous_accuracy": round(random.uniform(0.85, 0.95), 3),
            "current_accuracy": round(random.uniform(0.75, 0.85), 3),
            "degradation_period_hours": random.randint(12, 48),
        },
        "MODEL_ERROR": {
            "error_rate": round(random.uniform(0.1, 0.25), 3),
            "error_types": ["timeout", "memory_error", "connection_error"],
            "affected_endpoints": [f"{model_name}-endpoint"],
            "error_count_last_hour": random.randint(50, 200),
        },
    }

    base_metadata.update(type_specific_metadata.get(incident_type, {}))
    return base_metadata


def create_mock_incident_with_history(
    incident_type: str,
    severity: str,
    model_name: str,
    include_diagnosis: bool = True,
    include_remediation: bool = True,
    include_actions: bool = True,
) -> Incident:
    """Create a mock incident with optional history components"""

    # Convert string types to enum values
    incident_type_enum = IncidentType(incident_type)
    severity_enum = IncidentSeverity(severity)

    # Generate basic incident details
    incident = Incident(
        incident_type=incident_type_enum,
        severity=severity_enum,
        status=IncidentStatus.DETECTED,
        source_model_name=model_name,
        source_endpoint=f"{model_name}-endpoint",
        detection_agent="sentinel-monitor",
        title=f"{incident_type} detected in {model_name}",
        description=f"Automated detection of {incident_type.lower()} issue in {model_name} requiring investigation",
        metadata=_generate_mock_metadata(incident_type, model_name),
    )

    # Add diagnosis if requested
    if include_diagnosis:
        incident.diagnosis = create_mock_diagnosis(incident_type)
        incident.update_status(IncidentStatus.DIAGNOSING)

        # Add diagnostic agent action
        incident.add_action(
            AgentAction(
                agent_name="diagnostic-agent",
                action_type="DIAGNOSIS",
                timestamp=datetime.now(),
                input_data={
                    "incident_id": incident.incident_id,
                    "incident_type": incident_type,
                },
                output_data={"diagnosis_id": incident.diagnosis.diagnosis_id},
                success=True,
                duration_seconds=random.uniform(10, 60),
            )
        )

    # Add remediation plan if requested and we have diagnosis
    if include_remediation and include_diagnosis:
        incident.remediation_plan = create_mock_remediation_plan(incident_type)
        incident.update_status(IncidentStatus.PLANNING)

        # Add remediation agent action
        incident.add_action(
            AgentAction(
                agent_name="remediation-agent",
                action_type="PLANNING",
                timestamp=datetime.now(),
                input_data={
                    "incident_id": incident.incident_id,
                    "diagnosis_id": incident.diagnosis.diagnosis_id,
                },
                output_data={"plan_id": incident.remediation_plan.plan_id},
                success=True,
                duration_seconds=random.uniform(15, 90),
            )
        )

    # Add agent actions if requested
    if include_actions:
        actions = create_mock_actions(incident_type)
        for action in actions:
            incident.add_action(action)

    # Mark as resolved if we have full history
    if include_diagnosis and include_remediation and include_actions:
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now()
        incident.resolution_summary = (
            f"Mock {incident_type.lower()} incident resolved successfully"
        )

    return incident


def create_batch_mock_incidents(count: int = 5) -> List[Incident]:
    """Create a batch of mock incidents for testing"""

    incident_types = [
        "DATA_DRIFT",
        "PERFORMANCE_DEGRADATION",
        "MODEL_ERROR",
        "CONCEPT_DRIFT",
        "ANOMALY_DETECTION",
    ]
    severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    model_names = [
        "customer-churn-model",
        "fraud-detection-model",
        "recommendation-engine",
        "price-prediction-model",
    ]

    incidents = []

    for i in range(count):
        incident_type = random.choice(incident_types)
        severity = random.choice(severities)
        model_name = random.choice(model_names)

        # Create incident with some randomness in history
        include_diagnosis = random.choice([True, False])
        include_remediation = include_diagnosis and random.choice([True, False])
        include_actions = include_remediation and random.choice([True, False])

        incident = create_mock_incident_with_history(
            incident_type=incident_type,
            severity=severity,
            model_name=model_name,
            include_diagnosis=include_diagnosis,
            include_remediation=include_remediation,
            include_actions=include_actions,
        )

        # Randomize creation time
        incident.created_at = datetime.now() - timedelta(
            hours=random.randint(1, 72), minutes=random.randint(0, 59)
        )

        incidents.append(incident)

    return incidents


def get_mock_incident_scenarios() -> Dict[str, Dict[str, Any]]:
    """Get predefined mock incident scenarios for demonstrations"""

    return {
        "data_drift_customer_churn": {
            "incident_type": "DATA_DRIFT",
            "severity": "MEDIUM",
            "model_name": "customer-churn-model",
            "description": "Significant data drift detected in customer demographics features due to new marketing campaign targeting different age groups",
        },
        "performance_degradation_fraud": {
            "incident_type": "PERFORMANCE_DEGRADATION",
            "severity": "HIGH",
            "model_name": "fraud-detection-model",
            "description": "Model accuracy dropped from 95% to 87% over the past 24 hours due to emerging fraud patterns",
        },
        "model_error_recommendations": {
            "incident_type": "MODEL_ERROR",
            "severity": "CRITICAL",
            "model_name": "recommendation-engine",
            "description": "Model endpoint returning 500 errors for 15% of requests due to memory leak in serving infrastructure",
        },
        "concept_drift_pricing": {
            "incident_type": "CONCEPT_DRIFT",
            "severity": "HIGH",
            "model_name": "price-prediction-model",
            "description": "Concept drift detected in price prediction model due to market volatility and supply chain disruptions",
        },
        "anomaly_detection_user_behavior": {
            "incident_type": "ANOMALY_DETECTION",
            "severity": "MEDIUM",
            "model_name": "user-behavior-model",
            "description": "Anomalous user behavior patterns detected, potentially indicating data quality issues or system compromise",
        },
    }
