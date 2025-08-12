from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json


class IncidentStatus(Enum):
    """Incident status enumeration"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


class IncidentType(Enum):
    """Incident type enumeration"""
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MODEL_ERROR = "model_error"
    CONCEPT_DRIFT = "concept_drift"
    ANOMALY_DETECTION = "anomaly_detection"


class IncidentSeverity(Enum):
    """Incident severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Alias for backward compatibility
Severity = IncidentSeverity


class ActionType(Enum):
    """Action type enumeration"""
    DIAGNOSTIC = "diagnostic"
    REMEDIATION = "remediation"
    MONITORING = "monitoring"
    NOTIFICATION = "notification"


class ActionStatus(Enum):
    """Action status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DiagnosisResult:
    """Diagnosis result data class"""
    diagnosis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_cause: str = ""
    confidence_score: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    technical_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RemediationPlan:
    """Remediation plan data class"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration_minutes: int = 0
    required_resources: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentAction:
    """Agent action data class"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str = ""
    agent_id: str = ""
    action_type: ActionType = ActionType.DIAGNOSTIC
    name: str = ""
    description: str = ""
    status: ActionStatus = ActionStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Incident:
    """Main incident data class"""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: IncidentStatus = IncidentStatus.OPEN
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    incident_type: str = ""
    source: str = ""
    category: str = ""
    affected_systems: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    created_by: str = ""
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution_summary: Optional[str] = None
    resolution_time_seconds: Optional[int] = None
    response_time_seconds: Optional[int] = None
    
    # Related data
    diagnosis: Optional[DiagnosisResult] = None
    remediation_plan: Optional[RemediationPlan] = None
    actions: List[AgentAction] = field(default_factory=list)
    
    # Relationships
    parent_incident_id: Optional[str] = None
    child_incidents: List[str] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def update_status(self, new_status: IncidentStatus) -> None:
        """Update the incident status"""
        self.status = new_status
        self.updated_at = datetime.now()
        if new_status == IncidentStatus.RESOLVED:
            self.resolved_at = datetime.now()
    
    def add_action(self, action: AgentAction) -> None:
        """Add an action to the incident"""
        action.incident_id = self.incident_id
        self.actions.append(action)
        self.updated_at = datetime.now()
    
    def get_actions_by_status(self, status: ActionStatus) -> List[AgentAction]:
        """Get actions by status"""
        return [action for action in self.actions if action.status == status]
    
    def get_actions_by_type(self, action_type: ActionType) -> List[AgentAction]:
        """Get actions by type"""
        return [action for action in self.actions if action.action_type == action_type]
    
    def is_overdue(self, sla_seconds: int = 3600) -> bool:
        """Check if incident is overdue based on SLA"""
        if not self.created_at:
            return False
        
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > sla_seconds
    
    def calculate_response_time(self) -> None:
        """Calculate response time if first action exists"""
        if not self.actions or self.response_time_seconds is not None:
            return
        
        first_action = min(self.actions, key=lambda x: x.created_at)
        if self.created_at:
            self.response_time_seconds = int((first_action.created_at - self.created_at).total_seconds())
    
    def get_action_summary(self) -> Dict[str, int]:
        """Get summary of actions by status"""
        summary = {}
        for status in ActionStatus:
            summary[status.value] = len(self.get_actions_by_status(status))
        return summary
    
    def add_related_incident(self, incident_id: str) -> None:
        """Add related incident"""
        if incident_id not in self.related_incidents:
            self.related_incidents.append(incident_id)
            self.updated_at = datetime.now()
    
    def set_parent_incident(self, parent_id: str) -> None:
        """Set parent incident"""
        self.parent_incident_id = parent_id
        self.updated_at = datetime.now()
    
    def add_child_incident(self, child_id: str) -> None:
        """Add child incident"""
        if child_id not in self.child_incidents:
            self.child_incidents.append(child_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Incident':
        """Create instance from dictionary"""
        return cls(**data)


# Factory functions for creating incident objects

def create_incident(title: str, description: str, severity: Severity, 
                   source: str, category: str, incident_type: str, 
                   created_by: str, **kwargs) -> Incident:
    """Factory function to create incidents"""
    return Incident(
        title=title,
        description=description,
        severity=severity,
        source=source,
        category=category,
        incident_type=incident_type,
        created_by=created_by,
        **kwargs
    )


def create_agent_action(incident_id: str, agent_id: str, action_type: ActionType,
                       name: str, description: str, **kwargs) -> AgentAction:
    """Factory function to create agent actions"""
    return AgentAction(
        incident_id=incident_id,
        agent_id=agent_id,
        action_type=action_type,
        name=name,
        description=description,
        **kwargs
    )


def create_diagnostic_action(incident_id: str, agent_id: str, name: str, 
                           description: str, **kwargs) -> AgentAction:
    """Factory function to create diagnostic actions"""
    return create_agent_action(
        incident_id=incident_id,
        agent_id=agent_id,
        action_type=ActionType.DIAGNOSTIC,
        name=name,
        description=description,
        **kwargs
    )


def create_remediation_action(incident_id: str, agent_id: str, name: str,
                            description: str, **kwargs) -> AgentAction:
    """Factory function to create remediation actions"""
    return create_agent_action(
        incident_id=incident_id,
        agent_id=agent_id,
        action_type=ActionType.REMEDIATION,
        name=name,
        description=description,
        **kwargs
    )


# Utility functions for serialization and validation

def serialize_incident(incident: Incident) -> str:
    """Serialize incident to JSON string"""
    return incident.json()


def deserialize_incident(data: str) -> Incident:
    """Deserialize JSON string to incident"""
    parsed_data = json.loads(data)
    return Incident.from_dict(parsed_data)


def serialize_agent_action(action: AgentAction) -> str:
    """Serialize agent action to JSON string"""
    return action.json()


def deserialize_agent_action(data: str) -> AgentAction:
    """Deserialize JSON string to agent action"""
    parsed_data = json.loads(data)
    return AgentAction.from_dict(parsed_data)


def validate_incident_data(data: Dict[str, Any]) -> bool:
    """Validate incident data against schema"""
    try:
        Incident(**data)
        return True
    except Exception:
        return False


def validate_agent_action_data(data: Dict[str, Any]) -> bool:
    """Validate agent action data against schema"""
    try:
        AgentAction(**data)
        return True
    except Exception:
        return False


# Helper functions for incident management

def get_incident_priority_score(incident: Incident) -> int:
    """Calculate priority score for incident triage"""
    severity_scores = {
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4
    }
    
    base_score = severity_scores.get(incident.severity, 1)
    
    # Increase score for incidents affecting multiple systems
    if len(incident.affected_systems) > 1:
        base_score += 1
    
    # Increase score for incidents with many affected services
    if len(incident.affected_services) > 2:
        base_score += 1
    
    # Increase score for escalated incidents
    if incident.status == IncidentStatus.ESCALATED:
        base_score += 2
    
    return min(base_score, 10)  # Cap at 10


def should_auto_escalate(incident: Incident, max_duration_hours: int = 4) -> bool:
    """Check if incident should be auto-escalated"""
    if not incident.created_at:
        return False
    
    duration_hours = (datetime.now() - incident.created_at).total_seconds() / 3600
    
    # Auto-escalate critical incidents after 1 hour
    if incident.severity == Severity.CRITICAL and duration_hours > 1:
        return True
    
    # Auto-escalate high severity incidents after 2 hours
    if incident.severity == Severity.HIGH and duration_hours > 2:
        return True
    
    # Auto-escalate any incident after max duration
    if duration_hours > max_duration_hours:
        return True
    
    return False


def get_incident_metrics(incidents: List[Incident]) -> Dict[str, Any]:
    """Calculate metrics for a list of incidents"""
    if not incidents:
        return {}
    
    total_incidents = len(incidents)
    resolved_incidents = [i for i in incidents if i.status == IncidentStatus.RESOLVED]
    
    # Calculate resolution times for resolved incidents
    resolution_times = []
    for incident in resolved_incidents:
        if incident.resolution_time_seconds:
            resolution_times.append(incident.resolution_time_seconds)
    
    metrics = {
        'total_incidents': total_incidents,
        'resolved_incidents': len(resolved_incidents),
        'resolution_rate': len(resolved_incidents) / total_incidents if total_incidents > 0 else 0,
        'average_resolution_time_seconds': sum(resolution_times) / len(resolution_times) if resolution_times else 0,
        'severity_breakdown': {},
        'status_breakdown': {}
    }
    
    # Severity breakdown
    for severity in Severity:
        count = len([i for i in incidents if i.severity == severity])
        metrics['severity_breakdown'][severity.value] = count
    
    # Status breakdown
    for status in IncidentStatus:
        count = len([i for i in incidents if i.status == status])
        metrics['status_breakdown'][status.value] = count
    
    return metrics
