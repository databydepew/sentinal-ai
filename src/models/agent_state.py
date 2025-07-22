from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
import uuid


class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentCapability:
    """Agent capability data class"""
    name: str
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class AgentState(BaseModel):
    """Agent state model"""
    agent_id: str
    incident_id: str
    status: str
    last_updated: datetime
    
    # Additional fields for conductor agent
    active_incidents: List[str] = field(default_factory=list)
    registered_agents: Dict[str, Any] = field(default_factory=dict)
    total_incidents_processed: int = 0
    incidents_resolved_24h: int = 0
    average_resolution_time_minutes: float = 0.0
    capabilities: List[AgentCapability] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def update_status(self, status: AgentStatus, message: str = "") -> None:
        """Update agent status"""
        self.status = status.value if isinstance(status, AgentStatus) else status
        self.last_updated = datetime.now()
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def record_error(self, error_message: str) -> None:
        """Record an error message"""
        self.errors.append(f"{datetime.now().isoformat()}: {error_message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Convert from dictionary for deserialization"""
        self.agent_id = data.get("agent_id")
        self.incident_id = data.get("incident_id")
        self.status = data.get("status")
        self.last_updated = data.get("last_updated")


def serialize_agent_state(agent_state: AgentState) -> str:
    """Serialize agent state to JSON string"""
    return agent_state.model_dump()


def deserialize_agent_state(data: str) -> AgentState:
    """Deserialize JSON string to agent state"""
    parsed_data =AgentState.model_validate_json(data)
    return parsed_data


def validate_agent_state_data(data: Dict[str, Any]) -> bool:
    """Validate agent state data against schema"""
    try:
        AgentState(**data)
        return True
    except Exception:
        return False


def create_agent_state(agent_id: str, incident_id: str = "", status: str = "idle") -> AgentState:
    """Factory function to create agent state"""
    return AgentState(
        agent_id=agent_id,
        incident_id=incident_id,
        status=status,
        last_updated=datetime.now()
    )


def create_agent_capability(name: str, description: str, enabled: bool = True, **kwargs) -> AgentCapability:
    """Factory function to create agent capability"""
    return AgentCapability(
        name=name,
        description=description,
        enabled=enabled,
        parameters=kwargs
    )
