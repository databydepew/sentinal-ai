# Agent Development Guide

This document provides comprehensive information for developing custom agents in the Sentinel AI system.

## Table of Contents

- [Overview](#overview)
- [Agent Architecture](#agent-architecture)
- [Development Environment Setup](#development-environment-setup)
- [Creating a Custom Agent](#creating-a-custom-agent)
- [Agent Capabilities](#agent-capabilities)
- [Communication Patterns](#communication-patterns)
- [Testing Agents](#testing-agents)
- [Deployment](#deployment)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

Sentinel AI uses a plugin-based agent architecture that allows developers to create custom agents for specialized incident management tasks. Agents are autonomous components that can:

- Process incidents independently
- Communicate with other agents
- Execute actions on external systems
- Learn from historical data
- Provide specialized capabilities

### Agent Types

- **Core Agents**: Built-in agents (Conductor, Diagnosis, Remediation, Verification, Reporting)
- **Custom Agents**: User-developed agents for specific use cases
- **Third-party Agents**: Community-developed agents

## Agent Architecture

### Base Agent Class

All agents inherit from the `BaseAgent` class:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging

from src.models.incident import Incident, AgentAction
from src.models.agent_state import AgentState

class BaseAgent(ABC):
    """Base class for all Sentinel AI agents"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
    @abstractmethod
    async def process_incident(self, incident: Incident) -> AgentAction:
        """Process an incident and return the action taken"""
        pass
    
    @abstractmethod
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Handle inter-agent messages"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return self.capabilities
```

## Development Environment Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Project Structure

```
src/agents/
├── __init__.py
├── base_agent.py
├── conductor.py
├── diagnosis.py
├── remediation.py
├── verification.py
├── reporting.py
└── custom/
    ├── __init__.py
    ├── my_custom_agent.py
    └── another_agent.py
```

## Creating a Custom Agent

### Step 1: Define Agent Class

```python
# src/agents/custom/security_scanner_agent.py
from typing import List, Dict, Any
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.incident import Incident, AgentAction, ActionType, ActionStatus

class SecurityScannerAgent(BaseAgent):
    """Agent for security vulnerability scanning"""
    
    def __init__(self):
        super().__init__(
            agent_id="security_scanner",
            capabilities=[
                "vulnerability_scanning",
                "security_analysis",
                "compliance_checking"
            ]
        )
    
    async def process_incident(self, incident: Incident) -> AgentAction:
        """Process security-related incidents"""
        action = AgentAction(
            incident_id=incident.incident_id,
            agent_id=self.agent_id,
            action_type=ActionType.DIAGNOSTIC,
            name="Security Vulnerability Scan",
            description="Scanning for security vulnerabilities",
            status=ActionStatus.IN_PROGRESS,
            started_at=datetime.now()
        )
        
        try:
            # Perform security scan
            scan_results = await self._perform_security_scan(incident)
            
            # Update action with results
            action.output_data = {
                "vulnerabilities_found": len(scan_results.get("vulnerabilities", [])),
                "recommendations": await self._generate_recommendations(scan_results)
            }
            action.success = True
            action.status = ActionStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            action.error_message = str(e)
            action.success = False
            action.status = ActionStatus.FAILED
        
        finally:
            action.completed_at = datetime.now()
        
        return action
    
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Handle inter-agent messages"""
        message_type = message.get("type")
        
        if message_type == "scan_request":
            await self._handle_scan_request(message)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
    
    async def _perform_security_scan(self, incident: Incident) -> Dict[str, Any]:
        """Perform security vulnerability scan"""
        # Implementation details
        return {"vulnerabilities": []}
    
    async def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate remediation recommendations"""
        return ["Apply security patches", "Update configurations"]
    
    async def _handle_scan_request(self, message: Dict[str, Any]) -> None:
        """Handle scan request from other agents"""
        # Implementation for handling scan requests
        pass
```

### Step 2: Register Agent

```python
# src/agents/registry.py
from typing import Dict, Type
from src.agents.base_agent import BaseAgent
from src.agents.custom.security_scanner_agent import SecurityScannerAgent

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "security_scanner": SecurityScannerAgent,
    # Add other custom agents here
}

def create_agent(agent_type: str) -> BaseAgent:
    """Create agent instance"""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return AGENT_REGISTRY[agent_type]()
```

## Agent Capabilities

### Defining Capabilities

```python
class AgentCapability:
    """Agent capability definition"""
    
    def __init__(self, name: str, description: str, required_permissions: List[str] = None):
        self.name = name
        self.description = description
        self.required_permissions = required_permissions or []

# Example capabilities
CAPABILITIES = {
    "vulnerability_scanning": AgentCapability(
        name="vulnerability_scanning",
        description="Scan systems for security vulnerabilities",
        required_permissions=["read:systems", "execute:scans"]
    ),
    "model_retraining": AgentCapability(
        name="model_retraining",
        description="Retrain ML models with new data",
        required_permissions=["read:data", "write:models", "execute:training"]
    )
}
```

## Communication Patterns

### Message Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    RESULT_NOTIFICATION = "result_notification"
    COLLABORATION_REQUEST = "collaboration_request"
    ESCALATION = "escalation"

@dataclass
class AgentMessage:
    message_type: MessageType
    sender_id: str
    recipient_id: str
    incident_id: Optional[str]
    payload: Dict[str, Any]
    timestamp: datetime
```

## Testing Agents

### Unit Testing

```python
# tests/agents/test_security_scanner_agent.py
import pytest
from unittest.mock import Mock, AsyncMock

from src.agents.custom.security_scanner_agent import SecurityScannerAgent
from src.models.incident import Incident, IncidentSeverity, IncidentStatus

@pytest.fixture
def security_agent():
    return SecurityScannerAgent()

@pytest.fixture
def sample_incident():
    return Incident(
        incident_id="test-incident-123",
        title="Security vulnerability detected",
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.OPEN,
        affected_systems=["web-app-prod"]
    )

@pytest.mark.asyncio
async def test_process_incident_success(security_agent, sample_incident):
    # Process incident
    action = await security_agent.process_incident(sample_incident)
    
    # Assertions
    assert action.success is True
    assert "vulnerabilities_found" in action.output_data
    assert len(action.output_data["recommendations"]) > 0
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile.security-agent
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy agent code
COPY src/ src/
COPY config/ config/

# Set environment variables
ENV PYTHONPATH=/app
ENV AGENT_TYPE=security_scanner

# Run agent
CMD ["python", "-m", "src.agents.runner", "--agent-type", "security_scanner"]
```

### Kubernetes Deployment

```yaml
# k8s/security-agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-scanner-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: security-scanner-agent
  template:
    metadata:
      labels:
        app: security-scanner-agent
    spec:
      containers:
      - name: security-scanner
        image: sentinel-ai/security-scanner:latest
        env:
        - name: AGENT_ID
          value: "security-scanner"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## Best Practices

### Error Handling

```python
class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class AgentTimeoutError(AgentError):
    """Agent operation timeout"""
    pass

# Use structured error handling
async def process_with_retry(self, func, max_retries=3):
    """Execute function with retry logic"""
    for attempt in range(max_retries):
        try:
            return await func()
        except AgentTimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Logging and Monitoring

```python
import structlog
from prometheus_client import Counter, Histogram

# Structured logging
logger = structlog.get_logger()

# Metrics
incident_counter = Counter(
    'agent_incidents_processed_total', 
    'Total incidents processed', 
    ['agent_id', 'status']
)
processing_time = Histogram(
    'agent_processing_duration_seconds', 
    'Time spent processing incidents', 
    ['agent_id']
)

class MonitoredAgent(BaseAgent):
    async def process_incident(self, incident: Incident) -> AgentAction:
        start_time = time.time()
        
        try:
            logger.info(
                "Processing incident",
                agent_id=self.agent_id,
                incident_id=incident.incident_id
            )
            
            action = await super().process_incident(incident)
            incident_counter.labels(agent_id=self.agent_id, status='success').inc()
            return action
            
        except Exception as e:
            incident_counter.labels(agent_id=self.agent_id, status='error').inc()
            logger.error("Incident processing failed", error=str(e))
            raise
            
        finally:
            processing_time.labels(agent_id=self.agent_id).observe(
                time.time() - start_time
            )
```

## Examples

### Data Quality Agent

```python
class DataQualityAgent(BaseAgent):
    """Agent for data quality monitoring and validation"""
    
    def __init__(self):
        super().__init__(
            agent_id="data_quality",
            capabilities=[
                "data_validation",
                "quality_scoring",
                "anomaly_detection"
            ]
        )
    
    async def process_incident(self, incident: Incident) -> AgentAction:
        # Validate data quality
        quality_score = await self._calculate_quality_score(incident)
        
        # Detect anomalies
        anomalies = await self._detect_anomalies(incident)
        
        return AgentAction(
            incident_id=incident.incident_id,
            agent_id=self.agent_id,
            action_type=ActionType.DIAGNOSTIC,
            name="Data Quality Analysis",
            output_data={
                "quality_score": quality_score,
                "anomalies": anomalies
            },
            success=True
        )
    
    async def _calculate_quality_score(self, incident: Incident) -> float:
        # Implementation for quality scoring
        return 0.85
    
    async def _detect_anomalies(self, incident: Incident) -> List[Dict[str, Any]]:
        # Implementation for anomaly detection
        return []
```

This guide provides the foundation for developing custom agents in Sentinel AI. Follow these patterns and best practices to create robust, scalable agents that integrate seamlessly with the system.
