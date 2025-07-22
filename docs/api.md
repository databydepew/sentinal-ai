# Sentinel AI API Documentation

This document provides comprehensive information about the Sentinel AI API endpoints, data models, and usage patterns.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Data Models](#data-models)
- [Incident Management API](#incident-management-api)
- [Agent Management API](#agent-management-api)
- [Monitoring API](#monitoring-api)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

Sentinel AI provides a RESTful API for managing ML incidents, orchestrating AI agents, and monitoring system health. The API is designed to be intuitive and follows standard HTTP conventions.

**Base URL**: `https://api.sentinel-ai.com/v1`

## Authentication

All API requests require authentication using API keys or OAuth 2.0 tokens.

```http
Authorization: Bearer <your-api-token>
Content-Type: application/json
```

## Data Models

### Incident

```json
{
  "incident_id": "string (UUID)",
  "title": "string",
  "description": "string",
  "status": "open|in_progress|resolved|closed|escalated",
  "severity": "low|medium|high|critical",
  "incident_type": "string",
  "source": "string",
  "category": "string",
  "affected_systems": ["string"],
  "affected_services": ["string"],
  "created_by": "string",
  "assigned_to": "string|null",
  "created_at": "datetime (ISO 8601)",
  "updated_at": "datetime (ISO 8601)",
  "resolved_at": "datetime|null (ISO 8601)",
  "resolution_summary": "string|null",
  "resolution_time_seconds": "integer|null",
  "response_time_seconds": "integer|null",
  "diagnosis": "DiagnosisResult|null",
  "remediation_plan": "RemediationPlan|null",
  "actions": ["AgentAction"],
  "metadata": {},
  "tags": ["string"]
}
```

### DiagnosisResult

```json
{
  "diagnosis_id": "string (UUID)",
  "root_cause": "string",
  "confidence_score": "float (0.0-1.0)",
  "contributing_factors": ["string"],
  "recommended_actions": ["string"],
  "technical_details": {},
  "created_at": "datetime (ISO 8601)"
}
```

### RemediationPlan

```json
{
  "plan_id": "string (UUID)",
  "description": "string",
  "steps": [
    {
      "action": "string",
      "duration_minutes": "integer",
      "resources": {}
    }
  ],
  "estimated_duration_minutes": "integer",
  "required_resources": {},
  "risk_assessment": {},
  "created_at": "datetime (ISO 8601)"
}
```

### AgentAction

```json
{
  "action_id": "string (UUID)",
  "incident_id": "string (UUID)",
  "agent_id": "string",
  "action_type": "diagnostic|remediation|monitoring|notification",
  "name": "string",
  "description": "string",
  "status": "pending|in_progress|completed|failed|cancelled",
  "input_data": {},
  "output_data": {},
  "success": "boolean",
  "error_message": "string|null",
  "duration_seconds": "float|null",
  "created_at": "datetime (ISO 8601)",
  "started_at": "datetime|null (ISO 8601)",
  "completed_at": "datetime|null (ISO 8601)"
}
```

## Incident Management API

### Create Incident

```http
POST /incidents
```

**Request Body:**

```json
{
  "title": "Model accuracy degradation detected",
  "description": "Customer churn model accuracy dropped from 95% to 87%",
  "severity": "high",
  "incident_type": "performance_degradation",
  "source": "monitoring_system",
  "category": "ml_model",
  "affected_systems": ["customer-churn-model"],
  "affected_services": ["prediction-api"],
  "created_by": "system"
}
```

**Response:**

```json
{
  "incident_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "open",
  "created_at": "2024-01-15T10:30:00Z",
  "message": "Incident created successfully"
}
```

### Get Incident

```http
GET /incidents/{incident_id}
```

**Response:**

```json
{
  "incident_id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "Model accuracy degradation detected",
  "status": "in_progress",
  "severity": "high",
  "diagnosis": {
    "root_cause": "Data drift in customer demographics",
    "confidence_score": 0.85
  },
  "actions": [
    {
      "action_id": "456e7890-e89b-12d3-a456-426614174001",
      "agent_id": "diagnosis_agent",
      "status": "completed",
      "name": "Data drift analysis"
    }
  ]
}
```

### List Incidents

```http
GET /incidents?status=open&severity=high&limit=10&offset=0
```

**Query Parameters:**

- `status` (optional): Filter by incident status
- `severity` (optional): Filter by severity level
- `assigned_to` (optional): Filter by assignee
- `limit` (optional): Number of results per page (default: 20, max: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**

```json
{
  "incidents": [
    {
      "incident_id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Model accuracy degradation detected",
      "status": "open",
      "severity": "high",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_count": 1,
  "has_more": false
}
```

### Update Incident

```http
PUT /incidents/{incident_id}
```

**Request Body:**

```json
{
  "status": "resolved",
  "assigned_to": "john.doe@company.com",
  "resolution_summary": "Model retrained with recent data, accuracy restored to 94%"
}
```

### Delete Incident

```http
DELETE /incidents/{incident_id}
```

## Agent Management API

### Trigger Agent Action

```http
POST /incidents/{incident_id}/actions
```

**Request Body:**

```json
{
  "agent_id": "remediation_agent",
  "action_type": "remediation",
  "name": "Retrain model",
  "description": "Retrain the customer churn model with recent data",
  "input_data": {
    "model_name": "customer-churn-model",
    "training_data_path": "gs://bucket/recent_data.csv"
  }
}
```

**Response:**

```json
{
  "action_id": "789e0123-e89b-12d3-a456-426614174002",
  "status": "pending",
  "message": "Agent action queued successfully"
}
```

### Get Agent Action Status

```http
GET /actions/{action_id}
```

**Response:**

```json
{
  "action_id": "789e0123-e89b-12d3-a456-426614174002",
  "status": "in_progress",
  "progress": 45,
  "estimated_completion": "2024-01-15T12:00:00Z",
  "output_data": {
    "current_step": "Data preprocessing",
    "steps_completed": 2,
    "total_steps": 5
  }
}
```

## Monitoring API

### Get System Health

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "message_queue": "healthy",
    "ai_agents": "healthy",
    "monitoring": "healthy"
  },
  "metrics": {
    "active_incidents": 5,
    "agents_running": 12,
    "avg_response_time_ms": 150
  }
}
```

### Get Incident Metrics

```http
GET /metrics/incidents?period=7d
```

**Response:**

```json
{
  "period": "7d",
  "total_incidents": 25,
  "resolved_incidents": 20,
  "resolution_rate": 0.8,
  "avg_resolution_time_hours": 2.5,
  "severity_breakdown": {
    "critical": 2,
    "high": 8,
    "medium": 12,
    "low": 3
  },
  "top_incident_types": [
    {"type": "performance_degradation", "count": 10},
    {"type": "data_drift", "count": 8},
    {"type": "model_error", "count": 7}
  ]
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid incident severity level",
    "details": {
      "field": "severity",
      "allowed_values": ["low", "medium", "high", "critical"]
    }
  },
  "request_id": "req_123456789"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (e.g., duplicate incident)
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Examples

### Complete Incident Workflow

```bash
# 1. Create an incident
curl -X POST https://api.sentinel-ai.com/v1/incidents \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Model accuracy degradation",
    "description": "Fraud detection model accuracy dropped",
    "severity": "high",
    "incident_type": "performance_degradation"
  }'

# 2. Trigger diagnosis
curl -X POST https://api.sentinel-ai.com/v1/incidents/123e4567-e89b-12d3-a456-426614174000/actions \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "diagnosis_agent",
    "action_type": "diagnostic",
    "name": "Analyze model performance"
  }'

# 3. Check incident status
curl -X GET https://api.sentinel-ai.com/v1/incidents/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer $API_TOKEN"

# 4. Resolve incident
curl -X PUT https://api.sentinel-ai.com/v1/incidents/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "resolved",
    "resolution_summary": "Model retrained successfully"
  }'
```

### Python SDK Example

```python
from sentinel_ai import SentinelClient

# Initialize client
client = SentinelClient(api_token="your-api-token")

# Create incident
incident = client.incidents.create(
    title="Data drift detected",
    description="Customer behavior patterns have shifted",
    severity="medium",
    incident_type="data_drift"
)

# Trigger diagnosis
action = client.incidents.trigger_action(
    incident_id=incident.incident_id,
    agent_id="diagnosis_agent",
    action_type="diagnostic"
)

# Wait for completion and get results
result = client.actions.wait_for_completion(action.action_id)
print(f"Diagnosis complete: {result.output_data}")
```

## Rate Limits

- **Standard tier**: 1000 requests per hour
- **Premium tier**: 10000 requests per hour
- **Enterprise tier**: Custom limits

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## Webhooks

Sentinel AI supports webhooks for real-time notifications:

```json
{
  "event": "incident.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "incident_id": "123e4567-e89b-12d3-a456-426614174000",
    "severity": "high",
    "status": "open"
  }
}
```

### Supported Events

- `incident.created`
- `incident.updated`
- `incident.resolved`
- `action.started`
- `action.completed`
- `action.failed`