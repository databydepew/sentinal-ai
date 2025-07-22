# Sentinel AI Configuration Guide

This document provides comprehensive information about configuring Sentinel AI for different environments and use cases.

## Table of Contents

- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Agent Configuration](#agent-configuration)
- [Database Configuration](#database-configuration)
- [Cloud Provider Setup](#cloud-provider-setup)
- [Monitoring Configuration](#monitoring-configuration)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

Sentinel AI uses a hierarchical configuration system that supports:

- YAML configuration files for structured settings
- Environment variables for secrets and environment-specific values
- Command-line arguments for runtime overrides
- Dynamic configuration updates through the API

### Configuration Priority

1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration files
4. Default values (lowest priority)

## Configuration Files

### Main Configuration File

**Location**: `config.yaml`

```yaml
# Sentinel AI Configuration
app:
  name: "sentinel-ai"
  version: "1.0.0"
  environment: "production"  # development, staging, production
  debug: false
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# API Server Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_request_size: 10485760  # 10MB
  timeout: 30
  cors:
    enabled: true
    origins:
      - "http://localhost:3000"
      - "https://dashboard.sentinel-ai.com"
    methods: ["GET", "POST", "PUT", "DELETE"]
    headers: ["*"]

# Database Configuration
database:
  # Primary database for incidents and metadata
  primary:
    type: "postgresql"
    host: "${DB_HOST:localhost}"
    port: 5432
    database: "sentinel_ai"
    username: "${DB_USER:sentinel}"
    password: "${DB_PASSWORD}"
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    ssl_mode: "require"
  
  # Metrics database for time-series data
  metrics:
    type: "influxdb"
    host: "${INFLUX_HOST:localhost}"
    port: 8086
    database: "sentinel_metrics"
    username: "${INFLUX_USER:admin}"
    password: "${INFLUX_PASSWORD}"
    retention_policy: "30d"

# Redis Configuration
redis:
  host: "${REDIS_HOST:localhost}"
  port: 6379
  password: "${REDIS_PASSWORD}"
  database: 0
  max_connections: 50
  socket_timeout: 5
  socket_connect_timeout: 5

# Message Queue Configuration
message_queue:
  type: "redis"  # redis, rabbitmq, kafka
  host: "${REDIS_HOST:localhost}"
  port: 6379
  password: "${REDIS_PASSWORD}"
  max_retries: 3
  retry_delay: 5
  dead_letter_queue: true

# Agent Configuration
agents:
  conductor:
    enabled: true
    max_concurrent_incidents: 100
    escalation_timeout: 3600  # 1 hour
    human_approval_timeout: 7200  # 2 hours
    
  diagnosis:
    enabled: true
    instances: 3
    timeout: 1800  # 30 minutes
    max_retries: 2
    
  remediation:
    enabled: true
    instances: 2
    timeout: 3600  # 1 hour
    max_retries: 1
    require_approval: true
    
  verification:
    enabled: true
    instances: 2
    timeout: 1800  # 30 minutes
    
  reporting:
    enabled: true
    instances: 1
    timeout: 600  # 10 minutes

# Cloud Provider Configuration
cloud:
  provider: "gcp"  # gcp, aws, azure
  
  gcp:
    project_id: "${GCP_PROJECT_ID}"
    region: "${GCP_REGION:us-central1}"
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"
    
    vertex_ai:
      location: "${GCP_REGION:us-central1}"
      model_endpoint: "projects/${GCP_PROJECT_ID}/locations/${GCP_REGION}/endpoints/"
      
    bigquery:
      dataset: "sentinel_ai_data"
      location: "${GCP_REGION:us-central1}"
      
    storage:
      bucket: "${GCS_BUCKET:sentinel-ai-storage}"
      
  aws:
    region: "${AWS_REGION:us-west-2}"
    access_key_id: "${AWS_ACCESS_KEY_ID}"
    secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    
  azure:
    subscription_id: "${AZURE_SUBSCRIPTION_ID}"
    resource_group: "${AZURE_RESOURCE_GROUP}"
    location: "${AZURE_LOCATION:eastus}"

# Monitoring Configuration
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: "/metrics"
    
  logging:
    level: "INFO"
    format: "json"  # json, text
    file: "/var/log/sentinel-ai/app.log"
    max_size: 100  # MB
    max_files: 10
    
  tracing:
    enabled: true
    service_name: "sentinel-ai"
    jaeger_endpoint: "${JAEGER_ENDPOINT:http://localhost:14268/api/traces}"

# Security Configuration
security:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expiration: 3600  # 1 hour
    
  api_keys:
    enabled: true
    header_name: "X-API-Key"
    
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100
    
  encryption:
    key: "${ENCRYPTION_KEY}"
    algorithm: "AES-256-GCM"

# Notification Configuration
notifications:
  email:
    enabled: true
    smtp_host: "${SMTP_HOST}"
    smtp_port: 587
    username: "${SMTP_USER}"
    password: "${SMTP_PASSWORD}"
    from_address: "noreply@sentinel-ai.com"
    
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channel: "#incidents"
    
  teams:
    enabled: false
    webhook_url: "${TEAMS_WEBHOOK_URL}"
    
  pagerduty:
    enabled: false
    integration_key: "${PAGERDUTY_INTEGRATION_KEY}"

# Feature Flags
features:
  auto_remediation: true
  human_approval_required: true
  advanced_analytics: true
  multi_tenant: false
  audit_logging: true
```

### Agent-Specific Configuration

**Location**: `config/agents/`

#### Conductor Agent (`config/agents/conductor.yaml`)

```yaml
conductor:
  # Orchestration settings
  max_concurrent_incidents: 100
  incident_processing_timeout: 7200  # 2 hours
  
  # Escalation rules
  escalation:
    auto_escalate_after: 3600  # 1 hour
    escalation_levels:
      - level: 1
        timeout: 1800  # 30 minutes
        notify: ["team-lead@company.com"]
      - level: 2
        timeout: 3600  # 1 hour
        notify: ["manager@company.com"]
      - level: 3
        timeout: 7200  # 2 hours
        notify: ["director@company.com"]
  
  # Human approval workflows
  approval:
    required_for:
      - "high_risk_remediation"
      - "production_changes"
      - "data_modifications"
    timeout: 7200  # 2 hours
    approvers:
      - "ops-team@company.com"
      - "security-team@company.com"
  
  # Agent coordination
  coordination:
    max_parallel_agents: 5
    agent_timeout: 1800  # 30 minutes
    retry_failed_agents: true
```

#### Diagnosis Agent (`config/agents/diagnosis.yaml`)

```yaml
diagnosis:
  # Analysis settings
  analysis:
    timeout: 1800  # 30 minutes
    confidence_threshold: 0.7
    max_root_causes: 3
    
  # Data drift detection
  data_drift:
    enabled: true
    statistical_tests:
      - "kolmogorov_smirnov"
      - "chi_square"
      - "jensen_shannon"
    threshold: 0.05
    
  # Performance analysis
  performance:
    enabled: true
    metrics:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1_score"
    degradation_threshold: 0.05  # 5% drop
    
  # Log analysis
  log_analysis:
    enabled: true
    max_log_lines: 10000
    error_patterns:
      - "ERROR"
      - "CRITICAL"
      - "EXCEPTION"
      - "TIMEOUT"
```

#### Remediation Agent (`config/agents/remediation.yaml`)

```yaml
remediation:
  # Execution settings
  execution:
    timeout: 3600  # 1 hour
    max_retries: 2
    rollback_on_failure: true
    
  # Model retraining
  retraining:
    enabled: true
    auto_trigger: false  # Requires approval
    training_timeout: 7200  # 2 hours
    validation_required: true
    
  # Infrastructure actions
  infrastructure:
    enabled: true
    allowed_actions:
      - "scale_up"
      - "restart_service"
      - "update_config"
    forbidden_actions:
      - "delete_data"
      - "terminate_instance"
    
  # Safety checks
  safety:
    pre_execution_checks: true
    backup_before_changes: true
    canary_deployment: true
    rollback_plan_required: true
```

## Environment Variables

### Required Environment Variables

```bash
# Database
export DB_HOST="your-db-host"
export DB_USER="sentinel_user"
export DB_PASSWORD="your-secure-password"

# Redis
export REDIS_HOST="your-redis-host"
export REDIS_PASSWORD="your-redis-password"

# Cloud Provider (GCP example)
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GCS_BUCKET="your-storage-bucket"

# Security
export JWT_SECRET_KEY="your-jwt-secret-key"
export ENCRYPTION_KEY="your-encryption-key"

# Notifications
export SMTP_HOST="smtp.gmail.com"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### Optional Environment Variables

```bash
# Performance tuning
export WORKER_PROCESSES="4"
export MAX_CONNECTIONS="100"
export CACHE_TTL="3600"

# Feature flags
export ENABLE_AUTO_REMEDIATION="true"
export ENABLE_AUDIT_LOGGING="true"
export ENABLE_ADVANCED_ANALYTICS="false"

# Monitoring
export PROMETHEUS_PORT="9090"
export JAEGER_ENDPOINT="http://jaeger:14268/api/traces"
export LOG_LEVEL="INFO"
```

## Agent Configuration

### Agent Capabilities

Each agent can be configured with specific capabilities:

```yaml
agent_capabilities:
  diagnosis:
    - "data_drift_analysis"
    - "performance_analysis"
    - "log_analysis"
    - "anomaly_detection"
    
  remediation:
    - "model_retraining"
    - "infrastructure_scaling"
    - "configuration_updates"
    - "rollback_operations"
    
  verification:
    - "model_validation"
    - "integration_testing"
    - "performance_testing"
    - "health_checks"
```

### Agent Resource Limits

```yaml
resource_limits:
  diagnosis:
    cpu: "2000m"  # 2 CPU cores
    memory: "4Gi"
    timeout: 1800  # 30 minutes
    
  remediation:
    cpu: "1000m"  # 1 CPU core
    memory: "2Gi"
    timeout: 3600  # 1 hour
    
  verification:
    cpu: "1000m"
    memory: "2Gi"
    timeout: 1800  # 30 minutes
```

## Database Configuration

### PostgreSQL Setup

```sql
-- Create database and user
CREATE DATABASE sentinel_ai;
CREATE USER sentinel_user WITH PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE sentinel_ai TO sentinel_user;

-- Create extensions
\c sentinel_ai;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

### Database Connection Pool

```yaml
database_pool:
  min_connections: 5
  max_connections: 20
  connection_timeout: 30
  idle_timeout: 300
  max_lifetime: 3600
```

### Migration Configuration

```yaml
migrations:
  auto_migrate: false  # Set to true for development
  migration_path: "./migrations"
  backup_before_migrate: true
```

## Cloud Provider Setup

### Google Cloud Platform

```bash
# Install and configure gcloud CLI
curl https://sdk.cloud.google.com | bash
gcloud auth login
gcloud config set project your-project-id

# Create service account
gcloud iam service-accounts create sentinel-ai-service \
    --display-name="Sentinel AI Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:sentinel-ai-service@your-project-id.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:sentinel-ai-service@your-project-id.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

# Create and download key
gcloud iam service-accounts keys create service-account.json \
    --iam-account=sentinel-ai-service@your-project-id.iam.gserviceaccount.com
```

### AWS Configuration

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# AWS Access Key ID: your-access-key
# AWS Secret Access Key: your-secret-key
# Default region: us-west-2
# Default output format: json
```

### Azure Configuration

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login and set subscription
az login
az account set --subscription your-subscription-id
```

## Monitoring Configuration

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sentinel-ai'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Sentinel AI Metrics",
    "panels": [
      {
        "title": "Active Incidents",
        "type": "stat",
        "targets": [
          {
            "expr": "sentinel_ai_active_incidents_total"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sentinel_ai_agent_processing_duration_seconds[5m])"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### SSL/TLS Configuration

```yaml
ssl:
  enabled: true
  cert_file: "/etc/ssl/certs/sentinel-ai.crt"
  key_file: "/etc/ssl/private/sentinel-ai.key"
  ca_file: "/etc/ssl/certs/ca.crt"
  protocols: ["TLSv1.2", "TLSv1.3"]
  ciphers: "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
```

### Authentication Configuration

```yaml
authentication:
  providers:
    - name: "local"
      type: "database"
      enabled: true
      
    - name: "oauth2"
      type: "oauth2"
      enabled: true
      client_id: "${OAUTH_CLIENT_ID}"
      client_secret: "${OAUTH_CLIENT_SECRET}"
      authorization_url: "https://accounts.google.com/o/oauth2/auth"
      token_url: "https://oauth2.googleapis.com/token"
      
    - name: "ldap"
      type: "ldap"
      enabled: false
      server: "ldap://your-ldap-server:389"
      base_dn: "dc=company,dc=com"
```

## Performance Tuning

### Application Performance

```yaml
performance:
  # Connection pooling
  database_pool_size: 20
  redis_pool_size: 50
  
  # Caching
  cache_ttl: 3600  # 1 hour
  cache_max_size: 1000
  
  # Request handling
  max_request_size: 10485760  # 10MB
  request_timeout: 30
  
  # Background processing
  worker_processes: 4
  max_queue_size: 1000
  batch_size: 100
```

### Memory Configuration

```yaml
memory:
  # JVM settings (if using Java components)
  heap_size: "2g"
  gc_algorithm: "G1GC"
  
  # Python settings
  max_memory_per_worker: "1g"
  garbage_collection_threshold: 0.8
```

## Troubleshooting

### Common Configuration Issues

1. **Database Connection Failures**
   ```bash
   # Check database connectivity
   pg_isready -h $DB_HOST -p 5432 -U $DB_USER
   
   # Test connection
   psql -h $DB_HOST -U $DB_USER -d sentinel_ai -c "SELECT 1;"
   ```

2. **Redis Connection Issues**
   ```bash
   # Test Redis connection
   redis-cli -h $REDIS_HOST -p 6379 -a $REDIS_PASSWORD ping
   ```

3. **Cloud Provider Authentication**
   ```bash
   # GCP: Test authentication
   gcloud auth application-default print-access-token
   
   # AWS: Test credentials
   aws sts get-caller-identity
   
   # Azure: Test login
   az account show
   ```

### Configuration Validation

```bash
# Validate configuration file
python -m sentinel_ai.config.validator --config config.yaml

# Test database migrations
python -m sentinel_ai.database.migrate --dry-run

# Validate agent configurations
python -m sentinel_ai.agents.validator --config config/agents/
```

### Logging Configuration for Debugging

```yaml
logging:
  level: "DEBUG"
  format: "detailed"
  handlers:
    - type: "file"
      filename: "/var/log/sentinel-ai/debug.log"
      max_size: 100MB
      backup_count: 5
    - type: "console"
      stream: "stdout"
  
  loggers:
    sentinel_ai.agents: "DEBUG"
    sentinel_ai.database: "INFO"
    sentinel_ai.api: "DEBUG"
```

This configuration guide provides comprehensive setup instructions for all aspects of Sentinel AI. Adjust the values according to your specific environment and requirements.
