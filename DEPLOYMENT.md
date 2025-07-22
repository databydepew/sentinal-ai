# Sentinel AI - Quickstart Deployment Guide

This guide will help you deploy Sentinel AI from demo to production in Google Cloud Platform.

## 🚀 Deployment Overview

Sentinel AI can be deployed in three stages:
1. **Local Demo** - Run the demonstration locally (already working)
2. **Development Environment** - Deploy to GCP for testing with real data
3. **Production Environment** - Full production deployment with monitoring

## 📋 Prerequisites

### Required Tools
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Docker (for containerized deployment)
- Terraform (optional, for infrastructure as code)

### Google Cloud Requirements
- GCP Project with billing enabled
- Required APIs enabled (see API Setup section)
- Service account with appropriate permissions
- Vertex AI region configured

## 🔧 Stage 1: Local Demo (Already Working)

If you haven't run the demo yet:

```bash
# Clone and setup
git clone <repository-url>
cd sentinel-ai
uv sync

# Run the demo
python examples/demo_conductor.py
```

The demo demonstrates all core functionality with mock data.

## ☁️ Stage 2: Development Environment Setup

### Step 1: GCP Project Setup

```bash
# Set your project ID
export PROJECT_ID="your-sentinel-ai-project"
export REGION="us-central1"

# Login and set project
gcloud auth login
gcloud config set project $PROJECT_ID
```

### Step 2: Enable Required APIs

```bash
# Enable required Google Cloud APIs
gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    pubsub.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com
```

### Step 3: Create Service Account

```bash
# Create service account
gcloud iam service-accounts create sentinel-ai-sa \
    --display-name="Sentinel AI Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:sentinel-ai-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:sentinel-ai-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:sentinel-ai-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.admin"

# Create and download key
gcloud iam service-accounts keys create sentinel-ai-key.json \
    --iam-account=sentinel-ai-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### Step 4: Configure BigQuery Datasets

```bash
# Create datasets for Sentinel AI
bq mk --dataset --location=$REGION $PROJECT_ID:sentinel_ai_incidents
bq mk --dataset --location=$REGION $PROJECT_ID:sentinel_ai_metrics
bq mk --dataset --location=$REGION $PROJECT_ID:sentinel_ai_models
```

### Step 5: Update Configuration

Edit `src/config/settings.py` to include your GCP settings:

```python
# Update these values in settings.py
GCP_PROJECT_ID = "your-sentinel-ai-project"
GCP_REGION = "us-central1"
BIGQUERY_DATASET = "sentinel_ai_incidents"
VERTEX_AI_LOCATION = "us-central1"

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="./sentinel-ai-key.json"
export GCP_PROJECT_ID="your-sentinel-ai-project"
```

### Step 6: Test Development Environment

```bash
# Run with real GCP integration
python examples/demo_conductor.py --use-gcp
```

## 🏭 Stage 3: Production Deployment

### Option A: Cloud Run Deployment

#### Step 1: Create Dockerfile

```dockerfile
# Create Dockerfile in project root
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Run the application
CMD ["python", "-m", "src.main"]
```

#### Step 2: Build and Deploy

```bash
# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/sentinel-ai

# Deploy to Cloud Run
gcloud run deploy sentinel-ai \
    --image gcr.io/$PROJECT_ID/sentinel-ai \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars GCP_PROJECT_ID=$PROJECT_ID \
    --service-account sentinel-ai-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### Option B: GKE Deployment

#### Step 1: Create GKE Cluster

```bash
# Create GKE cluster
gcloud container clusters create sentinel-ai-cluster \
    --zone $REGION-a \
    --num-nodes 3 \
    --enable-autorepair \
    --enable-autoupgrade \
    --workload-pool=$PROJECT_ID.svc.id.goog
```

#### Step 2: Deploy with Kubernetes

```yaml
# Create k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentinel-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentinel-ai
  template:
    metadata:
      labels:
        app: sentinel-ai
    spec:
      serviceAccountName: sentinel-ai-ksa
      containers:
      - name: sentinel-ai
        image: gcr.io/PROJECT_ID/sentinel-ai:latest
        ports:
        - containerPort: 8080
        env:
        - name: GCP_PROJECT_ID
          value: "PROJECT_ID"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/var/secrets/google/key.json"
        volumeMounts:
        - name: google-cloud-key
          mountPath: /var/secrets/google
      volumes:
      - name: google-cloud-key
        secret:
          secretName: sentinel-ai-key
```

```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml
```

## 📊 Monitoring and Observability

### Step 1: Set up Cloud Monitoring

```bash
# Create monitoring dashboard
gcloud alpha monitoring dashboards create --config-from-file=monitoring-dashboard.json
```

### Step 2: Configure Alerting

```bash
# Create alert policies for key metrics
gcloud alpha monitoring policies create --policy-from-file=alert-policies.yaml
```

### Step 3: Set up Logging

```python
# Update logging configuration in settings.py
LOGGING_CONFIG = {
    "version": 1,
    "handlers": {
        "cloud_logging": {
            "class": "google.cloud.logging.handlers.CloudLoggingHandler",
            "client": "google.cloud.logging.Client()"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["cloud_logging"]
    }
}
```

## 🔒 Security Configuration

### Step 1: Secret Management

```bash
# Store sensitive configuration in Secret Manager
gcloud secrets create sentinel-ai-config --data-file=config.json

# Grant access to service account
gcloud secrets add-iam-policy-binding sentinel-ai-config \
    --member="serviceAccount:sentinel-ai-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### Step 2: Network Security

```bash
# Create VPC for secure networking
gcloud compute networks create sentinel-ai-vpc --subnet-mode regional

# Create firewall rules
gcloud compute firewall-rules create sentinel-ai-allow-internal \
    --network sentinel-ai-vpc \
    --allow tcp,udp,icmp \
    --source-ranges 10.0.0.0/8
```

## 🚦 Health Checks and Readiness

### Application Health Endpoint

Add to your main application:

```python
# src/main.py
from flask import Flask, jsonify
from src.agents.conductor import ConductorAgent

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/ready')
def readiness_check():
    # Check if all agents are ready
    conductor = ConductorAgent.get_instance()
    if conductor and conductor.running:
        return jsonify({"status": "ready"})
    return jsonify({"status": "not ready"}), 503
```

## 📈 Scaling Configuration

### Horizontal Pod Autoscaler (GKE)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentinel-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentinel-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Cloud Run Scaling

```bash
# Configure Cloud Run scaling
gcloud run services update sentinel-ai \
    --min-instances=1 \
    --max-instances=10 \
    --concurrency=100 \
    --cpu=2 \
    --memory=4Gi
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy Sentinel AI
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Cloud SDK
      uses: google-github-actions/setup-gcloud@v0
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Build and Deploy
      run: |
        gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT_ID }}/sentinel-ai
        gcloud run deploy sentinel-ai --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/sentinel-ai
```

## 🧪 Testing in Production

### Step 1: Canary Deployment

```bash
# Deploy canary version
gcloud run deploy sentinel-ai-canary \
    --image gcr.io/$PROJECT_ID/sentinel-ai:canary \
    --traffic 10
```

### Step 2: Load Testing

```bash
# Use Apache Bench for load testing
ab -n 1000 -c 10 https://sentinel-ai-url/health
```

## 📚 Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Check service account permissions
   gcloud projects get-iam-policy $PROJECT_ID
   ```

2. **API Quota Limits**
   ```bash
   # Check quota usage
   gcloud compute project-info describe --project=$PROJECT_ID
   ```

3. **Memory Issues**
   ```bash
   # Increase Cloud Run memory
   gcloud run services update sentinel-ai --memory=2Gi
   ```

### Debugging Commands

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=sentinel-ai"

# Check service status
gcloud run services describe sentinel-ai --region=$REGION

# Monitor metrics
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision"
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies**
   ```bash
   uv sync --upgrade
   ```

2. **Monitor Costs**
   ```bash
   gcloud billing budgets list
   ```

3. **Security Updates**
   ```bash
   gcloud components update
   ```

### Getting Help

- Check the [main README](README.md) for project overview
- Review logs in Cloud Logging
- Monitor metrics in Cloud Monitoring
- Create issues in the repository for bugs

---

**🎉 Congratulations!** You've successfully deployed Sentinel AI to production. The system is now ready to monitor and manage your ML models autonomously.

For advanced configuration and customization, refer to the detailed documentation in the `docs/` directory.
