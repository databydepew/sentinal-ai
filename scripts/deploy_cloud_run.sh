#!/bin/bash

# Sentinel AI - Cloud Run Deployment Script
# This script deploys Sentinel AI to Google Cloud Run

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    print_status "Loaded environment variables from .env"
else
    print_error ".env file not found. Please run setup_gcp.sh first."
    exit 1
fi

# Check required variables
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCP_REGION" ]; then
    print_error "Required environment variables not set. Please run setup_gcp.sh first."
    exit 1
fi

# Configuration
SERVICE_NAME="sentinel-ai"
IMAGE_NAME="gcr.io/${GCP_PROJECT_ID}/${SERVICE_NAME}"
REGION="$GCP_REGION"

print_status "Deploying Sentinel AI to Cloud Run..."
print_status "Project: $GCP_PROJECT_ID"
print_status "Region: $REGION"
print_status "Service: $SERVICE_NAME"
print_status "Image: $IMAGE_NAME"

# Build the container image
print_status "Building container image..."
gcloud builds submit --tag "$IMAGE_NAME" --project="$GCP_PROJECT_ID"

if [ $? -eq 0 ]; then
    print_success "Container image built successfully"
else
    print_error "Failed to build container image"
    exit 1
fi

# Deploy to Cloud Run
print_status "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE_NAME" \
    --platform=managed \
    --region="$REGION" \
    --allow-unauthenticated \
    --port=8080 \
    --memory=2Gi \
    --cpu=2 \
    --timeout=3600 \
    --concurrency=10 \
    --max-instances=10 \
    --set-env-vars="GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_REGION=${GCP_REGION}" \
    --service-account="sentinel-ai-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
    --project="$GCP_PROJECT_ID"

if [ $? -eq 0 ]; then
    print_success "Deployment to Cloud Run completed successfully"
    
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --platform=managed \
        --region="$REGION" \
        --project="$GCP_PROJECT_ID" \
        --format="value(status.url)")
    
    print_success "Service deployed at: $SERVICE_URL"
    
    # Test the deployment
    print_status "Testing deployment..."
    if curl -f "${SERVICE_URL}/health" >/dev/null 2>&1; then
        print_success "Health check passed"
    else
        print_warning "Health check failed (service may still be starting up)"
    fi
    
    echo
    echo "=================================================="
    echo "    CLOUD RUN DEPLOYMENT COMPLETED"
    echo "=================================================="
    echo
    echo "Service URL: $SERVICE_URL"
    echo "Service Name: $SERVICE_NAME"
    echo "Project: $GCP_PROJECT_ID"
    echo "Region: $REGION"
    echo
    echo "Next steps:"
    echo "1. Test the service: curl ${SERVICE_URL}/health"
    echo "2. View logs: gcloud logs tail --follow --service=${SERVICE_NAME}"
    echo "3. Monitor in Console: https://console.cloud.google.com/run"
    echo
    echo "=================================================="
    
else
    print_error "Failed to deploy to Cloud Run"
    exit 1
fi
