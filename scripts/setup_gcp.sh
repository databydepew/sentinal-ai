#!/bin/bash

# Sentinel AI - GCP Setup Script
# This script sets up the Google Cloud Platform environment for Sentinel AI deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_PROJECT_ID=""
DEFAULT_REGION="us-central1"
DEFAULT_ZONE="us-central1-a"

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to prompt for input with default
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local result
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " result
        echo "${result:-$default}"
    else
        read -p "$prompt: " result
        echo "$result"
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command_exists gcloud; then
        print_error "Google Cloud SDK (gcloud) is not installed."
        print_status "Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if bq is installed
    if ! command_exists bq; then
        print_error "BigQuery CLI (bq) is not installed."
        print_status "Please install it with: gcloud components install bq"
        exit 1
    fi
    
    # Check if uv is installed
    if ! command_exists uv; then
        print_error "uv package manager is not installed."
        print_status "Please install it from: https://docs.astral.sh/uv/"
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to setup authentication
setup_authentication() {
    print_status "Setting up authentication..."
    
    # Check if user is already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        local active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        print_success "Already authenticated as: $active_account"
        
        read -p "Do you want to use this account? (y/n): " use_current
        if [[ $use_current != "y" && $use_current != "Y" ]]; then
            gcloud auth login
        fi
    else
        print_status "No active authentication found. Starting login process..."
        gcloud auth login
    fi
    
    # Set up application default credentials
    print_status "Setting up application default credentials..."
    gcloud auth application-default login
    
    print_success "Authentication setup complete"
}

# Function to configure project
configure_project() {
    print_status "Configuring GCP project..."
    
    # Get current project if any
    local current_project=$(gcloud config get-value project 2>/dev/null || echo "")
    
    if [ -n "$current_project" ]; then
        print_status "Current project: $current_project"
        read -p "Do you want to use this project? (y/n): " use_current
        if [[ $use_current == "y" || $use_current == "Y" ]]; then
            PROJECT_ID="$current_project"
        else
            PROJECT_ID=$(prompt_with_default "Enter GCP Project ID" "$DEFAULT_PROJECT_ID")
        fi
    else
        PROJECT_ID=$(prompt_with_default "Enter GCP Project ID" "$DEFAULT_PROJECT_ID")
    fi
    
    # Set the project
    print_status "Setting project to: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
    
    # Set region and zone
    REGION=$(prompt_with_default "Enter region" "$DEFAULT_REGION")
    ZONE=$(prompt_with_default "Enter zone" "$DEFAULT_ZONE")
    
    gcloud config set compute/region "$REGION"
    gcloud config set compute/zone "$ZONE"
    
    print_success "Project configuration complete"
    print_status "Project ID: $PROJECT_ID"
    print_status "Region: $REGION"
    print_status "Zone: $ZONE"
}

# Function to enable APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    local apis=(
        "aiplatform.googleapis.com"
        "bigquery.googleapis.com"
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "secretmanager.googleapis.com"
        "pubsub.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "storage.googleapis.com"
        "iam.googleapis.com"
        "cloudresourcemanager.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        print_status "Enabling $api..."
        if gcloud services enable "$api" --project="$PROJECT_ID"; then
            print_success "Enabled $api"
        else
            print_warning "Failed to enable $api (may already be enabled)"
        fi
    done
    
    print_success "API enablement complete"
}

# Function to create service account
create_service_account() {
    print_status "Creating service account for Sentinel AI..."
    
    local sa_name="sentinel-ai-sa"
    local sa_display_name="Sentinel AI Service Account"
    local sa_email="${sa_name}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Check if service account already exists
    if gcloud iam service-accounts describe "$sa_email" --project="$PROJECT_ID" >/dev/null 2>&1; then
        print_warning "Service account $sa_email already exists"
        read -p "Do you want to recreate it? (y/n): " recreate
        if [[ $recreate == "y" || $recreate == "Y" ]]; then
            print_status "Deleting existing service account..."
            gcloud iam service-accounts delete "$sa_email" --project="$PROJECT_ID" --quiet
        else
            print_status "Using existing service account"
            return 0
        fi
    fi
    
    # Create service account
    print_status "Creating service account: $sa_email"
    gcloud iam service-accounts create "$sa_name" \
        --display-name="$sa_display_name" \
        --project="$PROJECT_ID"
    
    # Grant necessary roles
    local roles=(
        "roles/aiplatform.user"
        "roles/bigquery.admin"
        "roles/pubsub.admin"
        "roles/storage.admin"
        "roles/secretmanager.admin"
        "roles/monitoring.editor"
        "roles/logging.admin"
        "roles/run.admin"
        "roles/cloudbuild.builds.builder"
    )
    
    print_status "Granting IAM roles to service account..."
    for role in "${roles[@]}"; do
        print_status "Granting $role..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role"
    done
    
    # Create and download service account key
    local key_file="sentinel-ai-key.json"
    print_status "Creating service account key: $key_file"
    gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$sa_email" \
        --project="$PROJECT_ID"
    
    print_success "Service account setup complete"
    print_status "Service account: $sa_email"
    print_status "Key file: $key_file"
    print_warning "Keep the key file secure and do not commit it to version control!"
}

# Function to create BigQuery datasets
create_bigquery_datasets() {
    print_status "Creating BigQuery datasets..."
    
    local datasets=(
        "sentinel_ai_incidents"
        "sentinel_ai_metrics"
        "sentinel_ai_models"
        "sentinel_ai_logs"
    )
    
    for dataset in "${datasets[@]}"; do
        print_status "Creating dataset: $dataset"
        if bq mk --dataset --location="$REGION" "${PROJECT_ID}:${dataset}"; then
            print_success "Created dataset: $dataset"
        else
            print_warning "Dataset $dataset may already exist"
        fi
    done
    
    print_success "BigQuery datasets setup complete"
}

# Function to create Cloud Storage buckets
create_storage_buckets() {
    print_status "Creating Cloud Storage buckets..."
    
    local buckets=(
        "${PROJECT_ID}-sentinel-staging"
        "${PROJECT_ID}-sentinel-models"
        "${PROJECT_ID}-sentinel-logs"
    )
    
    for bucket in "${buckets[@]}"; do
        print_status "Creating bucket: $bucket"
        if gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$bucket/"; then
            print_success "Created bucket: $bucket"
        else
            print_warning "Bucket $bucket may already exist"
        fi
    done
    
    print_success "Cloud Storage buckets setup complete"
}

# Function to create Pub/Sub topics
create_pubsub_topics() {
    print_status "Creating Pub/Sub topics and subscriptions..."
    
    local topics=(
        "sentinel-ai-incidents"
        "sentinel-ai-alerts"
        "sentinel-ai-metrics"
        "sentinel-ai-agent-messages"
    )
    
    for topic in "${topics[@]}"; do
        print_status "Creating topic: $topic"
        if gcloud pubsub topics create "$topic" --project="$PROJECT_ID"; then
            print_success "Created topic: $topic"
            
            # Create subscription for each topic
            local subscription="${topic}-sub"
            print_status "Creating subscription: $subscription"
            if gcloud pubsub subscriptions create "$subscription" \
                --topic="$topic" \
                --project="$PROJECT_ID"; then
                print_success "Created subscription: $subscription"
            else
                print_warning "Subscription $subscription may already exist"
            fi
        else
            print_warning "Topic $topic may already exist"
        fi
    done
    
    print_success "Pub/Sub setup complete"
}

# Function to setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring and alerting..."
    
    # Create notification channel (email)
    read -p "Enter email address for alerts: " alert_email
    if [ -n "$alert_email" ]; then
        print_status "Creating notification channel for: $alert_email"
        # Note: This would typically be done via API or Terraform
        print_status "Please manually create notification channels in the Cloud Console"
        print_status "Go to: Monitoring > Alerting > Notification Channels"
    fi
    
    print_success "Monitoring setup guidance provided"
}

# Function to generate configuration
generate_config() {
    print_status "Generating configuration files..."
    
    # Create environment file
    cat > .env << EOF
# Sentinel AI Environment Configuration
# Generated on $(date)

# GCP Configuration
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
GCP_ZONE=$ZONE
GOOGLE_APPLICATION_CREDENTIALS=./sentinel-ai-key.json

# BigQuery Configuration
BIGQUERY_DATASET_INCIDENTS=sentinel_ai_incidents
BIGQUERY_DATASET_METRICS=sentinel_ai_metrics
BIGQUERY_DATASET_MODELS=sentinel_ai_models
BIGQUERY_DATASET_LOGS=sentinel_ai_logs

# Cloud Storage Configuration
STORAGE_BUCKET_STAGING=${PROJECT_ID}-sentinel-staging
STORAGE_BUCKET_MODELS=${PROJECT_ID}-sentinel-models
STORAGE_BUCKET_LOGS=${PROJECT_ID}-sentinel-logs

# Pub/Sub Configuration
PUBSUB_TOPIC_INCIDENTS=sentinel-ai-incidents
PUBSUB_TOPIC_ALERTS=sentinel-ai-alerts
PUBSUB_TOPIC_METRICS=sentinel-ai-metrics
PUBSUB_TOPIC_MESSAGES=sentinel-ai-agent-messages

# Vertex AI Configuration
VERTEX_AI_LOCATION=$REGION
VERTEX_AI_STAGING_BUCKET=${PROJECT_ID}-sentinel-staging
EOF

    # Update config.yaml with GCP settings
    if [ -f "config.yaml" ]; then
        print_status "Updating config.yaml with GCP settings..."
        # Create backup
        cp config.yaml config.yaml.backup
        
        # Update the configuration (this is a simple approach)
        sed -i.bak "s/project_id: .*/project_id: \"$PROJECT_ID\"/" config.yaml
        sed -i.bak "s/region: .*/region: \"$REGION\"/" config.yaml
        sed -i.bak "s/zone: .*/zone: \"$ZONE\"/" config.yaml
        
        print_success "Updated config.yaml"
    fi
    
    print_success "Configuration files generated"
    print_status "Environment file: .env"
    print_status "Updated config: config.yaml"
}

# Function to verify setup
verify_setup() {
    print_status "Verifying GCP setup..."
    
    # Check authentication
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        print_success "✓ Authentication verified"
    else
        print_error "✗ Authentication failed"
        return 1
    fi
    
    # Check project access
    if gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        print_success "✓ Project access verified"
    else
        print_error "✗ Cannot access project: $PROJECT_ID"
        return 1
    fi
    
    # Check service account
    local sa_email="sentinel-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    if gcloud iam service-accounts describe "$sa_email" --project="$PROJECT_ID" >/dev/null 2>&1; then
        print_success "✓ Service account verified"
    else
        print_error "✗ Service account not found: $sa_email"
        return 1
    fi
    
    # Check key file
    if [ -f "sentinel-ai-key.json" ]; then
        print_success "✓ Service account key file found"
    else
        print_error "✗ Service account key file not found"
        return 1
    fi
    
    print_success "GCP setup verification complete"
    return 0
}

# Main function
main() {
    echo "=================================================="
    echo "    SENTINEL AI - GCP SETUP SCRIPT"
    echo "=================================================="
    echo
    echo "This script will set up your Google Cloud Platform"
    echo "environment for Sentinel AI deployment."
    echo
    echo "The script will:"
    echo "• Check prerequisites"
    echo "• Set up authentication"
    echo "• Configure GCP project"
    echo "• Enable required APIs"
    echo "• Create service account and IAM roles"
    echo "• Create BigQuery datasets"
    echo "• Create Cloud Storage buckets"
    echo "• Create Pub/Sub topics"
    echo "• Generate configuration files"
    echo
    echo "=================================================="
    echo
    
    read -p "Do you want to continue? (y/n): " continue_setup
    if [[ $continue_setup != "y" && $continue_setup != "Y" ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Run setup steps
    check_prerequisites
    setup_authentication
    configure_project
    enable_apis
    create_service_account
    create_bigquery_datasets
    create_storage_buckets
    create_pubsub_topics
    setup_monitoring
    generate_config
    
    # Verify setup
    if verify_setup; then
        echo
        echo "=================================================="
        echo "    GCP SETUP COMPLETED SUCCESSFULLY!"
        echo "=================================================="
        echo
        echo "Next steps:"
        echo "1. Review the generated .env file"
        echo "2. Run the deployment script: python deploy_all_agents.py"
        echo "3. Test the system with: uv run python examples/demo_conductor.py"
        echo
        echo "Important files created:"
        echo "• sentinel-ai-key.json (keep secure!)"
        echo "• .env (environment configuration)"
        echo "• config.yaml.backup (backup of original config)"
        echo
        echo "GCP Resources created:"
        echo "• Project: $PROJECT_ID"
        echo "• Service Account: sentinel-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com"
        echo "• BigQuery Datasets: sentinel_ai_*"
        echo "• Storage Buckets: ${PROJECT_ID}-sentinel-*"
        echo "• Pub/Sub Topics: sentinel-ai-*"
        echo
        echo "=================================================="
    else
        echo
        echo "=================================================="
        echo "    SETUP VERIFICATION FAILED"
        echo "=================================================="
        echo
        echo "Please check the errors above and run the script again."
        exit 1
    fi
}

# Run main function
main "$@"
