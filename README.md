# Sentinel AI - Autonomous Predictive MLOps Guardian System

Sentinel AI is an autonomous MLOps "guardian" built on Google Cloud Vertex AI that moves beyond reactive alerting to create a **predictive, self-healing, and continuously optimizing ecosystem** for machine learning models. The system prevents incidents before they occur through advanced pattern analysis and multi-agent orchestration.

## 🎉 Project Status

**✅ PRODUCTION READY** - Sentinel AI is fully deployed on GCP with predictive capabilities!

### What's Working:
- ✅ **Complete Multi-Agent System**: All 6 specialized agents operational (Conductor, Diagnostic, Verification, Reporting, Remediation, **Predictive**)
- ✅ **GCP Production Deployment**: Successfully deployed and running on Google Cloud Platform
- ✅ **Predictive Incident Prevention**: Proactive analysis prevents incidents before they occur
- ✅ **Application Default Credentials**: Secure authentication configured for GCP resources
- ✅ **End-to-End Demonstrations**: Both reactive and predictive scenarios fully functional
- ✅ **Agent Communication**: Message bus and agent registry for inter-agent coordination
- ✅ **Graceful Startup/Shutdown**: Proper initialization and cleanup of all components
- ✅ **Comprehensive Logging**: Structured logging and monitoring throughout the system

### Predictive Capabilities:
- **🔮 Data Drift Prediction**: Analyzes feature distribution trends to predict critical drift
- **⚡ Performance Degradation Prediction**: Forecasts system failures before they occur
- **💾 Resource Exhaustion Prediction**: Prevents capacity issues through usage pattern analysis
- **🤖 Model Staleness Prediction**: Predicts optimal retraining schedules based on performance decay
- **🎯 Proactive Alert System**: Time-to-incident estimation with prevention recommendations
- **🛠️ Automated Prevention**: Actionable steps to prevent predicted incidents

### Production Deployment:
- ✅ **GCP Resources**: BigQuery datasets, Cloud Storage buckets, Pub/Sub topics configured
- ✅ **Cloud Run Service**: Auto-scaling deployment with 4GB RAM, 1 CPU
- ✅ **API Integration**: Vertex AI, BigQuery, Pub/Sub, Cloud Storage, Monitoring APIs enabled
- ✅ **Security**: Application Default Credentials for secure authentication
- ✅ **Monitoring**: Real-time logging and performance tracking

## 🎯 Mission

Maximize model performance and business value while minimizing human intervention and operational costs through intelligent automation and multi-agent orchestration.

## Architecture

### Core Principles

- **Proactive & Predictive**: Shift from "model has drifted" to "model will drift, here's the plan"
- **Generative Diagnosis & Solutions**: Leverage Gemini's reasoning for root cause analysis and novel solutions
- **Business-Aware Optimization**: Every action evaluated for cost-benefit and business impact
- **Governed Autonomy**: Configurable human-in-the-loop checkpoints for safety and control
- **Explainable MLOps**: Complex technical events translated to human-readable summaries
- **Adaptive Learning**: System learns from past incidents to improve future decisions

### Multi-Agent System

The system consists of 6 specialized agents orchestrated by a central conductor:

#### 🎼 Conductor Agent (The Orchestrator)

- Central "brain" managing incident lifecycle (DETECTED → DIAGNOSING → RESOLVED)
- Intelligent task delegation to specialized agents
- State management and governance rule application
- **Status**: ✅ **Deployed and Operational**
- **Tools**: Vertex AI Agent Builder, LangChain, Vertex AI Vector Search

#### 🔮 Predictive Agent (The Oracle) - **NEW!**

- **Proactive incident prevention** through pattern analysis and trend forecasting
- Predicts data drift, performance degradation, resource exhaustion, and model staleness
- Generates time-to-incident estimates with prevention recommendations
- **Status**: ✅ **Deployed and Operational**
- **Tools**: Statistical Analysis, Machine Learning Models, Gemini Pro

#### 🔍 Diagnostic Agent (The Detective)

- Uses Gemini's reasoning to diagnose root causes of incidents
- Synthesizes monitoring data, model lineage, and historical context
- **Status**: ✅ **Deployed and Operational**
- **Tools**: Gemini Pro, Vertex AI ML Metadata, Vertex AI Workbench

#### 🛠️ Remediation Agent (The Strategist)

- Formulates concrete remediation plans for identified incidents
- Suggests retraining, hyperparameter optimization, or feature engineering
- **Status**: ✅ **Deployed and Operational**
- **Tools**: Gemini Pro, Vertex AI Pipelines, Vertex AI Vizier

#### ✅ Verification Agent (The Validator)

- Validates remediation effectiveness and system health
- Ensures fixes resolve issues without introducing new problems
- **Status**: ✅ **Deployed and Operational**
- **Tools**: Vertex AI Model Monitoring, BigQuery ML, Statistical Analysis

#### 📊 Reporting Agent (The Communicator)

- Generates human-readable reports and summaries
- Translates technical incidents into business-impact language
- **Status**: ✅ **Deployed and Operational**
- **Tools**: Gemini Pro, BigQuery Analytics, Visualization APIs
- **Tools**: BigQuery, Gemini Pro

#### ⚙️ Verification & Rollout Agent (The Engineer)

- Executes approved remediation plans
- Manages training pipelines, model evaluation, and safe production rollouts
- **Tools**: Vertex AI Pipelines, Model Registry, Endpoints

#### 📊 Reporting & Feedback Agent (The Communicator)

- Generates natural language summaries for stakeholders
- Makes system actions explainable and transparent
- **Tools**: Gemini Pro, Looker, BigQuery

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Google Cloud Project with Vertex AI API enabled (for production)
- Service account with appropriate permissions (for production)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentinel-ai

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Running the Demo

The project includes a fully functional demonstration that showcases the complete Sentinel AI system:

```bash
# Run the conductor agent demonstration
python examples/demo_conductor.py
```

The demo will:

- Initialize all core components (MessageBus, AgentRegistry, ConductorAgent)
- Create and start 4 specialized agents (remediation, verification, diagnostic, reporting)
- Run 3 demonstration scenarios:
  1. **Data Drift Detection** - Simulates model drift incident
  2. **Performance Degradation** - Simulates model performance issues
  3. **Model Error Handling** - Simulates model execution errors
- Show complete incident lifecycle management
- Demonstrate multi-agent orchestration and communication

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud SDK (`gcloud`)
- `uv` package manager (recommended) or `pip`

### 1. Clone and Setup
```bash
git clone <repository-url>
cd sentinal-ai

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. GCP Authentication
```bash
# Login and set up Application Default Credentials
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Run Demonstrations

**Reactive Demo (Original System):**
```bash
uv run python examples/demo_conductor.py
```

**Predictive Demo (NEW!):**
```bash
uv run python examples/demo_predictive.py
```

### 4. Deploy to GCP
```bash
# Automated GCP setup and deployment
./scripts/setup_gcp.sh
uv run python deploy_all_agents.py
```

## 🏭 Production Deployment

### Current Deployment Status
- **✅ GCP Project**: `mdepew-assets`
- **✅ Cloud Run Service**: `mdepew-agent` (auto-scaling, 4GB RAM, 1 CPU)
- **✅ Authentication**: Application Default Credentials
- **✅ APIs Enabled**: Vertex AI, BigQuery, Pub/Sub, Cloud Storage, Monitoring
- **✅ Resources Created**: BigQuery datasets, Storage buckets, Pub/Sub topics

### Access Your Deployed System
- **Cloud Run Service**: https://mdepew-agent-194822035697.us-central1.run.app
- **GCP Console**: https://console.cloud.google.com/run?project=mdepew-assets
- **BigQuery Data**: https://console.cloud.google.com/bigquery?project=mdepew-assets
- **Monitoring**: https://console.cloud.google.com/monitoring?project=mdepew-assets

### Deployment Scripts
- `scripts/setup_gcp.sh` - Automated GCP resource setup
- `deploy_all_agents.py` - Complete agent deployment and testing
- `scripts/deploy_cloud_run.sh` - Cloud Run deployment
- `Dockerfile` - Production container image

## 📁 Project Structure

```
sentinel-ai/
├── src/
│   ├── agents/           # Agent implementations
│   ├── services/         # Vertex AI service integrations
│   ├── models/           # Data models and schemas
│   ├── communication/    # Agent communication patterns
│   ├── governance/       # Autonomy and governance controls
│   ├── config/           # Configuration management
│   └── utils/            # Utility functions
├── examples/             # Usage examples and demos
├── tests/                # Test suite
├── docs/                 # Documentation
├── config.yaml           # Main configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

The system uses Python-based configuration management located in `src/config/settings.py`. The configuration includes:

### Current Configuration Structure:
- **GCP Settings**: Project ID, regions, service configurations
- **Agent Configuration**: Individual agent settings and thresholds  
- **Governance**: Autonomy levels and approval requirements
- **Logging**: Structured logging with contextual information
- **Communication**: Message bus and agent registry settings

### Key Configuration Files:
- `src/config/settings.py` - Main configuration management
- `pyproject.toml` - Project metadata and dependencies
- `requirements.txt` - Python package dependencies

### Demo Configuration:
The demo runs with default mock configurations and doesn't require GCP credentials. For production deployment, update the settings in `src/config/settings.py` with your specific GCP project details.

## 🤖 Operational Workflow

1. **Detect & Alert**: Drift Agent detects anomaly, alerts Conductor
2. **Diagnose**: Conductor invokes Diagnostic Agent for root cause analysis
3. **Plan**: Remediation Agent generates concrete action plan
4. **Analyze**: Economist Agent performs cost-benefit analysis
5. **Govern & Decide**: Conductor checks autonomy level and governance rules
6. **Execute & Verify**: Verification Agent executes plan and manages rollout
7. **Report & Learn**: Reporting Agent provides summaries, system learns from incident

## 🛡️ Governance & Safety

- **Autonomy Levels**: Report Only, Supervised Execution, Fully Autonomous
- **Human-in-the-Loop**: Configurable checkpoints for critical operations
- **Cost Controls**: Automatic cost-benefit analysis before expensive operations
- **Audit Trail**: Complete logging of all decisions and actions

## 🔍 Monitoring & Observability

- Prometheus metrics for system health
- Structured logging with correlation IDs
- Health check endpoints
- Integration with Google Cloud Monitoring

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

## Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [Agent Development Guide](docs/agent-development.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:

- Create an issue in the repository
- Check the [documentation](docs/)
- Review the [examples](examples/)

---

**Sentinel AI** - Autonomous MLOps for the Future 🚀
