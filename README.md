# Sentinel AI - Autonomous MLOps Guardian System

Sentinel AI is an autonomous MLOps "guardian" built on Google Cloud Vertex AI that moves beyond simple reactive alerting to create a proactive, self-healing, and continuously optimizing ecosystem for machine learning models.

## 📊 Project Status

**✅ DEMO READY** - The Sentinel AI system is fully functional and includes a working demonstration!

### What's Working:
- ✅ **Complete Multi-Agent System**: All 4 specialized agents (remediation, verification, diagnostic, reporting) are operational
- ✅ **Conductor Agent Orchestration**: Central coordination and incident lifecycle management
- ✅ **Mock Incident Processing**: End-to-end demonstration with 3 realistic ML incident scenarios
- ✅ **Agent Communication**: Message bus and agent registry for inter-agent coordination
- ✅ **Graceful Startup/Shutdown**: Proper initialization and cleanup of all components
- ✅ **Logging and Monitoring**: Structured logging throughout the system

### Current Capabilities:
- **Data Drift Detection Simulation**: Demonstrates how the system handles model drift incidents
- **Performance Degradation Handling**: Shows response to model performance issues
- **Model Error Processing**: Illustrates error handling and recovery workflows
- **Multi-Agent Coordination**: Real-time communication between specialized agents
- **Incident Lifecycle Management**: Complete workflow from detection to resolution

### Production Readiness:
- 🟡 **Core Functionality**: Fully working demo with mock data
- 🟡 **GCP Integration**: Service clients implemented but require production configuration
- 🟡 **Code Quality**: Some lint warnings and unused imports remain (non-blocking)
- 🔴 **Production Deployment**: Requires GCP setup and configuration for live data

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

The system consists of specialized agents orchestrated by a central conductor:

#### 🎼 Conductor Agent (The Orchestrator)

- Central "brain" managing incident lifecycle (DETECTED → DIAGNOSING → RESOLVED)
- Intelligent task delegation to specialized agents
- State management and governance rule application
- **Tools**: Vertex AI Agent Builder, LangChain, Vertex AI Vector Search

#### 👁️ Drift & Anomaly Detection Agent (The Eyes)

- Continuous monitoring for data drift, concept drift, and performance anomalies
- **Tools**: Vertex AI Model Monitoring, BigQuery ML, Pub/Sub

#### 🔍 Diagnostic & Root Cause Agent (The Detective)

- Uses Gemini's reasoning to diagnose root causes
- Synthesizes monitoring data, model lineage, and historical context
- **Tools**: Gemini Pro, Vertex AI ML Metadata, Vertex AI Workbench

#### 🎯 Remediation & Optimization Planning Agent (The Strategist)

- Formulates concrete remediation plans
- Suggests retraining, hyperparameter optimization, or feature engineering
- **Tools**: Gemini Pro, Vertex AI Pipelines, Vertex AI Vizier

#### 💰 Economist Agent (The Analyst)

- Cost-benefit analysis of proposed remediation plans
- Estimates GCP costs vs. business impact of model degradation
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

### Production Setup

For production deployment, configure the following:

```bash
# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Configure settings in src/config/settings.py
# Update GCP project details, regions, and service configurations
```

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
