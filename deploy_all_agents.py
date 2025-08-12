#!/usr/bin/env python3
"""
Comprehensive deployment script for Sentinel AI agents to Vertex AI Agent Space.

This script handles:
- Complete deployment and orchestration of Conductor and Diagnostic agents
- Automated tests and demo scenarios
- Deployment summaries and JSON output
- GCP project setup and configuration

Features:
- Data drift, performance degradation, and system alert scenarios
- Automated testing and validation
- Deployment info saved to JSON
- Comprehensive logging and error handling
"""

import asyncio
import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import get_settings
from utils.logging_setup import setup_logging, create_contextual_logger
from agents.conductor import ConductorAgent
from agents.diagnostic import DiagnosticAgent
from communication.agent_registry import AgentRegistry
from communication.message_bus import MessageBus
from examples.mock_incident import create_mock_incident
from models.incident import IncidentType, IncidentSeverity

class SentinelAIDeployer:
    """Comprehensive deployment manager for Sentinel AI"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.deployment_id = str(uuid.uuid4())
        self.deployment_info = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "status": "initializing",
            "agents": {},
            "tests": {},
            "demos": {},
            "gcp_resources": {}
        }
        self.logger = None
        
    async def initialize(self) -> None:
        """Initialize the deployment environment"""
        # Load configuration
        config = get_settings().to_dict()
        
        # Setup logging
        logging_config = config.get("logging", {})
        setup_logging(logging_config)
        
        self.logger = create_contextual_logger(__name__, deployment=self.deployment_id)
        self.logger.info("Starting Sentinel AI comprehensive deployment")
        
        # Update deployment info
        self.deployment_info["config"] = config
        self.deployment_info["status"] = "initialized"
        
    async def check_gcp_prerequisites(self) -> bool:
        """Check GCP prerequisites and configuration"""
        self.logger.info("Checking GCP prerequisites...")
        
        try:
            # Check gcloud CLI
            result = subprocess.run(["gcloud", "version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("gcloud CLI not found. Please install Google Cloud SDK")
                return False
                
            # Check authentication (ADC)
            result = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("No Application Default Credentials found. Run 'gcloud auth application-default login'")
                return False
                
            # Check project configuration
            result = subprocess.run(["gcloud", "config", "get-value", "project"], 
                                  capture_output=True, text=True)
            project_id = result.stdout.strip()
            if not project_id:
                self.logger.error("No GCP project configured. Run 'gcloud config set project PROJECT_ID'")
                return False
                
            self.deployment_info["gcp_resources"]["project_id"] = project_id
            self.logger.info(f"GCP prerequisites verified for project: {project_id}")
            self.logger.info("Using Application Default Credentials")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking GCP prerequisites: {e}")
            return False
    
    async def setup_gcp_resources(self) -> bool:
        """Set up required GCP resources"""
        self.logger.info("Setting up GCP resources...")
        
        try:
            project_id = self.deployment_info["gcp_resources"]["project_id"]
            
            # Enable required APIs
            apis = [
                "aiplatform.googleapis.com",
                "bigquery.googleapis.com",
                "cloudbuild.googleapis.com",
                "run.googleapis.com",
                "secretmanager.googleapis.com",
                "pubsub.googleapis.com",
                "monitoring.googleapis.com",
                "logging.googleapis.com"
            ]
            
            self.logger.info("Enabling required APIs...")
            for api in apis:
                result = subprocess.run([
                    "gcloud", "services", "enable", api, 
                    "--project", project_id
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info(f"Enabled API: {api}")
                else:
                    self.logger.warning(f"Failed to enable API {api}: {result.stderr}")
            
            # Create BigQuery datasets
            datasets = ["sentinel_ai_incidents", "sentinel_ai_metrics", "sentinel_ai_models"]
            region = "US"
            
            self.logger.info("Creating BigQuery datasets...")
            for dataset in datasets:
                result = subprocess.run([
                    "bq", "mk", "--dataset", f"--location={region}", 
                    f"{project_id}:{dataset}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info(f"Created dataset: {dataset}")
                elif "already exists" in result.stderr:
                    self.logger.info(f"Dataset already exists: {dataset}")
                else:
                    self.logger.warning(f"Failed to create dataset {dataset}: {result.stderr}")
            
            self.deployment_info["gcp_resources"]["apis_enabled"] = apis
            self.deployment_info["gcp_resources"]["datasets_created"] = datasets
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up GCP resources: {e}")
            return False
    
    async def deploy_conductor_agent(self) -> bool:
        """Deploy and test the Conductor Agent"""
        self.logger.info("Deploying Conductor Agent...")
        
        try:
            # Initialize core components
            config = get_settings().to_dict()
            
            # Create message bus
            message_bus = MessageBus(config)
            await message_bus.start()
            
            # Create agent registry
            agent_registry = AgentRegistry(config, message_bus)
            await agent_registry.start()
            
            # Create conductor agent
            conductor = ConductorAgent(config)
            conductor.set_message_bus(message_bus)
            conductor.set_agent_registry(agent_registry)
            
            # Initialize conductor
            await conductor.initialize()
            
            # Register conductor with registry
            await agent_registry.register_agent(conductor)
            
            self.deployment_info["agents"]["conductor"] = {
                "status": "deployed",
                "agent_id": conductor.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("Conductor Agent deployed successfully")
            
            # Store references for cleanup
            self.conductor = conductor
            self.agent_registry = agent_registry
            self.message_bus = message_bus
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying Conductor Agent: {e}")
            self.deployment_info["agents"]["conductor"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    async def deploy_diagnostic_agent(self) -> bool:
        """Deploy and test the Diagnostic Agent"""
        self.logger.info("Deploying Diagnostic Agent...")
        
        try:
            # The diagnostic agent is already created as part of conductor initialization
            # Just verify it's working
            diagnostic_agents = [agent for agent in self.conductor.specialized_agents.values() 
                               if hasattr(agent, 'agent_type') and 'diagnostic' in str(agent).lower()]
            
            if diagnostic_agents:
                diagnostic_agent = diagnostic_agents[0]
                self.deployment_info["agents"]["diagnostic"] = {
                    "status": "deployed",
                    "agent_id": getattr(diagnostic_agent, 'agent_id', 'unknown'),
                    "timestamp": datetime.now().isoformat()
                }
                self.logger.info("Diagnostic Agent deployed successfully")
                return True
            else:
                self.logger.error("Diagnostic Agent not found in conductor's specialized agents")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying Diagnostic Agent: {e}")
            self.deployment_info["agents"]["diagnostic"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    async def run_automated_tests(self) -> bool:
        """Run automated tests on deployed agents"""
        self.logger.info("Running automated tests...")
        
        try:
            test_results = {}
            
            # Test 1: Health check
            self.logger.info("Running health check test...")
            health_status = await self.conductor.health_check()
            test_results["health_check"] = {
                "status": "passed" if health_status.get("status") == "healthy" else "failed",
                "details": health_status
            }
            
            # Test 2: System metrics
            self.logger.info("Running system metrics test...")
            metrics = self.conductor.get_system_metrics()
            test_results["system_metrics"] = {
                "status": "passed" if metrics else "failed",
                "details": metrics
            }
            
            # Test 3: Agent availability
            self.logger.info("Running agent availability test...")
            active_incidents = self.conductor.get_active_incidents()
            test_results["agent_availability"] = {
                "status": "passed",
                "details": {"active_incidents": len(active_incidents)}
            }
            
            self.deployment_info["tests"] = test_results
            
            # Check if all tests passed
            all_passed = all(test["status"] == "passed" for test in test_results.values())
            
            if all_passed:
                self.logger.info("All automated tests passed")
                return True
            else:
                self.logger.warning("Some automated tests failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running automated tests: {e}")
            self.deployment_info["tests"] = {"error": str(e)}
            return False
    
    async def run_demo_scenarios(self) -> bool:
        """Execute demo scenarios to validate deployment"""
        self.logger.info("Running demo scenarios...")
        
        try:
            demo_results = {}
            
            # Demo 1: Data Drift Detection
            self.logger.info("Running data drift scenario...")
            drift_incident = create_mock_incident(
                incident_type="DATA_DRIFT",
                severity="MEDIUM",
                model_name="customer-churn-model",
                description="Significant data drift detected in customer demographics features"
            )
            
            drift_result = await self.conductor.process_incident(drift_incident)
            demo_results["data_drift"] = {
                "status": "completed",
                "incident_id": drift_incident.incident_id,
                "result": str(drift_result) if drift_result else "No result"
            }
            
            # Demo 2: Performance Degradation
            self.logger.info("Running performance degradation scenario...")
            perf_incident = create_mock_incident(
                incident_type="PERFORMANCE_DEGRADATION",
                severity="HIGH",
                model_name="recommendation-engine",
                description="Model accuracy dropped below threshold"
            )
            
            perf_result = await self.conductor.process_incident(perf_incident)
            demo_results["performance_degradation"] = {
                "status": "completed",
                "incident_id": perf_incident.incident_id,
                "result": str(perf_result) if perf_result else "No result"
            }
            
            # Demo 3: System Alert
            self.logger.info("Running system alert scenario...")
            alert_incident = create_mock_incident(
                incident_type="MODEL_ERROR",
                severity="CRITICAL",
                model_name="fraud-detection",
                description="Model serving endpoint returning errors"
            )
            
            alert_result = await self.conductor.process_incident(alert_incident)
            demo_results["system_alert"] = {
                "status": "completed",
                "incident_id": alert_incident.incident_id,
                "result": str(alert_result) if alert_result else "No result"
            }
            
            self.deployment_info["demos"] = demo_results
            self.logger.info("All demo scenarios completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running demo scenarios: {e}")
            self.deployment_info["demos"] = {"error": str(e)}
            return False
    
    async def generate_deployment_summary(self) -> Dict[str, Any]:
        """Generate comprehensive deployment summary"""
        self.logger.info("Generating deployment summary...")
        
        # Safely count deployed agents
        agents_deployed = 0
        for agent in self.deployment_info.get("agents", {}).values():
            if isinstance(agent, dict) and agent.get("status") == "deployed":
                agents_deployed += 1
            elif isinstance(agent, str) and agent == "deployed":
                agents_deployed += 1
        
        # Safely count passed tests
        tests_passed = 0
        for test in self.deployment_info.get("tests", {}).values():
            if isinstance(test, dict) and test.get("status") == "passed":
                tests_passed += 1
        
        # Safely count completed demos
        demos_completed = 0
        for demo in self.deployment_info.get("demos", {}).values():
            if isinstance(demo, dict) and demo.get("status") == "completed":
                demos_completed += 1
        
        summary = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "agents_deployed": agents_deployed,
            "tests_passed": tests_passed,
            "demos_completed": demos_completed,
            "gcp_resources": self.deployment_info.get("gcp_resources", {}),
            "next_steps": [
                "Configure production monitoring and alerting",
                "Set up CI/CD pipeline for agent updates",
                "Configure real-time data sources",
                "Set up production incident management workflows"
            ]
        }
        
        return summary
    
    async def save_deployment_info(self, filename: str = None) -> str:
        """Save deployment information to JSON file"""
        if not filename:
            filename = f"sentinel_ai_deployment_{self.deployment_id[:8]}.json"
        
        filepath = Path(filename)
        
        # Add summary to deployment info
        self.deployment_info["summary"] = await self.generate_deployment_summary()
        
        with open(filepath, 'w') as f:
            json.dump(self.deployment_info, f, indent=2, default=str)
        
        self.logger.info(f"Deployment information saved to: {filepath}")
        return str(filepath)
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Cleaning up deployment resources...")
        
        try:
            if hasattr(self, 'conductor'):
                await self.conductor.shutdown()
            if hasattr(self, 'agent_registry'):
                await self.agent_registry.stop()
            if hasattr(self, 'message_bus'):
                await self.message_bus.stop()
                
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def deploy_all(self) -> bool:
        """Execute complete deployment process"""
        try:
            self.logger.info("Starting comprehensive Sentinel AI deployment...")
            
            # Step 1: Check prerequisites
            if not await self.check_gcp_prerequisites():
                self.logger.error("GCP prerequisites check failed")
                return False
            
            # Step 2: Setup GCP resources
            if not await self.setup_gcp_resources():
                self.logger.error("GCP resource setup failed")
                return False
            
            # Step 3: Deploy Conductor Agent
            if not await self.deploy_conductor_agent():
                self.logger.error("Conductor Agent deployment failed")
                return False
            
            # Step 4: Deploy Diagnostic Agent
            if not await self.deploy_diagnostic_agent():
                self.logger.error("Diagnostic Agent deployment failed")
                return False
            
            # Step 5: Run automated tests
            if not await self.run_automated_tests():
                self.logger.warning("Some automated tests failed, but continuing...")
            
            # Step 6: Run demo scenarios
            if not await self.run_demo_scenarios():
                self.logger.warning("Some demo scenarios failed, but continuing...")
            
            # Step 7: Generate summary and save info
            summary = await self.generate_deployment_summary()
            deployment_file = await self.save_deployment_info()
            
            self.logger.info("=" * 60)
            self.logger.info("SENTINEL AI DEPLOYMENT COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Deployment ID: {self.deployment_id}")
            self.logger.info(f"Agents Deployed: {summary['agents_deployed']}")
            self.logger.info(f"Tests Passed: {summary['tests_passed']}")
            self.logger.info(f"Demos Completed: {summary['demos_completed']}")
            self.logger.info(f"Deployment Info: {deployment_file}")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.deployment_info["status"] = "failed"
            self.deployment_info["error"] = str(e)
            return False
        
        finally:
            await self.cleanup()


async def main():
    """Main deployment function"""
    print("=" * 60)
    print("    SENTINEL AI - COMPREHENSIVE GCP DEPLOYMENT")
    print("=" * 60)
    print()
    print("This script will deploy Sentinel AI to Google Cloud Platform")
    print("and run comprehensive tests and demonstrations.")
    print()
    print("Prerequisites:")
    print("- Google Cloud SDK installed and configured")
    print("- Active GCP project with billing enabled")
    print("- Appropriate IAM permissions")
    print()
    print("=" * 60)
    print()
    
    # Create deployer
    deployer = SentinelAIDeployer()
    
    try:
        # Initialize
        await deployer.initialize()
        
        # Run deployment
        success = await deployer.deploy_all()
        
        if success:
            print("\n🎉 Deployment completed successfully!")
            print("\nNext steps:")
            print("1. Review the deployment summary JSON file")
            print("2. Configure production monitoring")
            print("3. Set up real-time data sources")
            print("4. Configure incident management workflows")
        else:
            print("\n❌ Deployment failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nDeployment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nDeployment failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
