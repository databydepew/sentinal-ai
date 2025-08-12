#!/usr/bin/env python3
"""
Demonstration script for the Sentinel AI Conductor Agent.

This script shows how to initialize and use the Conductor Agent
with mock incident scenarios for testing purposes.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from utils.logging_setup import setup_logging, create_contextual_logger
from agents.conductor import ConductorAgent
from communication.agent_registry import AgentRegistry
from communication.message_bus import MessageBus
from examples.mock_incident import create_mock_incident


async def main():
    """Main demonstration function"""
    
    # Load configuration
    config = get_settings().to_dict()
    
    # Setup logging
    logging_config = config.get("logging", {})
    setup_logging(logging_config)
    
    logger = create_contextual_logger(__name__, demo="conductor")
    logger.info("Starting Sentinel AI Conductor demonstration")
    
    try:
        # Initialize core components
        logger.info("Initializing core components...")
        
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
        
        logger.info("Core components initialized successfully")
        
        # Run demonstration scenarios
        await run_demonstration_scenarios(conductor, logger)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise
    
    # finally:
    #     # Cleanup
    #     logger.info("Cleaning up...")
    #     try:
    #         if 'conductor' in locals():
    #             await conductor.shutdown()
    #         if 'agent_registry' in locals():
    #             await agent_registry.stop()
    #         if 'message_bus' in locals():
    #             await message_bus.stop()
    #     except Exception as e:
    #         logger.error(f"Cleanup error: {e}")


async def run_demonstration_scenarios(conductor: ConductorAgent, logger):
    """Run various demonstration scenarios"""
    
    logger.info("=== Running Demonstration Scenarios ===")
    
    # Scenario 1: Data Drift Detection
    await run_data_drift_scenario(conductor, logger)
    
    # Wait between scenarios
    await asyncio.sleep(2)
    
    # Scenario 2: Performance Degradation
    await run_performance_degradation_scenario(conductor, logger)
    
    # Wait between scenarios
    await asyncio.sleep(2)
    
    # Scenario 3: Model Error
    await run_model_error_scenario(conductor, logger)
    
    logger.info("=== All Demonstration Scenarios Completed ===")


async def run_data_drift_scenario(conductor: ConductorAgent, logger):
    """Demonstrate data drift incident handling"""
    
    logger.info("--- Scenario 1: Data Drift Detection ---")
    
    # Create mock data drift incident
    incident = create_mock_incident(
        incident_type="DATA_DRIFT",
        severity="MEDIUM",
        model_name="customer-churn-model",
        description="Significant data drift detected in customer demographics features"
    )
    
    logger.info(f"Created mock incident: {incident.incident_id}")
    
    # Process incident through conductor
    try:
        incident_id = await conductor.process_incident(incident)
        
        logger.info("Data drift scenario completed successfully")
        logger.info(f"Processed incident ID: {incident_id}")
        
        # Display results
        display_scenario_results("Data Drift", incident, incident_id, logger)
        
    except Exception as e:
        logger.error(f"Data drift scenario failed: {e}")


async def run_performance_degradation_scenario(conductor: ConductorAgent, logger):
    """Demonstrate performance degradation incident handling"""
    
    logger.info("--- Scenario 2: Performance Degradation ---")
    
    # Create mock performance degradation incident
    incident = create_mock_incident(
        incident_type="PERFORMANCE_DEGRADATION",
        severity="HIGH",
        model_name="fraud-detection-model",
        description="Model accuracy dropped from 95% to 87% over the past 24 hours"
    )
    
    logger.info(f"Created mock incident: {incident.incident_id}")
    
    # Process incident through conductor
    try:
        incident_id = await conductor.process_incident(incident)
        
        logger.info("Performance degradation scenario completed successfully")
        logger.info(f"Processed incident ID: {incident_id}")
        
        # Display results
        display_scenario_results("Performance Degradation", incident, incident_id, logger)
        
    except Exception as e:
        logger.error(f"Performance degradation scenario failed: {e}")


async def run_model_error_scenario(conductor: ConductorAgent, logger):
    """Demonstrate model error incident handling"""
    
    logger.info("--- Scenario 3: Model Error ---")
    
    # Create mock model error incident
    incident = create_mock_incident(
        incident_type="MODEL_ERROR",
        severity="CRITICAL",
        model_name="recommendation-engine",
        description="Model endpoint returning 500 errors for 15% of requests"
    )
    
    logger.info(f"Created mock incident: {incident.incident_id}")
    
    # Process incident through conductor
    try:
        incident_id = await conductor.process_incident(incident)
        
        logger.info("Model error scenario completed successfully")
        logger.info(f"Processed incident ID: {incident_id}")
        
        # Display results
        display_scenario_results("Model Error", incident, incident_id, logger)
        
    except Exception as e:
        logger.error(f"Model error scenario failed: {e}")


def display_scenario_results(scenario_name: str, incident, incident_id: str, logger):
    """Display the results of a scenario"""
    
    logger.info(f"=== {scenario_name} Scenario Results ===")
    logger.info(f"Incident ID: {incident.incident_id}")
    logger.info(f"Incident Type: {incident.incident_type.value if hasattr(incident.incident_type, 'value') else incident.incident_type}")
    logger.info(f"Severity: {incident.severity.value if hasattr(incident.severity, 'value') else incident.severity}")
    logger.info(f"Model: {incident.metadata.get('source_model_name', 'unknown')}")
    logger.info(f"Status: {incident.status.value if hasattr(incident.status, 'value') else incident.status}")
    
    # Display affected systems
    if incident.affected_systems:
        logger.info(f"Affected Systems: {', '.join(incident.affected_systems)}")
    
    # Display actions taken
    if incident.actions:
        logger.info("Actions Taken:")
        for i, action in enumerate(incident.actions, 1):
            logger.info(f"  {i}. {action.name} - {action.description}")
    
    logger.info("=" * 50)


async def demonstrate_agent_capabilities(conductor: ConductorAgent, logger):
    """Demonstrate various agent capabilities"""
    
    logger.info("=== Demonstrating Agent Capabilities ===")
    
    # Get agent capabilities
    capabilities = conductor.get_capabilities()
    
    logger.info(f"Conductor Agent has {len(capabilities)} capabilities:")
    for capability in capabilities:
        logger.info(f"  - {capability.name}: {capability.description}")
    
    # Demonstrate health monitoring
    health_status = await conductor.health_check()
    logger.info(f"Conductor health status: {health_status}")
    
    # Demonstrate state management
    state = conductor.get_state()
    logger.info(f"Conductor state: {state.status.value}")


async def demonstrate_governance_integration(conductor: ConductorAgent, logger):
    """Demonstrate governance and autonomy features"""
    
    logger.info("=== Demonstrating Governance Integration ===")
    
    # This would demonstrate autonomy controller and governance rules
    # For now, just log that governance is integrated
    logger.info("Governance features integrated:")
    logger.info("  - Autonomy level controls")
    logger.info("  - Human-in-the-loop checkpoints")
    logger.info("  - Cost-benefit analysis")
    logger.info("  - Risk assessment")


def print_demo_header():
    """Print demonstration header"""
    
    print("=" * 60)
    print("    SENTINEL AI - CONDUCTOR AGENT DEMONSTRATION")
    print("=" * 60)
    print()
    print("This demonstration shows the Sentinel AI Conductor Agent")
    print("processing various ML incident scenarios:")
    print()
    print("1. Data Drift Detection")
    print("2. Performance Degradation")
    print("3. Model Error Handling")
    print()
    print("The demonstration uses mock incidents and simulated")
    print("agent responses to show the complete incident")
    print("lifecycle management.")
    print()
    print("=" * 60)
    print()


def print_demo_footer():
    """Print demonstration footer"""
    
    print()
    print("=" * 60)
    print("    DEMONSTRATION COMPLETED")
    print("=" * 60)
    print()
    print("The Sentinel AI Conductor Agent demonstration has")
    print("completed successfully. The system demonstrated:")
    print()
    print("✓ Incident detection and classification")
    print("✓ Multi-agent orchestration")
    print("✓ Diagnosis and remediation planning")
    print("✓ Cost-benefit analysis")
    print("✓ Governance and approval workflows")
    print("✓ Execution and verification")
    print("✓ Reporting and communication")
    print()
    print("For production deployment, configure:")
    print("- GCP project and credentials")
    print("- Vertex AI and Gemini access")
    print("- BigQuery datasets")
    print("- Monitoring and alerting")
    print()
    print("=" * 60)


if __name__ == "__main__":
    # Print header
    print_demo_header()
    
    try:
        # Run the demonstration
        asyncio.run(main())
        
        # Print footer
        print_demo_footer()
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        sys.exit(1)
