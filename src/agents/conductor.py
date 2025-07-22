import logging
from typing import Dict, List, Optional, Any
import uuid
import asyncio
from datetime import datetime

from models.incident import Incident, AgentAction
from services.gemini_client import GeminiClient
from agents.remediation import RemediationAgent
from agents.verification import VerificationAgent
from agents.diagnostic import DiagnosticAgent
from agents.reporting import ReportingAgent
from models.agent_state import AgentState

logger = logging.getLogger(__name__)


class ConductorAgent:
    """Agent responsible for orchestrating incident remediation"""

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the conductor agent

        Args:
            gemini_client: Client for Gemini API
        """
        self.gemini_client = gemini_client
        self.incident_memory = {}  # In-memory storage for incidents
        self.state = AgentState(
            agent_id=str(uuid.uuid4()),
            incident_id="",  # No incident initially
            status="idle",   # Start in idle status
            last_updated=datetime.now()
        )
        self.state.active_incidents = []
        self.state.registered_agents = {}
        self.state.total_incidents_processed = 0
        self.state.incidents_resolved_24h = 0
        self.state.average_resolution_time_minutes = 0
        self.running = False
        self.task_queue = asyncio.Queue()
        self.vertex_client = None  # Will be initialized later

        # Initialize specialized agents
        self.specialized_agents = {}

        # Communication components (will be set by demo)
        self.message_bus = None
        self.agent_registry = None
        
        logger.info("ConductorAgent initialized")

    async def initialize(self) -> None:
        """Initialize the conductor agent and its components"""
        logger.info("Starting ConductorAgent initialization...")
        
        # Initialize specialized agents
        logger.info("Creating specialized agents...")
        self.specialized_agents = {
            "remediation": RemediationAgent(self.gemini_client),
            "verification": VerificationAgent(
                {"project_id": "sentinel-ai", "location": "us-central1"}
            ),
            "diagnostic": DiagnosticAgent(self.gemini_client),
            "reporting": ReportingAgent(self.gemini_client),
        }
        logger.info(f"Created {len(self.specialized_agents)} specialized agents")

        # Start all specialized agents
        logger.info("Starting specialized agents...")
        for agent_name, agent in self.specialized_agents.items():
            try:
                logger.info(f"Starting {agent_name} agent...")
                await agent.start()
                self.state.registered_agents[agent.agent_id] = agent
                logger.info(f"Successfully started {agent_name} agent")
            except Exception as e:
                logger.error(f"Failed to start {agent_name} agent: {e}")
                raise

        self.running = True
        logger.info("ConductorAgent fully initialized")
    
    def set_message_bus(self, message_bus):
        """Set the message bus for communication"""
        self.message_bus = message_bus
        logger.info("Message bus set for ConductorAgent")
    
    def set_agent_registry(self, agent_registry):
        """Set the agent registry for agent management"""
        self.agent_registry = agent_registry
        logger.info("Agent registry set for ConductorAgent")
    
    async def shutdown(self):
        """Shutdown the conductor agent and cleanup resources"""
        logger.info("Shutting down ConductorAgent...")
        self.running = False
        
        # Stop all specialized agents
        for agent_name, agent in self.specialized_agents.items():
            try:
                if hasattr(agent, 'stop'):
                    await agent.stop()
                logger.info(f"Stopped {agent_name} agent")
            except Exception as e:
                logger.error(f"Error stopping {agent_name} agent: {e}")
        
        logger.info("ConductorAgent shutdown complete")

    async def _call_verification_agent(self, incident: Incident) -> Dict[str, Any]:
        """Call the verification agent for plan execution"""
        # Get verification agent
        verification_agent = self.specialized_agents.get("verification")
        if not verification_agent:
            logger.error("Verification agent not available")
            return {"error": "Verification agent not available"}

        # Process incident with verification agent
        result = await verification_agent.process_incident(incident)
        return result

    async def _request_human_approval(self, incident: Incident) -> None:
        """Request human approval for incident remediation"""
        # Placeholder for human approval request
        # In real implementation, this would send notifications via email, Slack, etc.

        logger.info(f"Human approval requested for incident: {incident.incident_id}")

        # Generate approval request using Gemini
        approval_request = await self._generate_approval_request(incident)

        # Send notification (placeholder)
        await self._send_approval_notification(incident, approval_request)

    async def _generate_approval_request(self, incident: Incident) -> str:
        """Generate human-readable approval request using Gemini"""
        try:
            prompt = f"""
            Generate a concise approval request for the following ML incident:
            
            Incident: {incident.title}
            Type: {incident.incident_type.value}
            Severity: {incident.severity.value}
            Model: {incident.source_model_name}
            
            Diagnosis: {incident.diagnosis.root_cause if incident.diagnosis else 'Pending'}
            Remediation Plan: {incident.remediation_plan.description if incident.remediation_plan else 'Pending'}
            Cost: ${incident.cost_benefit_analysis.estimated_cost_usd if incident.cost_benefit_analysis else 'TBD'}
            Business Impact: ${incident.cost_benefit_analysis.estimated_business_impact_usd if incident.cost_benefit_analysis else 'TBD'}
            
            Create a brief, business-focused approval request that explains:
            1. What happened
            2. What we plan to do
            3. Cost and business impact
            4. Risk level
            """

            response = await self.gemini_client.generate_text(prompt)
            return response

        except Exception as e:
            logger.error(f"Failed to generate approval request: {e}")
            return f"Approval required for incident {incident.incident_id}: {incident.title}"

    async def _generate_incident_report(self, incident: Incident) -> None:
        """Generate final incident report using reporting agent"""
        logger.info(f"Generating incident report for: {incident.incident_id}")

        action = AgentAction(
            agent_name="ConductorAgent",
            action_type="generate_report",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=False,
        )

        try:
            # Delegate to reporting agent
            reporting_agent = self.specialized_agents.get("reporting")
            if reporting_agent:
                # Call reporting agent
                report = await self._call_reporting_agent(incident)

                # Store report metadata
                if not incident.metadata:
                    incident.metadata = {}
                incident.metadata["final_report"] = report

            action.success = True
            action.output_data = {"report_generated": True}

        except Exception as e:
            logger.error(f"Failed to generate incident report: {e}")
            action.output_data = {"error": str(e)}

        finally:
            incident.add_action(action)

    async def _call_reporting_agent(self, incident: Incident) -> Dict[str, Any]:
        """Call the reporting agent for incident summary generation"""
        # Get reporting agent
        reporting_agent = self.specialized_agents.get("reporting")
        if not reporting_agent:
            logger.error("Reporting agent not available")
            return {"error": "Reporting agent not available"}

        # Process incident with reporting agent
        result = await reporting_agent.process_incident(incident)
        return result

    async def _handle_agent_callback(self, callback_data: Dict[str, Any]) -> None:
        """Handle callbacks from specialized agents"""
        agent_id = callback_data.get("agent_id")
        incident_id = callback_data.get("incident_id")
        action_type = callback_data.get("action_type")
        result = callback_data.get("result", {})

        logger.info(
            f"Received callback from agent {agent_id} for incident {incident_id}"
        )

        # Get incident from memory
        incident = self.incident_memory.get(incident_id)
        if not incident:
            logger.error(f"Incident {incident_id} not found in memory")
            return

        # Handle different action types
        if action_type == "diagnosis_complete":
            # Update incident with diagnosis
            incident.diagnosis = result.get("diagnosis")
            incident.update_status("DIAGNOSED")

            # Generate remediation plan
            await self._generate_remediation_plan(incident)

        elif action_type == "remediation_plan_complete":
            # Update incident with remediation plan
            incident.remediation_plan = result.get("remediation_plan")
            incident.update_status("REMEDIATION_PLANNED")

            # Request approval if needed
            if incident.requires_human_approval:
                await self._request_human_approval(incident)
            else:
                # Execute remediation plan
                await self._execute_remediation_plan(incident)

        elif action_type == "remediation_execution_complete":
            # Update incident with execution results
            execution_success = result.get("execution_success", False)

            if execution_success:
                incident.update_status("RESOLVED")
                incident.resolved_at = datetime.now()

                # Generate final report
                await self._generate_incident_report(incident)
            else:
                incident.update_status("REMEDIATION_FAILED")

                # Log failure details
                if not incident.metadata:
                    incident.metadata = {}
                incident.metadata["remediation_failure"] = result.get(
                    "error", "Unknown error"
                )

    async def process_incident(self, incident: Incident) -> str:
        """Process a new incident

        Args:
            incident: The incident to process

        Returns:
            Incident ID
        """
        # Store incident in memory
        self.incident_memory[incident.incident_id] = incident
        self.state.active_incidents.append(incident.incident_id)

        # Create initial action
        action = AgentAction(
            agent_name="ConductorAgent",
            action_type="receive_incident",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=True,
        )
        incident.add_action(action)

        # Start diagnostic process
        await self._diagnose_incident(incident)

        return incident.incident_id

    async def _diagnose_incident(self, incident: Incident) -> None:
        """Diagnose an incident using the diagnostic agent"""
        # Get diagnostic agent
        diagnostic_agent = self.specialized_agents.get("diagnostic")
        if not diagnostic_agent:
            logger.error("Diagnostic agent not available")
            return

        # Update incident status
        incident.update_status("DIAGNOSING")

        # Create diagnostic action
        action = AgentAction(
            agent_name="ConductorAgent",
            action_type="start_diagnosis",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=True,
        )
        incident.add_action(action)

        try:
            # Process incident with diagnostic agent
            result = await diagnostic_agent.process_incident(incident)

            # Update incident with diagnosis
            incident.diagnosis = result.get("diagnosis")
            incident.update_status("DIAGNOSED")

            # Generate remediation plan
            await self._generate_remediation_plan(incident)

        except Exception as e:
            logger.error(f"Failed to diagnose incident {incident.incident_id}: {e}")
            incident.update_status("DIAGNOSIS_FAILED")

    async def _generate_remediation_plan(self, incident: Incident) -> None:
        """Generate remediation plan using the remediation agent"""
        # Get remediation agent
        remediation_agent = self.specialized_agents.get("remediation")
        if not remediation_agent:
            logger.error("Remediation agent not available")
            return

        # Update incident status
        incident.update_status("PLANNING_REMEDIATION")

        # Create remediation planning action
        action = AgentAction(
            agent_name="ConductorAgent",
            action_type="start_remediation_planning",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=True,
        )
        incident.add_action(action)

        try:
            # Process incident with remediation agent
            result = await remediation_agent.process_incident(incident)

            # Update incident with remediation plan
            incident.remediation_plan = result.get("remediation_plan")
            incident.update_status("REMEDIATION_PLANNED")

            # Request approval if needed
            if incident.requires_human_approval:
                await self._request_human_approval(incident)
            else:
                # Execute remediation plan
                await self._execute_remediation_plan(incident)

        except Exception as e:
            logger.error(
                f"Failed to generate remediation plan for incident {incident.incident_id}: {e}"
            )
            incident.update_status("REMEDIATION_PLANNING_FAILED")

    async def _execute_remediation_plan(self, incident: Incident) -> None:
        """Execute remediation plan using the verification agent"""
        # Get verification agent
        verification_agent = self.specialized_agents.get("verification")
        if not verification_agent:
            logger.error("Verification agent not available")
            return

        # Update incident status
        incident.update_status("EXECUTING_REMEDIATION")

        # Create remediation execution action
        action = AgentAction(
            agent_name="ConductorAgent",
            action_type="start_remediation_execution",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=True,
        )
        incident.add_action(action)

        try:
            # Process incident with verification agent
            result = await verification_agent.process_incident(incident)

            # Update incident based on execution result
            execution_success = result.get("execution_success", False)

            if execution_success:
                incident.update_status("RESOLVED")
                incident.resolved_at = datetime.now()

                # Generate final report
                await self._generate_incident_report(incident)
            else:
                incident.update_status("REMEDIATION_FAILED")

                # Log failure details
                if not incident.metadata:
                    incident.metadata = {}
                incident.metadata["remediation_failure"] = result.get(
                    "error", "Unknown error"
                )

        except Exception as e:
            logger.error(
                f"Failed to execute remediation plan for incident {incident.incident_id}: {e}"
            )
            incident.update_status("REMEDIATION_EXECUTION_FAILED")

    async def approve_incident(self, incident_id: str, approver: str) -> bool:
        """Approve an incident for execution (called by external approval system)"""
        try:
            incident = self.incident_memory.get(incident_id)
            if not incident:
                logger.error(f"Incident {incident_id} not found")
                return False

            # Update incident with approval info
            incident.approved_by = approver
            incident.approved_at = datetime.now()

            # Create approval action
            action = AgentAction(
                agent_name="ExternalSystem",
                action_type="approve_remediation",
                timestamp=datetime.now(),
                input_data={"incident_id": incident_id, "approver": approver},
                output_data={},
                success=True,
            )
            incident.add_action(action)

            # Execute remediation plan
            await self._execute_remediation_plan(incident)

            logger.info(f"Incident {incident_id} approved by {approver}")
            return True

        except Exception as e:
            logger.error(f"Error approving incident {incident_id}: {e}")
            return False

    async def reject_incident(
        self, incident_id: str, rejector: str, reason: str
    ) -> bool:
        """Reject an incident (called by external approval system)"""
        try:
            incident = self.incident_memory.get(incident_id)
            if not incident:
                logger.error(f"Incident {incident_id} not found")
                return False

            # Update incident with rejection info
            incident.update_status("REJECTED")

            # Create rejection action
            action = AgentAction(
                agent_name="ExternalSystem",
                action_type="reject_remediation",
                timestamp=datetime.now(),
                input_data={
                    "incident_id": incident_id,
                    "rejector": rejector,
                    "reason": reason,
                },
                output_data={},
                success=True,
            )
            incident.add_action(action)

            # Store rejection details in metadata
            if not incident.metadata:
                incident.metadata = {}
            incident.metadata["rejection"] = {
                "rejector": rejector,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Incident {incident_id} rejected by {rejector}")
            return True

        except Exception as e:
            logger.error(f"Error rejecting incident {incident_id}: {e}")
            return False

    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an incident"""
        incident = self.incident_memory.get(incident_id)
        if not incident:
            return None

        return {
            "incident_id": incident.incident_id,
            "status": incident.status.value,
            "severity": incident.severity.value,
            "created_at": incident.created_at.isoformat(),
            "updated_at": incident.updated_at.isoformat(),
            "title": incident.title,
            "source_model": incident.source_model_name,
            "requires_approval": incident.requires_human_approval,
            "approved_by": incident.approved_by,
            "resolution_summary": incident.resolution_summary,
            "duration_minutes": incident.get_duration_minutes(),
        }

    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get list of all active incidents"""
        active_incidents = []

        for incident_id in self.state.active_incidents:
            incident_status = self.get_incident_status(incident_id)
            if incident_status:
                active_incidents.append(incident_status)

        return active_incidents

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics and health information"""
        return {
            "conductor_state": {
                "active_incidents": len(self.state.active_incidents),
                "total_processed": self.state.total_incidents_processed,
                "incidents_resolved_24h": self.state.incidents_resolved_24h,
                "average_resolution_time": self.state.average_resolution_time_minutes,
            },
            "agent_registry": {
                "total_agents": len(self.state.registered_agents),
                "available_agents": len(
                    [
                        a
                        for a in self.state.registered_agents.values()
                        if a.state.is_available()
                    ]
                ),
                "agents_by_type": {
                    agent_type: len(
                        [
                            a
                            for a in self.state.registered_agents.values()
                            if a.agent_type == agent_type
                        ]
                    )
                    for agent_type in [
                        "diagnostic",
                        "remediation",
                        "economist",
                        "verification",
                        "reporting",
                    ]
                },
            },
            "system_health": {
                "running": self.running,
                "queue_size": self.task_queue.qsize()
                if hasattr(self.task_queue, "qsize")
                else 0,
                "memory_incidents": len(self.incident_memory),
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            # Check service connections
            vertex_health = {"status": "unknown"}
            if self.vertex_client:
                vertex_health = await self.vertex_client.health_check()

            gemini_health = {"status": "unknown"}
            if self.gemini_client:
                gemini_health = await self.gemini_client.health_check()

            # Check agent availability
            available_agents = [
                a
                for a in self.state.registered_agents.values()
                if a.state.is_available()
            ]

            return {
                "status": "healthy" if self.running else "stopped",
                "services": {"vertex_ai": vertex_health, "gemini": gemini_health},
                "agents": {
                    "available": len(available_agents),
                    "total": len(self.state.registered_agents),
                },
                "incidents": {
                    "active": len(self.state.active_incidents),
                    "in_memory": len(self.incident_memory),
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
