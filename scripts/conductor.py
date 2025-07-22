from typing import Dict, List, Optional, Any
import uuid
from src.models.incident import (
    Incident,
    IncidentStatus,
    DiagnosisResult,
    RemediationPlan,
    CostBenefitAnalysis,
)
from src.models.agent import AgentAction
import asyncio

from datetime import datetime

from src.services.gemini_client import GeminiClient

import logging

logger = logging.getLogger(__name__)


class ConductorAgent:
    """Agent responsible for orchestrating incident remediation"""

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the conductor agent

        Args:
            gemini_client: Client for Gemini API
        """
        self.gemini_client = gemini_client

    async def _call_verification_agent(self, incident: Incident) -> Dict[str, Any]:
        """Call the verification agent for plan execution"""
        # Placeholder for actual verification agent call

        return {"execution_id": str(uuid.uuid4()), "status": "started"}

    async def _request_human_approval(self, incident: Incident) -> None:
        """Request human approval for incident remediation"""
        # Placeholder for human approval request
        # In real implementation, this would send notifications via email, Slack, etc.

        logger.info(f"Human approval requested for incident: {incident.incident_id}")

        # Generate approval request using Gemini
        await self._generate_approval_request(incident)

        # Send notification (placeholder)
        # await self._send_approval_notification(incident)

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
            timestamp=datetime.datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=False,
        )

        try:
            # Delegate to reporting agent
            reporting_agent = self.state.get_agent_by_type("reporting")
            if reporting_agent and reporting_agent.is_available():
                # Call reporting agent (placeholder for actual implementation)
                report = await self._call_reporting_agent(incident)

                # Store report metadata
                incident.metadata["final_report"] = report

            action.success = True
            action.output_data = {"report_generated": True}

        except Exception as e:
            action.error_message = str(e)
            logger.error(
                f"Report generation failed for incident {incident.incident_id}: {e}"
            )

        finally:
            incident.add_action(action)

    async def _call_reporting_agent(self, incident: Incident) -> Dict[str, Any]:
        """Call the reporting agent for incident summary generation"""
        # Placeholder for actual reporting agent call

        return {
            "report_id": str(uuid.uuid4()),
            "summary": f"Incident {incident.incident_id} resolved successfully",
            "format": "markdown",
            "generated_at": datetime.now().isoformat(),
        }

    async def _handle_agent_callback(self, callback_data: Dict[str, Any]) -> None:
        """Handle callbacks from specialized agents"""
        try:
            agent_type = callback_data.get("agent_type")
            incident_id = callback_data.get("incident_id")
            result = callback_data.get("result")

            if not all([agent_type, incident_id, result]):
                logger.error("Invalid callback data received")
                return

            incident = self.incident_memory.get(incident_id)
            if not incident:
                logger.error(f"Incident {incident_id} not found for callback")
                return

            # Process callback based on agent type
            if agent_type == "diagnostic":
                incident.diagnosis = DiagnosisResult(**result)
                incident.update_status(IncidentStatus.PLANNING)

            elif agent_type == "remediation":
                incident.remediation_plan = RemediationPlan(**result)
                incident.update_status(IncidentStatus.ANALYZING)

            elif agent_type == "economist":
                incident.cost_benefit_analysis = CostBenefitAnalysis(**result)
                incident.update_status(IncidentStatus.AWAITING_APPROVAL)

            elif agent_type == "verification":
                if result.get("status") == "completed":
                    incident.update_status(IncidentStatus.VERIFYING)
                elif result.get("status") == "verified":
                    incident.update_status(IncidentStatus.RESOLVED)

            # Continue processing
            await self.task_queue.put(("process_incident", incident_id))

        except Exception as e:
            logger.error(f"Error handling agent callback: {e}")

    def approve_incident(self, incident_id: str, approver: str) -> bool:
        """Approve an incident for execution (called by external approval system)"""
        try:
            incident = self.incident_memory.get(incident_id)
            if not incident:
                logger.error(f"Incident {incident_id} not found for approval")
                return False

            if incident.status != IncidentStatus.AWAITING_APPROVAL:
                logger.error(f"Incident {incident_id} not in approval state")
                return False

            # Record approval
            incident.approved_by = approver
            incident.approved_at = datetime.now()
            incident.requires_human_approval = False
            incident.update_status(IncidentStatus.EXECUTING)

            # Queue for continued processing
            asyncio.create_task(self.task_queue.put(("process_incident", incident_id)))

            logger.info(f"Incident {incident_id} approved by {approver}")
            return True

        except Exception as e:
            logger.error(f"Error approving incident {incident_id}: {e}")
            return False

    def reject_incident(self, incident_id: str, rejector: str, reason: str) -> bool:
        """Reject an incident (called by external approval system)"""
        try:
            incident = self.incident_memory.get(incident_id)
            if not incident:
                logger.error(f"Incident {incident_id} not found for rejection")
                return False

            # Record rejection
            incident.update_status(IncidentStatus.FAILED)
            incident.final_outcome = f"Rejected by {rejector}: {reason}"

            # Complete the incident
            self.state.complete_incident(incident_id)

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
                "available_agents": len(self.state.get_available_agents()),
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
            vertex_health = await self.vertex_client.health_check()
            gemini_health = await self.gemini_client.health_check()

            # Check agent availability
            available_agents = self.state.get_available_agents()

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
