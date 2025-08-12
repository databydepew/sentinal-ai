from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
import logging

from models.incident import Incident, AgentAction
from services.gemini_client import GeminiClient
from agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class ReportingAgent(BaseAgent):
    """Agent responsible for generating reports and notifications for ML incidents"""

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the reporting agent

        Args:
            gemini_client: Client for Gemini API
        """
        super().__init__({"max_concurrent_tasks": 10}, "reporting")
        self.gemini_client = gemini_client
        self.report_templates = self._load_report_templates()

    async def initialize(self) -> None:
        """Initialize the agent with required services"""
        # No additional initialization required
        pass

    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent provides"""
        return [
            AgentCapability(
                name="generate_report", description="Generate incident reports"
            ),
            AgentCapability(
                name="create_notifications",
                description="Create stakeholder notifications",
            ),
            AgentCapability(
                name="generate_dashboard_data", description="Generate dashboard data"
            ),
        ]

    async def process_incident(self, incident: Incident) -> Dict[str, Any]:
        """Process an incident by generating reports and notifications

        Args:
            incident: The incident to process

        Returns:
            Dict containing the reporting results
        """
        # Create reporting action
        action = AgentAction(
            agent_id="ReportingAgent",
            action_type="generate_report",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=False,
        )

        try:
            # Generate comprehensive report package
            report_package = await self._generate_report_package(incident)

            # Generate communications
            communications = await self._generate_communications(incident)

            # Generate dashboard data
            dashboard_data = await self._generate_dashboard_data(incident)

            # Store report for historical analysis
            await self._store_report(report_package)

            # Update action with success
            action.success = True
            action.output_data = {
                "report_id": report_package["report_id"],
                "communications_generated": len(communications["notifications"]),
            }

            # Add action to incident
            incident.add_action(action)

            return {
                "report_package": report_package,
                "communications": communications,
                "dashboard_data": dashboard_data,
            }

        except Exception as e:
            action.output_data = {"error": str(e)}
            incident.add_action(action)
            raise

    async def _generate_report_package(self, incident: Incident) -> Dict[str, Any]:
        """Generate comprehensive report package for the incident"""

        report_id = str(uuid.uuid4())

        # Generate different report types
        executive_summary = await self._generate_executive_summary(incident)
        technical_report = await self._generate_technical_report(incident)
        lessons_learned = await self._generate_lessons_learned(incident)

        # Calculate KPIs
        kpis = await self._calculate_kpis(incident)

        return {
            "report_id": report_id,
            "incident_id": incident.incident_id,
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "technical_report": technical_report,
            "lessons_learned": lessons_learned,
            "kpis": kpis,
            "timeline": self._build_timeline_data(incident),
            "status_progression": self._build_status_progression_data(incident),
        }

    async def _generate_executive_summary(self, incident: Incident) -> str:
        """Generate executive summary using Gemini"""

        prompt = f"""
        Generate an executive summary for this ML incident:
        
        Incident: {incident.title}
        Type: {incident.incident_type.value}
        Severity: {incident.severity.value}
        Model: {incident.source_model_name}
        Duration: {incident.get_duration_minutes() or 'Ongoing'} minutes
        Status: {incident.status.value}
        
        Root Cause: {incident.diagnosis.root_cause if incident.diagnosis else 'Unknown'}
        
        Create a concise executive summary that includes:
        1. Brief incident overview
        2. Business impact
        3. Resolution summary
        4. Key recommendations
        
        Keep it business-focused, concise, and avoid technical jargon.
        """

        return await self.gemini_client.generate_text(prompt)

    async def _generate_technical_report(self, incident: Incident) -> str:
        """Generate technical report using Gemini"""

        prompt = f"""
        Generate a technical incident report for this ML incident:
        
        Incident: {incident.title}
        Type: {incident.incident_type.value}
        Severity: {incident.severity.value}
        Model: {incident.source_model_name}
        Duration: {incident.get_duration_minutes() or 'Ongoing'} minutes
        Status: {incident.status.value}
        
        Diagnosis: {json.dumps(incident.diagnosis.__dict__ if incident.diagnosis else {}, indent=2)}
        Remediation Plan: {json.dumps(incident.remediation_plan.__dict__ if incident.remediation_plan else {}, indent=2)}
        
        Create a detailed technical report that includes:
        1. Detailed incident description
        2. Technical root cause analysis
        3. Remediation steps taken
        4. Technical recommendations
        5. System impact details
        
        Use appropriate technical language for an engineering audience.
        """

        return await self.gemini_client.generate_text(prompt)

    async def _generate_lessons_learned(self, incident: Incident) -> List[str]:
        """Generate lessons learned from the incident"""

        prompt = f"""
        Generate lessons learned from this ML incident:
        
        Incident: {incident.title}
        Type: {incident.incident_type.value}
        Severity: {incident.severity.value}
        Model: {incident.source_model_name}
        Root Cause: {incident.diagnosis.root_cause if incident.diagnosis else 'Unknown'}
        
        List 5-7 specific lessons learned that could prevent similar incidents in the future.
        Format as a JSON array of strings.
        """

        response = await self.gemini_client.generate_text(prompt)

        try:
            # Extract JSON array from response
            response = response.strip()
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_content = response[start_idx:end_idx]
                lessons = json.loads(json_content)
                return lessons
            else:
                # Fallback if JSON parsing fails
                return [
                    "Improve monitoring systems",
                    "Enhance testing procedures",
                    "Review deployment processes",
                ]

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return [
                "Improve monitoring systems",
                "Enhance testing procedures",
                "Review deployment processes",
            ]

    async def _generate_communications(self, incident: Incident) -> Dict[str, Any]:
        """Generate communications for stakeholders"""

        communications = {
            "notifications": [],
            "status_updates": [],
            "follow_up_actions": [],
        }

        # Generate notifications for different stakeholder groups
        stakeholder_groups = ["executives", "engineering", "product", "support"]

        for group in stakeholder_groups:
            notification = await self._generate_group_notification(incident, group)
            communications["notifications"].append(notification)

        # Generate status updates
        if incident.status.value == "RESOLVED":
            status_update = {
                "type": "resolution_notification",
                "title": f"Incident {incident.incident_id} Resolved",
                "message": f"The {incident.incident_type.value.lower()} incident affecting {incident.source_model_name} has been resolved.",
                "timestamp": datetime.now().isoformat(),
            }
            communications["status_updates"].append(status_update)

        # Generate follow-up actions
        communications["follow_up_actions"] = [
            "Schedule post-incident review meeting",
            "Update monitoring thresholds based on lessons learned",
            "Review and update incident response procedures",
            "Communicate resolution to affected stakeholders",
        ]

        return communications

    async def _generate_group_notification(
        self, incident: Incident, group: str
    ) -> Dict[str, Any]:
        """Generate notification for specific stakeholder group"""

        group_templates = {
            "executives": {
                "subject": f"ML Incident Resolution Update - {incident.source_model_name}",
                "tone": "business-focused",
                "details": "high-level",
            },
            "engineering": {
                "subject": f"Technical Incident Report - {incident.incident_id}",
                "tone": "technical",
                "details": "detailed",
            },
            "product": {
                "subject": f"Product Impact Update - {incident.source_model_name}",
                "tone": "product-focused",
                "details": "moderate",
            },
            "support": {
                "subject": f"Customer Impact Resolution - {incident.incident_id}",
                "tone": "customer-focused",
                "details": "moderate",
            },
        }

        template = group_templates.get(group, group_templates["engineering"])

        # Generate notification content using Gemini
        prompt = f"""
        Generate a {template['tone']} notification for {group} about this ML incident:
        
        Incident: {incident.title}
        Status: {incident.status.value}
        Model: {incident.source_model_name}
        Duration: {incident.get_duration_minutes() or 'Ongoing'} minutes
        
        Create a brief notification with:
        1. Clear subject line
        2. Concise summary appropriate for {group}
        3. Key actions taken
        4. Current status
        5. Next steps if applicable
        
        Keep it {template['details']} level and {template['tone']}.
        """

        content = await self.gemini_client.generate_text(prompt)

        return {
            "stakeholder_group": group,
            "subject": template["subject"],
            "content": content,
            "priority": "normal"
            if incident.severity.value in ["LOW", "MEDIUM"]
            else "high",
            "channels": self._get_notification_channels(group),
            "generated_at": datetime.now().isoformat(),
        }

    def _get_notification_channels(self, group: str) -> List[str]:
        """Get appropriate notification channels for stakeholder group"""

        channel_mapping = {
            "executives": ["email", "slack"],
            "engineering": ["slack", "email", "pagerduty"],
            "product": ["email", "slack"],
            "support": ["email", "ticketing_system"],
        }

        return channel_mapping.get(group, ["email"])

    async def _generate_dashboard_data(self, incident: Incident) -> Dict[str, Any]:
        """Generate metrics and data for monitoring dashboards"""

        dashboard_data = {
            "dashboard_id": str(uuid.uuid4()),
            "incident_id": incident.incident_id,
            "generated_at": datetime.now().isoformat(),
            "metrics": await self._extract_dashboard_metrics(incident),
            "charts": await self._generate_chart_data(incident),
            "alerts": await self._generate_alert_data(incident),
            "timeline": self._build_timeline_data(incident),
        }

        return dashboard_data

    async def _extract_dashboard_metrics(self, incident: Incident) -> Dict[str, Any]:
        """Extract metrics for dashboard display"""

        metrics = {
            "incident": {
                "duration_minutes": incident.get_duration_minutes() or 0,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "type": incident.incident_type.value,
            },
            "model": {
                "name": incident.source_model_name,
                "version": incident.metadata.get("model_version", "unknown")
                if incident.metadata
                else "unknown",
            },
        }

        # Add diagnosis metrics if available
        if incident.diagnosis:
            metrics["diagnosis"] = {
                "confidence_score": incident.diagnosis.confidence_score,
                "contributing_factors_count": len(
                    incident.diagnosis.contributing_factors
                ),
            }

        # Add remediation metrics if available
        if incident.remediation_plan:
            metrics["remediation"] = {
                "steps_count": len(incident.remediation_plan.steps),
                "estimated_duration_hours": incident.remediation_plan.estimated_duration_hours,
                "risk_level": incident.remediation_plan.risk_level,
            }

        return metrics

    async def _generate_chart_data(self, incident: Incident) -> Dict[str, Any]:
        """Generate chart data for dashboard visualization"""

        # This would be enhanced with actual metrics in a real implementation

        # Example performance chart (before/after remediation)
        performance_chart = {
            "chart_type": "line",
            "title": "Model Performance",
            "x_axis": "Time",
            "y_axis": "Accuracy",
            "series": [
                {"name": "Before Remediation", "data": [0.92, 0.91, 0.89, 0.85, 0.82]},
                {"name": "After Remediation", "data": [0.82, 0.87, 0.90, 0.92, 0.93]},
            ],
        }

        # Example incident timeline chart
        timeline_chart = {
            "chart_type": "timeline",
            "title": "Incident Timeline",
            "data": self._build_timeline_data(incident),
        }

        return {"performance": performance_chart, "timeline": timeline_chart}

    def _build_timeline_data(self, incident: Incident) -> List[Dict[str, Any]]:
        """Build timeline data for visualization"""

        timeline = []

        # Add incident creation
        timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "event": "Incident Detected",
                "description": f"Incident {incident.incident_id} detected",
            }
        )

        # Add actions to timeline
        for action in incident.actions:
            timeline.append(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "event": action.action_type,
                    "description": f"{action.agent_name} performed {action.action_type}",
                    "success": action.success,
                }
            )

        # Add resolution if resolved
        if incident.resolved_at:
            timeline.append(
                {
                    "timestamp": incident.resolved_at.isoformat(),
                    "event": "Incident Resolved",
                    "description": incident.resolution_summary
                    or "Incident resolved successfully",
                }
            )

        return timeline

    def _build_status_progression_data(
        self, incident: Incident
    ) -> List[Dict[str, Any]]:
        """Build status progression data"""

        # Track status changes through actions
        status_progression = [
            {"timestamp": incident.created_at.isoformat(), "status": "DETECTED"}
        ]

        # This would be enhanced with actual status tracking
        if incident.resolved_at:
            status_progression.append(
                {"timestamp": incident.resolved_at.isoformat(), "status": "RESOLVED"}
            )

        return status_progression

    async def _generate_alert_data(self, incident: Incident) -> List[Dict[str, Any]]:
        """Generate alert data for dashboard"""

        alerts = []

        # High severity alert
        if incident.severity.value in ["HIGH", "CRITICAL"]:
            alerts.append(
                {
                    "alert_type": "high_severity",
                    "message": f"High severity incident: {incident.title}",
                    "severity": incident.severity.value,
                    "timestamp": incident.created_at.isoformat(),
                }
            )

        # Long duration alert
        duration = incident.get_duration_minutes() or 0
        if duration > 60:  # More than 1 hour
            alerts.append(
                {
                    "alert_type": "long_duration",
                    "message": f"Incident duration exceeds 1 hour: {duration:.0f} minutes",
                    "severity": "MEDIUM",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return alerts

    async def _calculate_kpis(self, incident: Incident) -> Dict[str, Any]:
        """Calculate key performance indicators"""

        kpis = {
            "mttr": incident.get_duration_minutes() or 0,  # Mean Time To Resolution
            "incident_count": 1,  # This incident
            "severity_distribution": {incident.severity.value: 1},
            "resolution_rate": 1.0 if incident.status.value == "RESOLVED" else 0.0,
        }

        # Add cost metrics if available
        if incident.cost_benefit_analysis:
            kpis["cost_metrics"] = {
                "remediation_cost": incident.cost_benefit_analysis.estimated_cost_usd,
                "business_impact_avoided": incident.cost_benefit_analysis.estimated_business_impact_usd,
                "roi": incident.cost_benefit_analysis.cost_benefit_ratio,
            }

        return kpis

    async def _store_report(self, report_package: Dict[str, Any]) -> None:
        """Store report in BigQuery for historical analysis"""

        try:
            # Placeholder for BigQuery storage
            # In real implementation, this would store the report
            logger.info(f"Storing report {report_package['report_id']} in BigQuery")

        except Exception as e:
            logger.error(f"Failed to store report: {e}")

    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load report templates for different audiences and formats"""

        return {
            "executive_summary": {
                "format": "business",
                "sections": ["overview", "impact", "resolution", "recommendations"],
                "tone": "executive",
            },
            "technical_report": {
                "format": "technical",
                "sections": [
                    "technical_details",
                    "root_cause",
                    "remediation",
                    "lessons_learned",
                ],
                "tone": "engineering",
            },
            "stakeholder_update": {
                "format": "communication",
                "sections": ["status", "actions", "next_steps"],
                "tone": "informative",
            },
            "dashboard_summary": {
                "format": "metrics",
                "sections": ["kpis", "charts", "alerts"],
                "tone": "data_driven",
            },
        }
