from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
import logging

from src.models.incident import Incident, AgentAction, CostBenefitAnalysis
from src.services.gemini_client import GeminiClient
from src.agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class EconomistAgent(BaseAgent):
    """Agent responsible for cost-benefit analysis of ML incident remediation"""

    def __init__(
        self, gemini_client: GeminiClient, cost_benefit_threshold: float = 1.5
    ):
        """Initialize the economist agent

        Args:
            gemini_client: Client for Gemini API
            cost_benefit_threshold: Threshold for cost-benefit ratio to recommend remediation
        """
        super().__init__({"max_concurrent_tasks": 5}, "economist")
        self.gemini_client = gemini_client
        self.cost_benefit_threshold = cost_benefit_threshold
        self.cost_models = self._load_cost_models()
        self.pricing_data = self._load_pricing_data()

    async def initialize(self) -> None:
        """Initialize the agent with required services"""
        # No additional initialization required
        pass

    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent provides"""
        return [
            AgentCapability(
                name="cost_benefit_analysis",
                description="Perform cost-benefit analysis",
            ),
            AgentCapability(
                name="resource_estimation", description="Estimate resource requirements"
            ),
            AgentCapability(
                name="business_impact_assessment", description="Assess business impact"
            ),
        ]

    async def process_incident(self, incident: Incident) -> Dict[str, Any]:
        """Process an incident by performing cost-benefit analysis

        Args:
            incident: The incident to process

        Returns:
            Dict containing the cost-benefit analysis results
        """
        # Create economist action
        action = AgentAction(
            agent_name="EconomistAgent",
            action_type="cost_benefit_analysis",
            timestamp=datetime.now(),
            input_data={"incident_id": incident.incident_id},
            output_data={},
            success=False,
        )

        try:
            # Perform cost-benefit analysis
            analysis = await self._perform_cost_benefit_analysis(incident)

            # Create cost-benefit analysis object
            cost_benefit = CostBenefitAnalysis(
                estimated_cost_usd=analysis["remediation_cost"],
                estimated_business_impact_usd=analysis["business_impact"],
                cost_benefit_ratio=analysis["cost_benefit_ratio"],
                recommendation=analysis["recommendation"],
                confidence_score=analysis["confidence_score"],
            )

            # Attach analysis to incident
            incident.cost_benefit_analysis = cost_benefit

            # Update action with success
            action.success = True
            action.output_data = {
                "cost_benefit_ratio": analysis["cost_benefit_ratio"],
                "recommendation": analysis["recommendation"],
            }

            # Add action to incident
            incident.add_action(action)

            return analysis

        except Exception as e:
            action.output_data = {"error": str(e)}
            incident.add_action(action)
            raise

    async def _perform_cost_benefit_analysis(
        self, incident: Incident
    ) -> Dict[str, Any]:
        """Perform cost-benefit analysis for incident remediation

        Args:
            incident: The incident to analyze

        Returns:
            Dict with analysis results
        """
        # Estimate remediation cost
        remediation_cost = await self._estimate_remediation_cost(incident)

        # Estimate business impact
        business_impact = await self._estimate_business_impact(incident)

        # Calculate cost-benefit ratio
        cost_benefit_ratio = (
            business_impact / remediation_cost if remediation_cost > 0 else float("inf")
        )

        # Determine recommendation
        recommendation = (
            "PROCEED"
            if cost_benefit_ratio >= self.cost_benefit_threshold
            else "EVALUATE"
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            incident, remediation_cost, business_impact
        )

        return {
            "incident_id": incident.incident_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "remediation_cost": remediation_cost,
            "business_impact": business_impact,
            "cost_benefit_ratio": cost_benefit_ratio,
            "recommendation": recommendation,
            "confidence_score": confidence_score,
            "sensitivity_analysis": {
                "cost_range": {
                    "optimistic": remediation_cost * 0.8,
                    "pessimistic": remediation_cost * 1.5,
                },
                "impact_range": {
                    "optimistic": business_impact * 1.2,
                    "pessimistic": business_impact * 0.8,
                },
            },
            "recommendation_rationale": self._generate_recommendation_rationale(
                cost_benefit_ratio, remediation_cost, business_impact
            ),
        }

    async def _estimate_remediation_cost(self, incident: Incident) -> float:
        """Estimate the cost of remediation

        Args:
            incident: The incident to analyze

        Returns:
            Estimated cost in USD
        """
        # Base cost calculation
        base_cost = 0.0

        # Add engineering time cost if remediation plan exists
        if incident.remediation_plan:
            # Estimate engineer hours based on remediation steps
            engineer_hours = (
                len(incident.remediation_plan.steps) * 2
            )  # 2 hours per step as baseline

            # Add complexity factor
            if incident.remediation_plan.risk_level == "HIGH":
                engineer_hours *= 1.5

            # Calculate engineering cost
            engineering_cost = (
                engineer_hours * self.pricing_data["engineer_cost_per_hour"]
            )
            base_cost += engineering_cost

            # Add compute resources cost if needed
            if "retraining" in incident.remediation_plan.description.lower():
                compute_hours = 8  # Baseline for retraining
                compute_cost = (
                    compute_hours * self.pricing_data["compute_per_unit_hour"]
                )

                # Add GPU cost if likely needed
                if "deep learning" in incident.remediation_plan.description.lower():
                    compute_cost += (
                        compute_hours * self.pricing_data["gpu_per_unit_hour"]
                    )

                base_cost += compute_cost
        else:
            # If no remediation plan, use incident type to estimate
            if incident.incident_type.value == "DATA_DRIFT":
                base_cost = 5 * self.pricing_data["engineer_cost_per_hour"]
            elif incident.incident_type.value == "PERFORMANCE_DEGRADATION":
                base_cost = 10 * self.pricing_data["engineer_cost_per_hour"]
            elif incident.incident_type.value == "MODEL_ERROR":
                base_cost = 15 * self.pricing_data["engineer_cost_per_hour"]
            else:
                base_cost = 8 * self.pricing_data["engineer_cost_per_hour"]

        # Add opportunity cost
        opportunity_cost = (
            4 * self.pricing_data["opportunity_cost_per_hour"]
        )  # 4 hours of delay cost

        return base_cost + opportunity_cost

    async def _estimate_business_impact(self, incident: Incident) -> float:
        """Estimate the business impact of the incident

        Args:
            incident: The incident to analyze

        Returns:
            Estimated business impact in USD
        """
        # Base impact calculation
        base_impact = 0.0

        # Impact multiplier based on severity
        severity_multiplier = {
            "LOW": 1.0,
            "MEDIUM": 3.0,
            "HIGH": 10.0,
            "CRITICAL": 25.0,
        }.get(incident.severity.value, 1.0)

        # Impact based on incident duration (actual or estimated)
        duration_hours = (
            incident.get_duration_minutes() / 60
            if incident.get_duration_minutes()
            else 24
        )

        # Calculate impact based on model usage
        requests_per_hour = (
            incident.metadata.get("requests_per_hour", 1000)
            if incident.metadata
            else 1000
        )
        impact_per_request = 0.01  # Base impact per affected request

        # Calculate total impact
        base_impact = (
            requests_per_hour
            * duration_hours
            * impact_per_request
            * severity_multiplier
        )

        # Add reputation cost for high severity incidents
        if incident.severity.value in ["HIGH", "CRITICAL"]:
            reputation_cost = (
                base_impact * 0.5
            )  # 50% additional impact for reputation damage
            base_impact += reputation_cost

        return base_impact

    def _calculate_confidence_score(
        self, incident: Incident, remediation_cost: float, business_impact: float
    ) -> float:
        """Calculate confidence score for the analysis

        Args:
            incident: The incident being analyzed
            remediation_cost: Estimated remediation cost
            business_impact: Estimated business impact

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence
        confidence = 0.7

        # Adjust based on available information
        if incident.diagnosis:
            confidence += 0.1

        if incident.remediation_plan:
            confidence += 0.1

        # Reduce confidence for extreme values
        if remediation_cost > 10000 or business_impact > 50000:
            confidence -= 0.1

        # Ensure confidence is within bounds
        return max(0.1, min(confidence, 0.95))

    def _generate_recommendation_rationale(
        self, cost_benefit_ratio: float, remediation_cost: float, business_impact: float
    ) -> str:
        """Generate rationale for the recommendation"""

        if cost_benefit_ratio >= self.cost_benefit_threshold:
            return (
                f"Cost-benefit ratio of {cost_benefit_ratio:.2f} exceeds threshold of {self.cost_benefit_threshold}. "
                f"Expected business impact (${business_impact:.2f}) significantly outweighs remediation cost (${remediation_cost:.2f})."
            )
        else:
            return (
                f"Cost-benefit ratio of {cost_benefit_ratio:.2f} is below threshold of {self.cost_benefit_threshold}. "
                f"Remediation cost (${remediation_cost:.2f}) may not be justified by business impact (${business_impact:.2f})."
            )

    def _load_cost_models(self) -> Dict[str, Any]:
        """Load cost estimation models"""
        return {
            "compute_cost_model": {
                "base_rate_per_hour": 0.50,
                "gpu_multiplier": 10.0,
                "memory_multiplier": 0.1,
            },
            "storage_cost_model": {
                "base_rate_per_gb_hour": 0.001,
                "backup_multiplier": 0.5,
            },
            "business_impact_model": {
                "revenue_impact_multiplier": 0.05,
                "customer_satisfaction_weight": 0.3,
                "operational_cost_weight": 0.2,
            },
        }

    def _load_pricing_data(self) -> Dict[str, float]:
        """Load current pricing data for GCP resources"""
        return {
            "compute_per_unit_hour": 2.50,
            "storage_per_gb_hour": 0.02,
            "gpu_per_unit_hour": 25.00,
            "engineer_cost_per_hour": 150.00,
            "opportunity_cost_per_hour": 50.00,
        }
