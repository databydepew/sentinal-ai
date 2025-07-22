from typing import Dict, List, Optional, Any
import uuid
import json


from models.incident import Incident, RemediationPlan
from services.gemini_client import GeminiClient

class RemediationAgent:
    """Agent responsible for generating and managing remediation plans for incidents"""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize the remediation agent
        
        Args:
            gemini_client: Client for Gemini API
        """
        self.gemini_client = gemini_client
        self.remediation_templates = self._load_remediation_templates()
        self.agent_id = str(uuid.uuid4())
        self.running = False
    
    async def start(self):
        """Start the remediation agent"""
        self.running = True
        # Agent-specific initialization can go here
        pass
    
    async def stop(self):
        """Stop the remediation agent"""
        self.running = False
        # Agent-specific cleanup can go here
        pass
    
    async def generate_remediation_plan(self, incident: Incident) -> Optional[RemediationPlan]:
        """Generate a remediation plan for an incident
        
        Args:
            incident: The incident to generate a plan for
            
        Returns:
            A remediation plan or None if generation fails
        """
        # Determine appropriate plan types based on incident
        plan_types = self._determine_plan_types(incident)
        
        # Generate plans for each type
        plans = []
        for plan_type in plan_types:
            plan = await self._generate_plan_by_type(incident, plan_type)
            if plan:
                # Add a unique ID to the plan
                plan.plan_id = str(uuid.uuid4())
                plans.append(plan)
        
        # Rank plans and return the best one
        if not plans:
            return None
            
        ranked_plans = await self._rank_remediation_plans(plans, incident)
        return ranked_plans[0] if ranked_plans else None
    
    def _determine_plan_types(self, incident: Incident) -> List[str]:
        """Determine appropriate remediation plan types based on incident
        
        Args:
            incident: The incident to analyze
            
        Returns:
            List of plan types to generate
        """
        incident_type = incident.incident_type.value
        
        if incident_type == "DATA_DRIFT":
            return ["retraining", "feature_engineering"]
        elif incident_type == "PERFORMANCE_DEGRADATION":
            return ["hyperparameter_optimization", "architecture_modification"]
        elif incident_type == "MODEL_ERROR":
            return ["data_pipeline_fix", "retraining"]
        else:
            # Default to all plan types
            return list(self.remediation_templates.keys())
    
    async def _generate_plan_by_type(self, incident: Incident, plan_type: str) -> Optional[RemediationPlan]:
        """Generate a specific type of remediation plan"""
        
        template = self.remediation_templates.get(plan_type)
        if not template:
            return None
        
        # Build plan generation prompt
        prompt = self._build_plan_prompt(incident, plan_type, template)
        
        # Generate plan using Gemini
        plan_response = await self.gemini_client.generate_text(prompt)
        
        # Parse response into RemediationPlan
        plan = self._parse_plan_response(plan_response, plan_type)
        
        return plan
    
    def _build_plan_prompt(self, incident: Incident, plan_type: str, template: Dict[str, Any]) -> str:
        """Build plan generation prompt for Gemini"""
        
        prompt = f"""
        Generate a {plan_type} remediation plan for the following ML incident:
        
        INCIDENT DETAILS:
        - Type: {incident.incident_type.value}
        - Severity: {incident.severity.value}
        - Model: {incident.source_model_name}
        - Description: {incident.description}
        
        DIAGNOSIS (if available):
        {incident.diagnosis.root_cause if incident.diagnosis else 'No diagnosis available'}
        
        TEMPLATE GUIDANCE:
        {template.get('description', '')}
        
        Please provide a detailed remediation plan with:
        1. Clear description of the approach
        2. Step-by-step implementation plan
        3. Estimated duration for each step
        4. Required resources (compute, storage, etc.)
        5. Risk assessment
        6. Rollback plan if applicable
        
        Format as JSON:
        {{
            "description": "Brief description of the remediation approach",
            "steps": [
                {{"action": "step description", "duration_minutes": 30, "resources": {{}}}},
                ...
            ],
            "estimated_duration_minutes": 120,
            "required_resources": {{"compute_units": 5, "storage_gb": 50}},
            "risk_level": "LOW|MEDIUM|HIGH",
            "rollback_plan": {{"description": "rollback approach", "steps": []}}
        }}
        """
        
        return prompt
    
    def _parse_plan_response(self, response: str, plan_type: str) -> RemediationPlan:
        """Parse Gemini response into RemediationPlan"""
        try:
            plan_data = json.loads(response)
            
            return RemediationPlan(
                description=plan_data.get("description", f"{plan_type} remediation plan"),
                steps=plan_data.get("steps", []),
                estimated_duration_minutes=plan_data.get("estimated_duration_minutes", 60),
                required_resources=plan_data.get("required_resources", {}),
                risk_level=plan_data.get("risk_level", "MEDIUM"),
                rollback_plan=plan_data.get("rollback_plan")
            )
            
        except json.JSONDecodeError:
            # Fallback plan if parsing fails
            return RemediationPlan(
                description=f"Generated {plan_type} plan (parsing failed)",
                steps=[
                    {"action": "Review incident details", "duration_minutes": 15},
                    {"action": "Implement remediation", "duration_minutes": 60},
                    {"action": "Validate solution", "duration_minutes": 30}
                ],
                estimated_duration_minutes=105,
                required_resources={"compute_units": 2, "storage_gb": 10},
                risk_level="MEDIUM"
            )
    
    async def _rank_remediation_plans(self, plans: List[RemediationPlan], incident: Incident) -> List[RemediationPlan]:
        """Rank remediation plans by effectiveness and feasibility"""
        
        # Score each plan
        scored_plans = []
        for plan in plans:
            score = await self._score_remediation_plan(plan, incident)
            scored_plans.append((score, plan))
        
        # Sort by score (highest first)
        scored_plans.sort(key=lambda x: x[0], reverse=True)
        
        return [plan for score, plan in scored_plans]
    
    async def _score_remediation_plan(self, plan: RemediationPlan, incident: Incident) -> float:
        """Score a remediation plan based on multiple factors"""
        score = 0.0
        
        # Effectiveness score (based on plan type and incident type)
        effectiveness = self._calculate_effectiveness_score(plan, incident)
        score += effectiveness * 0.4
        
        # Feasibility score (based on complexity and resources)
        feasibility = self._calculate_feasibility_score(plan)
        score += feasibility * 0.3
        
        # Risk score (lower risk = higher score)
        risk_score = self._calculate_risk_score(plan)
        score += (1.0 - risk_score) * 0.2
        
        # Time score (faster = higher score)
        time_score = self._calculate_time_score(plan)
        score += time_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_effectiveness_score(self, plan: RemediationPlan, incident: Incident) -> float:
        """Calculate effectiveness score based on plan-incident match"""
        # Simple heuristic based on plan description keywords
        description = plan.description.lower()
        incident_type = incident.incident_type.value.lower()
        
        if "retrain" in description and "drift" in incident_type:
            return 0.9
        elif "hyperparameter" in description and "performance" in incident_type:
            return 0.8
        elif "feature" in description and "data" in incident_type:
            return 0.7
        else:
            return 0.6
    
    def _calculate_feasibility_score(self, plan: RemediationPlan) -> float:
        """Calculate feasibility score based on complexity"""
        # Score based on number of steps and resource requirements
        step_score = max(0.2, 1.0 - (len(plan.steps) - 3) * 0.1)
        
        compute_units = plan.required_resources.get("compute_units", 1)
        resource_score = max(0.2, 1.0 - (compute_units - 5) * 0.1)
        
        return (step_score + resource_score) / 2
    
    def _calculate_risk_score(self, plan: RemediationPlan) -> float:
        """Calculate risk score (0.0 = low risk, 1.0 = high risk)"""
        risk_mapping = {
            "LOW": 0.2,
            "MEDIUM": 0.5,
            "HIGH": 0.8
        }
        return risk_mapping.get(plan.risk_level, 0.5)
    
    def _calculate_time_score(self, plan: RemediationPlan) -> float:
        """Calculate time score (faster = higher score)"""
        # Normalize duration to 0-1 scale (assuming max 480 minutes = 8 hours)
        max_duration = 480
        normalized_duration = min(plan.estimated_duration_minutes / max_duration, 1.0)
        return 1.0 - normalized_duration
    
    async def _generate_implementation_timeline(self, plan: Optional[RemediationPlan]) -> Dict[str, Any]:
        """Generate implementation timeline for a remediation plan"""
        if not plan:
            return {"error": "No plan provided"}
        
        timeline = {
            "plan_id": plan.plan_id,
            "total_duration_minutes": plan.estimated_duration_minutes,
            "phases": [],
            "milestones": [],
            "dependencies": []
        }
        
        current_time = 0
        for i, step in enumerate(plan.steps):
            duration = step.get("duration_minutes", 30)
            
            phase = {
                "phase_number": i + 1,
                "description": step.get("action", f"Step {i + 1}"),
                "start_minute": current_time,
                "end_minute": current_time + duration,
                "duration_minutes": duration,
                "resources_required": step.get("resources", {})
            }
            
            timeline["phases"].append(phase)
            current_time += duration
            
            # Add milestone for significant phases
            if duration > 60:  # Phases longer than 1 hour
                timeline["milestones"].append({
                    "milestone": f"Complete {step.get('action', f'Step {i + 1}')}",
                    "target_minute": current_time
                })
        
        return timeline
    
    async def _assess_remediation_risks(self, plan: Optional[RemediationPlan]) -> Dict[str, Any]:
        """Assess risks associated with a remediation plan"""
        if not plan:
            return {"error": "No plan provided"}
        
        risks = {
            "overall_risk_level": plan.risk_level,
            "identified_risks": [],
            "mitigation_strategies": [],
            "rollback_available": plan.rollback_plan is not None
        }
        
        # Identify risks based on plan characteristics
        if plan.estimated_duration_minutes > 240:  # > 4 hours
            risks["identified_risks"].append("Long execution time increases failure probability")
            risks["mitigation_strategies"].append("Break into smaller phases with checkpoints")
        
        if len(plan.steps) > 5:
            risks["identified_risks"].append("Complex plan with many steps")
            risks["mitigation_strategies"].append("Implement comprehensive monitoring at each step")
        
        compute_units = plan.required_resources.get("compute_units", 0)
        if compute_units > 10:
            risks["identified_risks"].append("High resource requirements")
            risks["mitigation_strategies"].append("Ensure resource availability before execution")
        
        if not plan.rollback_plan:
            risks["identified_risks"].append("No rollback plan available")
            risks["mitigation_strategies"].append("Create manual rollback procedures")
        
        return risks
    
    async def _estimate_resource_requirements(self, plan: Optional[RemediationPlan]) -> Dict[str, Any]:
        """Estimate detailed resource requirements for a remediation plan"""
        if not plan:
            return {"error": "No plan provided"}
        
        base_resources = plan.required_resources
        
        # Estimate additional resources
        estimated_resources = {
            "compute": {
                "cpu_cores": base_resources.get("compute_units", 1) * 2,
                "memory_gb": base_resources.get("compute_units", 1) * 4,
                "gpu_units": base_resources.get("gpu_units", 0)
            },
            "storage": {
                "temporary_gb": base_resources.get("storage_gb", 10),
                "backup_gb": base_resources.get("storage_gb", 10) * 0.5,
                "model_artifacts_gb": 5
            },
            "network": {
                "bandwidth_mbps": 100,
                "data_transfer_gb": base_resources.get("storage_gb", 10) * 0.2
            },
            "estimated_cost_usd": self._estimate_execution_cost(plan),
            "execution_environment": "vertex_ai_training"
        }
        
        return estimated_resources
    
    def _estimate_execution_cost(self, plan: RemediationPlan) -> float:
        """Estimate execution cost in USD"""
        # Simple cost estimation based on duration and resources
        base_cost_per_hour = 5.0
        hours = plan.estimated_duration_minutes / 60
        compute_multiplier = plan.required_resources.get("compute_units", 1) * 0.5
        
        return base_cost_per_hour * hours * (1 + compute_multiplier)
    
    def _load_remediation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load remediation plan templates"""
        return {
            "retraining": {
                "description": "Retrain the model with fresh data to address drift or performance issues",
                "typical_duration": 120,
                "risk_level": "LOW"
            },
            "hyperparameter_optimization": {
                "description": "Optimize model hyperparameters to improve performance",
                "typical_duration": 180,
                "risk_level": "MEDIUM"
            },
            "feature_engineering": {
                "description": "Modify or add features to improve model performance",
                "typical_duration": 240,
                "risk_level": "MEDIUM"
            },
            "architecture_modification": {
                "description": "Modify model architecture for better performance",
                "typical_duration": 360,
                "risk_level": "HIGH"
            },
            "data_pipeline_fix": {
                "description": "Fix issues in the data preprocessing pipeline",
                "typical_duration": 90,
                "risk_level": "LOW"
            }
        }
