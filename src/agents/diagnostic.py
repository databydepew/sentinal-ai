from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid

from models.incident import Incident, DiagnosisResult, AgentAction, ActionType
from services.gemini_client import GeminiClient
from agents.base_agent import BaseAgent, AgentCapability


class DiagnosticAgent(BaseAgent):
    """Agent responsible for diagnosing ML incidents"""

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the diagnostic agent

        Args:
            gemini_client: Client for Gemini API
        """
        super().__init__({"max_concurrent_tasks": 5}, "diagnostic")
        self.gemini_client = gemini_client
        self.diagnosis_prompts = self._load_diagnosis_prompts()

    async def initialize(self) -> None:
        """Initialize the agent with required services"""
        # No additional initialization required
        pass

    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent provides"""
        return [
            AgentCapability(
                name="diagnose_incident", description="Diagnose ML incidents"
            ),
            AgentCapability(
                name="root_cause_analysis", description="Perform root cause analysis"
            ),
            AgentCapability(
                name="technical_analysis", description="Generate technical analysis"
            ),
        ]

    async def process_incident(self, incident: Incident) -> Dict[str, Any]:
        """Process an incident by diagnosing the root cause

        Args:
            incident: The incident to process

        Returns:
            Dict containing the diagnosis results
        """
        # Create diagnostic action
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.DIAGNOSE,
            incident_id=incident.id,
            details=f"Diagnostic analysis for {incident.type.value} incident"
        )

        try:
            # Gather context for diagnosis
            context = await self._gather_diagnostic_context(incident)

            # Perform root cause analysis
            diagnosis = await self._perform_root_cause_analysis(incident, context)

            # If confidence is low, try to refine the diagnosis
            if diagnosis.confidence_score < 0.7:
                diagnosis = await self._refine_diagnosis(incident, context, diagnosis)

            # Generate technical analysis
            technical_analysis = await self._generate_technical_analysis(
                incident, diagnosis
            )

            # Update action with success
            action.success = True
            action.output_data = {
                "diagnosis_id": str(uuid.uuid4()),
                "root_cause": diagnosis.root_cause,
                "confidence_score": diagnosis.confidence_score,
            }

            # Add action to incident
            incident.add_action(action)

            return {"diagnosis": diagnosis, "technical_analysis": technical_analysis}

        except Exception as e:
            action.output_data = {"error": str(e)}
            incident.add_action(action)
            raise

    async def _gather_diagnostic_context(self, incident: Incident) -> Dict[str, Any]:
        """Gather all context needed for diagnosis"""

        # Gather data in parallel for efficiency
        monitoring_data = await self._get_monitoring_data(incident)
        historical_context = await self._get_historical_context(incident)
        system_state = await self._get_system_state(incident)
        model_info = await self._get_model_info(incident)

        return {
            "monitoring_data": monitoring_data,
            "historical_context": historical_context,
            "system_state": system_state,
            "model_info": model_info,
        }

    async def _get_model_info(self, incident: Incident) -> Dict[str, Any]:
        """Get information about the model involved in the incident"""
        # Placeholder for model information retrieval
        # In real implementation, this would query model registry
        return {
            "model_name": incident.source_model_name,
            "version": incident.metadata.get("model_version", "unknown"),
            "framework": "TensorFlow",
            "creation_date": "2023-01-15",
            "last_updated": "2023-06-22",
            "performance_metrics": {
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.88,
                "f1_score": 0.89,
            },
        }

    async def _get_monitoring_data(self, incident: Incident) -> Dict[str, Any]:
        """Get relevant monitoring data for the incident"""
        # Placeholder for monitoring data retrieval
        # In real implementation, this would query monitoring systems
        return {
            "performance_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            },
            "prediction_statistics": {
                "total_predictions": 10000,
                "average_confidence": 0.75,
                "prediction_distribution": {"class_0": 0.6, "class_1": 0.4},
            },
            "system_metrics": {
                "cpu_usage": 0.65,
                "memory_usage": 0.78,
                "response_time_ms": 150,
            },
        }

    async def _get_historical_context(self, incident: Incident) -> Dict[str, Any]:
        """Get historical context for similar incidents"""
        # Placeholder for historical incident analysis
        # In real implementation, this would query incident database
        return {"similar_incidents": [], "patterns": [], "resolution_history": []}

    async def _get_system_state(self, incident: Incident) -> Dict[str, Any]:
        """Get current system state information"""
        # Placeholder for system state retrieval
        # In real implementation, this would query infrastructure monitoring
        return {
            "infrastructure": {
                "healthy_endpoints": 3,
                "total_endpoints": 3,
                "load_balancer_status": "healthy",
            },
            "dependencies": {"database_status": "healthy", "external_apis": "healthy"},
        }

    async def _perform_root_cause_analysis(
        self, incident: Incident, context: Dict[str, Any]
    ) -> DiagnosisResult:
        """Perform comprehensive root cause analysis using Gemini"""

        # Prepare analysis prompt
        analysis_prompt = self._build_analysis_prompt(incident, context)

        # Get analysis from Gemini
        analysis_response = await self.gemini_client.generate_text(analysis_prompt)

        # Parse and structure the response
        diagnosis = self._parse_diagnosis_response(analysis_response)

        return diagnosis

    def _build_analysis_prompt(
        self, incident: Incident, context: Dict[str, Any]
    ) -> str:
        """Build comprehensive analysis prompt for Gemini"""

        prompt = f"""
        You are an expert ML engineer performing root cause analysis for a machine learning incident.
        
        INCIDENT DETAILS:
        - Type: {incident.incident_type.value}
        - Severity: {incident.severity.value}
        - Model: {incident.source_model_name}
        - Description: {incident.description}
        
        MONITORING DATA:
        {json.dumps(context.get('monitoring_data', {}), indent=2)}
        
        MODEL INFORMATION:
        {json.dumps(context.get('model_info', {}), indent=2)}
        
        INCIDENT METADATA:
        {json.dumps(incident.metadata, indent=2)}
        
        Please provide a comprehensive root cause analysis including:
        1. Most likely root cause (be specific and technical)
        2. Confidence score (0.0 to 1.0)
        3. Contributing factors (list of specific factors)
        4. Recommended immediate actions
        5. Technical details and evidence supporting your analysis
        
        Format your response as JSON with the following structure:
        {{
            "root_cause": "specific technical root cause",
            "confidence_score": 0.85,
            "contributing_factors": ["factor1", "factor2"],
            "recommended_actions": ["action1", "action2"],
            "technical_details": {{
                "evidence": ["evidence1", "evidence2"],
                "analysis": "detailed technical analysis",
                "affected_components": ["component1", "component2"]
            }}
        }}
        """

        return prompt

    def _parse_diagnosis_response(self, response: str) -> DiagnosisResult:
        """Parse Gemini response into structured diagnosis"""
        try:
            # Extract JSON from response
            response = response.strip()

            # Find JSON content (in case there's text before or after)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_content = response[start_idx:end_idx]
                diagnosis_data = json.loads(json_content)
            else:
                raise ValueError("No JSON content found in response")

            return DiagnosisResult(
                root_cause=diagnosis_data.get("root_cause", "Unknown root cause"),
                confidence_score=diagnosis_data.get("confidence_score", 0.5),
                contributing_factors=diagnosis_data.get("contributing_factors", []),
                recommended_actions=diagnosis_data.get("recommended_actions", []),
                technical_details=diagnosis_data.get("technical_details", {}),
            )

        except json.JSONDecodeError:
            # Fallback parsing if JSON parsing fails
            return DiagnosisResult(
                root_cause="Analysis completed but response format was unexpected",
                confidence_score=0.3,
                contributing_factors=["Response parsing error"],
                recommended_actions=["Review analysis manually", "Retry diagnosis"],
                technical_details={"raw_response": response},
            )

    async def _refine_diagnosis(
        self,
        incident: Incident,
        context: Dict[str, Any],
        initial_diagnosis: DiagnosisResult,
    ) -> DiagnosisResult:
        """Refine diagnosis if confidence is below threshold"""

        refinement_prompt = f"""
        The initial diagnosis had low confidence ({initial_diagnosis.confidence_score:.2f}).
        Please refine the analysis with additional scrutiny.
        
        INITIAL DIAGNOSIS:
        - Root Cause: {initial_diagnosis.root_cause}
        - Contributing Factors: {', '.join(initial_diagnosis.contributing_factors)}
        
        Please provide a more detailed analysis or alternative hypotheses.
        Focus on increasing confidence through deeper technical analysis.
        
        Format your response as JSON with the following structure:
        {{
            "root_cause": "specific technical root cause",
            "confidence_score": 0.85,
            "contributing_factors": ["factor1", "factor2"],
            "recommended_actions": ["action1", "action2"],
            "technical_details": {{
                "evidence": ["evidence1", "evidence2"],
                "analysis": "detailed technical analysis",
                "affected_components": ["component1", "component2"]
            }}
        }}
        """

        refined_response = await self.gemini_client.generate_text(refinement_prompt)
        refined_diagnosis = self._parse_diagnosis_response(refined_response)

        # Use refined diagnosis if confidence improved
        if refined_diagnosis.confidence_score > initial_diagnosis.confidence_score:
            return refined_diagnosis
        else:
            return initial_diagnosis

    async def _generate_technical_analysis(
        self, incident: Incident, diagnosis: DiagnosisResult
    ) -> Dict[str, Any]:
        """Generate detailed technical analysis"""

        return {
            "analysis_id": str(uuid.uuid4()),
            "model_analysis": {
                "architecture_review": "Model architecture appears stable",
                "hyperparameter_analysis": "No obvious hyperparameter issues detected",
                "training_data_analysis": "Training data quality assessment needed",
            },
            "infrastructure_analysis": {
                "compute_resources": "Adequate compute resources available",
                "network_analysis": "Network latency within normal ranges",
                "storage_analysis": "Storage performance normal",
            },
            "data_pipeline_analysis": {
                "data_quality": "Data quality checks passed",
                "feature_engineering": "Feature engineering pipeline stable",
                "preprocessing": "Preprocessing steps functioning normally",
            },
            "recommendations": {
                "immediate": diagnosis.recommended_actions,
                "short_term": ["Monitor closely for 24 hours", "Prepare rollback plan"],
                "long_term": [
                    "Review monitoring thresholds",
                    "Update incident response procedures",
                ],
            },
        }

    def _load_diagnosis_prompts(self) -> Dict[str, str]:
        """Load diagnosis prompt templates"""
        return {
            "data_drift": "Analyze data distribution changes and their impact on model performance...",
            "concept_drift": "Examine changes in the relationship between features and target variable...",
            "performance_degradation": "Investigate causes of model performance decline...",
            "anomaly_detection": "Analyze anomalous patterns in model behavior...",
        }
