from typing import Dict, List, Optional, Any
import uuid
import asyncio

from datetime import datetime

from models.incident import Incident, AgentAction
from services.pipeline_client import PipelineClient
from services.model_registry_client import ModelRegistryClient
from agents.base_agent import BaseAgent, AgentCapability


class VerificationAgent(BaseAgent):
    """Agent responsible for verifying and executing remediation plans"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the verification agent

        Args:
            config: Agent configuration
        """
        super().__init__(config, "verification")
        self.pipeline_client = None
        self.model_registry_client = None
        self.config = config
        self.project_id = config.get("project_id", "")
        self.location = config.get("location", "us-central1")
        self.agent_id = str(uuid.uuid4())
        self.running = False

    async def initialize(self) -> None:
        """Initialize the agent with required services"""
        # Initialize pipeline client for executing remediation steps
        self.pipeline_client = PipelineClient(
            project_id=self.config.get("project_id"),
            location=self.config.get("location"),
        )

        # Initialize model registry client for model deployment
        self.model_registry_client = ModelRegistryClient(
            project_id=self.config.get("project_id"),
            location=self.config.get("location"),
        )

    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent provides"""
        return [
            AgentCapability(
                name="execute_remediation", description="Execute remediation plans"
            ),
            AgentCapability(
                name="validate_execution", description="Validate execution results"
            ),
            AgentCapability(
                name="deploy_models", description="Deploy models to production"
            ),
        ]

    async def process_incident(self, incident: Incident) -> Dict[str, Any]:
        """Process an incident by executing its remediation plan

        Args:
            incident: The incident to process

        Returns:
            Dict containing the execution results
        """
        if not incident.remediation_plan:
            return {"error": "No remediation plan available"}

        # Create execution context
        execution_context = {
            "execution_id": str(uuid.uuid4()),
            "incident_id": incident.incident_id,
            "model_name": incident.source_model_name,
            "remediation_plan": incident.remediation_plan,
            "created_at": datetime.now(),
            "steps_completed": [],
        }

        # Execute remediation steps
        execution_result = await self._execute_remediation_plan(execution_context)

        # Validate execution results
        validation_result = await self._validate_execution_results(
            execution_context, execution_result
        )

        # Deploy model if validation passed and plan requires it
        deployment_result = None
        if validation_result.get("validation_passed", False) and execution_result.get(
            "requires_deployment", False
        ):
            deployment_result = await self._deploy_model(
                execution_context, execution_result
            )

        # Generate execution report
        report = await self._generate_execution_report(
            execution_context, execution_result, validation_result, deployment_result
        )

        # Update incident with execution results
        self._update_incident_with_results(
            incident, execution_result, validation_result, deployment_result
        )

        return {
            "execution_id": execution_context["execution_id"],
            "status": "completed"
            if execution_result.get("success", False)
            else "failed",
            "validation_passed": validation_result.get("validation_passed", False),
            "deployed": deployment_result is not None,
            "report": report,
        }

    async def _execute_remediation_plan(
        self, execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute all steps in the remediation plan

        Args:
            execution_context: Context for execution

        Returns:
            Dict containing execution results
        """
        plan = execution_context["remediation_plan"]
        steps = plan.steps

        execution_result = {
            "success": False,
            "steps_completed": [],
            "step_results": [],
            "errors": [],
            "requires_deployment": False,
            "metrics": {},
            "artifacts_created": [],
        }

        for i, step in enumerate(steps):
            step_result = await self._execute_step(execution_context, step)

            execution_result["step_results"].append(step_result)

            if step_result.get("success", False):
                execution_result["steps_completed"].append(i)
                execution_context["steps_completed"].append(i)
            else:
                execution_result["errors"].append(
                    {"step": i, "error": step_result.get("error", "Unknown error")}
                )
                break

            # Check if step created artifacts that need deployment
            if step_result.get("created_model_artifact", False):
                execution_result["requires_deployment"] = True

            # Collect artifacts
            if "artifacts" in step_result:
                execution_result["artifacts_created"].extend(step_result["artifacts"])

            # Collect metrics
            if "metrics" in step_result:
                execution_result["metrics"].update(step_result["metrics"])

        # Mark as successful if all steps completed
        if len(execution_result["steps_completed"]) == len(steps):
            execution_result["success"] = True

        return execution_result

    async def _execute_step(
        self, execution_context: Dict[str, Any], step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single remediation step

        Args:
            execution_context: Context for execution
            step: Step to execute

        Returns:
            Dict containing step execution result
        """
        step_type = step.get("action", "").lower()

        try:
            # Execute step based on type
            if "retrain" in step_type:
                return await self._execute_retraining_step(execution_context, step)
            elif "hyperparameter" in step_type:
                return await self._execute_hyperparameter_step(execution_context, step)
            elif "feature" in step_type:
                return await self._execute_feature_engineering_step(
                    execution_context, step
                )
            elif "validate" in step_type or "test" in step_type:
                return await self._execute_validation_step(execution_context, step)
            else:
                return await self._execute_generic_step(execution_context, step)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_retraining_step(
        self, execution_context: Dict[str, Any], step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model retraining step"""

        model_name = execution_context["model_name"]

        # Create training job
        training_config = {
            "model_name": model_name,
            "training_data": step.get("training_data", "latest"),
            "hyperparameters": step.get("hyperparameters", {}),
        }

        training_job_id = await self.pipeline_client.submit_training_job(
            training_config
        )

        # Wait for completion (with timeout)
        training_result = await self.pipeline_client.wait_for_completion(
            training_job_id, timeout_minutes=120
        )

        return {
            "training_job_id": training_job_id,
            "model_artifacts": training_result.get("model_artifacts", {}),
            "training_metrics": training_result.get("metrics", {}),
            "training_duration_minutes": training_result.get("duration_minutes", 0),
            "created_model_artifact": True,
            "success": True,
        }

    async def _execute_hyperparameter_step(
        self, execution_context: Dict[str, Any], step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute hyperparameter optimization step"""

        model_name = execution_context["model_name"]

        # Create hyperparameter tuning job
        tuning_config = {
            "model_name": model_name,
            "parameter_space": step.get("parameter_space", {}),
            "optimization_goal": "maximize_accuracy",
        }

        tuning_job_id = await self.pipeline_client.submit_tuning_job(tuning_config)
        tuning_result = await self.pipeline_client.wait_for_completion(tuning_job_id)

        return {
            "tuning_job_id": tuning_job_id,
            "best_parameters": tuning_result.get("best_parameters", {}),
            "best_score": tuning_result.get("best_score", 0.0),
            "success": True,
        }

    async def _execute_feature_engineering_step(
        self, execution_context: Dict[str, Any], step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute feature engineering step"""

        # Create feature engineering pipeline
        feature_config = {
            "transformations": step.get("transformations", []),
            "new_features": step.get("new_features", []),
        }

        # Execute feature engineering
        feature_job_id = await self.pipeline_client.submit_feature_job(feature_config)
        feature_result = await self.pipeline_client.wait_for_completion(feature_job_id)

        return {
            "feature_job_id": feature_job_id,
            "feature_schema": feature_result.get("feature_schema", {}),
            "feature_statistics": feature_result.get("statistics", {}),
            "success": True,
        }

    async def _execute_validation_step(
        self, execution_context: Dict[str, Any], step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model validation step"""

        model_name = execution_context["model_name"]

        # Run validation pipeline
        validation_config = {
            "model_name": model_name,
            "validation_dataset": step.get("validation_dataset", "default"),
            "metrics": ["accuracy", "precision", "recall", "f1"],
        }

        validation_result = await self.pipeline_client.run_validation(validation_config)

        return {
            "validation_metrics": validation_result.get("metrics", {}),
            "validation_passed": validation_result.get("passed", False),
            "validation_report": validation_result.get("report", {}),
            "success": True,
        }

    async def _execute_generic_step(
        self, execution_context: Dict[str, Any], step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a generic step (fallback)"""

        # Simulate step execution
        await asyncio.sleep(5)  # Simulate work

        return {
            "step_type": "generic",
            "simulated": True,
            "message": f"Executed generic step: {step.get('action', 'unknown')}",
            "success": True,
        }

    async def _validate_execution_results(
        self, execution_context: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the results of remediation execution"""

        validation_result = {
            "validation_passed": False,
            "validation_score": 0.0,
            "validation_details": {},
            "issues_found": [],
        }

        # Check if execution was successful
        if not execution_result.get("success", False):
            validation_result["issues_found"].append(
                "Execution failed to complete all steps"
            )
            return validation_result

        # Check metrics against thresholds
        metrics = execution_result.get("metrics", {})
        if metrics:
            validation_score = self._calculate_validation_score(metrics)
            validation_result["validation_score"] = validation_score

            # Check if score meets threshold
            if validation_score >= 0.7:  # 70% threshold
                validation_result["validation_passed"] = True
            else:
                validation_result["issues_found"].append(
                    f"Validation score {validation_score:.2f} below threshold (0.7)"
                )

        # Check for specific issues in step results
        for i, step_result in enumerate(execution_result.get("step_results", [])):
            if step_result.get("validation_metrics"):
                for metric, value in step_result["validation_metrics"].items():
                    # Example: Check if accuracy is below threshold
                    if metric == "accuracy" and value < 0.8:
                        validation_result["issues_found"].append(
                            f"Step {i} accuracy ({value:.2f}) below threshold (0.8)"
                        )

        # If no issues found but no explicit pass, set to pass
        if (
            not validation_result["issues_found"]
            and not validation_result["validation_passed"]
        ):
            validation_result["validation_passed"] = True

        return validation_result

    def _calculate_validation_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall validation score from metrics"""
        # Default weights for common metrics
        weights = {
            "accuracy": 0.3,
            "precision": 0.2,
            "recall": 0.2,
            "f1": 0.3,
            "auc": 0.4,
            "mae": -0.2,  # Negative because lower is better
            "mse": -0.2,  # Negative because lower is better
            "rmse": -0.2,  # Negative because lower is better
        }

        score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric.lower() in weights:
                weight = weights[metric.lower()]

                # For metrics where lower is better (negative weight)
                if weight < 0:
                    # Convert to a 0-1 scale where 1 is best
                    # Assuming values typically range from 0 to 1
                    normalized_value = max(0, 1 - value)
                    score += abs(weight) * normalized_value
                else:
                    score += weight * value

                total_weight += abs(weight)

        # Return normalized score
        return score / total_weight if total_weight > 0 else 0.5

    async def _deploy_model(
        self, execution_context: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy the model using the configured rollout strategy"""

        # Determine deployment strategy
        strategy = self.config.get("deployment_strategy", "canary")

        deployment_result = {"strategy": strategy, "status": "pending"}

        try:
            # Deploy based on strategy
            if strategy == "canary":
                deployment_details = await self._deploy_canary(
                    execution_context, execution_result
                )
            elif strategy == "blue_green":
                deployment_details = await self._deploy_blue_green(
                    execution_context, execution_result
                )
            else:
                deployment_details = await self._deploy_direct(
                    execution_context, execution_result
                )

            deployment_result.update(deployment_details)
            deployment_result["status"] = "deployed"

        except Exception as e:
            deployment_result["status"] = "failed"
            deployment_result["error"] = str(e)

        return deployment_result

    async def _deploy_canary(
        self, execution_context: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy using canary strategy"""

        model_name = execution_context["model_name"]

        # Create canary deployment with 10% traffic
        canary_endpoint = await self.model_registry_client.create_endpoint(
            model_name=model_name, traffic_percentage=10
        )

        # Schedule gradual rollout
        rollout_schedule = [
            {"time_minutes": 30, "traffic_percentage": 25},
            {"time_minutes": 60, "traffic_percentage": 50},
            {"time_minutes": 120, "traffic_percentage": 100},
        ]

        await self.model_registry_client.schedule_rollout(
            endpoint_id=canary_endpoint["endpoint_id"],
            rollout_schedule=rollout_schedule,
        )

        return {
            "endpoint": canary_endpoint,
            "rollout_schedule": rollout_schedule,
            "canary_deployment": True,
        }

    async def _deploy_blue_green(
        self, execution_context: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy using blue-green strategy"""

        model_name = execution_context["model_name"]

        # Create new "green" environment
        green_endpoint = await self.model_registry_client.create_endpoint(
            model_name=model_name,
            traffic_percentage=0,  # Start with 0% traffic
        )

        # Schedule instant cutover after validation period
        cutover_schedule = {
            "validation_minutes": 60,  # Validate for 60 minutes before cutover
            "cutover_minutes": 5,  # Complete cutover within 5 minutes
        }

        await self.model_registry_client.schedule_cutover(
            endpoint_id=green_endpoint["endpoint_id"], cutover_schedule=cutover_schedule
        )

        return {
            "endpoint": green_endpoint,
            "cutover_schedule": cutover_schedule,
            "blue_green_deployment": True,
        }

    async def _deploy_direct(
        self, execution_context: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy directly to production"""

        model_name = execution_context["model_name"]

        # Deploy directly
        endpoint = await self.model_registry_client.create_endpoint(
            model_name=model_name, traffic_percentage=100
        )

        return {"endpoint": endpoint, "direct_deployment": True}

    async def _generate_execution_report(
        self,
        execution_context: Dict[str, Any],
        execution_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        deployment_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate comprehensive execution report"""

        return {
            "report_id": str(uuid.uuid4()),
            "execution_summary": {
                "execution_id": execution_context["execution_id"],
                "model_name": execution_context["model_name"],
                "total_steps": len(execution_context["remediation_plan"].steps),
                "completed_steps": len(execution_result.get("steps_completed", [])),
                "execution_status": execution_result.get("status", "unknown"),
                "execution_duration": self._calculate_execution_duration(
                    execution_context
                ),
            },
            "validation_summary": {
                "validation_passed": validation_result.get("validation_passed", False),
                "validation_score": validation_result.get("validation_score", 0.0),
                "issues_count": len(validation_result.get("issues_found", [])),
            },
            "deployment_summary": {
                "deployed": deployment_result is not None,
                "deployment_strategy": deployment_result.get("strategy")
                if deployment_result
                else None,
                "deployment_status": deployment_result.get("status")
                if deployment_result
                else None,
            },
            "artifacts": {
                "created_artifacts": execution_result.get("artifacts_created", []),
                "model_artifacts": execution_result.get("model_artifacts", {}),
                "metrics": execution_result.get("metrics", {}),
            },
            "recommendations": self._generate_post_execution_recommendations(
                execution_result, validation_result, deployment_result
            ),
        }

    def _calculate_execution_duration(self, execution_context: Dict[str, Any]) -> float:
        """Calculate execution duration in minutes"""

        created_at = execution_context.get("created_at")
        if not created_at:
            return 0.0

        duration = datetime.now() - created_at
        return duration.total_seconds() / 60

    def _generate_post_execution_recommendations(
        self,
        execution_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        deployment_result: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on execution results"""

        recommendations = []

        # Execution recommendations
        if execution_result.get("status") == "completed":
            recommendations.append(
                "Monitor model performance closely for the next 24 hours"
            )
        else:
            recommendations.append(
                "Investigate execution failures and consider rollback"
            )

        # Validation recommendations
        if not validation_result.get("validation_passed", False):
            recommendations.append("Review validation failures before proceeding")
            recommendations.append("Consider additional testing or model improvements")

        # Deployment recommendations
        if deployment_result and deployment_result.get("strategy") == "canary":
            recommendations.append("Monitor canary metrics before full rollout")
            recommendations.append("Prepare for quick rollback if issues arise")

        return recommendations

    def _update_incident_with_results(
        self,
        incident: Incident,
        execution_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        deployment_result: Optional[Dict[str, Any]],
    ) -> None:
        """Update incident with execution results"""

        # Add execution action to incident
        action = AgentAction(
            agent_id="VerificationAgent",
            action_type="execute_remediation",
            timestamp=datetime.now(),
            input_data={
                "remediation_plan_id": incident.remediation_plan.plan_id
                if incident.remediation_plan
                else None
            },
            output_data={
                "execution_success": execution_result.get("success", False),
                "validation_passed": validation_result.get("validation_passed", False),
                "deployed": deployment_result is not None,
            },
            success=execution_result.get("success", False),
        )

        incident.add_action(action)

        # Update incident metadata with execution results
        if not incident.metadata:
            incident.metadata = {}

        incident.metadata["remediation_execution"] = {
            "execution_id": execution_result.get("execution_id", str(uuid.uuid4())),
            "execution_success": execution_result.get("success", False),
            "validation_passed": validation_result.get("validation_passed", False),
            "deployed": deployment_result is not None,
            "execution_timestamp": datetime.now().isoformat(),
        }

        # If execution was successful and validation passed, update incident status
        if execution_result.get("success", False) and validation_result.get(
            "validation_passed", False
        ):
            incident.update_status("RESOLVED")
            incident.resolved_at = datetime.now()
            incident.resolution_summary = "Remediation plan executed successfully"
