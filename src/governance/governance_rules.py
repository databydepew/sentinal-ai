import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from models.governance import (
    GovernanceRule,
    RuleEvaluation,
    RuleStatus,
    RuleType,
    Severity,
    RulesEngineConfig
)

logger = logging.getLogger(__name__)


class GovernanceRulesEngine:
    """Engine for evaluating and managing governance rules"""
    
    def __init__(self, config: Optional[RulesEngineConfig] = None):
        self.config = config or RulesEngineConfig()
        self.rules: Dict[str, GovernanceRule] = {}
        self.evaluation_history: List[RuleEvaluation] = []
        self.running = False
        self.max_history_size = self.config.max_history_size
        self.evaluation_interval_minutes = self.config.evaluation_interval_minutes
        self._periodic_task = None
    
    async def initialize(self) -> None:
        """Initialize the rules engine"""
        self.running = True
        self._initialize_default_rules()
        
        # Start periodic evaluation if enabled
        if self.config.enable_periodic_evaluation:
            self._periodic_task = asyncio.create_task(self._periodic_evaluation())
        
        logger.info("Governance rules engine initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the rules engine"""
        self.running = False
        
        if self._periodic_task:
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Governance rules engine shutdown")
    
    async def add_rule(self, rule: GovernanceRule) -> None:
        """Add a governance rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added governance rule: {rule.rule_id}")
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove a governance rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed governance rule: {rule_id}")
            return True
        return False
    
    async def evaluate_rules(self, context: Dict[str, Any]) -> List[RuleEvaluation]:
        """Evaluate all rules against the given context"""
        evaluations = []
        
        # Evaluate each rule
        for rule in self.rules.values():
            try:
                evaluation = await self._evaluate_rule(rule, context)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                evaluations.append(RuleEvaluation(
                    rule_id=rule.rule_id,
                    status=RuleStatus.SKIPPED,
                    message=f"Rule evaluation error: {str(e)}",
                    context=context
                ))
        
        # Store evaluations in history
        self.evaluation_history.extend(evaluations)
        
        # Trim history if needed
        if len(self.evaluation_history) > self.max_history_size:
            self.evaluation_history = self.evaluation_history[-self.max_history_size:]
        
        return evaluations
    
    async def _evaluate_rule(self, rule: GovernanceRule, context: Dict[str, Any]) -> RuleEvaluation:
        """Evaluate a single rule against the context"""
        
        try:
            # Check rule condition
            condition_result = await rule.condition(context)
            
            if condition_result:
                # Rule condition passed
                evaluation = RuleEvaluation(
                    rule_id=rule.rule_id,
                    status=RuleStatus.PASSED,
                    message=f"Rule '{rule.name}' passed",
                    context=context
                )
            else:
                # Rule condition failed - execute action
                action_result = await rule.action(context)
                
                evaluation = RuleEvaluation(
                    rule_id=rule.rule_id,
                    status=RuleStatus.FAILED,
                    message=f"Rule '{rule.name}' failed: {action_result.get('message', 'No details')}",
                    context=context,
                    recommendations=action_result.get('recommendations', [])
                )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return RuleEvaluation(
                rule_id=rule.rule_id,
                status=RuleStatus.SKIPPED,
                message=f"Rule evaluation error: {str(e)}",
                context=context
            )
    
    async def _periodic_evaluation(self) -> None:
        """Background task for periodic rule evaluation"""
        
        while self.running:
            try:
                # Create evaluation context
                context = {
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_type": "periodic"
                }
                
                # Evaluate all rules
                evaluations = await self.evaluate_rules(context)
                
                # Log any failures
                failed_evaluations = [e for e in evaluations if e.status == RuleStatus.FAILED]
                if failed_evaluations:
                    logger.warning(f"Periodic evaluation found {len(failed_evaluations)} rule violations")
                
                # Wait for next evaluation
                await asyncio.sleep(self.evaluation_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in periodic evaluation: {e}")
                await asyncio.sleep(60)
    
    def _initialize_default_rules(self) -> None:
        """Initialize default governance rules"""
        
        # Resource limit rule
        resource_limit_rule = GovernanceRule(
            rule_id="resource_limit_check",
            rule_type=RuleType.RESOURCE_LIMIT,
            name="Resource Limit Check",
            description="Ensure resource usage stays within limits",
            condition=self._check_resource_limits,
            action=self._handle_resource_violation,
            severity="HIGH"
        )
        self.add_rule(resource_limit_rule)
        
        # Cost control rule
        cost_control_rule = GovernanceRule(
            rule_id="cost_control_check",
            rule_type=RuleType.COST_CONTROL,
            name="Cost Control Check",
            description="Monitor and control costs",
            condition=self._check_cost_limits,
            action=self._handle_cost_violation,
            severity="MEDIUM"
        )
        self.add_rule(cost_control_rule)
        
        # Security policy rule
        security_policy_rule = GovernanceRule(
            rule_id="security_policy_check",
            rule_type=RuleType.SECURITY_POLICY,
            name="Security Policy Check",
            description="Enforce security policies",
            condition=self._check_security_policies,
            action=self._handle_security_violation,
            severity="HIGH"
        )
        self.add_rule(security_policy_rule)
    
    async def _check_resource_limits(self, context: Dict[str, Any]) -> bool:
        """Check if resource usage is within limits"""
        
        # Get resource usage from context
        compute_units = context.get("compute_units", 0)
        storage_gb = context.get("storage_gb", 0)
        
        # Check against limits
        max_compute = self.rules_config.get("max_compute_units", 100)
        max_storage = self.rules_config.get("max_storage_gb", 1000)
        
        return compute_units <= max_compute and storage_gb <= max_storage
    
    async def _handle_resource_violation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource limit violation"""
        
        return {
            "message": "Resource usage exceeds limits",
            "recommendations": [
                "Reduce compute resource allocation",
                "Clean up unused storage",
                "Review resource optimization opportunities"
            ]
        }
    
    async def _check_cost_limits(self, context: Dict[str, Any]) -> bool:
        """Check if costs are within limits"""
        
        estimated_cost = context.get("estimated_cost_usd", 0.0)
        max_cost = self.rules_config.get("max_cost_usd", 5000.0)
        
        return estimated_cost <= max_cost
    
    async def _handle_cost_violation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cost limit violation"""
        
        return {
            "message": "Estimated cost exceeds budget limits",
            "recommendations": [
                "Review cost optimization opportunities",
                "Consider alternative approaches",
                "Request budget approval for high-cost operations"
            ]
        }
    
    async def _check_security_policies(self, context: Dict[str, Any]) -> bool:
        """Check security policy compliance"""
        
        # Check for sensitive data handling
        has_sensitive_data = context.get("has_sensitive_data", False)
        encryption_enabled = context.get("encryption_enabled", True)
        
        if has_sensitive_data and not encryption_enabled:
            return False
        
        return True
    
    async def _handle_security_violation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security policy violation"""
        
        return {
            "message": "Security policy violation detected",
            "recommendations": [
                "Enable encryption for sensitive data",
                "Review data handling procedures",
                "Implement additional security controls"
            ]
        }
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all governance rules"""
        return [rule.to_dict() for rule in self.rules.values()]
    
    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific governance rule"""
        if rule_id in self.rules:
            return self.rules[rule_id].to_dict()
        return None
    
    def get_evaluation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get rule evaluation history"""
        return [evaluation.to_dict() for evaluation in self.evaluation_history[-limit:]]
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule evaluation statistics"""
        total_rules = len(self.rules)
        active_rules = len([r for r in self.rules.values() if r.status == RuleStatus.ACTIVE])
        inactive_rules = total_rules - active_rules
        total_evaluations = sum(rule.evaluation_count for rule in self.rules.values())
        failed_evaluations = sum(rule.violation_count for rule in self.rules.values())
        
        # Type-level statistics
        type_stats = {}
        for rule_type in RuleType:
            type_rules = [r for r in self.rules.values() if r.type == rule_type]
            type_evaluations = []
            type_failures = []
            for r in type_rules:
                evals = [e for e in self.evaluation_history if e.rule_id == r.rule_id]
                type_evaluations.extend(evals)
                type_failures.extend([e for e in evals if e.status == RuleStatus.FAILED])
            
            type_stats[rule_type.value] = {
                "total_rules": len(type_rules),
                "active_rules": len([r for r in type_rules if r.status == RuleStatus.ACTIVE]),
                "total_evaluations": len(type_evaluations),
                "failed_evaluations": len(type_failures),
                "failure_rate": len(type_failures) / len(type_evaluations) if type_evaluations else 0.0
            }
        
        return {
            "total_rules": total_rules,
            "active_rules": active_rules,
            "inactive_rules": inactive_rules,
            "total_evaluations": total_evaluations,
            "failed_evaluations": failed_evaluations,
            "failure_rate": failed_evaluations / total_evaluations if total_evaluations > 0 else 0.0,
            "evaluation_history_size": len(self.evaluation_history),
            "by_rule_type": type_stats,
            "last_evaluation": self.evaluation_history[-1].evaluated_at.isoformat() if self.evaluation_history else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on rules engine"""
        
        try:
            health_status = {
                "healthy": True,
                "running": self.running,
                "total_rules": len(self.rules),
                "active_rules": len([r for r in self.rules.values() if r.status == RuleStatus.ACTIVE]),
                "evaluation_task_running": self._evaluation_task and not self._evaluation_task.done(),
                "evaluation_history_size": len(self.evaluation_history),
                "errors": []
            }
            
            # Check if evaluation task is running when it should be
            if self.config.enable_periodic_evaluation and not health_status["evaluation_task_running"] and self.running:
                health_status["healthy"] = False
                health_status["errors"].append("Periodic evaluation task is not running")
            
            # Check for recent evaluation failures
            recent_threshold = datetime.now() - timedelta(hours=1)
            recent_failures = [
                e for e in self.evaluation_history
                if e.evaluated_at > recent_threshold and not e.passed
            ]
            
            if len(recent_failures) > 10:  # More than 10 failures in the last hour
                health_status["healthy"] = False
                health_status["errors"].append(f"{len(recent_failures)} rule failures in the last hour")
            
            # Check for rules with no recent evaluations
            if self.config.enable_periodic_evaluation:
                stale_threshold = datetime.now() - timedelta(hours=2)
                stale_rules = []
                
                for rule in self.rules.values():
                    if rule.status == RuleStatus.ACTIVE:
                        recent_evaluations = [
                            e for e in self.evaluation_history
                            if e.rule_id == rule.rule_id and e.evaluated_at > stale_threshold
                        ]
                        if not recent_evaluations:
                            stale_rules.append(rule.rule_id)
                
                if stale_rules:
                    health_status["healthy"] = False
                    health_status["errors"].append(f"{len(stale_rules)} rules have not been evaluated recently")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "healthy": False,
                "running": self.running,
                "errors": [f"Health check failed: {str(e)}"]
            }
    
    async def update_rules_config(self, new_config: Dict[str, Any]) -> None:
        """Update rules configuration"""
        
        try:
            self.rules_config.update(new_config)
            logger.info(f"Updated rules configuration: {new_config}")
            
        except Exception as e:
            logger.error(f"Error updating rules configuration: {e}")
            raise
    
    async def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules for backup or migration"""
        
        try:
            exported_rules = []
            
            for rule in self.rules.values():
                rule_data = rule.to_dict()
                rule_data['export_timestamp'] = datetime.now().isoformat()
                exported_rules.append(rule_data)
            
            logger.info(f"Exported {len(exported_rules)} rules")
            return exported_rules
            
        except Exception as e:
            logger.error(f"Error exporting rules: {e}")
            raise
    
    async def import_rules(self, rules_data: List[Dict[str, Any]], 
                          overwrite_existing: bool = False) -> Dict[str, Any]:
        """Import rules from backup or migration"""
        
        try:
            imported_count = 0
            skipped_count = 0
            error_count = 0
            errors = []
            
            for rule_data in rules_data:
                try:
                    # Remove export timestamp if present
                    rule_data.pop('export_timestamp', None)
                    
                    # Create rule object
                    rule = GovernanceRule.from_dict(rule_data)
                    
                    # Check if rule already exists
                    if rule.rule_id in self.rules and not overwrite_existing:
                        skipped_count += 1
                        continue
                    
                    # Validate and add rule
                    if await self._validate_rule(rule):
                        async with self._lock:
                            self.rules[rule.rule_id] = rule
                        imported_count += 1
                    else:
                        error_count += 1
                        errors.append(f"Validation failed for rule {rule.rule_id}")
                        
                except Exception as e:
                    error_count += 1
                    errors.append(f"Error importing rule: {str(e)}")
            
            result = {
                "imported": imported_count,
                "skipped": skipped_count,
                "errors": error_count,
                "error_details": errors
            }
            
            logger.info(f"Import completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error importing rules: {e}")
            raise


# Utility functions for governance rules engine

def create_rules_engine(config: Optional[RulesEngineConfig] = None) -> GovernanceRulesEngine:
    """Factory function to create governance rules engine"""
    return GovernanceRulesEngine(config)


def get_default_rules_config() -> Dict[str, Any]:
    """Get default rules configuration"""
    return {
        "max_compute_units": 100,
        "max_storage_gb": 1000,
        "max_cost_usd": 5000.0,
        "enable_security_checks": True,
        "auto_remediation": False,
        "resource_threshold_cpu": 80.0,
        "resource_threshold_memory": 80.0,
        "min_data_retention_days": 30,
        "max_data_retention_days": 2555,
        "allowed_hours": list(range(6, 22)),  # 6 AM to 10 PM
        "restricted_environments": ["production"],
        "required_security_controls": ["encryption", "access_control", "audit_logging"]
    }


async def validate_rules_engine_config(config: RulesEngineConfig) -> List[str]:
    """Validate rules engine configuration"""
    
    errors = []
    
    # Validate evaluation interval
    if config.evaluation_interval_minutes <= 0:
        errors.append("Evaluation interval must be positive")
    
    if config.evaluation_interval_minutes > 1440:  # 24 hours
        errors.append("Evaluation interval cannot exceed 24 hours")
    
    # Validate history size
    if config.max_history_size <= 0:
        errors.append("Max history size must be positive")
    
    if config.max_history_size > 100000:  # Reasonable upper limit
        errors.append("Max history size is too large (max 100,000)")
    
    # Validate timeout
    if config.rule_timeout_seconds <= 0:
        errors.append("Rule timeout must be positive")
    
    if config.rule_timeout_seconds > 300:  # 5 minutes
        errors.append("Rule timeout cannot exceed 5 minutes")
    
    # Validate concurrency
    if config.max_concurrent_evaluations <= 0:
        errors.append("Max concurrent evaluations must be positive")
    
    if config.max_concurrent_evaluations > 100:
        errors.append("Max concurrent evaluations is too high (max 100)")
    
    return errors


def create_evaluation_context(agent_id: str, action_type: str, **kwargs) -> Dict[str, Any]:
    """Create evaluation context for rule evaluation"""
    
    context = {
        "agent_id": agent_id,
        "action_type": action_type,
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "on_demand"
    }
    
    # Add additional context
    context.update(kwargs)
    
    return context


async def evaluate_single_rule(rule: GovernanceRule, context: Dict[str, Any]) -> RuleEvaluation:
    """Evaluate a single rule independently"""
    
    # Create a temporary rules engine for evaluation
    engine = GovernanceRulesEngine()
    await engine.initialize()
    
    try:
        # Add the rule
        await engine.add_rule(rule)
        
        # Evaluate
        evaluations = await engine.evaluate_rules(context)
        
        return evaluations[0] if evaluations else RuleEvaluation(
            rule_id=rule.rule_id,
            status=RuleStatus.SKIPPED,
            message="No evaluation result",
            context=context
        )
        
    finally:
        await engine.shutdown()


def get_rule_evaluation_summary(evaluations: List[RuleEvaluation]) -> Dict[str, Any]:
    """Get summary of rule evaluations"""
    
    if not evaluations:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "violations": [],
            "recommendations": []
        }
    
    passed = len([e for e in evaluations if e.status == RuleStatus.PASSED])
    failed = len([e for e in evaluations if e.status == RuleStatus.FAILED])
    
    # Collect all violations and recommendations
    all_violations = []
    all_recommendations = []
    
    for evaluation in evaluations:
        all_violations.extend(evaluation.violations)
        all_recommendations.extend(evaluation.recommendations)
    
    return {
        "total": len(evaluations),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(evaluations),
        "violations": all_violations,
        "recommendations": list(set(all_recommendations)),  # Remove duplicates
        "severity_breakdown": _get_severity_breakdown(evaluations),
        "rule_breakdown": _get_rule_breakdown(evaluations)
    }


def _get_severity_breakdown(evaluations: List[RuleEvaluation]) -> Dict[str, int]:
    """Get breakdown of evaluations by severity"""
    
    breakdown = {severity.value: 0 for severity in Severity}
    
    for evaluation in evaluations:
        sev = evaluation.get_severity_level()
        breakdown[sev.value] += 1
    
    return breakdown


def _get_rule_breakdown(evaluations: List[RuleEvaluation]) -> Dict[str, Dict[str, Any]]:
    """Get breakdown of evaluations by rule"""
    
    breakdown = {}
    
    for evaluation in evaluations:
        rule_id = evaluation.rule_id
        
        if rule_id not in breakdown:
            breakdown[rule_id] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "violations": 0
            }
        
        breakdown[rule_id]["total"] += 1
        
        if evaluation.status == RuleStatus.PASSED:
            breakdown[rule_id]["passed"] += 1
        else:
            breakdown[rule_id]["failed"] += 1
            breakdown[rule_id]["violations"] += len(evaluation.violations)
    
    return breakdown
