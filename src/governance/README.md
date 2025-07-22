
#### 4. High Rule Violation Rates

**Symptoms:**

- Frequent rule violations across multiple agents
- High number of approval requests
- Agents frequently blocked from taking actions

**Diagnosis:**

```python
# Check rule statistics
stats = rules_engine.get_rule_statistics()
print(f"Overall failure rate: {stats['failure_rate']:.2%}")

# Check violations by rule type
for rule_type, type_stats in stats['by_rule_type'].items():
    if type_stats['failure_rate'] > 0.1:  # More than 10% failure rate
        print(f"High failure rate for {rule_type}: {type_stats['failure_rate']:.2%}")
```

**Solution:**

```python
# Review and adjust rule thresholds
await rules_engine.update_rules_config({
    "cost_threshold_usd": 2000.0,  # Increase if too restrictive
    "resource_threshold_cpu": 85.0,  # Adjust based on actual usage
})

# Consider rule exceptions for specific scenarios
exception_rule = GovernanceRule(
    name="Emergency Response Exception",
    rule_type=RuleType.OPERATIONAL_CONSTRAINT,
    parameters={
        "exception_conditions": ["incident_severity=CRITICAL"],
        "bypass_cost_limits": True
    }
)
```

#### 5. Performance Issues

**Symptoms:**

- Slow approval response times
- Rule evaluation timeouts
- High system resource usage

**Diagnosis:**

```python
# Check system performance
health = await rules_engine.health_check()
print(f"Evaluation task running: {health['evaluation_task_running']}")

autonomy_health = await autonomy_controller.health_check()
print(f"Pending approvals: {autonomy_health['pending_approvals']}")
```

**Solution:**

```python
# Optimize rule evaluation
await rules_engine.update_rules_config({
    "max_concurrent_evaluations": 20,  # Increase parallelism
    "rule_timeout_seconds": 60,  # Increase timeout
})

# Optimize approval processing
autonomy_controller.config.max_concurrent_requests = 20
autonomy_controller.config.cleanup_interval_minutes = 2  # More frequent cleanup
```

### Monitoring and Metrics

#### Key Performance Indicators (KPIs)

**Governance Effectiveness:**

- Approval request volume and trends
- Average approval response time
- Rule violation frequency by type and severity
- Agent autonomy utilization rates

**Operational Efficiency:**

- Percentage of actions requiring approval
- Time spent waiting for approvals
- Rule evaluation performance
- System resource utilization

**Compliance and Risk:**

- Policy violation incidents
- Audit trail completeness
- Risk threshold breach frequency
- Emergency override usage

#### Monitoring Implementation

```python
async def collect_governance_metrics():
    """Collect governance system metrics"""
    
    # Autonomy controller metrics
    autonomy_status = autonomy_controller.get_autonomy_status()
    autonomy_health = await autonomy_controller.health_check()
    
    # Rules engine metrics
    rules_stats = rules_engine.get_rule_statistics()
    rules_health = await rules_engine.health_check()
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "autonomy": {
            "level": autonomy_status["autonomy_level"],
            "pending_approvals": autonomy_status["pending_approvals_count"],
            "approval_history": autonomy_status["approval_history_count"],
            "healthy": autonomy_health["healthy"]
        },
        "rules": {
            "total_rules": rules_stats["total_rules"],
            "active_rules": rules_stats["active_rules"],
            "failure_rate": rules_stats["failure_rate"],
            "total_evaluations": rules_stats["total_evaluations"],
            "healthy": rules_health["healthy"]
        }
    }
    
    return metrics

# Set up periodic monitoring
async def governance_monitor():
    """Background monitoring task"""
    while True:
        try:
            metrics = await collect_governance_metrics()
            
            # Log metrics or send to monitoring system
            logger.info(f"Governance metrics: {metrics}")
            
            # Alert on issues
            if not metrics["autonomy"]["healthy"]:
                logger.error("Autonomy controller unhealthy")
            
            if not metrics["rules"]["healthy"]:
                logger.error("Rules engine unhealthy")
            
            if metrics["rules"]["failure_rate"] > 0.2:  # 20% failure rate
                logger.warning(f"High rule failure rate: {metrics['rules']['failure_rate']:.2%}")
            
            await asyncio.sleep(300)  # 5 minute intervals
            
        except Exception as e:
            logger.error(f"Governance monitoring error: {e}")
            await asyncio.sleep(60)
```

### Integration Examples

#### Complete Agent Integration

```python
from src.agents.base_agent import BaseAgent
from src.governance.autonomy_controller import AutonomyController
from src.governance.governance_rules import GovernanceRulesEngine
from src.models.governance import AutonomyLevel

class ProductionAgent(BaseAgent):
    """Example production agent with full governance integration"""
    
    def __init__(self, config):
        super().__init__(config, "production_agent")
        
        # Set conservative autonomy level for production
        self.set_autonomy_level(AutonomyLevel.CONDITIONAL_AUTONOMY)
        
        # Configure approval-required actions
        self.add_approval_required_action("deploy_model")
        self.add_approval_required_action("modify_production_data")
        self.add_approval_required_action("scale_resources")
        
    async def deploy_model(self, model_config):
        """Deploy model with governance integration"""
        
        # This will automatically trigger approval workflow
        # if required based on autonomy level and risk assessment
        incident = create_incident(
            title="Model Deployment Request",
            description=f"Deploy model: {model_config['name']}",
            severity=Severity.MEDIUM,
            source="production_agent",
            category="deployment",
            incident_type="deploy_model",
            created_by=self.agent_id
        )
        
        # Handle through base class (includes governance)
        task_id = await self.handle_incident(incident)
        
        return task_id
    
    async def process_incident(self, incident):
        """Process incident with production safeguards"""
        
        if incident.incident_type == "deploy_model":
            return await self._handle_model_deployment(incident)
        else:
            return await super().process_incident(incident)
    
    async def _handle_model_deployment(self, incident):
        """Handle model deployment with additional checks"""
        
        # Perform pre-deployment validation
        validation_result = await self._validate_deployment(incident)
        if not validation_result["valid"]:
            raise RuntimeError(f"Deployment validation failed: {validation_result['reason']}")
        
        # Execute deployment
        deployment_result = await self._execute_deployment(incident)
        
        # Post-deployment verification
        verification_result = await self._verify_deployment(incident)
        
        return {
            "status": "completed",
            "validation": validation_result,
            "deployment": deployment_result,
            "verification": verification_result
        }

# Initialize governance system
async def setup_governance():
    """Set up complete governance system"""
    
    # Initialize autonomy controller
    autonomy_controller = AutonomyController()
    await autonomy_controller.initialize()
    
    # Initialize rules engine
    rules_engine = GovernanceRulesEngine()
    await rules_engine.initialize()
    
    # Set up approval handler
    async def approval_handler(request):
        """Custom approval handler"""
        
        # Route based on action type and cost
        if request["action_type"] == "deploy_model":
            if request.get("estimated_cost_usd", 0) > 1000:
                return await route_to_senior_engineer(request)
            else:
                return await route_to_team_lead(request)
        
        # Default routing
        return await route_to_default_approver(request)
    
    # Register callbacks
    autonomy_controller.register_approval_callback(approval_handler)
    
    # Create and configure agent
    agent = ProductionAgent(config)
    agent.set_governance_callback(autonomy_controller.request_action_approval)
    
    return autonomy_controller, rules_engine, agent
```

#### Custom Rule Implementation

```python
async def setup_custom_rules():
    """Set up organization-specific governance rules"""
    
    # Business hours deployment rule
    business_hours_rule = GovernanceRule(
        name="Business Hours Deployment Only",
        description="Production deployments only allowed during business hours",
        rule_type=RuleType.OPERATIONAL_CONSTRAINT,
        conditions={"time_check": True, "environment_check": True},
        actions={"block_outside_hours": True, "escalate_if_emergency": True},
        parameters={
            "allowed_hours": list(range(9, 17)),  # 9 AM to 5 PM
            "timezone": "US/Pacific",
            "restricted_environments": ["production"],
            "emergency_override": True,
            "escalation_target": "on_call_engineer"
        },
        applicable_actions=["deploy_model", "scale_resources"],
        violation_severity=Severity.HIGH,
        created_by="devops_team"
    )
    
    # Cost escalation rule with tiered approval
    cost_escalation_rule = GovernanceRule(
        name="Tiered Cost Approval",
        description="Multi-tier approval based on estimated cost",
        rule_type=RuleType.COST_LIMIT,
        conditions={"cost_check": True},
        actions={"escalate_by_cost": True},
        parameters={
            "tier_1_limit": 1000.0,    # Team lead approval
            "tier_2_limit": 5000.0,    # Manager approval
            "tier_3_limit": 25000.0,   # Director approval
            "tier_4_limit": 100000.0,  # VP approval
            "escalation_targets": {
                "tier_1": "team_lead",
                "tier_2": "engineering_manager",
                "tier_3": "engineering_director",
                "tier_4": "vp_engineering"
            }
        },
        violation_severity=Severity.MEDIUM,
        created_by="finance_team"
    )
    
    # Data sensitivity rule
    data_sensitivity_rule = GovernanceRule(
        name="Sensitive Data Protection",
        description="Additional controls for sensitive data operations",
        rule_type=RuleType.SECURITY_POLICY,
        conditions={"data_sensitivity_check": True},
        actions={"require_additional_controls": True},
        parameters={
            "sensitivity_levels": ["PII", "PHI", "FINANCIAL"],
            "required_controls": ["encryption", "audit_logging", "access_control"],
            "restricted_operations": ["export", "copy", "transform"],
            "approval_required": True
        },
        applicable_actions=["process_data", "export_data", "transform_data"],
        violation_severity=Severity.CRITICAL,
        created_by="security_team"
    )
    
    # Add rules to engine
    await rules_engine.add_rule(business_hours_rule)
    await rules_engine.add_rule(cost_escalation_rule)
    await rules_engine.add_rule(data_sensitivity_rule)
    
    logger.info("Custom governance rules configured")
```

## Conclusion

The Sentinel AI governance system provides a comprehensive framework for managing autonomous agent operations with appropriate oversight and control. By implementing proper autonomy levels, approval workflows, and governance rules, organizations can ensure that AI agents operate safely and effectively within defined boundaries.

Key benefits include:

- **Risk Management**: Automated risk assessment and approval workflows
- **Compliance**: Built-in audit trails and policy enforcement
- **Flexibility**: Configurable autonomy levels and rule sets
- **Scalability**: Designed to handle multiple agents and high-volume operations
- **Transparency**: Complete visibility into agent actions and decisions

For additional support or questions about the governance system, please refer to the source code documentation or contact the development team.

## Quick Reference

### Environment Variables

```bash
# Core governance settings
AUTONOMY_LEVEL=SUPERVISED_AUTONOMY
APPROVAL_TIMEOUT_MINUTES=30
COST_THRESHOLD_USD=1000.0

# Performance tuning
MAX_CONCURRENT_EVALUATIONS=10
RULES_EVALUATION_INTERVAL=15
CLEANUP_INTERVAL_MINUTES=5
```

### Common Commands

```python
# Check system health
autonomy_health = await autonomy_controller.health_check()
rules_health = await rules_engine.health_check()

# Get pending approvals
pending = autonomy_controller.get_pending_approvals()

# Get rule statistics
stats = rules_engine.get_rule_statistics()

# Manual rule evaluation
evaluations = await rules_engine.evaluate_rules(context)
```

### Support and Resources

- **Source Code**: `src/governance/`
- **Configuration**: `src/config/settings.py`
- **Models**: `src/models/governance.py`
- **Examples**: See integration examples above
- **Logging**: Check application logs for governance events
