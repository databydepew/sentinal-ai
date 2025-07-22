from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid


class AutonomyLevel(Enum):
    """Autonomy level enumeration"""
    MANUAL = "manual"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"
    FULL_AUTONOMOUS = "full_autonomous"


class RuleType(Enum):
    """Rule type enumeration"""
    SECURITY_POLICY = "security_policy"
    COST_CONTROL = "cost_control"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    SAFETY = "safety"


@dataclass
class GovernanceRule:
    """Governance rule data class"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    rule_type: RuleType = RuleType.SECURITY_POLICY
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleEvaluation:
    """Rule evaluation result data class"""
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    agent_id: str = ""
    action_type: str = ""
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)


def create_security_policy_rule(
    name: str, policy_type: str, **kwargs
) -> GovernanceRule:
    """Factory function to create security policy rules"""
    return GovernanceRule(
        name=name,
        description=f"Security policy rule: {policy_type}",
        rule_type=RuleType.SECURITY_POLICY,
        conditions={"security_check": True},
        actions={"enforce_policy": True},
        parameters={"policy_type": policy_type},
        **kwargs,
    )


def create_rule_evaluation(
    rule_id: str, agent_id: str, action_type: str, passed: bool = True, **kwargs
) -> RuleEvaluation:
    """Factory function to create rule evaluations"""
    return RuleEvaluation(
        rule_id=rule_id,
        agent_id=agent_id,
        action_type=action_type,
        passed=passed,
        **kwargs,
    )
