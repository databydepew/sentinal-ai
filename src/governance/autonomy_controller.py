import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from governance.models import (
    ApprovalRequest,
    ApprovalStatus,
    AutonomyLevel,
    AutonomyConfig,
)
import logger


class AutonomyController:
    def __init__(self, config: AutonomyConfig):
        self.config = config
        self.autonomy_level = config.autonomy_level
        self.approval_timeout_minutes = config.approval_timeout_minutes
        self.risk_thresholds = config.risk_thresholds or {}
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self.approval_callbacks: List[Any] = []
        self.running = False
        self._cleanup_task = None
        self._lock = asyncio.Lock()

    def requires_approval(self, action_type: str, action_data: Dict[str, Any]) -> bool:
        # Always require approval in MANUAL mode
        if self.autonomy_level == AutonomyLevel.MANUAL:
            return True

        # Never require approval in FULL_AUTO mode
        if self.autonomy_level == AutonomyLevel.FULL_AUTO:
            return False

        # Check specific approval requirements
        risk_level = action_data.get("risk_level", "MEDIUM")
        cost_usd = action_data.get("estimated_cost_usd", 0.0)

        # High-risk actions require approval
        if risk_level == "HIGH" and self.risk_thresholds.get(
            "high_severity_requires_approval", True
        ):
            return True

        # High-cost actions require approval
        if cost_usd > self.risk_thresholds.get("cost_threshold_usd", 1000.0):
            return True

        # Production deployments require approval
        if (
            action_type == "deploy_model"
            and action_data.get("environment") == "production"
            and self.risk_thresholds.get(
                "production_deployment_requires_approval", True
            )
        ):
            return True

        # Incident severity-based approval
        incident_severity = action_data.get("incident_severity")
        if incident_severity in ["HIGH", "CRITICAL"] and self.risk_thresholds.get(
            "high_severity_requires_approval", True
        ):
            return True

        return False

    async def _create_approval_request(
        self, action_type: str, action_data: Dict[str, Any], requester_agent: str
    ) -> ApprovalRequest:
        """Create a new approval request"""

        request_id = str(uuid.uuid4())
        risk_level = action_data.get("risk_level", "MEDIUM")

        approval_request = ApprovalRequest(
            request_id=request_id,
            action_type=action_type,
            action_data=action_data,
            requester_agent=requester_agent,
            risk_level=risk_level,
            timeout_minutes=self.approval_timeout_minutes,
        )

        # Store the request
        self.pending_approvals[request_id] = approval_request

        # Notify approval callbacks
        await self._notify_approval_callbacks(approval_request)

        logger.info(f"Created approval request: {request_id}")
        return approval_request

    async def _wait_for_approval(self, approval_request: ApprovalRequest) -> bool:
        """Wait for approval or timeout"""

        timeout_seconds = approval_request.timeout_minutes * 60
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            # Check if request has been approved or rejected
            if approval_request.status == ApprovalStatus.APPROVED:
                logger.info(f"Approval request {approval_request.request_id} approved")
                return True
            elif approval_request.status == ApprovalStatus.REJECTED:
                logger.info(f"Approval request {approval_request.request_id} rejected")
                return False

            # Wait before checking again
            await asyncio.sleep(5)

        # Timeout reached
        approval_request.status = ApprovalStatus.EXPIRED
        logger.warning(f"Approval request {approval_request.request_id} expired")
        return False

    async def _notify_approval_callbacks(
        self, approval_request: ApprovalRequest
    ) -> None:
        """Notify registered callbacks about approval request"""

        for callback in self.approval_callbacks:
            try:
                await callback(
                    {"type": "approval_request", "request": approval_request.to_dict()}
                )
            except Exception as e:
                logger.error(f"Failed to notify approval callback: {e}")

    async def approve_request(self, request_id: str, approved_by: str) -> bool:
        """Approve a pending request"""

        if request_id not in self.pending_approvals:
            logger.warning(f"Approval request {request_id} not found")
            return False

        approval_request = self.pending_approvals[request_id]

        if approval_request.is_expired():
            approval_request.status = ApprovalStatus.EXPIRED
            logger.warning(f"Cannot approve expired request: {request_id}")
            return False

        approval_request.status = ApprovalStatus.APPROVED
        approval_request.approved_by = approved_by
        approval_request.approval_timestamp = datetime.now()

        # Move to history
        self.approval_history.append(approval_request)
        del self.pending_approvals[request_id]

        logger.info(f"Approved request {request_id} by {approved_by}")
        return True

    async def reject_request(
        self, request_id: str, rejected_by: str, reason: str
    ) -> bool:
        """Reject a pending request"""

        if request_id not in self.pending_approvals:
            logger.warning(f"Approval request {request_id} not found")
            return False

        approval_request = self.pending_approvals[request_id]

        if approval_request.is_expired():
            approval_request.status = ApprovalStatus.EXPIRED
            logger.warning(f"Cannot reject expired request: {request_id}")
            return False

        approval_request.status = ApprovalStatus.REJECTED
        approval_request.approved_by = rejected_by
        approval_request.approval_timestamp = datetime.now()
        approval_request.rejection_reason = reason

        # Move to history
        self.approval_history.append(approval_request)
        del self.pending_approvals[request_id]

        logger.info(f"Rejected request {request_id} by {rejected_by}: {reason}")
        return True

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request"""
        try:
            approval_request = self.pending_approvals.pop(request_id, None)
            if approval_request:
                approval_request.status = ApprovalStatus.CANCELLED
                self.approval_history.append(approval_request)
                logger.info(f"Cancelled approval request: {request_id}")
                return True
            else:
                logger.warning(
                    f"Approval request {request_id} not found for cancellation"
                )
                return False
        except Exception as e:
            logger.error(f"Error cancelling request {request_id}: {e}")
            return False

    async def _cleanup_expired_approvals(self) -> None:
        """Background task to clean up expired approval requests"""

        while self.running:
            try:
                expired_requests = []

                async with self._lock:
                    for request_id, approval_request in self.pending_approvals.items():
                        if approval_request.is_expired():
                            expired_requests.append(request_id)

                    # Clean up expired requests
                    for request_id in expired_requests:
                        approval_request = self.pending_approvals[request_id]
                        approval_request.status = ApprovalStatus.EXPIRED

                        # Move to history
                        self.approval_history.append(approval_request)
                        del self.pending_approvals[request_id]

                        logger.info(
                            f"Cleaned up expired approval request: {request_id}"
                        )

                # Wait before next cleanup
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in approval cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        return [request.to_dict() for request in self.pending_approvals.values()]

    def get_approval_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get approval history"""
        return [request.to_dict() for request in self.approval_history[-limit:]]

    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get current autonomy controller status"""
        return {
            "autonomy_level": self.autonomy_level.value,
            "approval_timeout_minutes": self.approval_timeout_minutes,
            "risk_thresholds": self.risk_thresholds,
            "pending_approvals_count": len(self.pending_approvals),
            "approval_history_count": len(self.approval_history),
            "running": self.running,
            "max_pending_approvals": self.config.max_pending_approvals,
            "cleanup_interval_minutes": self.config.cleanup_interval_minutes,
            "last_cleanup": datetime.now().isoformat(),
        }

    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific approval request"""

        try:
            async with self._lock:
                # Check pending requests
                if request_id in self.pending_approvals:
                    return self.pending_approvals[request_id].to_dict()

                # Check history
                for request in self.approval_history:
                    if request.id == request_id:
                        return request.to_dict()

                return None

        except Exception as e:
            logger.error(f"Error getting request status for {request_id}: {e}")
            return None

    async def update_risk_thresholds(self, new_thresholds: Dict[str, Any]) -> None:
        """Update risk thresholds"""

        try:
            self.risk_thresholds.update(new_thresholds)
            logger.info(f"Updated risk thresholds: {new_thresholds}")
        except Exception as e:
            logger.error(f"Error updating risk thresholds: {e}")
            raise

    async def bulk_approve_requests(
        self, request_ids: List[str], approved_by: str, reason: Optional[str] = None
    ) -> Dict[str, bool]:
        """Bulk approve multiple requests"""

        results = {}

        for request_id in request_ids:
            try:
                result = await self.approve_request(request_id, approved_by)
                results[request_id] = result
            except Exception as e:
                logger.error(f"Error bulk approving request {request_id}: {e}")
                results[request_id] = False

        return results

    async def bulk_reject_requests(
        self, request_ids: List[str], rejected_by: str, reason: str
    ) -> Dict[str, bool]:
        """Bulk reject multiple requests"""

        results = {}

        for request_id in request_ids:
            try:
                result = await self.reject_request(request_id, rejected_by, reason)
                results[request_id] = result
            except Exception as e:
                logger.error(f"Error bulk rejecting request {request_id}: {e}")
                results[request_id] = False

        return results

    def get_requests_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all requests (pending and historical) for a specific agent"""

        requests = []

        # Add pending requests
        for request in self.pending_approvals.values():
            if request.requester_agent == agent_id:
                requests.append(request.to_dict())

        # Add historical requests
        for request in self.approval_history:
            if request.requester_agent == agent_id:
                requests.append(request.to_dict())

        # Sort by creation time (newest first)
        requests.sort(key=lambda x: x.get("requested_at", ""), reverse=True)

        return requests

    def get_requests_by_status(self, status: ApprovalStatus) -> List[Dict[str, Any]]:
        """Get all requests with a specific status"""

        requests = []

        if status == ApprovalStatus.PENDING:
            requests = [
                request.to_dict() for request in self.pending_approvals.values()
            ]
        else:
            for request in self.approval_history:
                if request.status == status:
                    requests.append(request.to_dict())

        requests.sort(key=lambda x: x.get("requested_at", ""), reverse=True)
        return requests

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on autonomy controller"""

        try:
            health_status = {
                "healthy": True,
                "running": self.running,
                "pending_approvals": len(self.pending_approvals),
                "approval_history_size": len(self.approval_history),
                "cleanup_task_running": self._cleanup_task
                and not self._cleanup_task.done(),
                "errors": [],
            }

            # Check if cleanup task is running
            if not health_status["cleanup_task_running"] and self.running:
                health_status["healthy"] = False
                health_status["errors"].append("Cleanup task is not running")

            # Check for too many pending approvals
            if len(self.pending_approvals) >= self.config.max_pending_approvals:
                health_status["healthy"] = False
                health_status["errors"].append("Maximum pending approvals reached")

            # Check for very old pending approvals
            old_threshold = datetime.now() - timedelta(hours=24)
            old_requests = 0

            for request in self.pending_approvals.values():
                if request.requested_at < old_threshold:
                    old_requests += 1

            if old_requests > 0:
                health_status["healthy"] = False
                health_status["errors"].append(
                    f"{old_requests} requests older than 24 hours"
                )

            return health_status

        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "healthy": False,
                "running": self.running,
                "errors": [f"Health check failed: {str(e)}"],
            }


def create_autonomy_controller(
    autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED_AUTONOMY,
    approval_timeout_minutes: int = 30,
    risk_thresholds: Optional[Dict[str, Any]] = None,
) -> AutonomyController:
    """Factory function to create autonomy controller"""
    config = AutonomyConfig(
        autonomy_level=autonomy_level,
        approval_timeout_minutes=approval_timeout_minutes,
        risk_thresholds=risk_thresholds or {},
    )
    return AutonomyController(config)


def get_default_risk_thresholds() -> Dict[str, Any]:
    """Get default risk thresholds"""
    return {
        "high_severity_requires_approval": True,
        "cost_threshold_usd": 1000.0,
        "production_deployment_requires_approval": True,
        "resource_threshold_cpu": 80.0,
        "resource_threshold_memory": 80.0,
        "max_concurrent_actions": 5,
        "data_modification_requires_approval": True,
        "external_api_calls_require_approval": False,
    }


async def validate_autonomy_controller_config(config: AutonomyConfig) -> List[str]:
    """Validate autonomy controller configuration"""

    errors = []

    # Validate autonomy level
    if not isinstance(config.autonomy_level, AutonomyLevel):
        errors.append("Invalid autonomy level")

    # Validate timeout
    if config.approval_timeout_minutes <= 0:
        errors.append("Approval timeout must be positive")

    if config.approval_timeout_minutes > 1440:  # 24 hours
        errors.append("Approval timeout cannot exceed 24 hours")

    # Validate max pending approvals
    if config.max_pending_approvals <= 0:
        errors.append("Max pending approvals must be positive")

    # Validate cleanup interval
    if config.cleanup_interval_minutes <= 0:
        errors.append("Cleanup interval must be positive")

    # Validate risk thresholds
    if "cost_threshold_usd" in config.risk_thresholds:
        if config.risk_thresholds["cost_threshold_usd"] < 0:
            errors.append("Cost threshold cannot be negative")

    if "resource_threshold_cpu" in config.risk_thresholds:
        cpu_threshold = config.risk_thresholds["resource_threshold_cpu"]
        if not 0 <= cpu_threshold <= 100:
            errors.append("CPU threshold must be between 0 and 100")

    if "resource_threshold_memory" in config.risk_thresholds:
        memory_threshold = config.risk_thresholds["resource_threshold_memory"]
        if not 0 <= memory_threshold <= 100:
            errors.append("Memory threshold must be between 0 and 100")

    return errors
