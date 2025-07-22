from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
import uuid

from models.agent_state import (
    AgentState,
    AgentStatus,
    AgentCapability,
    create_agent_state,
    create_agent_capability,
)
from models.incident import (
    Incident,
    AgentAction,
    ActionType,
    ActionStatus,
    Severity,
    create_agent_action,
)
from models.governance import AutonomyLevel


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all Sentinel AI agents.

    Defines the common interface and behavior that all specialized agents
    must implement for proper integration with the Conductor Agent.
    """

    def __init__(self, config: Dict[str, Any], agent_type: str):
        self.config = config
        self.agent_type = agent_type
        self.agent_id = str(uuid.uuid4())
        self.agent_name = f"{agent_type}Agent"

        # Initialize agent state using proper model
        self.state = create_agent_state(
            agent_id=self.agent_id,
            incident_id="",  # No incident initially
            status="idle"    # Start in idle status
        )

        # Communication callbacks
        self.message_handlers: Dict[str, Callable] = {}
        self.conductor_callback: Optional[Callable] = None
        self.governance_callback: Optional[Callable] = None

        # Task management
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self.running = False

        # Governance integration
        self.autonomy_level = AutonomyLevel.SUPERVISED
        self.approval_required_actions = set()

        logger.info(f"Initialized {self.agent_name} with ID: {self.agent_id}")

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent with required services and capabilities.
        Must be implemented by each specialized agent.
        """
        pass

    @abstractmethod
    async def process_incident(self, incident: Incident) -> Dict[str, Any]:
        """
        Process an incident and return results.
        Must be implemented by each specialized agent.

        Args:
            incident: The incident to process

        Returns:
            Dict containing the processing results
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return list of capabilities this agent provides.
        Must be implemented by each specialized agent.

        Returns:
            List of AgentCapability objects
        """
        pass

    async def start(self) -> None:
        """Start the agent and begin processing"""
        try:
            await self.initialize()
            self.running = True
            self.state.update_status(AgentStatus.ACTIVE, "Agent started successfully")

            # Set capabilities
            capabilities = self.get_capabilities()
            for capability in capabilities:
                self.state.add_capability(capability)

            # Start health monitoring
            asyncio.create_task(self._health_monitor())

            logger.info(f"{self.agent_name} started successfully")

        except Exception as e:
            logger.error(f"Failed to start {self.agent_name}: {e}")
            self.state.update_status(AgentStatus.ERROR, f"Start failed: {str(e)}")
            self.state.record_error(str(e))
            raise

    async def stop(self) -> None:
        """Stop the agent and cleanup resources"""
        try:
            self.running = False
            self.state.update_status(AgentStatus.OFFLINE, "Agent stopping")

            # Cancel all current tasks
            for task_id, task in self.current_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self.current_tasks.clear()
            self.state.update_status(AgentStatus.OFFLINE, "Agent stopped")
            logger.info(f"{self.agent_name} stopped")

        except Exception as e:
            logger.error(f"Error stopping {self.agent_name}: {e}")
            self.state.record_error(str(e))
            raise

    async def handle_incident(self, incident: Incident) -> str:
        """
        Handle an incident by processing it and returning a task ID.

        Args:
            incident: The incident to handle

        Returns:
            Task ID for tracking the processing
        """
        try:
            if not self.state.is_healthy():
                raise RuntimeError(f"{self.agent_name} is not healthy")

            if self.state.status != AgentStatus.ACTIVE:
                raise RuntimeError(
                    f"{self.agent_name} is not active (status: {self.state.status.value})"
                )

            # Check if we can accept more tasks
            max_tasks = self.config.get("max_concurrent_tasks", 5)
            if len(self.current_tasks) >= max_tasks:
                raise RuntimeError(
                    f"{self.agent_name} cannot accept more tasks (current: {len(self.current_tasks)}, max: {max_tasks})"
                )

            task_id = str(uuid.uuid4())

            # Update agent state
            self.state.update_status(
                AgentStatus.BUSY, f"Processing incident: {incident.id}"
            )
            self.state.current_task = incident.id
            self.state.task_queue_size = len(self.current_tasks) + 1

            # Check if governance approval is required
            if await self._requires_governance_approval(incident):
                approval_granted = await self._request_governance_approval(
                    incident, task_id
                )
                if not approval_granted:
                    self.state.update_status(
                        AgentStatus.IDLE, "Governance approval denied"
                    )
                    raise RuntimeError(
                        f"Governance approval denied for incident: {incident.id}"
                    )

            # Create processing task
            task = asyncio.create_task(
                self._process_incident_with_tracking(incident, task_id)
            )
            self.current_tasks[task_id] = task

            logger.info(f"{self.agent_name} handling incident: {incident.id}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to handle incident {incident.id}: {e}")
            self.state.record_error(str(e))
            self.state.update_status(
                AgentStatus.ERROR, f"Failed to handle incident: {str(e)}"
            )
            raise

    async def _process_incident_with_tracking(
        self, incident: Incident, task_id: str
    ) -> None:
        """Process incident with proper error handling and state tracking"""
        start_time = datetime.now()

        # Create action record
        action = create_agent_action(
            incident_id=incident.id,
            agent_id=self.agent_id,
            action_type=ActionType.ANALYSIS,
            name=f"Process incident {incident.id}",
            description=f"Processing incident: {incident.title}",
        )

        try:
            # Start action execution
            action.start_execution()

            # Process the incident
            result = await self.process_incident(incident)

            # Complete action successfully
            action.complete_execution(result=result)

            # Update agent state
            self.state.clear_error()
            self.state.update_activity()

            # Notify conductor if callback is set
            if self.conductor_callback:
                await self.conductor_callback(
                    {
                        "agent_type": self.agent_type,
                        "agent_id": self.agent_id,
                        "incident_id": incident.id,
                        "task_id": task_id,
                        "result": result,
                        "status": "completed",
                        "action_id": action.id,
                    }
                )

            logger.info(f"{self.agent_name} completed incident: {incident.id}")

        except Exception as e:
            # Record failure in action
            action.fail_execution(str(e))

            # Update agent state
            self.state.record_error(str(e))

            logger.error(
                f"{self.agent_name} failed to process incident {incident.id}: {e}"
            )

            # Notify conductor of failure
            if self.conductor_callback:
                await self.conductor_callback(
                    {
                        "agent_type": self.agent_type,
                        "agent_id": self.agent_id,
                        "incident_id": incident.id,
                        "task_id": task_id,
                        "error": str(e),
                        "status": "failed",
                        "action_id": action.id,
                    }
                )

        finally:
            # Add action to incident
            try:
                incident.add_action(action)
            except Exception as e:
                logger.warning(f"Could not add action to incident: {e}")

            # Clean up task tracking
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]

            # Update agent state
            self.state.current_task = None
            self.state.task_queue_size = len(self.current_tasks)

            # Update status based on remaining tasks
            if len(self.current_tasks) == 0:
                self.state.update_status(AgentStatus.IDLE, "No active tasks")
            else:
                self.state.update_status(
                    AgentStatus.BUSY, f"Processing {len(self.current_tasks)} tasks"
                )

    def set_conductor_callback(self, callback: Callable) -> None:
        """Set callback function for communicating with conductor"""
        self.conductor_callback = callback
        logger.info(f"{self.agent_name} conductor callback set")

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
        logger.info(f"{self.agent_name} registered handler for: {message_type}")

    async def handle_message(
        self, message_type: str, message_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming messages from other agents or the conductor"""
        if message_type not in self.message_handlers:
            logger.warning(
                f"{self.agent_name} no handler for message type: {message_type}"
            )
            return {"error": f"No handler for message type: {message_type}"}

        try:
            handler = self.message_handlers[message_type]
            result = await handler(message_data)
            return {"success": True, "result": result}

        except Exception as e:
            logger.error(
                f"{self.agent_name} error handling message {message_type}: {e}"
            )
            self.state.record_error(str(e))
            return {"error": str(e)}

    def set_governance_callback(self, callback: Callable) -> None:
        """Set callback function for governance approval requests"""
        self.governance_callback = callback
        logger.info(f"{self.agent_name} governance callback set")

    def set_autonomy_level(self, level: AutonomyLevel) -> None:
        """Set the autonomy level for this agent"""
        self.autonomy_level = level
        self.state.autonomy_level = level.value
        logger.info(f"{self.agent_name} autonomy level set to: {level.value}")

    def add_approval_required_action(self, action_type: str) -> None:
        """Add an action type that requires governance approval"""
        self.approval_required_actions.add(action_type)
        logger.info(f"{self.agent_name} added approval required action: {action_type}")

    def remove_approval_required_action(self, action_type: str) -> None:
        """Remove an action type from governance approval requirements"""
        self.approval_required_actions.discard(action_type)
        logger.info(
            f"{self.agent_name} removed approval required action: {action_type}"
        )

    async def _requires_governance_approval(self, incident: Incident) -> bool:
        """Check if incident processing requires governance approval"""
        try:
            # Always require approval for full human oversight
            if self.autonomy_level == AutonomyLevel.FULL_HUMAN_OVERSIGHT:
                return True

            # Check severity-based approval requirements
            if incident.severity in [Severity.HIGH, Severity.CRITICAL]:
                if self.autonomy_level in [
                    AutonomyLevel.HUMAN_APPROVAL_REQUIRED,
                    AutonomyLevel.CONDITIONAL_AUTONOMY,
                ]:
                    return True

            # Check if incident type requires approval
            if incident.incident_type in self.approval_required_actions:
                return True

            # Check for production environment
            if "production" in incident.affected_systems:
                if self.autonomy_level != AutonomyLevel.FULL_AUTONOMY:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking governance approval requirement: {e}")
            # Default to requiring approval on error
            return True

    async def _request_governance_approval(
        self, incident: Incident, task_id: str
    ) -> bool:
        """Request governance approval for incident processing"""
        try:
            if not self.governance_callback:
                logger.warning(
                    f"{self.agent_name} no governance callback set, denying approval"
                )
                return False

            approval_request = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "action_type": "process_incident",
                "action_description": f"Process incident: {incident.title}",
                "incident_id": incident.id,
                "task_id": task_id,
                "severity": incident.severity.value,
                "affected_systems": incident.affected_systems,
                "estimated_duration_minutes": 30,  # Default estimate
                "risk_level": incident.severity.value.upper(),
            }

            # Request approval through callback
            approval_response = await self.governance_callback(approval_request)

            if isinstance(approval_response, dict):
                return approval_response.get("approved", False)
            else:
                return bool(approval_response)

        except Exception as e:
            logger.error(f"Error requesting governance approval: {e}")
            return False

    async def _health_monitor(self) -> None:
        """Background task to monitor agent health"""
        while self.running:
            try:
                # Update health metrics
                self.state.health_metrics.update_heartbeat()
                self.state.update_activity()

                # Update system metrics (simulated for now)
                self.state.health_metrics.cpu_usage_percent = min(
                    100.0, len(self.current_tasks) * 10.0
                )
                self.state.health_metrics.memory_usage_mb = 256.0 + (
                    len(self.current_tasks) * 64.0
                )

                # Update performance metrics
                if len(self.current_tasks) > 0:
                    self.state.health_metrics.throughput_ops_per_sec = max(
                        0.1, 1.0 / len(self.current_tasks)
                    )
                else:
                    self.state.health_metrics.throughput_ops_per_sec = 1.0

                # Update health score based on current state
                self.state.health_metrics.update_health_score()

                # Check for stuck tasks
                await self._check_task_timeouts()

                # Update agent status based on health
                if not self.state.health_metrics.is_healthy():
                    if self.state.status != AgentStatus.ERROR:
                        self.state.update_status(
                            AgentStatus.ERROR, "Health check failed"
                        )
                elif (
                    len(self.current_tasks) == 0
                    and self.state.status == AgentStatus.BUSY
                ):
                    self.state.update_status(AgentStatus.IDLE, "No active tasks")

                # Sleep for health check interval
                await asyncio.sleep(30)  # 30 second intervals

            except Exception as e:
                logger.error(f"{self.agent_name} health monitor error: {e}")
                self.state.record_error(str(e))
                await asyncio.sleep(60)  # Longer sleep on error

    async def _check_task_timeouts(self) -> None:
        """Check for and handle task timeouts"""
        try:
            timeout_seconds = self.config.get(
                "task_timeout_seconds", 300
            )  # 5 minute default
            current_time = datetime.now()
            timed_out_tasks = []

            for task_id, task in list(self.current_tasks.items()):
                # Check if task has been running too long
                task_start_time = getattr(task, "_start_time", None)
                if not task_start_time:
                    # Set start time if not already set
                    task._start_time = current_time
                    continue

                elapsed_seconds = (current_time - task_start_time).total_seconds()
                if elapsed_seconds > timeout_seconds:
                    logger.warning(
                        f"{self.agent_name} cancelling timeout task: {task_id}"
                    )
                    task.cancel()
                    timed_out_tasks.append(task_id)

            # Clean up timed out tasks
            for task_id in timed_out_tasks:
                if task_id in self.current_tasks:
                    del self.current_tasks[task_id]

                # Update agent state
                self.state.record_error(
                    f"Task {task_id} timed out after {timeout_seconds} seconds"
                )

            # Update task queue size
            self.state.task_queue_size = len(self.current_tasks)

        except Exception as e:
            logger.error(f"Error checking task timeouts: {e}")
            self.state.record_error(str(e))

    def get_state(self) -> AgentState:
        """Get current agent state"""
        return self.state

    def get_status(self) -> Dict[str, Any]:
        """Get agent status summary"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.state.status.value,
            "running": self.running,
            "current_tasks": len(self.current_tasks),
            "task_queue_size": self.state.task_queue_size,
            "current_task": self.state.current_task,
            "capabilities": [cap.name for cap in self.state.capabilities],
            "available_capabilities": [
                cap.name for cap in self.state.get_available_capabilities()
            ],
            "last_heartbeat": self.state.health_metrics.last_heartbeat.isoformat(),
            "last_activity": self.state.last_activity.isoformat(),
            "is_healthy": self.state.is_healthy(),
            "health_score": self.state.health_metrics.health_score,
            "autonomy_level": self.autonomy_level.value,
            "error_count": self.state.error_count,
            "consecutive_errors": self.state.consecutive_errors,
            "last_error": self.state.last_error,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check and return status"""
        try:
            # Get current health status
            is_healthy = self.state.is_healthy()
            health_metrics = self.state.health_metrics

            # Perform additional checks
            checks = {
                "agent_running": self.running,
                "status_not_error": self.state.status != AgentStatus.ERROR,
                "low_error_rate": self.state.consecutive_errors < 5,
                "health_metrics_ok": health_metrics.is_healthy(),
                "task_load_ok": len(self.current_tasks)
                <= self.config.get("max_concurrent_tasks", 5),
                "recent_activity": (
                    datetime.now() - self.state.last_activity
                ).total_seconds()
                < 300,  # 5 minutes
            }

            # Calculate overall health
            passed_checks = sum(1 for check in checks.values() if check)
            health_percentage = (passed_checks / len(checks)) * 100

            return {
                "healthy": is_healthy and all(checks.values()),
                "health_percentage": health_percentage,
                "status": self.state.status.value,
                "current_tasks": len(self.current_tasks),
                "max_tasks": self.config.get("max_concurrent_tasks", 5),
                "last_heartbeat": health_metrics.last_heartbeat.isoformat(),
                "last_activity": self.state.last_activity.isoformat(),
                "uptime_seconds": (
                    datetime.now() - self.state.created_at
                ).total_seconds(),
                "error_count": self.state.error_count,
                "consecutive_errors": self.state.consecutive_errors,
                "health_score": health_metrics.health_score,
                "cpu_usage_percent": health_metrics.cpu_usage_percent,
                "memory_usage_mb": health_metrics.memory_usage_mb,
                "throughput_ops_per_sec": health_metrics.throughput_ops_per_sec,
                "checks": checks,
                "autonomy_level": self.autonomy_level.value,
                "governance_enabled": self.governance_callback is not None,
            }

        except Exception as e:
            logger.error(f"{self.agent_name} health check failed: {e}")
            self.state.record_error(str(e))
            return {
                "healthy": False,
                "error": str(e),
                "status": self.state.status.value,
                "current_tasks": len(self.current_tasks),
            }


class AgentError(Exception):
    """Base exception for agent-related errors"""

    pass


class AgentNotAvailableError(AgentError):
    """Raised when an agent is not available to process requests"""

    pass


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out"""

    pass


class AgentConfigurationError(AgentError):
    """Raised when there's an agent configuration issue"""

    pass
