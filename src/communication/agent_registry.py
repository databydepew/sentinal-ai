import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from agents.base_agent import BaseAgent, AgentStatus

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for managing and coordinating multiple agents"""
    
    def __init__(self, config: Dict[str, Any], message_bus=None):
        self.config = config
        self.message_bus = message_bus
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, List[str]] = {}
        self.round_robin_counters: Dict[str, int] = {}
        self.load_balancer_strategy = config.get("load_balancer_strategy", "round_robin")
        self.health_check_interval = config.get("health_check_interval_minutes", 5)
        self.running = False  # Track if registry is running
        self.unhealthy_threshold_minutes = config.get("unhealthy_threshold_minutes", 10)
        self._health_monitor_task = None
        self._running = False
    
    async def start(self) -> None:
        """Start the agent registry"""
        self._running = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        logger.info("Agent registry started")
    
    async def stop(self) -> None:
        """Stop the agent registry"""
        self._running = False
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
        
        # Stop all registered agents
        for agent in self.agents.values():
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping agent {agent.agent_id}: {e}")
        
        logger.info("Agent registry stopped")
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register a new agent"""
        try:
            agent_id = agent.agent_id
            agent_type = agent.agent_type
            
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered")
                return False
            
            # Register the agent
            self.agents[agent_id] = agent
            
            # Add to type mapping
            if agent_type not in self.agent_types:
                self.agent_types[agent_type] = []
            self.agent_types[agent_type].append(agent_id)
            
            logger.info(f"Registered agent: {agent.agent_name} ({agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: The ID of the agent to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
            
            agent = self.agents[agent_id]
            
            # Stop the agent
            await agent.stop()
            
            # Remove from registry
            del self.agents[agent_id]
            
            # Remove from type mapping
            if agent.agent_type in self.agent_types:
                if agent_id in self.agent_types[agent.agent_type]:
                    self.agent_types[agent.agent_type].remove(agent_id)
                
                # Clean up empty type lists
                if not self.agent_types[agent.agent_type]:
                    del self.agent_types[agent.agent_type]
                    if agent.agent_type in self.round_robin_counters:
                        del self.round_robin_counters[agent.agent_type]
            
            logger.info(f"Unregistered agent: {agent.agent_name} ({agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        if agent_type not in self.agent_types:
            return []
        
        return [self.agents[agent_id] for agent_id in self.agent_types[agent_type] 
                if agent_id in self.agents]
    
    def get_available_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Get an available agent of the specified type using load balancing.
        
        Args:
            agent_type: The type of agent needed
            
        Returns:
            Available agent instance or None if none available
        """
        agents = self.get_agents_by_type(agent_type)
        available_agents = [agent for agent in agents if agent.state.is_available()]
        
        if not available_agents:
            return None
        
        # Apply load balancing strategy
        if self.load_balancer_strategy == "round_robin":
            return self._round_robin_select(agent_type, available_agents)
        elif self.load_balancer_strategy == "least_loaded":
            return self._least_loaded_select(available_agents)
        else:
            # Default to first available
            return available_agents[0]
    
    def _round_robin_select(self, agent_type: str, agents: List[BaseAgent]) -> BaseAgent:
        """Select agent using round-robin strategy"""
        if agent_type not in self.round_robin_counters:
            self.round_robin_counters[agent_type] = 0
        
        selected_agent = agents[self.round_robin_counters[agent_type] % len(agents)]
        self.round_robin_counters[agent_type] += 1
        
        return selected_agent
    
    def _least_loaded_select(self, agents: List[BaseAgent]) -> BaseAgent:
        """Select agent with least current load"""
        return min(agents, key=lambda agent: len(agent.state.current_tasks))
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    def get_agent_types(self) -> List[str]:
        """Get list of all registered agent types"""
        return list(self.agent_types.keys())
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status"""
        total_agents = len(self.agents)
        available_agents = sum(1 for agent in self.agents.values() 
                             if agent.state.is_available())
        
        type_summary = {}
        for agent_type, agent_ids in self.agent_types.items():
            agents = [self.agents[aid] for aid in agent_ids if aid in self.agents]
            type_summary[agent_type] = {
                "total": len(agents),
                "available": sum(1 for agent in agents if agent.state.is_available()),
                "busy": sum(1 for agent in agents if agent.state.status == AgentStatus.BUSY),
                "error": sum(1 for agent in agents if agent.state.status == AgentStatus.ERROR)
            }
        
        return {
            "registry_id": self.registry_id,
            "running": self.running,
            "total_agents": total_agents,
            "available_agents": available_agents,
            "agent_types": type_summary,
            "load_balancer_strategy": self.load_balancer_strategy,
            "health_check_interval": self.health_check_interval
        }
    
    async def _health_monitor(self) -> None:
        """Background task to monitor agent health"""
        while self.running:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval * 2)
    
    async def _check_agent_health(self) -> None:
        """Check health of all registered agents"""
        unhealthy_agents = []
        current_time = datetime.now()
        
        for agent_id, agent in self.agents.items():
            try:
                # Check if agent is responsive
                health_result = await agent.health_check()
                
                if not health_result.get("healthy", False):
                    logger.warning(f"Agent {agent_id} reported unhealthy: {health_result}")
                
                # Check heartbeat timeout
                last_heartbeat = agent.state.health_metrics.last_heartbeat
                time_since_heartbeat = (current_time - last_heartbeat).total_seconds() / 60
                
                if time_since_heartbeat > self.unhealthy_threshold_minutes:
                    logger.warning(f"Agent {agent_id} heartbeat timeout: {time_since_heartbeat:.1f} minutes")
                    unhealthy_agents.append(agent_id)
                
            except Exception as e:
                logger.error(f"Health check failed for agent {agent_id}: {e}")
                unhealthy_agents.append(agent_id)
        
        # Handle unhealthy agents
        for agent_id in unhealthy_agents:
            await self._handle_unhealthy_agent(agent_id)
    
    async def _handle_unhealthy_agent(self, agent_id: str) -> None:
        """Handle an unhealthy agent"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                return
            
            logger.warning(f"Handling unhealthy agent: {agent_id}")
            
            # Mark as error state
            agent.state.status = AgentStatus.ERROR
            
            # Optionally restart the agent
            restart_unhealthy = self.config.get("restart_unhealthy_agents", False)
            if restart_unhealthy:
                logger.info(f"Attempting to restart unhealthy agent: {agent_id}")
                try:
                    await agent.stop()
                    await agent.start()
                    logger.info(f"Successfully restarted agent: {agent_id}")
                except Exception as e:
                    logger.error(f"Failed to restart agent {agent_id}: {e}")
                    # Consider unregistering if restart fails
                    await self.unregister_agent(agent_id)
            
        except Exception as e:
            logger.error(f"Error handling unhealthy agent {agent_id}: {e}")
    
    async def broadcast_message(self, message_type: str, message_data: Dict[str, Any], 
                              agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Broadcast a message to all agents or agents of a specific type.
        
        Args:
            message_type: Type of message to send
            message_data: Message payload
            agent_type: Optional filter by agent type
            
        Returns:
            Dict mapping agent_id to response
        """
        target_agents = self.agents.values()
        if agent_type:
            target_agents = self.get_agents_by_type(agent_type)
        
        responses = {}
        
        # Send message to all target agents concurrently
        tasks = []
        for agent in target_agents:
            task = asyncio.create_task(
                self._send_message_to_agent(agent, message_type, message_data)
            )
            tasks.append((agent.agent_id, task))
        
        # Collect responses
        for agent_id, task in tasks:
            try:
                response = await task
                responses[agent_id] = response
            except Exception as e:
                logger.error(f"Failed to send message to agent {agent_id}: {e}")
                responses[agent_id] = {"error": str(e)}
        
        return responses
    
    async def _send_message_to_agent(self, agent: BaseAgent, message_type: str, 
                                   message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a specific agent"""
        try:
            return await agent.handle_message(message_type, message_data)
        except Exception as e:
            logger.error(f"Error sending message to agent {agent.agent_id}: {e}")
            return {"error": str(e)}
