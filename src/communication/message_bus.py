import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Message:
    """Message data class"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None
    correlation_id: Optional[str] = None


class MessageBus:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.bus_id = str(uuid.uuid4())
        self.running = False
        self.agent_callbacks = {}
        self.message_queues = {}
        self.subscribers = {}
        
        # Message tracking
        self.pending_messages: Dict[str, Message] = {}
        self.message_history: List[Message] = []
        self.delivery_confirmations: Dict[str, Set[str]] = {}
        self.failed_deliveries: Dict[str, List[str]] = {}
        
        # Configuration
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.default_ttl_seconds = self.config.get("default_ttl_seconds", 300)
        self.max_queue_size = self.config.get("max_queue_size", 100)
        self.enable_persistence = self.config.get("enable_persistence", False)
        
        # Async queues and tasks
        self.broadcast_queue = None
        self.cleanup_task = None
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self) -> None:
        """Start the message bus"""
        self.running = True
        self.broadcast_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        
        logger.info(f"Message bus {self.bus_id} started")
    
    async def stop(self) -> None:
        """Stop the message bus"""
        self.running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        for task in self.processing_tasks.values():
            task.cancel()
        
        logger.info(f"Message bus {self.bus_id} stopped")
    
    async def register_agent(self, agent_id: str, callback: Callable) -> bool:
        """Register an agent with the message bus"""
        try:
            if agent_id in self.agent_callbacks:
                logger.warning(f"Agent {agent_id} already registered")
                return False
            
            self.agent_callbacks[agent_id] = callback
            self.message_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
            
            # Start processing task for this agent
            self.processing_tasks[agent_id] = asyncio.create_task(
                self._process_agent_queue(agent_id)
            )
            
            logger.info(f"Registered agent {agent_id} with message bus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message bus"""
        # Remove from subscribers
        for message_type, subscribers in self.subscribers.items():
            subscribers.discard(agent_id)
        
        # Remove callback and queue
        if agent_id in self.agent_callbacks:
            del self.agent_callbacks[agent_id]
        
        if agent_id in self.message_queues:
            del self.message_queues[agent_id]
        
        logger.info(f"Unregistered agent {agent_id} from message bus")
    
    def subscribe(self, agent_id: str, message_type: str) -> bool:
        """
        Subscribe an agent to a specific message type.
        
        Args:
            agent_id: ID of the agent subscribing
            message_type: Type of message to subscribe to
            
        Returns:
            True if subscription successful, False otherwise
        """
        if agent_id not in self.agent_callbacks:
            logger.error(f"Agent {agent_id} not registered with message bus")
            return False
        
        if message_type not in self.subscribers:
            self.subscribers[message_type] = set()
        
        self.subscribers[message_type].add(agent_id)
        logger.info(f"Agent {agent_id} subscribed to message type: {message_type}")
        return True
    
    def unsubscribe(self, agent_id: str, message_type: str) -> bool:
        """
        Unsubscribe an agent from a specific message type.
        
        Args:
            agent_id: ID of the agent unsubscribing
            message_type: Type of message to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if message_type in self.subscribers:
            self.subscribers[message_type].discard(agent_id)
            
            # Clean up empty subscription lists
            if not self.subscribers[message_type]:
                del self.subscribers[message_type]
            
            logger.info(f"Agent {agent_id} unsubscribed from message type: {message_type}")
            return True
        
        return False
    
    async def publish(self, message: Message) -> str:
        """
        Publish a message to the bus.
        
        Args:
            message: Message to publish
            
        Returns:
            Message ID for tracking
        """
        # Set TTL if not specified
        if message.ttl_seconds is None:
            message.ttl_seconds = self.default_ttl_seconds
        
        # Add to pending messages
        self.pending_messages[message.message_id] = message
        
        # Add to message history
        self._add_to_history(message)
        
        # Route the message
        if message.recipient_id:
            # Direct message to specific agent
            await self._route_direct_message(message)
        else:
            # Broadcast message
            await self._route_broadcast_message(message)
        
        logger.debug(f"Published message {message.message_id} of type {message.message_type}")
        return message.message_id
    
    async def send_direct_message(self, sender_id: str, recipient_id: str, 
                                message_type: str, payload: Dict[str, Any],
                                priority: MessagePriority = MessagePriority.NORMAL,
                                correlation_id: Optional[str] = None) -> str:
        """
        Send a direct message to a specific agent.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            correlation_id: Optional correlation ID for request/response tracking
            
        Returns:
            Message ID for tracking
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )
        
        return await self.publish(message)
    
    async def broadcast_message(self, sender_id: str, message_type: str, 
                              payload: Dict[str, Any],
                              priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """
        Broadcast a message to all subscribers of a message type.
        
        Args:
            sender_id: ID of the sending agent
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            
        Returns:
            Message ID for tracking
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=None,  # None indicates broadcast
            payload=payload,
            priority=priority
        )
        
        return await self.publish(message)
    
    async def _route_direct_message(self, message: Message) -> None:
        """Route a direct message to a specific agent"""
        recipient_id = message.recipient_id
        
        if recipient_id not in self.message_queues:
            logger.error(f"Recipient {recipient_id} not found for message {message.message_id}")
            self._mark_delivery_failed(message.message_id, recipient_id)
            return
        
        try:
            queue = self.message_queues[recipient_id]
            await queue.put(message)
            logger.debug(f"Routed direct message {message.message_id} to {recipient_id}")
            
        except Exception as e:
            logger.error(f"Failed to route message {message.message_id} to {recipient_id}: {e}")
            self._mark_delivery_failed(message.message_id, recipient_id)
    
    async def _route_broadcast_message(self, message: Message) -> None:
        """Route a broadcast message to all subscribers"""
        message_type = message.message_type
        
        if message_type not in self.subscribers:
            logger.warning(f"No subscribers for message type: {message_type}")
            return
        
        # Add to broadcast queue for processing
        await self.broadcast_queue.put(message)
    
    async def _process_broadcast_queue(self) -> None:
        """Background task to process broadcast messages"""
        while self.running:
            try:
                # Wait for broadcast message
                message = await asyncio.wait_for(self.broadcast_queue.get(), timeout=1.0)
                
                # Send to all subscribers
                subscribers = self.subscribers.get(message.message_type, set())
                
                for subscriber_id in subscribers:
                    if subscriber_id in self.message_queues:
                        try:
                            queue = self.message_queues[subscriber_id]
                            await queue.put(message)
                            self._mark_delivery_confirmed(message.message_id, subscriber_id)
                            
                        except Exception as e:
                            logger.error(f"Failed to deliver broadcast message {message.message_id} to {subscriber_id}: {e}")
                            self._mark_delivery_failed(message.message_id, subscriber_id)
                
                logger.debug(f"Processed broadcast message {message.message_id} to {len(subscribers)} subscribers")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing broadcast queue: {e}")
    
    async def _process_agent_queue(self, agent_id: str) -> None:
        """Background task to process messages for a specific agent"""
        queue = self.message_queues[agent_id]
        callback = self.agent_callbacks[agent_id]
        
        while self.running and agent_id in self.agent_callbacks:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Deliver message to agent
                try:
                    await callback(message)
                    self._mark_delivery_confirmed(message.message_id, agent_id)
                    logger.debug(f"Delivered message {message.message_id} to agent {agent_id}")
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed to handle message {message.message_id}: {e}")
                    self._mark_delivery_failed(message.message_id, agent_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing queue for agent {agent_id}: {e}")
    
    def _mark_delivery_confirmed(self, message_id: str, agent_id: str) -> None:
        """Mark message delivery as confirmed"""
        if message_id not in self.delivery_confirmations:
            self.delivery_confirmations[message_id] = set()
        
        self.delivery_confirmations[message_id].add(agent_id)
    
    def _mark_delivery_failed(self, message_id: str, agent_id: str) -> None:
        """Mark message delivery as failed"""
        if message_id not in self.failed_deliveries:
            self.failed_deliveries[message_id] = []
        
        self.failed_deliveries[message_id].append(agent_id)
    
    def _add_to_history(self, message: Message) -> None:
        """Add message to history with size limit"""
        self.message_history.append(message)
        
        # Maintain history size limit
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)
    
    async def _cleanup_expired_messages(self) -> None:
        """Background task to clean up expired messages"""
        while self.running:
            try:
                current_time = datetime.now()
                expired_messages = []
                
                for message_id, message in self.pending_messages.items():
                    if message.ttl_seconds:
                        age_seconds = (current_time - message.timestamp).total_seconds()
                        if age_seconds > message.ttl_seconds:
                            expired_messages.append(message_id)
                
                # Remove expired messages
                for message_id in expired_messages:
                    del self.pending_messages[message_id]
                    
                    # Clean up delivery tracking
                    self.delivery_confirmations.pop(message_id, None)
                    self.failed_deliveries.pop(message_id, None)
                
                if expired_messages:
                    logger.debug(f"Cleaned up {len(expired_messages)} expired messages")
                
                # Sleep for cleanup interval
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"Error during message cleanup: {e}")
                await asyncio.sleep(120)  # Longer sleep on error
    
    def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get delivery status of a message"""
        if message_id not in self.pending_messages:
            return None
        
        message = self.pending_messages[message_id]
        confirmed_deliveries = self.delivery_confirmations.get(message_id, set())
        failed_deliveries = self.failed_deliveries.get(message_id, [])
        
        return {
            "message_id": message_id,
            "message_type": message.message_type,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority.value,
            "confirmed_deliveries": list(confirmed_deliveries),
            "failed_deliveries": failed_deliveries,
            "total_confirmations": len(confirmed_deliveries),
            "total_failures": len(failed_deliveries)
        }
    
    def get_bus_metrics(self) -> Dict[str, Any]:
        """Get message bus metrics and statistics"""
        total_agents = len(self.agent_callbacks)
        total_subscriptions = sum(len(subs) for subs in self.subscribers.values())
        
        queue_sizes = {
            agent_id: queue.qsize() if hasattr(queue, 'qsize') else 0
            for agent_id, queue in self.message_queues.items()
        }
        
        return {
            "bus_id": self.bus_id,
            "running": self.running,
            "agents": {
                "total_registered": total_agents,
                "total_subscriptions": total_subscriptions,
                "queue_sizes": queue_sizes
            },
            "messages": {
                "pending": len(self.pending_messages),
                "history_size": len(self.message_history),
                "broadcast_queue_size": self.broadcast_queue.qsize() if hasattr(self.broadcast_queue, 'qsize') else 0
            },
            "delivery": {
                "confirmed_messages": len(self.delivery_confirmations),
                "failed_messages": len(self.failed_deliveries)
            },
            "configuration": {
                "max_history_size": self.max_history_size,
                "default_ttl_seconds": self.default_ttl_seconds,
                "max_queue_size": self.max_queue_size,
                "enable_persistence": self.enable_persistence
            }
        }
    
    def get_subscription_info(self) -> Dict[str, List[str]]:
        """Get current subscription information"""
        return {
            message_type: list(subscribers)
            for message_type, subscribers in self.subscribers.items()
        }
