from typing import Dict, List, Optional, Any, AsyncGenerator
import logging
import asyncio
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Custom exception for Gemini client errors"""
    pass


class GeminiConfig:
    """Configuration class for Gemini client"""
    def __init__(self, model_name: str = "gemini-pro", **kwargs):
        self.model_name = model_name
        self.base_url = kwargs.get("base_url", "")
        self.temperature = kwargs.get("temperature", 0.2)
        self.top_p = kwargs.get("top_p", 0.95)
        self.top_k = kwargs.get("top_k", 40)
        self.max_output_tokens = kwargs.get("max_output_tokens", 1024)
        self.safety_settings = kwargs.get("safety_settings", {})
        self.timeout_seconds = kwargs.get("timeout_seconds", 30)
        self.max_retries = kwargs.get("max_retries", 3)
        self.max_concurrent_requests = kwargs.get("max_concurrent_requests", 10)
        self.rate_limit_requests_per_minute = kwargs.get("rate_limit_requests_per_minute", 60)
        self.enable_simulation = kwargs.get("enable_simulation", True)


class GeminiClient:
    """Client for interacting with Google's Gemini API"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-pro",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """Initialize the Gemini client

        Args:
            api_key: API key for Gemini
            model_name: Name of the model to use
            max_retries: Maximum number of retries for API calls
            timeout: Timeout in seconds for API calls
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.initialized = False
        self.last_health_check = None

        # Default generation config
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        # Default safety settings
        self.safety_settings = {
            "harassment": "block_medium_and_above",
            "hate_speech": "block_medium_and_above",
            "sexually_explicit": "block_medium_and_above",
            "dangerous_content": "block_medium_and_above",
        }

        logger.info(f"GeminiClient initialized with model {model_name}")

    async def initialize(self) -> None:
        """Initialize the client and test connection"""
        try:
            # Test connection with a simple prompt
            await self.generate_text("Hello, are you working?")
            self.initialized = True
            logger.info("GeminiClient successfully initialized and connected")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiClient: {e}")
            raise

    async def generate_text(
        self, prompt: str, generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text using Gemini API

        Args:
            prompt: The prompt to send to Gemini
            generation_config: Optional custom generation config

        Returns:
            Generated text response
        """
        if not self.initialized:
            logger.warning("GeminiClient not initialized, initializing now")
            await self.initialize()

        config = generation_config or self.generation_config

        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                # In a real implementation, this would make an API call to Gemini
                # For now, we'll simulate a response
                logger.debug(
                    f"Sending prompt to Gemini (attempt {attempt+1}): {prompt[:50]}..."
                )

                # Simulate API call delay
                await asyncio.sleep(0.5)

                # Simulate response based on prompt
                if "error" in prompt.lower() and attempt < self.max_retries - 1:
                    raise Exception("Simulated API error")

                response = self._simulate_gemini_response(prompt)
                return response

            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"All {self.max_retries} attempts to call Gemini API failed"
                    )
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

    def _simulate_gemini_response(self, prompt: str) -> str:
        """Simulate a response from Gemini API for development purposes

        Args:
            prompt: The prompt sent to the API

        Returns:
            Simulated response
        """
        # Extract keywords from prompt to generate contextual response
        prompt_lower = prompt.lower()

        if "diagnose" in prompt_lower or "root cause" in prompt_lower:
            return """
            {
                "root_cause": "Feature distribution drift in input data",
                "confidence_score": 0.87,
                "contributing_factors": [
                    "Recent data pipeline changes",
                    "Seasonal variation in user behavior",
                    "Missing values in critical features"
                ],
                "recommended_actions": [
                    "Retrain model with recent data",
                    "Add data quality checks to pipeline",
                    "Implement feature monitoring"
                ],
                "technical_details": {
                    "evidence": [
                        "KL divergence of 0.42 between training and production distributions",
                        "15% increase in null values for feature 'user_activity'",
                        "Correlation shift between features 'x' and 'y'"
                    ],
                    "analysis": "The model is experiencing data drift due to recent changes in the data pipeline combined with seasonal variations in user behavior.",
                    "affected_components": ["data_ingestion", "feature_preprocessing"]
                }
            }
            """

        elif "remediation" in prompt_lower or "plan" in prompt_lower:
            return """
            {
                "plan_description": "Comprehensive plan to address data drift and retrain model",
                "steps": [
                    {
                        "step_id": "1",
                        "description": "Validate data pipeline integrity",
                        "estimated_duration_minutes": 60,
                        "technical_details": "Run validation tests on data pipeline to ensure proper functioning"
                    },
                    {
                        "step_id": "2",
                        "description": "Fix data quality issues",
                        "estimated_duration_minutes": 120,
                        "technical_details": "Address missing values and outliers in critical features"
                    },
                    {
                        "step_id": "3",
                        "description": "Retrain model with recent data",
                        "estimated_duration_minutes": 180,
                        "technical_details": "Use last 3 months of data with updated feature engineering"
                    },
                    {
                        "step_id": "4",
                        "description": "Validate model performance",
                        "estimated_duration_minutes": 60,
                        "technical_details": "Run A/B test comparing new model against current production model"
                    }
                ],
                "estimated_duration_hours": 7,
                "required_resources": ["Data Scientist", "ML Engineer", "DevOps"],
                "risk_level": "MEDIUM",
                "rollback_plan": "Revert to previous model version if performance degrades"
            }
            """

        elif "notification" in prompt_lower or "report" in prompt_lower:
            return """
            # Incident Resolution Report
            
            ## Summary
            The model performance degradation incident affecting the recommendation system has been successfully resolved. Root cause was identified as data drift in key features, and remediation involved retraining the model with recent data and implementing improved monitoring.
            
            ## Impact
            - Duration: 4.5 hours
            - Affected users: Approximately 12% of total user base
            - Business impact: Minimal revenue loss, estimated at $2,300
            
            ## Resolution
            The issue was resolved by implementing a four-step remediation plan that addressed data quality issues and retrained the model with recent data. Performance has returned to expected levels with a 15% improvement over the degraded state.
            
            ## Recommendations
            1. Implement automated drift detection
            2. Improve data quality monitoring
            3. Schedule regular model retraining
            4. Update incident response procedures
            """

        else:
            return "I've processed your request and generated a response based on the available information. If you need more specific details or have follow-up questions, please provide additional context."

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Gemini API connection

        Returns:
            Dict with health status information
        """
        if not self.initialized:
            return {"status": "not initialized", "initialized": False}

        start_time = time.time()

        try:
            # Test basic functionality
            test_response = await self.generate_text("Health check test")

            response_time_ms = (time.time() - start_time) * 1000
            self.last_health_check = datetime.now()

            return {
                "status": "healthy",
                "model": self.config.model_name,
                "initialized": self.initialized,
                "response_time_ms": response_time_ms,
                "last_check": self.last_health_check.isoformat(),
                "statistics": self.get_statistics(),
                "rate_limit_status": self._get_rate_limit_status(),
                "session_status": "active"
                if self._session and not self._session.closed
                else "inactive",
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.model_name,
                "initialized": self.initialized,
                "last_check": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
            }

    def _get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)
        recent_requests = [t for t in self._request_times if t > cutoff_time]

        return {
            "requests_last_minute": len(recent_requests),
            "limit_per_minute": self.config.rate_limit_requests_per_minute,
            "remaining_requests": max(
                0, self.config.rate_limit_requests_per_minute - len(recent_requests)
            ),
            "reset_time": (cutoff_time + timedelta(minutes=1)).isoformat()
            if recent_requests
            else None,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and configuration"""
        return {
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "generation_config": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
            },
            "safety_settings": self.config.safety_settings,
            "timeout_seconds": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "rate_limit_per_minute": self.config.rate_limit_requests_per_minute,
            "simulation_mode": self.config.enable_simulation,
            "initialized": self.initialized,
            "last_health_check": self.last_health_check.isoformat()
            if self.last_health_check
            else None,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get client usage statistics"""
        return {
            "total_requests": self._stats["total_requests"],
            "successful_requests": self._stats["successful_requests"],
            "failed_requests": self._stats["failed_requests"],
            "success_rate": self._get_success_rate(),
            "average_response_time_ms": self._stats["average_response_time_ms"],
            "total_tokens_processed": self._stats["total_tokens"],
            "rate_limit_status": self._get_rate_limit_status(),
        }

    def update_generation_config(self, **kwargs) -> None:
        """Update generation configuration parameters

        Args:
            **kwargs: Configuration parameters to update
        """
        valid_params = ["temperature", "top_p", "top_k", "max_output_tokens"]

        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self.config, param, value)
                logger.info(f"Updated {param} to {value}")
            else:
                logger.warning(f"Invalid generation config parameter: {param}")

    def update_safety_settings(self, settings: Dict[str, str]) -> None:
        """Update safety settings

        Args:
            settings: New safety settings to apply
        """
        valid_categories = [
            "harassment",
            "hate_speech",
            "sexually_explicit",
            "dangerous_content",
        ]
        valid_levels = [
            "block_none",
            "block_few",
            "block_some",
            "block_most",
            "block_medium_and_above",
        ]

        for category, level in settings.items():
            if category in valid_categories and level in valid_levels:
                self.config.safety_settings[category] = level
                logger.info(f"Updated safety setting {category} to {level}")
            else:
                logger.warning(f"Invalid safety setting: {category}={level}")

    async def batch_generate(
        self, prompts: List[str], generation_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate text for multiple prompts concurrently

        Args:
            prompts: List of prompts to process
            generation_config: Optional generation configuration

        Returns:
            List of generated responses in the same order as input prompts
        """
        if not self.initialized:
            raise GeminiClientError("Client not initialized. Call initialize() first.")

        if not prompts:
            return []

        # Create tasks for concurrent processing
        tasks = []
        for prompt in prompts:
            task = asyncio.create_task(self.generate_text(prompt, generation_config))
            tasks.append(task)

        try:
            # Wait for all tasks to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Batch generation failed for prompt {i}: {response}")
                    results.append(f"Error: {str(response)}")
                else:
                    results.append(response)

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise GeminiClientError(f"Batch generation failed: {str(e)}")

    async def stream_generate(
        self, prompt: str, generation_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response (simulated for development)

        Args:
            prompt: The prompt to send to Gemini
            generation_config: Optional generation configuration

        Yields:
            Chunks of generated text
        """
        if not self.initialized:
            raise GeminiClientError("Client not initialized. Call initialize() first.")

        # For simulation, we'll break the response into chunks
        full_response = await self.generate_text(prompt, generation_config)

        # Split response into chunks for streaming simulation
        chunk_size = 50
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i : i + chunk_size]
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "average_response_time_ms": 0.0,
        }
        logger.info("Statistics reset")

    async def validate_config(self) -> List[str]:
        """Validate current configuration

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate API key
        if not self.config.api_key:
            errors.append("API key is required")
        elif len(self.config.api_key) < 10:
            errors.append("API key appears to be invalid (too short)")

        # Validate model name
        if not self.config.model_name:
            errors.append("Model name is required")

        # Validate generation config
        if not 0 <= self.config.temperature <= 2:
            errors.append("Temperature must be between 0 and 2")

        if not 0 <= self.config.top_p <= 1:
            errors.append("Top-p must be between 0 and 1")

        if self.config.top_k < 1:
            errors.append("Top-k must be at least 1")

        if self.config.max_output_tokens < 1:
            errors.append("Max output tokens must be at least 1")

        # Validate timeouts and limits
        if self.config.timeout_seconds < 1:
            errors.append("Timeout must be at least 1 second")

        if self.config.max_retries < 0:
            errors.append("Max retries cannot be negative")

        if self.config.max_concurrent_requests < 1:
            errors.append("Max concurrent requests must be at least 1")

        if self.config.rate_limit_requests_per_minute < 1:
            errors.append("Rate limit must be at least 1 request per minute")

        return errors


# Utility functions and factory methods


def create_gemini_client(
    api_key: str, model_name: str = "gemini-pro", **kwargs
) -> GeminiClient:
    """Factory function to create a Gemini client

    Args:
        api_key: API key for Gemini
        model_name: Model name to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured GeminiClient instance
    """
    config = GeminiConfig(api_key=api_key, model_name=model_name, **kwargs)
    return GeminiClient(config)


@asynccontextmanager
async def gemini_client_context(api_key: str, model_name: str = "gemini-pro", **kwargs):
    """Async context manager for Gemini client

    Args:
        api_key: API key for Gemini
        model_name: Model name to use
        **kwargs: Additional configuration parameters

    Yields:
        Initialized GeminiClient instance
    """
    client = create_gemini_client(api_key, model_name, **kwargs)
    try:
        await client.initialize()
        yield client
    finally:
        await client.shutdown()


def get_default_generation_config() -> Dict[str, Any]:
    """Get default generation configuration"""
    return {"temperature": 0.2, "top_p": 0.95, "top_k": 40, "max_output_tokens": 1024}


def get_default_safety_settings() -> Dict[str, str]:
    """Get default safety settings"""
    return {
        "harassment": "block_medium_and_above",
        "hate_speech": "block_medium_and_above",
        "sexually_explicit": "block_medium_and_above",
        "dangerous_content": "block_medium_and_above",
    }


async def validate_gemini_config(config: GeminiConfig) -> List[str]:
    """Validate Gemini configuration

    Args:
        config: Configuration to validate

    Returns:
        List of validation errors
    """
    client = GeminiClient(config)
    return await client.validate_config()


# Example usage and testing utilities
async def test_gemini_client(
    api_key: str, test_prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Test Gemini client functionality

    Args:
        api_key: API key for testing
        test_prompts: Optional list of test prompts

    Returns:
        Test results
    """
    if test_prompts is None:
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain the concept of governance in AI systems.",
        ]

    results = {
        "initialization": False,
        "health_check": False,
        "text_generation": False,
        "batch_generation": False,
        "errors": [],
    }

    try:
        async with gemini_client_context(api_key, enable_simulation=True) as client:
            # Test initialization
            results["initialization"] = client.initialized

            # Test health check
            health = await client.health_check()
            results["health_check"] = health["status"] == "healthy"

            # Test single text generation
            try:
                response = await client.generate_text(test_prompts[0])
                results["text_generation"] = len(response) > 0
            except Exception as e:
                results["errors"].append(f"Text generation failed: {str(e)}")

            # Test batch generation
            try:
                batch_responses = await client.batch_generate(test_prompts[:2])
                results["batch_generation"] = len(batch_responses) == 2
            except Exception as e:
                results["errors"].append(f"Batch generation failed: {str(e)}")

            # Get final statistics
            results["statistics"] = client.get_statistics()

    except Exception as e:
        results["errors"].append(f"Client test failed: {str(e)}")

    return results
