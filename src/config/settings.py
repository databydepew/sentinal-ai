import os
import json
import yaml
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class Settings:
    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        self.config_file = config_file
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._config = {
            "agents": {
                "conductor": {
                    "max_concurrent_incidents": 10,
                    "incident_timeout_minutes": 120
                },
                "drift_detection": {
                    "monitoring_interval_minutes": 30,
                    "drift_threshold": 0.1,
                    "anomaly_threshold": 2.0
                },
                "diagnostic": {
                    "max_diagnosis_attempts": 3,
                    "confidence_threshold": 0.8
                },
                "remediation": {
                    "max_remediation_plans": 5,
                    "plan_timeout_minutes": 30,
                    "task_timeout_seconds": int(os.getenv("REMEDIATION_TASK_TIMEOUT", "1800"))
                },
                "economist": {
                    "cost_benefit_threshold": float(os.getenv("ECONOMIST_COST_BENEFIT", "1.2")),
                    "business_impact_weight": float(os.getenv("ECONOMIST_IMPACT_WEIGHT", "0.7")),
                    "max_concurrent_tasks": int(os.getenv("ECONOMIST_MAX_TASKS", "2")),
                    "task_timeout_seconds": int(os.getenv("ECONOMIST_TASK_TIMEOUT", "600"))
                },
                "verification": {
                    "rollout_strategy": os.getenv("VERIFICATION_ROLLOUT", "canary"),
                    "canary_percentage": int(os.getenv("VERIFICATION_CANARY_PCT", "10")),
                    "verification_timeout_minutes": int(os.getenv("VERIFICATION_TIMEOUT", "45")),
                    "max_concurrent_tasks": int(os.getenv("VERIFICATION_MAX_TASKS", "2")),
                    "task_timeout_seconds": int(os.getenv("VERIFICATION_TASK_TIMEOUT", "1200"))
                },
                "reporting": {
                    "report_format": os.getenv("REPORT_FORMAT", "markdown"),
                    "include_technical_details": os.getenv("INCLUDE_TECH_DETAILS", "true").lower() == "true",
                    "max_concurrent_tasks": int(os.getenv("REPORTING_MAX_TASKS", "3")),
                    "task_timeout_seconds": int(os.getenv("REPORTING_TASK_TIMEOUT", "300"))
                }
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                "file": os.getenv("LOG_FILE", None)
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            current = self._config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception:
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        self._set_nested_value(key_path, value)
    
    def get_gcp_config(self) -> Dict[str, Any]:
        """Get GCP configuration"""
        return self.get("gcp", {})
    
    def get_governance_config(self) -> Dict[str, Any]:
        """Get governance configuration"""
        return self.get("governance", {})
    
    def get_agents_config(self) -> Dict[str, Any]:
        """Get agents configuration"""
        return self.get("agents", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get("logging", {})
    
    def get_autonomy_controller_config(self) -> Dict[str, Any]:
        """Get autonomy controller configuration"""
        return self.get("governance.autonomy_controller", {})
    
    def get_rules_engine_config(self) -> Dict[str, Any]:
        """Get rules engine configuration"""
        return self.get("governance.rules_engine", {})
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini client configuration"""
        return self.get("services.gemini", {})
    
    def validate_governance_config(self) -> List[str]:
        """
        Validate governance-specific configuration.
        
        Returns:
            List of governance validation error messages
        """
        errors = []
        governance_config = self.get_governance_config()
        
        # Check required governance settings
        if not governance_config:
            errors.append("Governance configuration is missing")
            return errors
        
        # Validate autonomy controller config
        autonomy_config = self.get_autonomy_controller_config()
        if autonomy_config:
            max_pending = autonomy_config.get("max_pending_approvals")
            if max_pending is not None and (not isinstance(max_pending, int) or max_pending <= 0):
                errors.append("autonomy_controller.max_pending_approvals must be a positive integer")
            
            cleanup_interval = autonomy_config.get("cleanup_interval_minutes")
            if cleanup_interval is not None and (not isinstance(cleanup_interval, (int, float)) or cleanup_interval <= 0):
                errors.append("autonomy_controller.cleanup_interval_minutes must be a positive number")
        
        # Validate rules engine config
        rules_config = self.get_rules_engine_config()
        if rules_config:
            rule_timeout = rules_config.get("rule_timeout_seconds")
            if rule_timeout is not None and (not isinstance(rule_timeout, (int, float)) or rule_timeout <= 0):
                errors.append("rules_engine.rule_timeout_seconds must be a positive number")
            
            max_concurrent = rules_config.get("max_concurrent_evaluations")
            if max_concurrent is not None and (not isinstance(max_concurrent, int) or max_concurrent <= 0):
                errors.append("rules_engine.max_concurrent_evaluations must be a positive integer")
        
        return errors
    
    def validate_services_config(self) -> List[str]:
        """
        Validate services configuration.
        
        Returns:
            List of services validation error messages
        """
        errors = []
        
        # Validate Gemini configuration
        gemini_config = self.get_gemini_config()
        if gemini_config:
            api_key = gemini_config.get("api_key")
            if api_key and len(api_key) < 10:
                errors.append("services.gemini.api_key appears to be invalid (too short)")
            
            timeout = gemini_config.get("timeout_seconds")
            if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
                errors.append("services.gemini.timeout_seconds must be a positive number")
            
            max_retries = gemini_config.get("max_retries")
            if max_retries is not None and (not isinstance(max_retries, int) or max_retries < 0):
                errors.append("services.gemini.max_retries must be a non-negative integer")
            
            temperature = gemini_config.get("temperature")
            if temperature is not None and (not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2):
                errors.append("services.gemini.temperature must be between 0 and 2")
            
            top_p = gemini_config.get("top_p")
            if top_p is not None and (not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1):
                errors.append("services.gemini.top_p must be between 0 and 1")
        
        return errors
    
    def validate_config(self) -> List[str]:
        """
        Validate the configuration and return any errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate GCP configuration
        gcp_config = self.get_gcp_config()
        if not gcp_config.get("project_id"):
            errors.append("GCP project_id is required")
        
        if not gcp_config.get("region"):
            errors.append("GCP region is required")
        
        # Validate governance configuration
        governance_errors = self.validate_governance_config()
        errors.extend(governance_errors)
        
        # Validate services configuration
        services_errors = self.validate_services_config()
        errors.extend(services_errors)
        
        # Validate agent configurations
        agents_config = self.get_agents_config()
        for agent_name, agent_config in agents_config.items():
            if not isinstance(agent_config, dict):
                errors.append(f"Agent {agent_name} configuration must be a dictionary")
                continue
            
            # Validate max concurrent tasks
            max_tasks = agent_config.get("max_concurrent_tasks")
            if max_tasks is not None and (not isinstance(max_tasks, int) or max_tasks <= 0):
                errors.append(f"Agent {agent_name}.max_concurrent_tasks must be a positive integer")
            
            # Validate task timeout
            task_timeout = agent_config.get("task_timeout_seconds")
            if task_timeout is not None and (not isinstance(task_timeout, (int, float)) or task_timeout <= 0):
                errors.append(f"Agent {agent_name}.task_timeout_seconds must be a positive number")
        
        # Validate logging configuration
        logging_config = self.get_logging_config()
        log_level = logging_config.get("level")
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level and log_level not in valid_log_levels:
            errors.append(f"Invalid logging level: {log_level}. Must be one of: {', '.join(valid_log_levels)}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary"""
        return self._config.copy()
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
        """
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(self._config, f, indent=2)
                else:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_configuration()
        logger.info("Configuration reloaded")
    
    def get_environment(self) -> str:
        """Get the current environment"""
        return self.environment
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration"""
        return {
            "environment": self.environment,
            "config_file": self.config_file,
            "gcp_project": self.get("gcp.project_id"),
            "gcp_region": self.get("gcp.region"),
            "autonomy_level": self.get("governance.autonomy_level"),
            "agents_count": len(self.get_agents_config()),
            "validation_errors": len(self.validate_config()),
            "governance_enabled": bool(self.get_governance_config()),
            "services_configured": {
                "gemini": bool(self.get("services.gemini.api_key")),
            },
            "logging_level": self.get("logging.level"),
            "rules_engine_enabled": self.get("governance.rules_engine.enable_periodic_evaluation", False)
        }
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides"""
        env_config = {}
        
        if self.is_development():
            env_config.update({
                "services.gemini.enable_simulation": True,
                "governance.rules_engine.enable_periodic_evaluation": False,
                "logging.level": "DEBUG"
            })
        elif self.is_production():
            env_config.update({
                "services.gemini.enable_simulation": False,
                "governance.rules_engine.enable_periodic_evaluation": True,
                "logging.level": "INFO",
                "governance.autonomy_level": "SUPERVISED_AUTONOMY"
            })
        
        return env_config
    
    def apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        env_config = self.get_environment_specific_config()
        
        for key_path, value in env_config.items():
            current_value = self.get(key_path)
            if current_value is None:  # Only apply if not already set
                self.set(key_path, value)
                logger.info(f"Applied environment override: {key_path} = {value}")
    
    def validate_required_config(self) -> List[str]:
        """Validate that all required configuration is present"""
        errors = []
        required_configs = [
            "gcp.project_id",
            "gcp.region",
            "governance.autonomy_level"
        ]
        
        for config_path in required_configs:
            if not self.get(config_path):
                errors.append(f"Required configuration missing: {config_path}")
        
        # Environment-specific requirements
        if self.is_production():
            prod_required = [
                "services.gemini.api_key"
            ]
            for config_path in prod_required:
                if not self.get(config_path):
                    errors.append(f"Production environment requires: {config_path}")
        
        return errors
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        return self.get(f"agents.{agent_name}", {})
    
    def update_agent_config(self, agent_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific agent"""
        current_config = self.get_agent_config(agent_name)
        current_config.update(config_updates)
        self.set(f"agents.{agent_name}", current_config)
        logger.info(f"Updated configuration for agent: {agent_name}")

# Global settings instance
_settings_instance: Optional[Settings] = None

def get_settings(config_file: Optional[str] = None, environment: Optional[str] = None) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        config_file: Optional config file path
        environment: Optional environment name
        
    Returns:
        Settings instance
    """
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = Settings(config_file, environment)
    
    return _settings_instance

def reload_settings() -> None:
    """Reload the global settings instance"""
    global _settings_instance
    
    if _settings_instance:
        _settings_instance.reload()

def reset_settings() -> None:
    """Reset the global settings instance"""
    global _settings_instance
    _settings_instance = None
