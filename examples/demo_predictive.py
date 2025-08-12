"""
Predictive Sentinel AI Demo - Proactive Incident Prevention

This demo shows how Sentinel AI can predict and prevent incidents before they occur,
transforming from reactive to proactive MLOps management.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import numpy as np

# Setup path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.predictive_agent import PredictiveAgent, PredictiveAlert, PredictionConfidence
from models.incident import IncidentType, IncidentSeverity
from utils.logging_setup import setup_logging


class PredictiveDemo:
    """Demo class for predictive capabilities"""
    
    def __init__(self):
        self.logger = setup_logging("predictive_demo")
        self.config = {
            "gemini": {
                "api_key": "demo_key",
                "model": "gemini-pro"
            }
        }
        
    async def run_predictive_demo(self):
        """Run comprehensive predictive demo scenarios"""
        
        print("🔮 " + "="*60)
        print("🔮 SENTINEL AI - PREDICTIVE MLOPS DEMO")
        print("🔮 " + "="*60)
        print()
        
        # Initialize predictive agent
        predictive_agent = PredictiveAgent(self.config)
        await predictive_agent.initialize()
        
        print("✅ Predictive Agent initialized")
        print("🔍 Analyzing historical patterns for future incident prediction...")
        print()
        
        # Run different predictive scenarios
        await self._demo_data_drift_prediction(predictive_agent)
        await self._demo_performance_prediction(predictive_agent)
        await self._demo_resource_prediction(predictive_agent)
        await self._demo_model_staleness_prediction(predictive_agent)
        
        print("🎯 " + "="*60)
        print("🎯 PREDICTIVE ANALYSIS COMPLETE")
        print("🎯 " + "="*60)
        
    async def _demo_data_drift_prediction(self, agent: PredictiveAgent):
        """Demo data drift prediction scenario"""
        
        print("📊 SCENARIO 1: Data Drift Prediction")
        print("-" * 40)
        
        # Simulate historical feature statistics showing drift acceleration
        metrics_data = {
            "feature_statistics": {
                "user_age": {
                    "current_drift": 0.18,
                    "drift_history": [0.05, 0.08, 0.12, 0.15, 0.18],  # Accelerating drift
                    "baseline_mean": 35.2,
                    "current_mean": 38.7
                },
                "purchase_amount": {
                    "current_drift": 0.22,
                    "drift_history": [0.08, 0.13, 0.17, 0.20, 0.22],  # Critical drift
                    "baseline_mean": 127.50,
                    "current_mean": 156.80
                }
            }
        }
        
        alerts = await agent.analyze_predictive_patterns(metrics_data)
        drift_alerts = [a for a in alerts if a.predicted_incident_type == IncidentType.DATA_DRIFT]
        
        for alert in drift_alerts:
            self._display_predictive_alert(alert)
            
        print()
        
    async def _demo_performance_prediction(self, agent: PredictiveAgent):
        """Demo performance degradation prediction"""
        
        print("⚡ SCENARIO 2: Performance Degradation Prediction")
        print("-" * 40)
        
        # Simulate degrading performance metrics
        metrics_data = {
            "performance_metrics": {
                "inference_latency": [0.85, 0.87, 0.89, 0.92, 0.95, 0.98, 1.02, 1.05],  # Degrading
                "accuracy": [0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87],  # Declining
                "throughput": [1000, 980, 960, 940, 920, 900, 880, 860]  # Decreasing
            }
        }
        
        alerts = await agent.analyze_predictive_patterns(metrics_data)
        perf_alerts = [a for a in alerts if a.predicted_incident_type == IncidentType.PERFORMANCE_DEGRADATION]
        
        for alert in perf_alerts:
            self._display_predictive_alert(alert)
            
        print()
        
    async def _demo_resource_prediction(self, agent: PredictiveAgent):
        """Demo resource exhaustion prediction"""
        
        print("💾 SCENARIO 3: Resource Exhaustion Prediction")
        print("-" * 40)
        
        # Simulate growing resource usage
        metrics_data = {
            "resource_usage": {
                "memory": [0.65, 0.68, 0.72, 0.75, 0.78, 0.82, 0.85, 0.88],  # Growing toward limit
                "cpu": [0.70, 0.72, 0.75, 0.77, 0.80, 0.83, 0.86, 0.89],  # High usage trend
                "disk": [0.60, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81]   # Steady growth
            }
        }
        
        alerts = await agent.analyze_predictive_patterns(metrics_data)
        resource_alerts = [a for a in alerts if a.predicted_incident_type == IncidentType.SYSTEM_ALERT]
        
        for alert in resource_alerts:
            self._display_predictive_alert(alert)
            
        print()
        
    async def _demo_model_staleness_prediction(self, agent: PredictiveAgent):
        """Demo model staleness prediction"""
        
        print("🤖 SCENARIO 4: Model Staleness Prediction")
        print("-" * 40)
        
        # Simulate aging model performance
        metrics_data = {
            "model_performance": {
                "fraud_detection_v2": {
                    "last_training_date": datetime.now() - timedelta(days=45),
                    "accuracy_history": [0.94, 0.93, 0.92, 0.90, 0.88, 0.86],  # Declining
                    "precision_history": [0.91, 0.90, 0.88, 0.86, 0.84, 0.82],
                    "recall_history": [0.89, 0.88, 0.86, 0.84, 0.82, 0.80]
                },
                "recommendation_engine_v3": {
                    "last_training_date": datetime.now() - timedelta(days=28),
                    "accuracy_history": [0.87, 0.86, 0.85, 0.83, 0.81, 0.79],  # Steady decline
                    "click_through_rate": [0.12, 0.11, 0.10, 0.09, 0.08, 0.07]
                }
            }
        }
        
        alerts = await agent.analyze_predictive_patterns(metrics_data)
        staleness_alerts = [a for a in alerts if a.predicted_incident_type == IncidentType.MODEL_DRIFT]
        
        for alert in staleness_alerts:
            self._display_predictive_alert(alert)
            
        print()
        
    def _display_predictive_alert(self, alert: PredictiveAlert):
        """Display a predictive alert in a formatted way"""
        
        # Color coding based on confidence
        confidence_colors = {
            PredictionConfidence.LOW: "🟡",
            PredictionConfidence.MEDIUM: "🟠", 
            PredictionConfidence.HIGH: "🔴",
            PredictionConfidence.CRITICAL: "🚨"
        }
        
        severity_colors = {
            IncidentSeverity.LOW: "🟢",
            IncidentSeverity.MEDIUM: "🟡",
            IncidentSeverity.HIGH: "🟠",
            IncidentSeverity.CRITICAL: "🔴"
        }
        
        print(f"{confidence_colors[alert.confidence]} PREDICTIVE ALERT")
        print(f"   Alert ID: {alert.alert_id}")
        print(f"   Predicted Incident: {alert.predicted_incident_type.value}")
        print(f"   Severity: {severity_colors[alert.predicted_severity]} {alert.predicted_severity.value}")
        print(f"   Confidence: {alert.confidence.value}")
        print(f"   Time to Incident: {self._format_timedelta(alert.time_to_incident)}")
        print(f"   Prevention Probability: {alert.prevention_probability*100:.1f}%")
        print()
        
        print("   🔍 Contributing Factors:")
        for factor in alert.contributing_factors:
            print(f"      • {factor}")
        print()
        
        print("   🛠️  Recommended Prevention Actions:")
        for i, action in enumerate(alert.recommended_actions, 1):
            print(f"      {i}. {action}")
        print()
        
        # Show prevention urgency
        urgency = self._calculate_urgency(alert)
        print(f"   ⚡ Urgency Level: {urgency}")
        print("   " + "-" * 50)
        print()
        
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta for display"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
            
    def _calculate_urgency(self, alert: PredictiveAlert) -> str:
        """Calculate urgency level for display"""
        hours_to_incident = alert.time_to_incident.total_seconds() / 3600
        
        if alert.confidence == PredictionConfidence.CRITICAL and hours_to_incident < 2:
            return "🚨 IMMEDIATE ACTION REQUIRED"
        elif alert.confidence in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL] and hours_to_incident < 6:
            return "🔴 HIGH URGENCY - Act within hours"
        elif hours_to_incident < 24:
            return "🟠 MEDIUM URGENCY - Act within day"
        else:
            return "🟡 LOW URGENCY - Plan prevention"


async def main():
    """Main demo function"""
    demo = PredictiveDemo()
    
    try:
        await demo.run_predictive_demo()
        
        print("🎉 PREDICTIVE DEMO COMPLETED!")
        print()
        print("💡 Key Benefits of Predictive MLOps:")
        print("   • Prevent incidents before they occur")
        print("   • Reduce mean time to resolution (MTTR)")
        print("   • Minimize business impact from ML failures")
        print("   • Enable proactive resource planning")
        print("   • Improve overall system reliability")
        print()
        print("🚀 Your Sentinel AI system is now PREDICTIVE!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
