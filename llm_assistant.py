"""
LLM-Powered Maintenance Assistant for Aircraft Engine RUL Prediction
Uses Google Gemini to generate natural language insights from engine data.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from utils import setup_logging
import config

logger = setup_logging(__name__)


class MaintenanceAssistant:
    """
    AI-powered maintenance assistant using Google Gemini.
    
    Generates human-readable explanations, summaries, and answers
    to free-form questions about fleet health and engine degradation.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the LLM assistant.
        
        Args:
            api_key: Google Gemini API key. Falls back to GEMINI_API_KEY env var.
            model_name: Gemini model to use.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        self._initialized = False
        
        self.system_prompt = (
            "You are an expert aerospace maintenance engineer AI assistant. "
            "You analyze aircraft turbofan engine sensor data and Remaining Useful Life (RUL) predictions "
            "from the NASA C-MAPSS dataset. Your role is to:\n"
            "1. Interpret engine health data and RUL predictions clearly.\n"
            "2. Recommend actionable maintenance decisions.\n"
            "3. Explain failure modes (HPC degradation, fan degradation) in plain language.\n"
            "4. Prioritize safety above cost savings.\n"
            "Always cite specific sensor values or RUL numbers when making recommendations. "
            "Use bullet points for clarity. Be concise but thorough."
        )
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        if not self.api_key:
            logger.warning(
                "No GEMINI_API_KEY found. LLM features will use fallback summaries. "
                "Set GEMINI_API_KEY environment variable to enable AI-powered insights."
            )
            self._initialized = False
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt
            )
            self._initialized = True
            logger.info(f"Gemini LLM initialized with model: {self.model_name}")
        except ImportError:
            logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._initialized = False
    
    def _build_fleet_context(self, predictions_df: pd.DataFrame) -> str:
        """
        Build a structured text context from fleet predictions data.
        
        Args:
            predictions_df: DataFrame with unit_id and RUL_pred columns.
            
        Returns:
            Formatted context string.
        """
        total = len(predictions_df)
        avg_rul = predictions_df['RUL_pred'].mean()
        min_rul = predictions_df['RUL_pred'].min()
        max_rul = predictions_df['RUL_pred'].max()
        
        critical = predictions_df[predictions_df['RUL_pred'] < config.MAINTENANCE_THRESHOLDS['critical']]
        warning = predictions_df[
            (predictions_df['RUL_pred'] >= config.MAINTENANCE_THRESHOLDS['critical']) &
            (predictions_df['RUL_pred'] < config.MAINTENANCE_THRESHOLDS['warning'])
        ]
        healthy = predictions_df[predictions_df['RUL_pred'] >= config.MAINTENANCE_THRESHOLDS['warning']]
        
        context_parts = [
            f"=== FLEET HEALTH DATA ===",
            f"Total engines monitored: {total}",
            f"Average RUL: {avg_rul:.1f} cycles",
            f"Min RUL: {min_rul:.1f} cycles | Max RUL: {max_rul:.1f} cycles",
            f"",
            f"STATUS BREAKDOWN:",
            f"  🔴 CRITICAL (<{config.MAINTENANCE_THRESHOLDS['critical']} cycles): {len(critical)} engines",
            f"  🟡 WARNING ({config.MAINTENANCE_THRESHOLDS['critical']}-{config.MAINTENANCE_THRESHOLDS['warning']} cycles): {len(warning)} engines",
            f"  🟢 HEALTHY (≥{config.MAINTENANCE_THRESHOLDS['warning']} cycles): {len(healthy)} engines",
        ]
        
        # Add critical engine details
        if len(critical) > 0:
            context_parts.append("\nCRITICAL ENGINES (immediate attention):")
            for _, row in critical.nsmallest(10, 'RUL_pred').iterrows():
                context_parts.append(f"  Engine {int(row['unit_id'])}: RUL = {row['RUL_pred']:.1f} cycles")
        
        # Add warning engine details
        if len(warning) > 0:
            context_parts.append("\nWARNING ENGINES:")
            for _, row in warning.nsmallest(5, 'RUL_pred').iterrows():
                context_parts.append(f"  Engine {int(row['unit_id'])}: RUL = {row['RUL_pred']:.1f} cycles")
        
        return "\n".join(context_parts)
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the Gemini LLM with error handling and fallback.
        
        Args:
            prompt: The full prompt to send.
            
        Returns:
            Generated text response.
        """
        if not self._initialized:
            return self._fallback_response(prompt)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"⚠️ AI analysis unavailable ({e}). Using rule-based summary."
    
    def _fallback_response(self, prompt: str) -> str:
        """Generate a basic rule-based response when LLM is unavailable."""
        return (
            "⚠️ **AI Assistant Offline** — Gemini API key not configured.\n\n"
            "Set `GEMINI_API_KEY` environment variable to enable AI-powered insights.\n\n"
            "**Basic Rule-Based Summary:**\n"
            "- Engines with RUL < 30 cycles require immediate grounding.\n"
            "- Engines with RUL 30-80 cycles should be scheduled for maintenance.\n"
            "- Monitor sensor_11 (HPC outlet pressure) and sensor_4 (LPT temperature) for anomalies."
        )
    
    def generate_fleet_summary(self, predictions_df: pd.DataFrame) -> str:
        """
        Generate a natural language fleet health summary.
        
        Args:
            predictions_df: DataFrame with unit_id and RUL_pred columns.
            
        Returns:
            Natural language summary string.
        """
        context = self._build_fleet_context(predictions_df)
        
        prompt = (
            f"{context}\n\n"
            "Based on the fleet data above, provide a concise executive summary including:\n"
            "1. Overall fleet health assessment\n"
            "2. Immediate actions required for critical engines\n"
            "3. This week's recommended maintenance schedule\n"
            "4. Key risk factors to monitor\n"
            "5. Estimated cost impact of recommended actions"
        )
        
        return self._call_llm(prompt)
    
    def explain_engine_health(self, 
                              unit_id: int,
                              rul_pred: float,
                              sensor_data: pd.Series = None) -> str:
        """
        Generate a detailed explanation for a specific engine's health.
        
        Args:
            unit_id: Engine unit identifier.
            rul_pred: Predicted RUL in cycles.
            sensor_data: Latest sensor readings (optional).
            
        Returns:
            Natural language explanation.
        """
        context_parts = [
            f"=== ENGINE {unit_id} HEALTH REPORT ===",
            f"Predicted RUL: {rul_pred:.1f} cycles",
        ]
        
        if rul_pred < config.MAINTENANCE_THRESHOLDS['critical']:
            context_parts.append(f"Status: 🔴 CRITICAL")
        elif rul_pred < config.MAINTENANCE_THRESHOLDS['warning']:
            context_parts.append(f"Status: 🟡 WARNING")
        else:
            context_parts.append(f"Status: 🟢 HEALTHY")
        
        if sensor_data is not None:
            context_parts.append("\nLatest Sensor Readings:")
            for sensor, value in sensor_data.items():
                if 'sensor' in str(sensor):
                    context_parts.append(f"  {sensor}: {value:.4f}")
        
        context = "\n".join(context_parts)
        
        prompt = (
            f"{context}\n\n"
            "Provide a detailed health assessment for this engine including:\n"
            "1. What the RUL prediction means in operational terms\n"
            "2. Which sensors are showing concerning trends (if data provided)\n"
            "3. Likely failure mode (HPC degradation vs fan degradation)\n"
            "4. Recommended maintenance action and timeline\n"
            "5. Safety considerations"
        )
        
        return self._call_llm(prompt)
    
    def answer_question(self, 
                        question: str,
                        predictions_df: pd.DataFrame = None) -> str:
        """
        Answer a free-form question about the fleet or engine health.
        
        Args:
            question: User's question in natural language.
            predictions_df: Fleet predictions for context (optional).
            
        Returns:
            Natural language answer.
        """
        context = ""
        if predictions_df is not None:
            context = self._build_fleet_context(predictions_df) + "\n\n"
        
        prompt = (
            f"{context}"
            f"USER QUESTION: {question}\n\n"
            "Answer the question based on the fleet data provided. "
            "If specific data is not available, provide general aerospace maintenance guidance. "
            "Be specific and actionable in your response."
        )
        
        return self._call_llm(prompt)
    
    def generate_maintenance_report(self,
                                     predictions_df: pd.DataFrame,
                                     cost_analysis: Dict = None) -> str:
        """
        Generate a comprehensive maintenance report in natural language.
        
        Args:
            predictions_df: DataFrame with predictions.
            cost_analysis: Cost comparison data (optional).
            
        Returns:
            Formatted report string.
        """
        context = self._build_fleet_context(predictions_df)
        
        cost_context = ""
        if cost_analysis:
            cost_context = (
                f"\n\nCOST ANALYSIS:\n"
                f"  Traditional maintenance cost: ${cost_analysis.get('traditional_cost', 'N/A'):,}\n"
                f"  Predictive maintenance cost: ${cost_analysis.get('predictive_cost', 'N/A'):,}\n"
                f"  Cost savings: ${cost_analysis.get('savings', 'N/A'):,}\n"
                f"  Fleet availability improvement: {cost_analysis.get('availability_improvement', 'N/A')}%"
            )
        
        prompt = (
            f"{context}{cost_context}\n\n"
            "Generate a professional maintenance report suitable for an airline operations manager. "
            "Include:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Fleet Status Overview\n"
            "3. Critical Actions Required (with priority ranking)\n"
            "4. Weekly Maintenance Schedule Recommendation\n"
            "5. Cost-Benefit Analysis\n"
            "6. Risk Assessment\n"
            "Format with clear headers and bullet points."
        )
        
        return self._call_llm(prompt)
    
    def chat(self, message: str, chat_history: List[Dict] = None,
             predictions_df: pd.DataFrame = None) -> str:
        """
        Multi-turn chat interface for interactive queries.
        
        Args:
            message: User's message.
            chat_history: Previous conversation turns.
            predictions_df: Current fleet data for context.
            
        Returns:
            Assistant's response.
        """
        # Build conversation context
        context = ""
        if predictions_df is not None:
            context = self._build_fleet_context(predictions_df) + "\n\n"
        
        history_text = ""
        if chat_history:
            for turn in chat_history[-5:]:  # Keep last 5 turns for context
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                history_text += f"{role.upper()}: {content}\n"
        
        prompt = (
            f"{context}"
            f"CONVERSATION HISTORY:\n{history_text}\n"
            f"USER: {message}\n\n"
            "ASSISTANT: "
        )
        
        return self._call_llm(prompt)


# ============================================================
# Standalone usage
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Maintenance Assistant")
    print("=" * 60)
    
    # Create sample predictions
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'unit_id': range(1, 21),
        'RUL_pred': np.random.randint(5, 150, size=20).astype(float)
    })
    
    assistant = MaintenanceAssistant()
    
    # Test fleet summary
    print("\n--- Fleet Summary ---")
    summary = assistant.generate_fleet_summary(sample_df)
    print(summary)
    
    # Test engine explanation
    print("\n--- Engine Health Explanation ---")
    explanation = assistant.explain_engine_health(unit_id=1, rul_pred=12.5)
    print(explanation)
    
    # Test Q&A
    print("\n--- Question Answering ---")
    answer = assistant.answer_question(
        "Which engines need immediate maintenance?",
        predictions_df=sample_df
    )
    print(answer)
