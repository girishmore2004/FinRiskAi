"""
Explainable AI implementation using SHAP and LIME
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from ..base_model import BaseMLModel
import logging

logger = logging.getLogger(__name__)

class CreditExplainer:
    """Advanced explainable AI for credit decisions"""
    
    def __init__(self, model, feature_names: List[str], training_data: pd.DataFrame):
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        # Initialize SHAP explainer
        if hasattr(model, 'predict_proba'):
            self.shap_explainer = shap.TreeExplainer(model)
        else:
            self.shap_explainer = shap.KernelExplainer(model.predict_proba, training_data.sample(100))
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data.values,
            feature_names=feature_names,
            class_names=['Reject', 'Approve'],
            mode='classification'
        )
    
    def explain_prediction(self, instance: pd.DataFrame, explanation_type: str = 'both') -> Dict[str, Any]:
        """Generate comprehensive explanation for a single prediction"""
        
        explanation = {
            'prediction': self.model.predict(instance)[0],
            'probability': self.model.predict_proba(instance)[0],
            'credit_score': self._calculate_credit_score(instance),
            'risk_grade': self._get_risk_grade(instance)
        }
        
        if explanation_type in ['shap', 'both']:
            explanation['shap'] = self._get_shap_explanation(instance)
        
        if explanation_type in ['lime', 'both']:
            explanation['lime'] = self._get_lime_explanation(instance)
        
        explanation['counterfactuals'] = self._generate_counterfactuals(instance)
        explanation['feature_contributions'] = self._get_feature_contributions(instance)
        
        return explanation
    
    def _get_shap_explanation(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        shap_values = self.shap_explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class SHAP values
        
        # Get top contributing features
        feature_contributions = []
        for i, feature in enumerate(self.feature_names):
            contribution = {
                'feature': feature,
                'value': float(instance.iloc[0, i]),
                'shap_value': float(shap_values[0, i]),
                'impact': 'positive' if shap_values[0, i] > 0 else 'negative'
            }
            feature_contributions.append(contribution)
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'base_value': float(self.shap_explainer.expected_value[1]) if isinstance(self.shap_explainer.expected_value, list) else float(self.shap_explainer.expected_value),
            'prediction_value': float(np.sum(shap_values[0])),
            'feature_contributions': feature_contributions[:10],  # Top 10 features
            'explanation_text': self._generate_shap_text(feature_contributions[:5])
        }
    
    def _get_lime_explanation(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate LIME explanation"""
        lime_exp = self.lime_explainer.explain_instance(
            instance.values[0],
            self.model.predict_proba,
            num_features=10
        )
        
        # Extract feature contributions
        feature_contributions = []
        for feature_idx, contribution in lime_exp.as_list():
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
            feature_contributions.append({
                'feature': feature_name,
                'contribution': contribution,
                'impact': 'positive' if contribution > 0 else 'negative'
            })
        
        return {
            'feature_contributions': feature_contributions,
            'explanation_text': self._generate_lime_text(feature_contributions),
            'local_prediction': lime_exp.local_pred[1]
        }
    
    def _generate_counterfactuals(self, instance: pd.DataFrame, num_counterfactuals: int = 3) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        counterfactuals = []
        current_prediction = self.model.predict_proba(instance)[0, 1]
        
        # Define possible changes for key features
        feature_changes = {
            'monthly_revenue': [1.1, 1.2, 1.5, 2.0],  # Increase revenue
            'years_in_business': [1, 2, 3, 5],  # Add years
            'average_balance': [1.2, 1.5, 2.0],  # Increase balance
            'gst_compliance_score': [0.1, 0.2, 0.3],  # Improve compliance
            'digital_transaction_ratio': [0.1, 0.2, 0.3]  # Increase digital adoption
        }
        
        for feature, multipliers in feature_changes.items():
            if feature in instance.columns:
                for multiplier in multipliers:
                    modified_instance = instance.copy()
                    original_value = instance[feature].iloc[0]
                    
                    if feature in ['gst_compliance_score', 'digital_transaction_ratio']:
                        new_value = min(1.0, original_value + multiplier)
                    elif feature == 'years_in_business':
                        new_value = original_value + multiplier
                    else:
                        new_value = original_value * multiplier
                    
                    modified_instance[feature] = new_value
                    new_prediction = self.model.predict_proba(modified_instance)[0, 1]
                    
                    if abs(new_prediction - current_prediction) > 0.05:  # Significant change
                        counterfactual = {
                            'feature_changed': feature,
                            'original_value': float(original_value),
                            'new_value': float(new_value),
                            'change_description': self._describe_change(feature, original_value, new_value),
                            'original_probability': float(current_prediction),
                            'new_probability': float(new_prediction),
                            'probability_change': float(new_prediction - current_prediction),
                            'new_credit_score': int(300 + (900 - 300) * new_prediction)
                        }
                        counterfactuals.append(counterfactual)
                        
                        if len(counterfactuals) >= num_counterfactuals:
                            break
            
            if len(counterfactuals) >= num_counterfactuals:
                break
        
        # Sort by impact
        counterfactuals.sort(key=lambda x: abs(x['probability_change']), reverse=True)
        return counterfactuals[:num_counterfactuals]
    
    def _get_feature_contributions(self, instance: pd.DataFrame) -> Dict[str, float]:
        """Get simplified feature contributions"""
        shap_values = self.shap_explainer.shap_values(instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            contributions[feature] = float(shap_values[0, i])
        
        return contributions
    
    def _calculate_credit_score(self, instance: pd.DataFrame) -> int:
        """Calculate credit score from probability"""
        probability = self.model.predict_proba(instance)[0, 1]
        return int(300 + (900 - 300) * probability)
    
    def _get_risk_grade(self, instance: pd.DataFrame) -> str:
        """Get risk grade from credit score"""
        score = self._calculate_credit_score(instance)
        if score >= 800: return 'A+'
        elif score >= 750: return 'A'
        elif score >= 700: return 'B+'
        elif score >= 650: return 'B'
        elif score >= 600: return 'C+'
        elif score >= 550: return 'C'
        else: return 'D'
    
    def _generate_shap_text(self, contributions: List[Dict]) -> str:
        """Generate human-readable explanation from SHAP values"""
        text_parts = []
        
        for contrib in contributions[:3]:  # Top 3 factors
            feature = contrib['feature']
            impact = "positively" if contrib['impact'] == 'positive' else "negatively"
            text_parts.append(f"{feature} contributes {impact} to the decision")
        
        return ". ".join(text_parts) + "."
    
    def _generate_lime_text(self, contributions: List[Dict]) -> str:
        """Generate human-readable explanation from LIME values"""
        positive_factors = [c for c in contributions if c['impact'] == 'positive']
        negative_factors = [c for c in contributions if c['impact'] == 'negative']
        
        text = "Key factors supporting approval: " + ", ".join([f['feature'] for f in positive_factors[:3]])
        if negative_factors:
            text += ". Factors against approval: " + ", ".join([f['feature'] for f in negative_factors[:2]])
        
        return text
    
    def _describe_change(self, feature: str, original: float, new: float) -> str:
        """Describe the change in human-readable terms"""
        if feature == 'monthly_revenue':
            return f"Increase monthly revenue from ₹{original:,.0f} to ₹{new:,.0f}"
        elif feature == 'years_in_business':
            return f"Increase business experience from {original:.1f} to {new:.1f} years"
        elif feature == 'average_balance':
            return f"Maintain higher average balance of ₹{new:,.0f} (from ₹{original:,.0f})"
        elif feature == 'gst_compliance_score':
            return f"Improve GST compliance score to {new:.2f} (from {original:.2f})"
        elif feature == 'digital_transaction_ratio':
            return f"Increase digital transactions to {new:.1%} (from {original:.1%})"
        else:
            return f"Change {feature} from {original:.2f} to {new:.2f}"