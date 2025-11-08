"""
Base classes for all ML models in FinRiskAI+
"""
import joblib
import pickle
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseMLModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.training_metadata = {}
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data format"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if self.is_trained:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training order
            X = X[self.feature_names]
        
        return X
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'version': self.version,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data.get('model_name', self.model_name)
        self.version = model_data.get('version', self.version)
        self.feature_names = model_data.get('feature_names', [])
        self.is_trained = model_data.get('is_trained', False)
        self.training_metadata = model_data.get('training_metadata', {})
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (if supported by model)"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}

class FeatureEngineer:
    """Feature engineering utilities"""
    
    @staticmethod
    def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create financial ratio features"""
        df_new = df.copy()
        
        # Revenue ratios
        if 'monthly_revenue' in df.columns and 'annual_revenue' in df.columns:
            df_new['revenue_consistency'] = df['monthly_revenue'] * 12 / df['annual_revenue']
        
        # Balance ratios
        if all(col in df.columns for col in ['average_balance', 'monthly_inflow']):
            df_new['balance_to_inflow_ratio'] = df['average_balance'] / (df['monthly_inflow'] + 1e-6)
        
        if all(col in df.columns for col in ['min_balance', 'max_balance']):
            df_new['balance_volatility'] = (df['max_balance'] - df['min_balance']) / (df['average_balance'] + 1e-6)
        
        # Cash flow ratios
        if all(col in df.columns for col in ['monthly_inflow', 'monthly_outflow']):
            df_new['cash_flow_ratio'] = df['monthly_inflow'] / (df['monthly_outflow'] + 1e-6)
            df_new['net_cash_flow'] = df['monthly_inflow'] - df['monthly_outflow']
        
        # Business metrics
        if all(col in df.columns for col in ['annual_revenue', 'employee_count']):
            df_new['revenue_per_employee'] = df['annual_revenue'] / (df['employee_count'] + 1e-6)
        
        # Loan-specific ratios
        if 'loan_amount' in df.columns:
            if 'annual_revenue' in df.columns:
                df_new['loan_to_revenue_ratio'] = df['loan_amount'] / (df['annual_revenue'] + 1e-6)
            if 'monthly_revenue' in df.columns:
                df_new['loan_to_monthly_revenue'] = df['loan_amount'] / (df['monthly_revenue'] + 1e-6)
        
        return df_new
    
    @staticmethod
    def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        df_new = df.copy()
        
        # One-hot encode categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in ['id', 'sme_profile_id', 'business_name', 'email', 'phone', 'address']:
                continue  # Skip ID and personal info columns
            
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col)
            df_new = pd.concat([df_new, dummies], axis=1)
        
        return df_new
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, date_column: str = 'created_at') -> pd.DataFrame:
        """Create time-based features"""
        df_new = df.copy()
        
        if date_column in df.columns:
            df_new[date_column] = pd.to_datetime(df_new[date_column])
            
            # Extract time components
            df_new[f'{date_column}_year'] = df_new[date_column].dt.year
            df_new[f'{date_column}_month'] = df_new[date_column].dt.month
            df_new[f'{date_column}_day'] = df_new[date_column].dt.day
            df_new[f'{date_column}_dayofweek'] = df_new[date_column].dt.dayofweek
            df_new[f'{date_column}_quarter'] = df_new[date_column].dt.quarter
            
            # Business days
            df_new[f'{date_column}_is_weekend'] = df_new[date_column].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Days since epoch (for trend analysis)
            df_new[f'{date_column}_days_since_epoch'] = (df_new[date_column] - pd.Timestamp('1970-01-01')).dt.days
        
        return df_new
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values"""
        df_new = df.copy()
        
        for col in df_new.columns:
            if df_new[col].isnull().sum() > 0:
                if df_new[col].dtype in ['float64', 'int64']:
                    if strategy == 'median':
                        df_new[col].fillna(df_new[col].median(), inplace=True)
                    elif strategy == 'mean':
                        df_new[col].fillna(df_new[col].mean(), inplace=True)
                    else:
                        df_new[col].fillna(0, inplace=True)
                else:
                    df_new[col].fillna('Unknown', inplace=True)
        
        return df_new

class ModelEvaluator:
    """Model evaluation utilities"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate classification metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, loan_amounts: np.ndarray = None) -> Dict[str, float]:
        """Calculate business-specific metrics"""
        # Approval rate
        approval_rate = np.mean(y_pred)
        
        # Default rate (assuming 1 = default, 0 = no default)
        default_rate = np.mean(y_true)
        
        # Precision for approvals (how many approved loans actually don't default)
        approved_mask = y_pred == 1
        if np.sum(approved_mask) > 0:
            approval_precision = 1 - np.mean(y_true[approved_mask])  # 1 - default rate among approved
        else:
            approval_precision = 0
        
        metrics = {
            'approval_rate': approval_rate,
            'default_rate': default_rate,
            'approval_precision': approval_precision
        }
        
        if loan_amounts is not None:
            # Revenue impact (approved loan amounts minus defaulted amounts)
            approved_amounts = loan_amounts[y_pred == 1]
            defaulted_amounts = loan_amounts[(y_pred == 1) & (y_true == 1)]
            
            total_approved = np.sum(approved_amounts)
            total_defaulted = np.sum(defaulted_amounts)
            
            metrics['total_approved_amount'] = total_approved
            metrics['total_defaulted_amount'] = total_defaulted
            metrics['net_revenue_impact'] = total_approved - total_defaulted
        
        return metrics