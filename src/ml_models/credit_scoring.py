"""
Main credit scoring model implementation
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cb
from .base_model import BaseMLModel, FeatureEngineer, ModelEvaluator
import logging

logger = logging.getLogger(__name__)

class CreditScoringModel(BaseMLModel):
    """Advanced credit scoring model using ensemble methods"""
    
    def __init__(self, model_type: str = 'xgboost', version: str = "1.0.0"):
        super().__init__(f"credit_scoring_{model_type}", version)
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        # Create engineered features
        df_processed = self.feature_engineer.create_ratio_features(df)
        df_processed = self.feature_engineer.create_time_features(df_processed)
        df_processed = self.feature_engineer.handle_missing_values(df_processed)
        
        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in ['id', 'sme_profile_id', 'business_name', 'email', 'phone', 'address', 'created_at']:
                continue
            
            if is_training:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_processed[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    
                    for unknown_value in unique_values - known_values:
                        df_processed.loc[df_processed[col] == unknown_value, col] = 'Unknown'
                    
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Remove non-numeric columns
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed = df_processed[numeric_columns]
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.fillna(0)
        
        return df_processed
    
    def _create_model(self) -> Any:
        """Create the appropriate model based on model_type"""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == 'catboost':
            return cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """Train the credit scoring model"""
        logger.info(f"Training {self.model_name} with {len(X)} samples")
        
        # Prepare features
        X_processed = self._prepare_features(X, is_training=True)
        self.feature_names = list(X_processed.columns)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features for certain models
        if self.model_type in ['logistic_regression']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Create and train model
        self.model = self._create_model()
        
        # Train with early stopping for gradient boosting models
        if self.model_type in ['xgboost', 'lightgbm']:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred_train = self.model.predict(X_train_scaled)
        y_proba_train = self.model.predict_proba(X_train_scaled)
        y_pred_val = self.model.predict(X_val_scaled)
        y_proba_val = self.model.predict_proba(X_val_scaled)
        
        train_metrics = self.evaluator.calculate_classification_metrics(
            y_train, y_pred_train, y_proba_train[:, 1]
        )
        val_metrics = self.evaluator.calculate_classification_metrics(
            y_val, y_pred_val, y_proba_val[:, 1]
        )
        
        # Store training metadata
        self.training_metadata = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(self.feature_names),
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'feature_importance': self.get_feature_importance()
        }
        
        logger.info(f"Training completed. Validation AUC: {val_metrics.get('auc_roc', 0):.4f}")
        
        return self.training_metadata
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict credit decisions (0 = reject, 1 = approve)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_processed = self._prepare_features(X, is_training=False)
        X_processed = X_processed[self.feature_names]  # Ensure correct feature order
        
        if self.model_type in ['logistic_regression']:
            X_processed = self.scaler.transform(X_processed)
        
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of approval"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_processed = self._prepare_features(X, is_training=False)
        X_processed = X_processed[self.feature_names]
        
        if self.model_type in ['logistic_regression']:
            X_processed = self.scaler.transform(X_processed)
        
        return self.model.predict_proba(X_processed)
    
    def calculate_credit_score(self, X: pd.DataFrame, scale_range: Tuple[int, int] = (300, 900)) -> np.ndarray:
        """Convert probability to credit score (300-900 scale)"""
        probabilities = self.predict_proba(X)[:, 1]  # Probability of approval
        
        # Convert to credit score scale
        min_score, max_score = scale_range
        credit_scores = min_score + (max_score - min_score) * probabilities
        
        return credit_scores.astype(int)
    
    def get_risk_grade(self, credit_scores: np.ndarray) -> List[str]:
        """Convert credit scores to risk grades"""
        risk_grades = []
        
        for score in credit_scores:
            if score >= 800:
                risk_grades.append('A+')
            elif score >= 750:
                risk_grades.append('A')
            elif score >= 700:
                risk_grades.append('B+')
            elif score >= 650:
                risk_grades.append('B')
            elif score >= 600:
                risk_grades.append('C+')
            elif score >= 550:
                risk_grades.append('C')
            else:
                risk_grades.append('D')
        
        return risk_grades

class EnsembleCreditModel:
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self, models: List[str] = None):
        if models is None:
            models = ['xgboost', 'lightgbm', 'random_forest']
        
        self.models = {}
        self.weights = {}
        
        for model_type in models:
            self.models[model_type] = CreditScoringModel(model_type)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model_results = model.train(X, y, **kwargs)
            results[name] = model_results
            
            # Set weight based on validation AUC
            val_auc = model_results['validation_metrics'].get('auc_roc', 0.5)
            self.weights[name] = val_auc
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        return results
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction using weighted average"""
        ensemble_proba = None
        
        for name, model in self.models.items():
            model_proba = model.predict_proba(X)
            weight = self.weights[name]
            
            if ensemble_proba is None:
                ensemble_proba = weight * model_proba
            else:
                ensemble_proba += weight * model_proba
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def calculate_credit_score(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate credit scores using ensemble"""
        probabilities = self.predict_proba(X)[:, 1]
        credit_scores = 300 + (900 - 300) * probabilities
        return credit_scores.astype(int)