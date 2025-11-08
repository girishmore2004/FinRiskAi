"""
Complete testing suite for FinRiskAI+ system
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import requests_mock
import tempfile
import os

# Import all modules to test
from src.ml_models.credit_scoring import CreditScoringModel
from src.ml_models.explainable_ai.explainer import CreditExplainer
from src.ml_models.forecasting.revenue_predictor import HybridRevenueForecaster
from src.data_ingestion.synthetic_data import SyntheticSMEDataGenerator
from src.blockchain.audit_logger import BlockchainAuditLogger

class TestCompleteSMEPlatform:
    """Comprehensive test suite for the entire platform"""
    
    @pytest.fixture
    def sample_sme_data(self):
        """Generate sample SME data for testing"""
        generator = SyntheticSMEDataGenerator(seed=42)
        return generator.generate_complete_dataset(sme_count=100, app_count=50)
    
    @pytest.fixture
    def trained_credit_model(self, sample_sme_data):
        """Fixture for trained credit scoring model"""
        model = CreditScoringModel(model_type='xgboost')
        
        # Prepare training data
        applications = sample_sme_data['credit_applications']
        sme_profiles = sample_sme_data['sme_profiles']
        
        # Merge data for features
        merged_data = applications.merge(sme_profiles, left_on='sme_profile_id', right_on='id')
        
        # Create target variable (1 for approved, 0 for rejected/review)
        y = (merged_data['ai_recommendation'] == 'APPROVE').astype(int)
        
        # Select feature columns
        feature_cols = ['monthly_revenue', 'annual_revenue', 'years_in_business', 
                       'employee_count', 'loan_amount', 'loan_tenure_months']
        X = merged_data[feature_cols]
        
        # Train model
        model.train(X, y)
        return model
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        generator = SyntheticSMEDataGenerator(seed=42)
        datasets = generator.generate_complete_dataset(sme_count=10, app_count=5)
        
        # Test data structure
        assert 'sme_profiles' in datasets
        assert 'financial_data' in datasets
        assert 'credit_applications' in datasets
        
        # Test data quality
        sme_profiles = datasets['sme_profiles']
        assert len(sme_profiles) == 10
        assert all(sme_profiles['monthly_revenue'] > 0)
        assert all(sme_profiles['years_in_business'] >= 0)
        
        credit_applications = datasets['credit_applications']
        assert len(credit_applications) == 5
        assert all(credit_applications['loan_amount'] > 0)
    
    def test_credit_scoring_model(self, trained_credit_model, sample_sme_data):
        """Test credit scoring model functionality"""
        model = trained_credit_model
        
        # Test prediction
        test_data = sample_sme_data['sme_profiles'].iloc[:5]
        feature_cols = ['monthly_revenue', 'annual_revenue', 'years_in_business', 
                       'employee_count']
        
        # Add missing columns with defaults
        test_features = test_data[feature_cols].copy()
        test_features['loan_amount'] = 500000
        test_features['loan_tenure_months'] = 24
        
        # Test predictions
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        credit_scores = model.calculate_credit_score(test_features)
        
        assert len(predictions) == 5
        assert len(probabilities) == 5
        assert len(credit_scores) == 5
        assert all(300 <= score <= 900 for score in credit_scores)
    
    def test_explainable_ai(self, trained_credit_model, sample_sme_data):
        """Test explainable AI functionality"""
        model = trained_credit_model
        training_data = sample_sme_data['sme_profiles'].iloc[:20]
        
        explainer = CreditExplainer(model, model.feature_names, training_data)
        
        # Test explanation generation
        test_instance = training_data.iloc[[0]]
        explanation = explainer.explain_prediction(test_instance)
        
        assert 'prediction' in explanation
        assert 'probability' in explanation
        assert 'credit_score' in explanation
        assert 'counterfactuals' in explanation
        assert len(explanation['counterfactuals']) > 0
    
    def test_revenue_forecasting(self):
        """Test revenue forecasting functionality"""
        forecaster = HybridRevenueForecaster(sequence_length=6)
        
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        revenues = np.random.normal(100000, 10000, 12)
        
        historical_data = pd.DataFrame({
            'date': dates,
            'revenue': revenues
        })
        
        # Test training (simplified)
        X_lstm, y_lstm = forecaster.prepare_lstm_data(historical_data)
        assert len(X_lstm) == len(y_lstm)
        assert X_lstm.shape[1] == 6  # sequence_length
        
        # Test prediction structure
        predictions = np.random.normal(100000, 5000, 6)  # Mock predictions
        confidence_intervals = forecaster.calculate_confidence_intervals(predictions)
        
        assert 'lower_bound' in confidence_intervals
        assert 'upper_bound' in confidence_intervals
        assert len(confidence_intervals['lower_bound']) == 6
    
    def test_blockchain_audit_logger(self):
        """Test blockchain audit logging"""
        logger = BlockchainAuditLogger()
        
        # Test audit record creation
        decision_data = {
            'decision': 'APPROVE',
            'credit_score': 750,
            'risk_grade': 'A',
            'confidence_score': 0.85,
            'explanation': {'test': 'data'}
        }
        
        # Test logging
        transaction_id = logger.log_credit_decision('test_app_123', decision_data)
        assert transaction_id is not None
        
        # Test verification
        verification = logger.verify_transaction(transaction_id)
        assert verification['found'] == True
        assert verification['transaction_id'] == transaction_id
    
    def test_api_endpoints_integration(self):
        """Test API endpoints with mocked responses"""
        with requests_mock.Mocker() as m:
            # Mock API responses
            m.post('http://localhost:8000/api/sme-profiles', 
                  json={'profile_id': 'test_123', 'status': 'success'})
            
            m.post('http://localhost:8000/api/credit-applications',
                  json={
                      'application_id': 'app_123',
                      'decision': 'APPROVE',
                      'credit_score': 750,
                      'risk_grade': 'A',
                      'recommendations': ['Maintain current performance']
                  })
            
            m.get('http://localhost:8000/api/dashboard/stats',
                 json={
                     'total_applications': 100,
                     'approved': 60,
                     'rejected': 30,
                     'processing': 10,
                     'approval_rate': 0.6
                 })
            
            # Test the mocked endpoints
            import requests
            
            # Test profile creation
            profile_response = requests.post(
                'http://localhost:8000/api/sme-profiles',
                json={'business_name': 'Test Business'}
            )
            assert profile_response.status_code == 200
            assert profile_response.json()['status'] == 'success'
            
            # Test application processing
            app_response = requests.post(
                'http://localhost:8000/api/credit-applications',
                json={'sme_profile_id': 'test_123', 'loan_amount': 500000}
            )
            assert app_response.status_code == 200
            assert app_response.json()['decision'] == 'APPROVE'
    
    def test_data_pipeline_integration(self, sample_sme_data):
        """Test complete data pipeline"""
        # Test data flow through the system
        sme_profiles = sample_sme_data['sme_profiles']
        financial_data = sample_sme_data['financial_data']
        applications = sample_sme_data['credit_applications']
        
        # Test data consistency
        profile_ids = set(sme_profiles['id'])
        financial_profile_ids = set(financial_data['sme_profile_id'])
        app_profile_ids = set(applications['sme_profile_id'])
        
        # All financial data should reference valid profiles
        assert financial_profile_ids.issubset(profile_ids)
        # All applications should reference valid profiles
        assert app_profile_ids.issubset(profile_ids)
        
        # Test data quality checks
        assert all(sme_profiles['monthly_revenue'] >= 0)
        assert all(applications['loan_amount'] > 0)
        assert all(financial_data['gst_compliance_score'] <= 1.0)