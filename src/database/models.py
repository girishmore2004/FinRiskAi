"""
Database models for FinRiskAI+ platform
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import os
from typing import Optional, Dict, Any
import uuid

Base = declarative_base()

class SMEProfile(Base):
    """SME Business Profile Model"""
    __tablename__ = 'sme_profiles'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    business_name = Column(String(255), nullable=False)
    business_type = Column(String(100), nullable=False)
    industry_sector = Column(String(100), nullable=False)
    registration_number = Column(String(50), unique=True, nullable=False)
    gst_number = Column(String(15), unique=True)
    pan_number = Column(String(10), nullable=False)
    
    # Contact Information
    owner_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(15), nullable=False)
    address = Column(Text, nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    pincode = Column(String(10), nullable=False)
    
    # Business Metrics
    monthly_revenue = Column(Float, default=0.0)
    annual_revenue = Column(Float, default=0.0)
    employee_count = Column(Integer, default=1)
    years_in_business = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    credit_applications = relationship("CreditApplication", back_populates="sme_profile")
    financial_data = relationship("FinancialData", back_populates="sme_profile")
    risk_assessments = relationship("RiskAssessment", back_populates="sme_profile")

class CreditApplication(Base):
    """Credit Application Model"""
    __tablename__ = 'credit_applications'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sme_profile_id = Column(String, ForeignKey('sme_profiles.id'), nullable=False)
    
    # Loan Details
    loan_amount = Column(Float, nullable=False)
    loan_purpose = Column(String(255), nullable=False)
    loan_tenure_months = Column(Integer, nullable=False)
    collateral_type = Column(String(100))
    collateral_value = Column(Float, default=0.0)
    
    # Application Status
    status = Column(String(50), default='PENDING')  # PENDING, APPROVED, REJECTED, PROCESSING
    application_date = Column(DateTime, default=datetime.utcnow)
    decision_date = Column(DateTime)
    
    # AI Decision Details
    ai_score = Column(Float)  # 0-1000 scale
    ai_recommendation = Column(String(50))  # APPROVE, REJECT, REVIEW
    confidence_score = Column(Float)  # 0-1 scale
    explanation = Column(JSON)  # SHAP values and explanations
    
    # Risk Metrics
    probability_of_default = Column(Float)
    risk_grade = Column(String(10))  # A+, A, B+, B, C+, C, D
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sme_profile = relationship("SMEProfile", back_populates="credit_applications")
    risk_assessment = relationship("RiskAssessment", uselist=False, back_populates="credit_application")

class FinancialData(Base):
    """Financial Data Model"""
    __tablename__ = 'financial_data'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sme_profile_id = Column(String, ForeignKey('sme_profiles.id'), nullable=False)
    
    # Banking Data
    bank_account_number = Column(String(20), nullable=False)
    bank_name = Column(String(255), nullable=False)
    account_type = Column(String(50), nullable=False)
    
    # Financial Metrics (Monthly)
    monthly_inflow = Column(Float, default=0.0)
    monthly_outflow = Column(Float, default=0.0)
    average_balance = Column(Float, default=0.0)
    min_balance = Column(Float, default=0.0)
    max_balance = Column(Float, default=0.0)
    
    # Transaction Patterns
    transaction_count = Column(Integer, default=0)
    digital_transaction_ratio = Column(Float, default=0.0)
    recurring_income_ratio = Column(Float, default=0.0)
    
    # GST Data
    gst_monthly_filing = Column(Boolean, default=False)
    gst_annual_turnover = Column(Float, default=0.0)
    gst_compliance_score = Column(Float, default=0.0)
    
    # Data Collection Period
    data_from_date = Column(DateTime, nullable=False)
    data_to_date = Column(DateTime, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sme_profile = relationship("SMEProfile", back_populates="financial_data")

class RiskAssessment(Base):
    """Risk Assessment Model"""
    __tablename__ = 'risk_assessments'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sme_profile_id = Column(String, ForeignKey('sme_profiles.id'), nullable=False)
    credit_application_id = Column(String, ForeignKey('credit_applications.id'))
    
    # Risk Scores
    overall_risk_score = Column(Float, nullable=False)  # 0-1000 scale
    financial_risk_score = Column(Float, nullable=False)
    business_risk_score = Column(Float, nullable=False)
    market_risk_score = Column(Float, nullable=False)
    operational_risk_score = Column(Float, nullable=False)
    
    # Alternative Data Scores
    social_media_sentiment = Column(Float, default=0.0)
    online_reviews_score = Column(Float, default=0.0)
    digital_footprint_score = Column(Float, default=0.0)
    
    # Model Information
    model_version = Column(String(50), nullable=False)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Feature Importance
    feature_importance = Column(JSON)  # SHAP values for each feature
    
    # Predictions
    predicted_revenue_6m = Column(Float)
    predicted_revenue_12m = Column(Float)
    cash_flow_forecast = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sme_profile = relationship("SMEProfile", back_populates="risk_assessments")
    credit_application = relationship("CreditApplication", back_populates="risk_assessment")

class ModelMetrics(Base):
    """Model Performance Metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Performance Metrics
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    auc_roc = Column(Float, nullable=False)
    
    # Business Metrics
    approval_rate = Column(Float)
    default_rate = Column(Float)
    revenue_impact = Column(Float)
    
    # Training Information
    training_data_size = Column(Integer, nullable=False)
    training_duration_minutes = Column(Float)
    feature_count = Column(Integer)
    
    # Deployment Information
    deployment_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    """Audit Log for Blockchain Integration"""
    __tablename__ = 'audit_logs'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Transaction Details
    transaction_hash = Column(String(66))  # Ethereum transaction hash
    block_number = Column(Integer)
    
    # Application Details
    application_id = Column(String, nullable=False)
    action_type = Column(String(50), nullable=False)  # DECISION, UPDATE, REVIEW
    
    # Decision Details
    decision_data = Column(JSON, nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    blockchain_timestamp = Column(DateTime)
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verification_hash = Column(String(64))