"""
FastAPI application for FinRiskAI+ platform
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from sqlalchemy.orm import Session

# Import our custom modules
from ..database.connection import get_db
from ..database.models import SMEProfile, CreditApplication, FinancialData, RiskAssessment
from ..ml_models.credit_scoring import CreditScoringModel, EnsembleCreditModel
from ..ml_models.explainable_ai.explainer import CreditExplainer
from ..ml_models.forecasting.revenue_predictor import HybridRevenueForecaster
from ..ml_models.federated_learning.federated_model import FederatedLearningSystem
from ..ml_models.nlp_sentiment.multilingual_analyzer import MultilingualSentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="FinRiskAI+ API",
    description="Next-Generation AI Lending Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize ML models (should be loaded from saved models in production)
credit_model = None
explainer = None
revenue_forecaster = None
sentiment_analyzer = MultilingualSentimentAnalyzer()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# =============================================================================

class SMEProfileCreate(BaseModel):
    business_name: str
    business_type: str
    industry_sector: str
    registration_number: str
    gst_number: Optional[str] = None
    pan_number: str
    owner_name: str
    email: str
    phone: str
    address: str
    city: str
    state: str
    pincode: str
    monthly_revenue: float = 0.0
    annual_revenue: float = 0.0
    employee_count: int = 1
    years_in_business: float = 0.0

class CreditApplicationCreate(BaseModel):
    sme_profile_id: str
    loan_amount: float
    loan_purpose: str
    loan_tenure_months: int
    collateral_type: Optional[str] = None
    collateral_value: float = 0.0

class CreditDecisionResponse(BaseModel):
    application_id: str
    decision: str  # APPROVE, REJECT, REVIEW
    credit_score: int
    risk_grade: str
    probability_of_default: float
    confidence_score: float
    explanation: Dict[str, Any]
    recommendations: List[str]

class RevenueForecastRequest(BaseModel):
    sme_profile_id: str
    forecast_periods: int = 12

class RevenueForecastResponse(BaseModel):
    sme_profile_id: str
    forecast_periods: int
    predictions: List[float]
    confidence_intervals: Dict[str, List[float]]
    risk_assessment: Dict[str, Any]

class SentimentAnalysisRequest(BaseModel):
    business_texts: Dict[str, List[str]]

class SentimentAnalysisResponse(BaseModel):
    overall_sentiment: str
    confidence: float
    source_analysis: Dict[str, Any]
    language_breakdown: Dict[str, int]
    key_insights: List[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token - simplified for demo"""
    # In production, implement proper JWT validation
    if not credentials.token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.token

# =============================================================================
# API ROUTES - SME PROFILE MANAGEMENT
# =============================================================================

@app.post("/api/sme-profiles", response_model=Dict[str, Any])
async def create_sme_profile(
    profile: SMEProfileCreate,
    db: Session = Depends(get_db)
):
    """Create a new SME profile"""
    try:
        # Create new SME profile
        db_profile = SMEProfile(
            business_name=profile.business_name,
            business_type=profile.business_type,
            industry_sector=profile.industry_sector,
            registration_number=profile.registration_number,
            gst_number=profile.gst_number,
            pan_number=profile.pan_number,
            owner_name=profile.owner_name,
            email=profile.email,
            phone=profile.phone,
            address=profile.address,
            city=profile.city,
            state=profile.state,
            pincode=profile.pincode,
            monthly_revenue=profile.monthly_revenue,
            annual_revenue=profile.annual_revenue,
            employee_count=profile.employee_count,
            years_in_business=profile.years_in_business
        )
        
        db.add(db_profile)
        db.commit()
        db.refresh(db_profile)
        
        return {
            "message": "SME profile created successfully",
            "profile_id": db_profile.id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error creating SME profile: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/sme-profiles/{profile_id}")
async def get_sme_profile(
    profile_id: str,
    db: Session = Depends(get_db)
):
    """Get SME profile by ID"""
    profile = db.query(SMEProfile).filter(SMEProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return {
        "id": profile.id,
        "business_name": profile.business_name,
        "business_type": profile.business_type,
        "industry_sector": profile.industry_sector,
        "owner_name": profile.owner_name,
        "email": profile.email,
        "city": profile.city,
        "state": profile.state,
        "monthly_revenue": profile.monthly_revenue,
        "annual_revenue": profile.annual_revenue,
        "employee_count": profile.employee_count,
        "years_in_business": profile.years_in_business,
        "created_at": profile.created_at
    }

# =============================================================================
# API ROUTES - CREDIT SCORING
# =============================================================================

@app.post("/api/credit-applications", response_model=CreditDecisionResponse)
async def create_credit_application(
    application: CreditApplicationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create and process credit application"""
    try:
        # Get SME profile
        sme_profile = db.query(SMEProfile).filter(
            SMEProfile.id == application.sme_profile_id
        ).first()
        
        if not sme_profile:
            raise HTTPException(status_code=404, detail="SME profile not found")
        
        # Create credit application
        db_application = CreditApplication(
            sme_profile_id=application.sme_profile_id,
            loan_amount=application.loan_amount,
            loan_purpose=application.loan_purpose,
            loan_tenure_months=application.loan_tenure_months,
            collateral_type=application.collateral_type,
            collateral_value=application.collateral_value,
            status="PROCESSING"
        )
        
        db.add(db_application)
        db.commit()
        db.refresh(db_application)
        
        # Process with AI model
        credit_decision = await process_credit_application(
            db_application, sme_profile, db
        )
        
        # Add to background tasks for blockchain logging
        background_tasks.add_task(
            log_decision_to_blockchain,
            db_application.id,
            credit_decision
        )
        
        return credit_decision
        
    except Exception as e:
        logger.error(f"Error processing credit application: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def process_credit_application(
    application: CreditApplication,
    sme_profile: SMEProfile,
    db: Session
) -> CreditDecisionResponse:
    """Process credit application with AI models"""
    
    # Prepare feature data
    feature_data = prepare_credit_features(application, sme_profile, db)
    
    # Load or use pre-trained model
    global credit_model, explainer
    if credit_model is None:
        credit_model = load_credit_model()
        explainer = load_explainer(credit_model)
    
    # Make prediction
    prediction = credit_model.predict(feature_data)[0]
    probability = credit_model.predict_proba(feature_data)[0]
    credit_score = credit_model.calculate_credit_score(feature_data)[0]
    risk_grade = credit_model.get_risk_grade([credit_score])[0]
    
    # Generate explanation
    explanation = explainer.explain_prediction(feature_data)
    
    # Make final decision
    if credit_score >= 700 and probability[1] > 0.7:
        decision = "APPROVE"
    elif credit_score >= 600 and probability[1] > 0.5:
        decision = "REVIEW"
    else:
        decision = "REJECT"
    
    # Calculate confidence
    confidence_score = max(probability)
    probability_of_default = 1 - probability[1]
    
    # Generate recommendations
    recommendations = generate_recommendations(
        decision, explanation, sme_profile
    )
    
    # Update application in database
    application.ai_score = float(credit_score)
    application.ai_recommendation = decision
    application.confidence_score = float(confidence_score)
    application.probability_of_default = float(probability_of_default)
    application.risk_grade = risk_grade
    application.explanation = explanation
    application.status = decision if decision != "REVIEW" else "PROCESSING"
    application.decision_date = datetime.utcnow()
    
    db.commit()
    
    return CreditDecisionResponse(
        application_id=application.id,
        decision=decision,
        credit_score=credit_score,
        risk_grade=risk_grade,
        probability_of_default=probability_of_default,
        confidence_score=confidence_score,
        explanation=explanation,
        recommendations=recommendations
    )

def prepare_credit_features(
    application: CreditApplication,
    sme_profile: SMEProfile,
    db: Session
) -> pd.DataFrame:
    """Prepare features for credit scoring model"""
    
    # Get financial data
    financial_data = db.query(FinancialData).filter(
        FinancialData.sme_profile_id == sme_profile.id
    ).order_by(FinancialData.created_at.desc()).first()
    
    # Create feature dictionary
    features = {
        'monthly_revenue': sme_profile.monthly_revenue,
        'annual_revenue': sme_profile.annual_revenue,
        'employee_count': sme_profile.employee_count,
        'years_in_business': sme_profile.years_in_business,
        'loan_amount': application.loan_amount,
        'loan_tenure_months': application.loan_tenure_months,
        'collateral_value': application.collateral_value,
        'has_collateral': 1 if application.collateral_value > 0 else 0,
        'loan_to_revenue_ratio': application.loan_amount / (sme_profile.annual_revenue + 1e-6),
        'revenue_per_employee': sme_profile.annual_revenue / (sme_profile.employee_count + 1e-6)
    }
    
    # Add financial features if available
    if financial_data:
        features.update({
            'monthly_inflow': financial_data.monthly_inflow,
            'monthly_outflow': financial_data.monthly_outflow,
            'average_balance': financial_data.average_balance,
            'transaction_count': financial_data.transaction_count,
            'digital_transaction_ratio': financial_data.digital_transaction_ratio,
            'gst_compliance_score': financial_data.gst_compliance_score,
            'cash_flow_ratio': financial_data.monthly_inflow / (financial_data.monthly_outflow + 1e-6),
            'balance_to_revenue_ratio': financial_data.average_balance / (sme_profile.monthly_revenue + 1e-6)
        })
    else:
        # Default values if no financial data
        default_financial = {
            'monthly_inflow': sme_profile.monthly_revenue,
            'monthly_outflow': sme_profile.monthly_revenue * 0.8,
            'average_balance': sme_profile.monthly_revenue * 2,
            'transaction_count': 50,
            'digital_transaction_ratio': 0.5,
            'gst_compliance_score': 0.7,
            'cash_flow_ratio': 1.25,
            'balance_to_revenue_ratio': 2.0
        }
        features.update(default_financial)
    
    # Add categorical features
    industry_categories = {
        'Manufacturing': [1, 0, 0, 0],
        'Retail': [0, 1, 0, 0],
        'Services': [0, 0, 1, 0],
        'Technology': [0, 0, 0, 1]
    }
    
    industry_encoding = industry_categories.get(sme_profile.industry_sector, [0, 0, 0, 0])
    features.update({
        'industry_manufacturing': industry_encoding[0],
        'industry_retail': industry_encoding[1],
        'industry_services': industry_encoding[2],
        'industry_technology': industry_encoding[3]
    })
    
    # City risk score (simplified)
    city_risk_scores = {
        'Mumbai': 0.9, 'Delhi': 0.85, 'Bangalore': 0.8, 'Chennai': 0.75,
        'Hyderabad': 0.7, 'Pune': 0.7, 'Kolkata': 0.65
    }
    features['city_business_score'] = city_risk_scores.get(sme_profile.city, 0.5)
    
    return pd.DataFrame([features])

def generate_recommendations(
    decision: str,
    explanation: Dict[str, Any],
    sme_profile: SMEProfile
) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if decision == "REJECT":
        recommendations.extend([
            "Improve business revenue by focusing on core profitable activities",
            "Build a stronger financial track record with consistent cash flows",
            "Consider providing additional collateral to reduce lending risk",
            "Improve GST compliance and maintain regular filing records"
        ])
    elif decision == "REVIEW":
        recommendations.extend([
            "Provide additional financial documentation to support the application",
            "Consider a co-applicant or guarantor to strengthen the application",
            "Demonstrate stable business operations over the next few months"
        ])
    else:  # APPROVE
        recommendations.extend([
            "Maintain current business performance and financial discipline",
            "Consider setting up automatic loan repayment to avoid any delays",
            "Keep building your business credit history for future financing needs"
        ])
    
    # Add specific recommendations based on explanation
    if 'counterfactuals' in explanation:
        for cf in explanation['counterfactuals'][:2]:
            recommendations.append(cf['change_description'])
    
    return recommendations[:5]  # Return top 5 recommendations

def load_credit_model():
    """Load pre-trained credit scoring model"""
    # In production, load from saved model file
    model = CreditScoringModel(model_type='xgboost')
    # model.load_model('models/credit_scoring_model.pkl')
    return model

def load_explainer(model):
    """Load explainer for the model"""
    # In production, load with actual training data
    training_data = pd.DataFrame()  # Load actual training data
    explainer = CreditExplainer(model, model.feature_names, training_data)
    return explainer

# =============================================================================
# API ROUTES - REVENUE FORECASTING
# =============================================================================

@app.post("/api/revenue-forecast", response_model=RevenueForecastResponse)
async def generate_revenue_forecast(
    request: RevenueForecastRequest,
    db: Session = Depends(get_db)
):
    """Generate revenue forecast for SME"""
    try:
        # Get SME profile
        sme_profile = db.query(SMEProfile).filter(
            SMEProfile.id == request.sme_profile_id
        ).first()
        
        if not sme_profile:
            raise HTTPException(status_code=404, detail="SME profile not found")
        
        # Get historical financial data
        financial_data = db.query(FinancialData).filter(
            FinancialData.sme_profile_id == request.sme_profile_id
        ).order_by(FinancialData.data_from_date).all()
        
        if not financial_data:
            raise HTTPException(status_code=404, detail="No financial data available")
        
        # Prepare historical data
        historical_df = pd.DataFrame([{
            'date': fd.data_from_date,
            'revenue': fd.monthly_inflow
        } for fd in financial_data])
        
        # Initialize and use forecaster
        global revenue_forecaster
        if revenue_forecaster is None:
            revenue_forecaster = HybridRevenueForecaster()
            # In production, load pre-trained model
            # revenue_forecaster.load_model('models/revenue_forecaster.pkl')
        
        # Generate predictions
        predictions = revenue_forecaster.predict(
            historical_df, periods=request.forecast_periods
        )
        
        # Calculate confidence intervals
        confidence_intervals = revenue_forecaster.calculate_confidence_intervals(predictions)
        
        # Assess forecast risk
        current_revenue = sme_profile.monthly_revenue
        risk_assessment = revenue_forecaster.assess_forecast_risk(predictions, current_revenue)
        
        return RevenueForecastResponse(
            sme_profile_id=request.sme_profile_id,
            forecast_periods=request.forecast_periods,
            predictions=predictions.tolist(),
            confidence_intervals={
                'lower_bound': confidence_intervals['lower_bound'].tolist(),
                'upper_bound': confidence_intervals['upper_bound'].tolist()
            },
            risk_assessment=risk_assessment
        )
        
    except Exception as e:
        logger.error(f"Error generating revenue forecast: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# =============================================================================
# API ROUTES - SENTIMENT ANALYSIS
# =============================================================================

@app.post("/api/sentiment-analysis", response_model=SentimentAnalysisResponse)
async def analyze_business_sentiment(
    request: SentimentAnalysisRequest
):
    """Analyze business reputation from text sources"""
    try:
        # Use global sentiment analyzer
        global sentiment_analyzer
        
        # Analyze reputation
        reputation_analysis = sentiment_analyzer.analyze_business_reputation(
            request.business_texts
        )
        
        return SentimentAnalysisResponse(
            overall_sentiment=reputation_analysis['overall_sentiment'],
            confidence=reputation_analysis['confidence'],
            source_analysis=reputation_analysis['source_analysis'],
            language_breakdown=reputation_analysis['language_breakdown'],
            key_insights=reputation_analysis['key_insights']
        )
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# =============================================================================
# API ROUTES - DASHBOARD AND ANALYTICS
# =============================================================================

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db)
):
    """Get dashboard statistics"""
    try:
        # Total applications
        total_applications = db.query(CreditApplication).count()
        
        # Applications by status
        approved = db.query(CreditApplication).filter(
            CreditApplication.status == "APPROVED"
        ).count()
        
        rejected = db.query(CreditApplication).filter(
            CreditApplication.status == "REJECTED"
        ).count()
        
        processing = db.query(CreditApplication).filter(
            CreditApplication.status == "PROCESSING"
        ).count()
        
        # Total loan amount
        total_loan_amount = db.query(
            db.func.sum(CreditApplication.loan_amount)
        ).filter(
            CreditApplication.status == "APPROVED"
        ).scalar() or 0
        
        # Average credit score
        avg_credit_score = db.query(
            db.func.avg(CreditApplication.ai_score)
        ).scalar() or 0
        
        # Recent applications
        recent_applications = db.query(CreditApplication).order_by(
            CreditApplication.created_at.desc()
        ).limit(10).all()
        
        return {
            "total_applications": total_applications,
            "approved": approved,
            "rejected": rejected,
            "processing": processing,
            "approval_rate": approved / total_applications if total_applications > 0 else 0,
            "total_loan_amount": float(total_loan_amount),
            "average_credit_score": float(avg_credit_score),
            "recent_applications": [
                {
                    "id": app.id,
                    "business_name": app.sme_profile.business_name,
                    "loan_amount": app.loan_amount,
                    "status": app.status,
                    "credit_score": app.ai_score,
                    "created_at": app.created_at
                }
                for app in recent_applications
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def log_decision_to_blockchain(application_id: str, decision_data: Dict[str, Any]):
    """Log credit decision to blockchain (simplified implementation)"""
    try:
        # In production, implement actual blockchain logging
        logger.info(f"Logging decision for application {application_id} to blockchain")
        
        # Create audit log entry
        audit_entry = {
            'application_id': application_id,
            'decision': decision_data.get('decision'),
            'credit_score': decision_data.get('credit_score'),
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': '1.0.0'
        }
        
        # Store in audit log (simplified)
        with open('logs/blockchain_audit.json', 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Error logging to blockchain: {e}")