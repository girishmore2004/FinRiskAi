"""
Streamlit Web Dashboard for FinRiskAI+
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="FinRiskAI+ Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000/api"

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left-color: #2ecc71;
    }
    .warning-card {
        border-left-color: #f39c12;
    }
    .danger-card {
        border-left-color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏦 FinRiskAI+")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard", "Credit Applications", "SME Profiles", "Analytics", "Model Performance"]
)

# =============================================================================
# DASHBOARD PAGE
# =============================================================================

if page == "Dashboard":
    st.title("📊 FinRiskAI+ Dashboard")
    
    # Fetch dashboard stats
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card success-card">
                    <h3>Total Applications</h3>
                    <h2>{}</h2>
                </div>
                """.format(stats['total_applications']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card success-card">
                    <h3>Approval Rate</h3>
                    <h2>{:.1%}</h2>
                </div>
                """.format(stats['approval_rate']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card warning-card">
                    <h3>Avg Credit Score</h3>
                    <h2>{:.0f}</h2>
                </div>
                """.format(stats['average_credit_score']), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card success-card">
                    <h3>Total Approved Amount</h3>
                    <h2>₹{:,.0f}</h2>
                </div>
                """.format(stats['total_loan_amount']), unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Application Status Distribution")
                
                # Pie chart for application status
                status_data = {
                    'Status': ['Approved', 'Rejected', 'Processing'],
                    'Count': [stats['approved'], stats['rejected'], stats['processing']]
                }
                
                fig_pie = px.pie(
                    values=status_data['Count'],
                    names=status_data['Status'],
                    color_discrete_sequence=['#2ecc71', '#e74c3c', '#f39c12']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Recent Applications")
                
                # Recent applications table
                recent_df = pd.DataFrame(stats['recent_applications'])
                if not recent_df.empty:
                    recent_df['created_at'] = pd.to_datetime(recent_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(
                        recent_df[['business_name', 'loan_amount', 'status', 'credit_score', 'created_at']],
                        use_container_width=True
                    )
                else:
                    st.info("No recent applications found")
        else:
            st.error("Failed to fetch dashboard data")
            
    except Exception as e:
        st.error(f"Error connecting to API: {e}")

# =============================================================================
# CREDIT APPLICATIONS PAGE
# =============================================================================

elif page == "Credit Applications":
    st.title("💳 Credit Applications")
    
    tab1, tab2 = st.tabs(["New Application", "Application History"])
    
    with tab1:
        st.subheader("Create New Credit Application")
        
        with st.form("credit_application_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                sme_profile_id = st.text_input("SME Profile ID")
                loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, step=10000)
                loan_purpose = st.selectbox(
                    "Loan Purpose",
                    ["Working Capital", "Equipment Purchase", "Business Expansion", 
                     "Inventory Purchase", "Marketing Campaign", "Technology Upgrade"]
                )
            
            with col2:
                loan_tenure_months = st.selectbox("Loan Tenure (Months)", [6, 12, 18, 24, 36, 48, 60])
                collateral_type = st.selectbox(
                    "Collateral Type",
                    ["None", "Property", "Equipment", "Inventory", "Vehicle"]
                )
                collateral_value = st.number_input("Collateral Value (₹)", min_value=0, step=10000)
            
            submitted = st.form_submit_button("Submit Application")
            
            if submitted:
                if sme_profile_id and loan_amount > 0:
                    try:
                        application_data = {
                            "sme_profile_id": sme_profile_id,
                            "loan_amount": loan_amount,
                            "loan_purpose": loan_purpose,
                            "loan_tenure_months": loan_tenure_months,
                            "collateral_type": collateral_type,
                            "collateral_value": collateral_value
                        }
                        
                        response = requests.post(
                            f"{API_BASE_URL}/credit-applications",
                            json=application_data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            st.success("✅ Application Processed Successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Decision", result['decision'])
                            with col2:
                                st.metric("Credit Score", result['credit_score'])
                            with col3:
                                st.metric("Risk Grade", result['risk_grade'])
                            
                            # Recommendations
                            st.subheader("Recommendations")
                            for i, rec in enumerate(result['recommendations'], 1):
                                st.write(f"{i}. {rec}")
                            
                            # Explanation
                            if 'explanation' in result:
                                st.subheader("Decision Explanation")
                                explanation = result['explanation']
                                
                                if 'shap' in explanation:
                                    st.write("**Key Contributing Factors:**")
                                    for factor in explanation['shap']['feature_contributions'][:5]:
                                        impact = "🟢 Positive" if factor['impact'] == 'positive' else "🔴 Negative"
                                        st.write(f"- {factor['feature']}: {impact} ({factor['shap_value']:.3f})")
                        else:
                            st.error("Failed to process application")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please fill in all required fields")
    
    with tab2:
        st.subheader("Application History")
        # Implementation for viewing application history would go here
        st.info("Application history feature coming soon...")

# =============================================================================
# SME PROFILES PAGE
# =============================================================================

elif page == "SME Profiles":
    st.title("🏢 SME Profiles")
    
    tab1, tab2 = st.tabs(["Create Profile", "View Profiles"])
    
    with tab1:
        st.subheader("Create New SME Profile")
        
        with st.form("sme_profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                business_name = st.text_input("Business Name")
                business_type = st.selectbox("Business Type", 
                    ["Private Limited", "Partnership", "Proprietorship", "LLP"])
                industry_sector = st.selectbox("Industry Sector",
                    ["Manufacturing", "Retail", "Services", "Technology", 
                     "Healthcare", "Food & Beverage", "Textiles", "Construction"])
                registration_number = st.text_input("Registration Number")
                gst_number = st.text_input("GST Number")
                pan_number = st.text_input("PAN Number")
            
            with col2:
                owner_name = st.text_input("Owner Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone")
                address = st.text_area("Address")
                city = st.text_input("City")
                state = st.text_input("State")
                pincode = st.text_input("Pincode")
            
            col3, col4 = st.columns(2)
            with col3:
                monthly_revenue = st.number_input("Monthly Revenue (₹)", min_value=0)
                annual_revenue = st.number_input("Annual Revenue (₹)", min_value=0)
            
            with col4:
                employee_count = st.number_input("Employee Count", min_value=1, value=1)
                years_in_business = st.number_input("Years in Business", min_value=0.0, step=0.1)
            
            submitted = st.form_submit_button("Create Profile")
            
            if submitted:
                try:
                    profile_data = {
                        "business_name": business_name,
                        "business_type": business_type,
                        "industry_sector": industry_sector,
                        "registration_number": registration_number,
                        "gst_number": gst_number,
                        "pan_number": pan_number,
                        "owner_name": owner_name,
                        "email": email,
                        "phone": phone,
                        "address": address,
                        "city": city,
                        "state": state,
                        "pincode": pincode,
                        "monthly_revenue": monthly_revenue,
                        "annual_revenue": annual_revenue,
                        "employee_count": employee_count,
                        "years_in_business": years_in_business
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/sme-profiles",
                        json=profile_data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Profile created successfully! ID: {result['profile_id']}")
                    else:
                        st.error("Failed to create profile")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("View SME Profiles")
        profile_id = st.text_input("Enter Profile ID to view details")
        
        if st.button("Load Profile") and profile_id:
            try:
                response = requests.get(f"{API_BASE_URL}/sme-profiles/{profile_id}")
                if response.status_code == 200:
                    profile = response.json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Business Name:** {profile['business_name']}")
                        st.write(f"**Business Type:** {profile['business_type']}")
                        st.write(f"**Industry:** {profile['industry_sector']}")
                        st.write(f"**Owner:** {profile['owner_name']}")
                        st.write(f"**Email:** {profile['email']}")
                    
                    with col2:
                        st.write(f"**Location:** {profile['city']}, {profile['state']}")
                        st.write(f"**Monthly Revenue:** ₹{profile['monthly_revenue']:,.0f}")
                        st.write(f"**Annual Revenue:** ₹{profile['annual_revenue']:,.0f}")
                        st.write(f"**Employees:** {profile['employee_count']}")
                        st.write(f"**Years in Business:** {profile['years_in_business']}")
                else:
                    st.error("Profile not found")
            except Exception as e:
                st.error(f"Error: {e}")

# =============================================================================
# ANALYTICS PAGE
# =============================================================================

elif page == "Analytics":
    st.title("📈 Analytics & Insights")
    
    # Revenue Forecasting Section
    st.subheader("📊 Revenue Forecasting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        sme_id_forecast = st.text_input("SME Profile ID for Forecast")
        forecast_periods = st.slider("Forecast Periods (Months)", 3, 24, 12)
        
        if st.button("Generate Forecast"):
            if sme_id_forecast:
                try:
                    forecast_data = {
                        "sme_profile_id": sme_id_forecast,
                        "forecast_periods": forecast_periods
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/revenue-forecast",
                        json=forecast_data
                    )
                    
                    if response.status_code == 200:
                        forecast_result = response.json()
                        
                        # Create forecast visualization
                        months = list(range(1, forecast_periods + 1))
                        predictions = forecast_result['predictions']
                        lower_bound = forecast_result['confidence_intervals']['lower_bound']
                        upper_bound = forecast_result['confidence_intervals']['upper_bound']
                        
                        with col2:
                            fig = go.Figure()
                            
                            # Add prediction line
                            fig.add_trace(go.Scatter(
                                x=months,
                                y=predictions,
                                mode='lines+markers',
                                name='Predicted Revenue',
                                line=dict(color='blue', width=3)
                            ))
                            
                            # Add confidence interval
                            fig.add_trace(go.Scatter(
                                x=months + months[::-1],
                                y=upper_bound + lower_bound[::-1],
                                fill='toself',
                                fillcolor='rgba(0,100,80,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip",
                                showlegend=True,
                                name='Confidence Interval'
                            ))
                            
                            fig.update_layout(
                                title="Revenue Forecast",
                                xaxis_title="Months",
                                yaxis_title="Revenue (₹)",
                                hovermode='x'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk assessment
                        risk_assessment = forecast_result['risk_assessment']
                        st.subheader("Risk Assessment")
                        
                        col3, col4, col5 = st.columns(3)
                        with col3:
                            st.metric("Risk Level", risk_assessment['risk_level'])
                        with col4:
                            st.metric("Growth Rate", f"{risk_assessment['growth_rate']:.1%}")
                        with col5:
                            st.metric("Volatility", f"{risk_assessment['volatility']:.2f}")
                            
                    else:
                        st.error("Failed to generate forecast")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Sentiment Analysis Section
    st.subheader("💬 Business Sentiment Analysis")
    
    with st.form("sentiment_form"):
        st.write("Enter business-related texts from different sources:")
        
        col1, col2 = st.columns(2)
        with col1:
            reviews = st.text_area("Customer Reviews", placeholder="Enter customer reviews...")
            social_media = st.text_area("Social Media Mentions", placeholder="Enter social media posts...")
        
        with col2:
            news = st.text_area("News Articles", placeholder="Enter news content...")
            testimonials = st.text_area("Testimonials", placeholder="Enter testimonials...")
        
        if st.form_submit_button("Analyze Sentiment"):
            business_texts = {}
            
            if reviews: business_texts['reviews'] = [reviews]
            if social_media: business_texts['social_media'] = [social_media]
            if news: business_texts['news'] = [news]
            if testimonials: business_texts['testimonials'] = [testimonials]
            
            if business_texts:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/sentiment-analysis",
                        json={"business_texts": business_texts}
                    )
                    
                    if response.status_code == 200:
                        sentiment_result = response.json()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Sentiment", sentiment_result['overall_sentiment'].title())
                        with col2:
                            st.metric("Confidence", f"{sentiment_result['confidence']:.1%}")
                        with col3:
                            dominant_lang = max(sentiment_result['language_breakdown'], 
                                              key=sentiment_result['language_breakdown'].get)
                            st.metric("Dominant Language", dominant_lang.title())
                        
                        # Key insights
                        st.subheader("Key Insights")
                        for insight in sentiment_result['key_insights']:
                            st.write(f"• {insight}")
                        
                        # Source analysis
                        if sentiment_result['source_analysis']:
                            st.subheader("Source-wise Analysis")
                            for source, analysis in sentiment_result['source_analysis'].items():
                                with st.expander(f"{source.title()} Analysis"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Positive", f"{analysis['sentiment_distribution']['positive']:.1%}")
                                    with col2:
                                        st.metric("Negative", f"{analysis['sentiment_distribution']['negative']:.1%}")
                                    with col3:
                                        st.metric("Neutral", f"{analysis['sentiment_distribution']['neutral']:.1%}")
                    else:
                        st.error("Failed to analyze sentiment")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================================
# MODEL PERFORMANCE PAGE
# =============================================================================

elif page == "Model Performance":
    st.title("🤖 Model Performance & Monitoring")
    
    # Model metrics (simulated data for demo)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "87.3%", "↑ 2.1%")
    with col2:
        st.metric("Precision", "84.6%", "↑ 1.5%")
    with col3:
        st.metric("Recall", "89.2%", "↑ 0.8%")
    with col4:
        st.metric("AUC-ROC", "0.912", "↑ 0.023")
    
    # Performance over time
    st.subheader("Model Performance Over Time")
    
    # Simulated performance data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    accuracy = np.random.normal(0.87, 0.02, len(dates))
    precision = np.random.normal(0.85, 0.02, len(dates))
    recall = np.random.normal(0.89, 0.02, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=accuracy, name='Accuracy', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=precision, name='Precision', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=dates, y=recall, name='Recall', line=dict(color='red')))
    
    fig.update_layout(
        title="Model Performance Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis=dict(range=[0.7, 1.0])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Simulated feature importance data
    features = ['Monthly Revenue', 'Years in Business', 'Credit History', 'Industry Risk', 
               'Cash Flow Ratio', 'Debt-to-Income', 'Collateral Value', 'GST Compliance']
    importance = [0.23, 0.18, 0.15, 0.12, 0.10, 0.08, 0.08, 0.06]
    
    fig_bar = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Credit Scoring Model"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Model drift detection
    st.subheader("Model Drift Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Data Drift Score", "0.23", "↓ 0.05")
        st.write("🟢 **Status:** No significant drift detected")
    
    with col2:
        st.metric("Concept Drift Score", "0.18", "↑ 0.02")
        st.write("🟡 **Status:** Minor drift detected - monitoring required")
