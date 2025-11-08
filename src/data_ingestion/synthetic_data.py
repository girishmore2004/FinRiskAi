"""
Synthetic SME data generator for testing and development
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
from typing import Dict, List, Tuple
import json
import uuid

fake = Faker('en_IN')  # Indian locale

class SyntheticSMEDataGenerator:
    """Generate realistic synthetic SME data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        Faker.seed(seed)
        
        # Industry sectors with risk profiles
        self.industries = {
            'Manufacturing': {'risk_multiplier': 1.0, 'revenue_volatility': 0.15},
            'Retail': {'risk_multiplier': 1.2, 'revenue_volatility': 0.25},
            'Services': {'risk_multiplier': 0.8, 'revenue_volatility': 0.10},
            'Technology': {'risk_multiplier': 0.7, 'revenue_volatility': 0.20},
            'Healthcare': {'risk_multiplier': 0.6, 'revenue_volatility': 0.08},
            'Food & Beverage': {'risk_multiplier': 1.1, 'revenue_volatility': 0.18},
            'Textiles': {'risk_multiplier': 1.3, 'revenue_volatility': 0.22},
            'Construction': {'risk_multiplier': 1.4, 'revenue_volatility': 0.30},
            'Agriculture': {'risk_multiplier': 1.5, 'revenue_volatility': 0.35},
            'Transportation': {'risk_multiplier': 1.1, 'revenue_volatility': 0.20}
        }
        
        # Indian cities with business activity levels
        self.cities = {
            'Mumbai': 0.9, 'Delhi': 0.85, 'Bangalore': 0.8, 'Chennai': 0.75,
            'Hyderabad': 0.7, 'Pune': 0.7, 'Kolkata': 0.65, 'Ahmedabad': 0.6,
            'Surat': 0.55, 'Jaipur': 0.5, 'Nagpur': 0.45, 'Indore': 0.4
        }
    
    def generate_sme_profiles(self, count: int = 1000) -> pd.DataFrame:
        """Generate SME business profiles"""
        profiles = []
        
        for _ in range(count):
            # Basic business information
            business_name = fake.company()
            industry = random.choice(list(self.industries.keys()))
            city = random.choice(list(self.cities.keys()))
            
            # Business metrics based on industry and location
            city_multiplier = self.cities[city]
            industry_info = self.industries[industry]
            
            # Revenue generation (log-normal distribution)
            base_revenue = np.random.lognormal(mean=13, sigma=1.5)  # ~300K to 50M annually
            monthly_revenue = base_revenue / 12 * city_multiplier
            annual_revenue = base_revenue * city_multiplier
            
            # Employee count based on revenue
            employee_count = max(1, int(np.log(annual_revenue / 100000) * 5))
            
            # Years in business (affects risk)
            years_in_business = np.random.exponential(scale=5)
            
            profile = {
                'id': str(uuid.uuid4()),
                'business_name': business_name,
                'business_type': random.choice(['Private Limited', 'Partnership', 'Proprietorship', 'LLP']),
                'industry_sector': industry,
                'registration_number': f"REG{fake.random_number(digits=10)}",
                'gst_number': f"{fake.random_number(digits=2)}{fake.bothify('?????')}{'#' * 6}",
                'pan_number': fake.bothify('?????####?'),
                'owner_name': fake.name(),
                'email': fake.email(),
                'phone': fake.phone_number(),
                'address': fake.address(),
                'city': city,
                'state': fake.state(),
                'pincode': fake.postcode(),
                'monthly_revenue': round(monthly_revenue, 2),
                'annual_revenue': round(annual_revenue, 2),
                'employee_count': employee_count,
                'years_in_business': round(years_in_business, 1),
                'created_at': fake.date_time_between(start_date='-2y', end_date='now'),
                'city_business_score': city_multiplier,
                'industry_risk_multiplier': industry_info['risk_multiplier'],
                'revenue_volatility': industry_info['revenue_volatility']
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def generate_financial_data(self, sme_profiles: pd.DataFrame) -> pd.DataFrame:
        """Generate financial data for each SME"""
        financial_data = []
        
        for _, sme in sme_profiles.iterrows():
            # Generate 6-12 months of financial data per SME
            months_of_data = random.randint(6, 12)
            
            for month_offset in range(months_of_data):
                base_date = datetime.now() - timedelta(days=30 * month_offset)
                
                # Monthly financial metrics with seasonality and trends
                monthly_revenue = sme['monthly_revenue']
                
                # Add seasonality (some industries have seasonal patterns)
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month_offset / 12)
                
                # Add random variation
                volatility = sme['revenue_volatility']
                revenue_variation = np.random.normal(1, volatility)
                
                actual_monthly_revenue = monthly_revenue * seasonal_factor * revenue_variation
                
                # Calculate other financial metrics
                monthly_outflow = actual_monthly_revenue * random.uniform(0.6, 0.9)
                net_flow = actual_monthly_revenue - monthly_outflow
                
                # Bank balance simulation
                avg_balance = max(0, net_flow * random.uniform(1.5, 4.0))
                min_balance = avg_balance * random.uniform(0.1, 0.5)
                max_balance = avg_balance * random.uniform(1.5, 3.0)
                
                # Transaction patterns
                transaction_count = int(actual_monthly_revenue / 5000) + random.randint(10, 100)
                digital_ratio = random.uniform(0.3, 0.9)  # Digital payment adoption
                recurring_ratio = random.uniform(0.2, 0.7)  # Recurring income pattern
                
                # GST compliance
                gst_compliance = random.uniform(0.6, 1.0)
                gst_filing = gst_compliance > 0.8
                
                financial_record = {
                    'id': str(uuid.uuid4()),
                    'sme_profile_id': sme['id'],
                    'bank_account_number': fake.random_number(digits=12),
                    'bank_name': random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB', 'BOB']),
                    'account_type': random.choice(['Current', 'Savings']),
                    'monthly_inflow': round(actual_monthly_revenue, 2),
                    'monthly_outflow': round(monthly_outflow, 2),
                    'average_balance': round(avg_balance, 2),
                    'min_balance': round(min_balance, 2),
                    'max_balance': round(max_balance, 2),
                    'transaction_count': transaction_count,
                    'digital_transaction_ratio': round(digital_ratio, 3),
                    'recurring_income_ratio': round(recurring_ratio, 3),
                    'gst_monthly_filing': gst_filing,
                    'gst_annual_turnover': round(sme['annual_revenue'], 2),
                    'gst_compliance_score': round(gst_compliance, 3),
                    'data_from_date': base_date.replace(day=1),
                    'data_to_date': base_date.replace(day=28),
                    'created_at': base_date
                }
                financial_data.append(financial_record)
        
        return pd.DataFrame(financial_data)
    
    def generate_credit_applications(self, sme_profiles: pd.DataFrame, count: int = 500) -> pd.DataFrame:
        """Generate credit applications"""
        applications = []
        
        # Select random SMEs for applications
        selected_smes = sme_profiles.sample(n=min(count, len(sme_profiles)))
        
        for _, sme in selected_smes.iterrows():
            # Loan amount based on revenue (typically 2-10x monthly revenue)
            loan_multiplier = random.uniform(2, 10)
            loan_amount = sme['monthly_revenue'] * loan_multiplier
            
            # Loan purpose affects risk
            loan_purposes = [
                'Working Capital', 'Equipment Purchase', 'Business Expansion',
                'Inventory Purchase', 'Marketing Campaign', 'Technology Upgrade',
                'Debt Consolidation', 'Emergency Funds'
            ]
            
            loan_purpose = random.choice(loan_purposes)
            
            # Loan tenure
            loan_tenure = random.choice([6, 12, 18, 24, 36, 48, 60])
            
            # Collateral
            has_collateral = random.choice([True, False])
            collateral_type = random.choice(['Property', 'Equipment', 'Inventory', 'Vehicle', 'None']) if has_collateral else 'None'
            collateral_value = loan_amount * random.uniform(0.8, 1.5) if has_collateral else 0.0
            
            # Calculate risk-based defaults
            risk_factors = {
                'years_in_business': max(0, (5 - sme['years_in_business']) / 5 * 0.3),
                'industry_risk': sme['industry_risk_multiplier'] * 0.2,
                'city_risk': (1 - sme['city_business_score']) * 0.2,
                'loan_to_revenue_ratio': min(1.0, (loan_amount / sme['annual_revenue']) * 0.3)
            }
            
            total_risk_score = sum(risk_factors.values())
            
            # Generate AI scores and decisions
            base_score = 800 - (total_risk_score * 300)  # 500-800 range
            ai_score = max(300, min(900, base_score + random.normal(0, 50)))
            
            probability_of_default = min(0.5, total_risk_score * 0.4 + random.uniform(0, 0.1))
            
            # Decision logic
            if ai_score >= 700 and probability_of_default < 0.15:
                ai_recommendation = 'APPROVE'
                status = random.choices(['APPROVED', 'PROCESSING'], weights=[0.8, 0.2])[0]
            elif ai_score >= 600 and probability_of_default < 0.25:
                ai_recommendation = 'REVIEW'
                status = random.choices(['PROCESSING', 'APPROVED', 'REJECTED'], weights=[0.5, 0.3, 0.2])[0]
            else:
                ai_recommendation = 'REJECT'
                status = random.choices(['REJECTED', 'PROCESSING'], weights=[0.8, 0.2])[0]
            
            # Risk grade
            if ai_score >= 750:
                risk_grade = random.choice(['A+', 'A'])
            elif ai_score >= 650:
                risk_grade = random.choice(['A', 'B+'])
            elif ai_score >= 550:
                risk_grade = random.choice(['B+', 'B'])
            else:
                risk_grade = random.choice(['C+', 'C', 'D'])
            
            confidence_score = random.uniform(0.7, 0.95)
            
            # Generate SHAP-like explanations
            explanation = {
                'top_factors': [
                    {'feature': 'monthly_revenue', 'impact': random.uniform(-0.2, 0.3), 'value': sme['monthly_revenue']},
                    {'feature': 'years_in_business', 'impact': random.uniform(-0.15, 0.25), 'value': sme['years_in_business']},
                    {'feature': 'industry_sector', 'impact': random.uniform(-0.2, 0.1), 'value': sme['industry_sector']},
                    {'feature': 'city_business_score', 'impact': random.uniform(-0.1, 0.2), 'value': sme['city_business_score']},
                    {'feature': 'loan_to_revenue_ratio', 'impact': random.uniform(-0.3, 0.0), 'value': loan_amount/sme['annual_revenue']}
                ],
                'risk_factors': risk_factors
            }
            
            application_date = fake.date_time_between(start_date='-6m', end_date='now')
            decision_date = application_date + timedelta(days=random.randint(1, 14)) if status != 'PROCESSING' else None
            
            application = {
                'id': str(uuid.uuid4()),
                'sme_profile_id': sme['id'],
                'loan_amount': round(loan_amount, 2),
                'loan_purpose': loan_purpose,
                'loan_tenure_months': loan_tenure,
                'collateral_type': collateral_type,
                'collateral_value': round(collateral_value, 2),
                'status': status,
                'application_date': application_date,
                'decision_date': decision_date,
                'ai_score': round(ai_score, 1),
                'ai_recommendation': ai_recommendation,
                'confidence_score': round(confidence_score, 3),
                'explanation': explanation,
                'probability_of_default': round(probability_of_default, 4),
                'risk_grade': risk_grade,
                'created_at': application_date
            }
            applications.append(application)
        
        return pd.DataFrame(applications)
    
    def generate_complete_dataset(self, sme_count: int = 1000, app_count: int = 500) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic dataset"""
        print(f"Generating {sme_count} SME profiles...")
        sme_profiles = self.generate_sme_profiles(sme_count)
        
        print("Generating financial data...")
        financial_data = self.generate_financial_data(sme_profiles)
        
        print(f"Generating {app_count} credit applications...")
        credit_applications = self.generate_credit_applications(sme_profiles, app_count)
        
        return {
            'sme_profiles': sme_profiles,
            'financial_data': financial_data,
            'credit_applications': credit_applications
        }
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str = 'data/synthetic'):
        """Save datasets to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in datasets.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} records to {filepath}")

# Usage example
if __name__ == "__main__":
    generator = SyntheticSMEDataGenerator()
    datasets = generator.generate_complete_dataset(sme_count=2000, app_count=1000)
    generator.save_datasets(datasets)