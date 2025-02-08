import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List

class TechnicianRecommender:
    def __init__(self):
        """Initialize the recommender system"""
        self.model = joblib.load('example_data/technician_model.joblib')
        with open('example_data/model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        self.technicians = pd.read_csv('example_data/technicians.csv')
        
        # Convert string lists to actual lists
        self.technicians['certifications'] = self.technicians['certifications'].apply(eval)
        self.technicians['languages'] = self.technicians['languages'].apply(eval)
        self.technicians['service_areas'] = self.technicians['service_areas'].apply(eval)
        
        # Create binary columns for certifications and languages
        self._create_binary_columns()
    
    def _create_binary_columns(self):
        """Create binary columns for certifications and languages"""
        # Certifications
        all_certs = set()
        for certs in self.technicians['certifications']:
            all_certs.update(certs)
        for cert in all_certs:
            col_name = f'has_{cert.lower().replace(" ", "_")}'
            self.technicians[col_name] = self.technicians['certifications'].apply(lambda x: cert in x)
        
        # Languages
        all_languages = set()
        for langs in self.technicians['languages']:
            all_languages.update(langs)
        for lang in all_languages:
            col_name = f'speaks_{lang.lower()}'
            self.technicians[col_name] = self.technicians['languages'].apply(lambda x: lang in x)
    
    def _validate_job_details(self, job_details: Dict) -> None:
        """Validate job details for required fields and correct formats"""
        required_fields = {
            'job_type': str,
            'initial_quote': (int, float),
            'emergency_service': bool,
            'weekend_service': bool,
            'property_type': str,
            'property_age': int,
            'zip_code': int
        }
        
        for field, expected_type in required_fields.items():
            if field not in job_details:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(job_details[field], expected_type):
                raise ValueError(f"Invalid type for {field}. Expected {expected_type}")
    
    def _filter_available_technicians(self, job_details: Dict) -> pd.DataFrame:
        """Filter technicians based on availability and constraints"""
        techs = self.technicians.copy()
        
        # Filter by service area
        techs = techs[techs['service_areas'].apply(lambda x: job_details['zip_code'] in x)]
        
        # Filter by required certifications based on job type
        if job_details['job_type'] == 'Water Heater':
            techs = techs[techs['has_water_heater_specialist'] == True]
        elif job_details['property_type'] == 'Commercial':
            techs = techs[techs['has_commercial_plumber'] == True]
        
        # Filter by emergency availability if needed
        if job_details['emergency_service'] and job_details['weekend_service']:
            techs = techs[techs['specialty'] == 'Emergency']
        
        return techs
    
    def predict_best_technicians(self, job_details: Dict, top_n: int = 3) -> List[Dict]:
        """Predict the best technicians for a given job"""
        # Validate input
        self._validate_job_details(job_details)
        
        # Filter available technicians
        available_techs = self._filter_available_technicians(job_details)
        
        if len(available_techs) == 0:
            raise ValueError("No technicians available for this job with the given constraints")
        
        # Prepare prediction data for each technician
        predictions = []
        for _, tech in available_techs.iterrows():
            # Create feature vector
            prediction_data = pd.DataFrame({
                'job_type': [job_details['job_type']],
                'initial_quote': [job_details['initial_quote']],
                'emergency_service': [job_details['emergency_service']],
                'weekend_service': [job_details['weekend_service']],
                'property_type': [job_details['property_type']],
                'property_age': [job_details['property_age']],
                'experience_level': [tech['experience_level']],
                'years_experience': [tech['years_experience']],
                'specialty': [tech['specialty']],
                'certification_level': [tech['certification_level']],
                'hourly_rate': [tech['hourly_rate']],
                'jobs_completed': [tech['jobs_completed']],
                'on_time_percentage': [tech['on_time_percentage']],
                'shift_preference': [tech['shift_preference']]
            })
            
            # Add binary columns
            for col in self.technicians.columns:
                if col.startswith(('has_', 'speaks_')):
                    prediction_data[col] = [tech[col]]
            
            # Predict value ratio
            predicted_ratio = self.model.predict(prediction_data)[0]
            predicted_value = job_details['initial_quote'] * predicted_ratio
            
            # Calculate expected profit
            expected_profit = predicted_value - job_details['initial_quote']
            
            # Create recommendation with explanation
            recommendation = {
                'tech_id': tech['tech_id'],
                'name': tech['name'],
                'experience_level': tech['experience_level'],
                'years_experience': tech['years_experience'],
                'specialty': tech['specialty'],
                'certifications': tech['certifications'],
                'languages': tech['languages'],
                'customer_rating': tech['avg_customer_rating'],
                'on_time_percentage': tech['on_time_percentage'],
                'predicted_value_ratio': predicted_ratio,
                'predicted_final_value': predicted_value,
                'expected_profit': expected_profit,
                'confidence_factors': self._get_confidence_factors(tech, job_details)
            }
            
            predictions.append(recommendation)
        
        # Sort by predicted value and return top N
        predictions.sort(key=lambda x: x['predicted_value_ratio'], reverse=True)
        return predictions[:top_n]
    
    def _get_confidence_factors(self, tech: pd.Series, job_details: Dict) -> List[str]:
        """Generate explanation factors for the recommendation"""
        factors = []
        
        # Experience-based factors
        if tech['years_experience'] > 10:
            factors.append(f"Has {tech['years_experience']} years of experience")
        
        # Certification factors
        relevant_certs = [cert for cert in tech['certifications'] if 
                         job_details['job_type'].lower() in cert.lower() or
                         job_details['property_type'].lower() in cert.lower()]
        if relevant_certs:
            factors.append(f"Holds relevant certifications: {', '.join(relevant_certs)}")
        
        # Performance factors
        if tech['on_time_percentage'] > 0.95:
            factors.append(f"Excellent on-time performance ({tech['on_time_percentage']*100:.1f}%)")
        if tech['avg_customer_rating'] > 4.5:
            factors.append(f"High customer satisfaction ({tech['avg_customer_rating']}/5.0)")
        
        # Specialization factors
        if tech['specialty'] == job_details['property_type']:
            factors.append(f"Specializes in {job_details['property_type']} properties")
        if job_details['emergency_service'] and tech['specialty'] == 'Emergency':
            factors.append("Experienced in emergency services")
        
        return factors

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"

def main():
    # Initialize recommender
    recommender = TechnicianRecommender()
    
    # Example job details
    example_jobs = [
        {
            'job_type': 'Water Heater',
            'initial_quote': 400.00,
            'emergency_service': True,
            'weekend_service': True,
            'property_type': 'Residential',
            'property_age': 15,
            'zip_code': 90001,
            'description': 'Emergency water heater replacement on weekend'
        },
        {
            'job_type': 'Sewer Line',
            'initial_quote': 800.00,
            'emergency_service': False,
            'weekend_service': False,
            'property_type': 'Commercial',
            'property_age': 30,
            'zip_code': 90020,
            'description': 'Commercial sewer line inspection and potential repair'
        }
    ]
    
    print("\nTechnician Recommendations System")
    print("================================")
    
    for job in example_jobs:
        print(f"\nJob Details:")
        print(f"Type: {job['job_type']}")
        print(f"Property: {job['property_type']}, Age: {job['property_age']} years")
        print(f"Initial Quote: {format_currency(job['initial_quote'])}")
        print(f"Description: {job['description']}")
        print(f"Emergency Service: {'Yes' if job['emergency_service'] else 'No'}")
        print(f"Weekend Service: {'Yes' if job['weekend_service'] else 'No'}")
        print("\nTop 3 Recommended Technicians:")
        
        try:
            recommendations = recommender.predict_best_technicians(job)
            
            for i, tech in enumerate(recommendations, 1):
                print(f"\n{i}. {tech['name']} ({tech['experience_level']})")
                print(f"   Experience: {tech['years_experience']} years")
                print(f"   Specialty: {tech['specialty']}")
                print(f"   Certifications: {', '.join(tech['certifications'])}")
                print(f"   Languages: {', '.join(tech['languages'])}")
                print(f"   Customer Rating: {tech['customer_rating']}/5.0")
                print(f"   On-time Percentage: {tech['on_time_percentage']*100:.1f}%")
                print(f"   Predicted Job Value: {format_currency(tech['predicted_final_value'])}")
                print(f"   Expected Profit: {format_currency(tech['expected_profit'])}")
                print("   Why this technician?")
                for factor in tech['confidence_factors']:
                    print(f"   - {factor}")
        
        except ValueError as e:
            print(f"\nError: {str(e)}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 