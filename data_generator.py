import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

def generate_zip_codes(num_zips=50):
    """Generate a set of nearby zip codes"""
    base_zip = 90000
    return [base_zip + i for i in range(num_zips)]

def generate_certifications():
    """Generate a list of plumbing certifications"""
    all_certs = [
        'Journeyman Plumber',
        'Master Plumber',
        'Commercial Plumber',
        'Residential Plumber',
        'Water Heater Specialist',
        'Gas Line Specialist',
        'Backflow Specialist',
        'Green Plumbing Specialist'
    ]
    # Each technician gets 1-4 certifications
    return random.sample(all_certs, random.randint(1, 4))

def generate_technicians(num_technicians=10):
    """Generate mock technician data"""
    experience_levels = ['Junior', 'Mid-level', 'Senior', 'Master']
    specialties = ['General', 'Residential', 'Commercial', 'Emergency']
    languages = ['English', 'Spanish', 'Chinese', 'Vietnamese', 'Korean']
    zip_codes = generate_zip_codes()
    shifts = ['Morning', 'Day', 'Evening', 'Flexible']
    
    technicians = []
    for i in range(num_technicians):
        # Base attributes
        experience_level = np.random.choice(experience_levels, p=[0.2, 0.3, 0.3, 0.2])
        years_experience = np.random.randint(1, 25)
        
        # More realistic certification assignment based on experience
        certifications = generate_certifications()
        if experience_level in ['Senior', 'Master']:
            certifications.append('Master Plumber')
        
        tech = {
            'tech_id': f'T{i+1:03d}',
            'name': f'Technician {i+1}',
            'experience_level': experience_level,
            'years_experience': years_experience,
            'specialty': np.random.choice(specialties),
            'avg_customer_rating': round(np.random.normal(4.2, 0.3), 1),
            'certification_level': np.random.randint(1, 5),
            'certifications': certifications,
            'languages': random.sample(languages, random.randint(1, 3)),
            'service_areas': random.sample(zip_codes, random.randint(5, 15)),
            'shift_preference': np.random.choice(shifts),
            'hourly_rate': 25 + (years_experience * 1.5),
            'jobs_completed': np.random.randint(50, 1000),
            'on_time_percentage': round(np.random.uniform(0.85, 0.99), 2)
        }
        technicians.append(tech)
    return pd.DataFrame(technicians)

def generate_job_history(technicians, num_jobs=1000):
    """Generate mock historical job data"""
    job_types = {
        'Pipe Leak': {
            'base_quote': (100, 300),
            'upsell_prob': 0.4,
            'upsell_range': (500, 2000),
            'common_issues': ['Corroded pipes', 'Joint failure', 'Frozen damage']
        },
        'Drain Clog': {
            'base_quote': (150, 400),
            'upsell_prob': 0.3,
            'upsell_range': (600, 1500),
            'common_issues': ['Tree roots', 'Grease buildup', 'Foreign objects']
        },
        'Water Heater': {
            'base_quote': (200, 600),
            'upsell_prob': 0.6,
            'upsell_range': (1000, 3000),
            'common_issues': ['Age replacement', 'Leaking tank', 'Failed element']
        },
        'Toilet Issue': {
            'base_quote': (100, 300),
            'upsell_prob': 0.2,
            'upsell_range': (400, 1200),
            'common_issues': ['Clog', 'Broken flush', 'Seal leak']
        },
        'Sewer Line': {
            'base_quote': (300, 800),
            'upsell_prob': 0.7,
            'upsell_range': (2000, 5000),
            'common_issues': ['Root invasion', 'Pipe collapse', 'Blockage']
        }
    }
    
    property_types = ['Single Family', 'Multi Family', 'Commercial', 'Industrial']
    zip_codes = generate_zip_codes()
    
    jobs = []
    start_date = datetime(2022, 1, 1)
    
    for i in range(num_jobs):
        job_type = np.random.choice(list(job_types.keys()))
        job_info = job_types[job_type]
        
        # Select technician and ensure job is in their service area
        tech = technicians.iloc[np.random.randint(len(technicians))]
        job_zip = np.random.choice(eval(tech['service_areas']))
        
        # Generate initial quote
        initial_quote = np.random.uniform(*job_info['base_quote'])
        
        # Property details
        property_type = np.random.choice(property_types)
        property_age = np.random.randint(0, 80)
        
        # Determine if upsell occurred (influenced by multiple factors)
        base_upsell_prob = job_info['upsell_prob']
        experience_factor = tech['years_experience'] / 25
        property_age_factor = min(1.0, property_age / 50)  # Older properties more likely to need work
        certification_factor = tech['certification_level'] / 4
        
        upsell_prob = base_upsell_prob * (experience_factor * 0.4 + property_age_factor * 0.3 + certification_factor * 0.3)
        
        if np.random.random() < upsell_prob:
            final_amount = np.random.uniform(*job_info['upsell_range'])
            additional_work = random.choice(job_info['common_issues'])
        else:
            final_amount = initial_quote * np.random.uniform(0.9, 1.1)
            additional_work = 'None'
        
        # Calculate duration based on job complexity and tech experience
        base_duration = np.random.uniform(1, 6)
        experience_modifier = 1 - (tech['years_experience'] / 50)  # More experience = faster
        job_duration = base_duration * experience_modifier
        
        job = {
            'job_id': f'J{i+1:04d}',
            'date': start_date + timedelta(days=np.random.randint(0, 365)),
            'tech_id': tech['tech_id'],
            'job_type': job_type,
            'property_type': property_type,
            'property_age': property_age,
            'zip_code': job_zip,
            'initial_quote': round(initial_quote, 2),
            'final_amount': round(final_amount, 2),
            'additional_work': additional_work,
            'customer_rating': min(5, max(1, np.random.normal(tech['avg_customer_rating'], 0.5))),
            'job_duration_hours': round(job_duration, 1),
            'emergency_service': random.choice([True, False]),
            'weekend_service': random.choice([True, False]),
            'parts_used': random.randint(1, 5),
            'customer_callback_required': random.random() < 0.1,
            'payment_method': np.random.choice(['Credit Card', 'Cash', 'Check', 'Financing'])
        }
        jobs.append(job)
    
    return pd.DataFrame(jobs)

def main():
    # Create directory for data
    import os
    os.makedirs('example_data', exist_ok=True)
    
    # Generate data
    technicians_df = generate_technicians()
    jobs_df = generate_job_history(technicians_df)
    
    # Save to CSV
    technicians_df.to_csv('example_data/technicians.csv', index=False)
    jobs_df.to_csv('example_data/job_history.csv', index=False)
    
    print(f"Generated {len(technicians_df)} technicians and {len(jobs_df)} historical jobs")
    print("Data saved to example_data/")

if __name__ == "__main__":
    main() 