import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

def generate_technicians(num_technicians=10):
    """Generate mock technician data"""
    experience_levels = ['Junior', 'Mid-level', 'Senior', 'Master']
    specialties = ['General', 'Residential', 'Commercial', 'Emergency']
    
    technicians = []
    for i in range(num_technicians):
        tech = {
            'tech_id': f'T{i+1:03d}',
            'name': f'Technician {i+1}',
            'experience_level': np.random.choice(experience_levels, p=[0.2, 0.3, 0.3, 0.2]),
            'years_experience': np.random.randint(1, 25),
            'specialty': np.random.choice(specialties),
            'avg_customer_rating': round(np.random.normal(4.2, 0.3), 1),
            'certification_level': np.random.randint(1, 5)
        }
        technicians.append(tech)
    return pd.DataFrame(technicians)

def generate_job_history(technicians, num_jobs=1000):
    """Generate mock historical job data"""
    job_types = {
        'Pipe Leak': {'base_quote': (100, 300), 'upsell_prob': 0.4, 'upsell_range': (500, 2000)},
        'Drain Clog': {'base_quote': (150, 400), 'upsell_prob': 0.3, 'upsell_range': (600, 1500)},
        'Water Heater': {'base_quote': (200, 600), 'upsell_prob': 0.6, 'upsell_range': (1000, 3000)},
        'Toilet Issue': {'base_quote': (100, 300), 'upsell_prob': 0.2, 'upsell_range': (400, 1200)},
        'Sewer Line': {'base_quote': (300, 800), 'upsell_prob': 0.7, 'upsell_range': (2000, 5000)}
    }
    
    jobs = []
    start_date = datetime(2022, 1, 1)
    
    for i in range(num_jobs):
        job_type = np.random.choice(list(job_types.keys()))
        job_info = job_types[job_type]
        
        # Select technician
        tech = technicians.iloc[np.random.randint(len(technicians))]
        
        # Generate initial quote
        initial_quote = np.random.uniform(*job_info['base_quote'])
        
        # Determine if upsell occurred (influenced by technician's experience and certification)
        upsell_modifier = (tech['years_experience'] / 25 + tech['certification_level'] / 4) / 2
        upsell_prob = job_info['upsell_prob'] * upsell_modifier
        
        if np.random.random() < upsell_prob:
            final_amount = np.random.uniform(*job_info['upsell_range'])
        else:
            final_amount = initial_quote * np.random.uniform(0.9, 1.1)  # Small variation
        
        job = {
            'job_id': f'J{i+1:04d}',
            'date': start_date + timedelta(days=np.random.randint(0, 365)),
            'tech_id': tech['tech_id'],
            'job_type': job_type,
            'initial_quote': round(initial_quote, 2),
            'final_amount': round(final_amount, 2),
            'customer_rating': min(5, max(1, np.random.normal(tech['avg_customer_rating'], 0.5))),
            'job_duration_hours': np.random.uniform(1, 6),
            'emergency_service': random.choice([True, False]),
            'weekend_service': random.choice([True, False])
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