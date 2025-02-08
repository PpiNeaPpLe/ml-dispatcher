import pandas as pd
import joblib
from datetime import datetime

def load_resources():
    """Load the trained model and necessary data"""
    model = joblib.load('example_data/technician_model.joblib')
    feature_names = joblib.load('example_data/feature_names.joblib')
    technicians = pd.read_csv('example_data/technicians.csv')
    return model, feature_names, technicians

def predict_best_technician(job_details, model, feature_names, technicians):
    """Predict the best technician for a given job"""
    # Validate job type
    if job_details['job_type'] not in feature_names['job_types']:
        raise ValueError(f"Invalid job type. Must be one of: {feature_names['job_types']}")
    
    # Create prediction data for each technician
    predictions = []
    for _, tech in technicians.iterrows():
        # Combine job details with technician details
        prediction_data = pd.DataFrame({
            'job_type': [job_details['job_type']],
            'initial_quote': [job_details['initial_quote']],
            'emergency_service': [job_details['emergency_service']],
            'weekend_service': [job_details['weekend_service']],
            'experience_level': [tech['experience_level']],
            'years_experience': [tech['years_experience']],
            'specialty': [tech['specialty']],
            'certification_level': [tech['certification_level']]
        })
        
        # Predict value ratio
        predicted_ratio = model.predict(prediction_data)[0]
        
        predictions.append({
            'tech_id': tech['tech_id'],
            'name': tech['name'],
            'experience_level': tech['experience_level'],
            'specialty': tech['specialty'],
            'predicted_value_ratio': predicted_ratio,
            'predicted_final_value': job_details['initial_quote'] * predicted_ratio,
            'customer_rating': tech['avg_customer_rating']
        })
    
    # Convert to DataFrame and sort by predicted value
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('predicted_value_ratio', ascending=False)
    
    return predictions_df

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def main():
    # Load necessary resources
    model, feature_names, technicians = load_resources()
    
    # Example job details
    example_jobs = [
        {
            'job_type': 'Pipe Leak',
            'initial_quote': 200.00,
            'emergency_service': False,
            'weekend_service': False,
            'description': 'Standard pipe leak repair under kitchen sink'
        },
        {
            'job_type': 'Water Heater',
            'initial_quote': 400.00,
            'emergency_service': True,
            'weekend_service': True,
            'description': 'Emergency water heater replacement on weekend'
        }
    ]
    
    # Make predictions for example jobs
    print("\nTechnician Recommendations System")
    print("================================")
    
    for job in example_jobs:
        print(f"\nJob Details:")
        print(f"Type: {job['job_type']}")
        print(f"Initial Quote: {format_currency(job['initial_quote'])}")
        print(f"Description: {job['description']}")
        print(f"Emergency Service: {'Yes' if job['emergency_service'] else 'No'}")
        print(f"Weekend Service: {'Yes' if job['weekend_service'] else 'No'}")
        print("\nTop 3 Recommended Technicians:")
        
        predictions = predict_best_technician(job, model, feature_names, technicians)
        top_3 = predictions.head(3)
        
        for idx, tech in top_3.iterrows():
            print(f"\n{idx + 1}. {tech['name']} ({tech['experience_level']})")
            print(f"   Specialty: {tech['specialty']}")
            print(f"   Customer Rating: {tech['customer_rating']}/5.0")
            print(f"   Predicted Value: {format_currency(tech['predicted_final_value'])}")
            print(f"   Value Multiplier: {tech['predicted_value_ratio']:.2f}x")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 