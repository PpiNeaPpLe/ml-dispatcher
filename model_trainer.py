import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_and_prepare_data():
    """Load and combine technician and job history data"""
    techs_df = pd.read_csv('example_data/technicians.csv')
    jobs_df = pd.read_csv('example_data/job_history.csv')
    
    # Merge datasets
    df = jobs_df.merge(techs_df, on='tech_id')
    
    # Calculate value generation metric (final_amount / initial_quote ratio)
    df['value_ratio'] = df['final_amount'] / df['initial_quote']
    
    return df

def create_feature_pipeline():
    """Create a preprocessing pipeline for features"""
    numeric_features = ['initial_quote', 'years_experience', 'certification_level']
    categorical_features = ['job_type', 'experience_level', 'specialty']
    binary_features = ['emergency_service', 'weekend_service']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])
    
    return preprocessor

def train_model():
    """Train the model on historical data"""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare features and target
    X = df[['job_type', 'initial_quote', 'emergency_service', 'weekend_service',
            'experience_level', 'years_experience', 'specialty', 'certification_level']]
    y = df['value_ratio']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train pipeline
    preprocessor = create_feature_pipeline()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model R² score on training data: {train_score:.3f}")
    print(f"Model R² score on test data: {test_score:.3f}")
    
    # Save the model
    joblib.dump(model, 'example_data/technician_model.joblib')
    print("Model saved to example_data/technician_model.joblib")
    
    # Save feature names for later use
    feature_names = {
        'job_types': X['job_type'].unique().tolist(),
        'experience_levels': X['experience_level'].unique().tolist(),
        'specialties': X['specialty'].unique().tolist()
    }
    joblib.dump(feature_names, 'example_data/feature_names.joblib')

if __name__ == "__main__":
    train_model() 