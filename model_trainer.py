import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

def load_and_prepare_data():
    """Load and combine technician and job history data"""
    techs_df = pd.read_csv('example_data/technicians.csv')
    jobs_df = pd.read_csv('example_data/job_history.csv')
    
    # Convert string lists to actual lists
    techs_df['certifications'] = techs_df['certifications'].apply(eval)
    techs_df['languages'] = techs_df['languages'].apply(eval)
    techs_df['service_areas'] = techs_df['service_areas'].apply(eval)
    
    # Create certification indicator columns
    all_certs = set()
    for certs in techs_df['certifications']:
        all_certs.update(certs)
    for cert in all_certs:
        techs_df[f'has_{cert.lower().replace(" ", "_")}'] = techs_df['certifications'].apply(lambda x: cert in x)
    
    # Create language indicator columns
    all_languages = set()
    for langs in techs_df['languages']:
        all_languages.update(langs)
    for lang in all_languages:
        techs_df[f'speaks_{lang.lower()}'] = techs_df['languages'].apply(lambda x: lang in x)
    
    # Merge datasets
    df = jobs_df.merge(techs_df, on='tech_id')
    
    # Calculate value metrics
    df['value_ratio'] = df['final_amount'] / df['initial_quote']
    df['profit_per_hour'] = (df['final_amount'] - df['initial_quote']) / df['job_duration_hours']
    
    return df

def create_feature_pipeline():
    """Create a preprocessing pipeline for features"""
    numeric_features = [
        'initial_quote', 'years_experience', 'certification_level',
        'property_age', 'hourly_rate', 'jobs_completed', 'on_time_percentage'
    ]
    
    categorical_features = [
        'job_type', 'experience_level', 'specialty', 'shift_preference',
        'property_type', 'payment_method'
    ]
    
    binary_features = [
        'emergency_service', 'weekend_service', 'customer_callback_required',
        'has_master_plumber', 'has_commercial_plumber', 'has_residential_plumber',
        'has_water_heater_specialist', 'has_gas_line_specialist',
        'speaks_english', 'speaks_spanish'
    ]
    
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])
    
    return preprocessor

def evaluate_model(model, X, y, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    # Training metrics
    train_pred = model.predict(X)
    train_mse = mean_squared_error(y, train_pred)
    train_r2 = r2_score(y, train_pred)
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    metrics = {
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return metrics

def train_model():
    """Train the model on historical data"""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Prepare features and targets
    feature_cols = [col for col in df.columns if col not in [
        'job_id', 'date', 'tech_id', 'name', 'final_amount', 'value_ratio',
        'profit_per_hour', 'customer_rating', 'certifications', 'languages',
        'service_areas', 'additional_work', 'zip_code'
    ]]
    
    X = df[feature_cols]
    y = df['value_ratio']  # Primary target
    y2 = df['profit_per_hour']  # Secondary target for multi-objective evaluation
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Create and train pipeline
    preprocessor = create_feature_pipeline()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    print("\nModel Performance Metrics:")
    print(f"Training R² Score: {metrics['train_r2']:.3f}")
    print(f"Test R² Score: {metrics['test_r2']:.3f}")
    print(f"Cross-validation Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']*2:.3f})")
    
    # Save the model and metadata
    print("\nSaving model and metadata...")
    joblib.dump(model, 'example_data/technician_model.joblib')
    
    metadata = {
        'feature_columns': feature_cols,
        'metrics': metrics,
        'model_params': model.get_params(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('example_data/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model and metadata saved to example_data/")

if __name__ == "__main__":
    train_model() 