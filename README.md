# Plumber Technician Assignment Optimizer

This project demonstrates how to use machine learning to optimize technician assignments for plumbing jobs, leading to increased revenue and dispatcher efficiency. This is a simple demonstration of the concept extracted from a larger project I'm working on.

## Business Impact
- **Increased Revenue**: By matching the right technician to each job based on historical performance, we maximize the potential for appropriate upselling and quality service
- **Improved Efficiency**: Reduces dispatcher decision time by providing data-driven recommendations
- **Better Customer Service**: Assigns technicians who historically perform best for specific job types

## How It Works
1. Historical data is used to train a machine learning model on:
   - Initial job types and quotes
   - Final job outcomes and payments
   - Technician characteristics and performance
   - Customer satisfaction metrics

2. For new appointments, the system predicts which technician is most likely to:
   - Successfully complete the job
   - Identify additional necessary repairs
   - Generate optimal revenue through appropriate upselling
   - Maintain high customer satisfaction

## Project Structure
- `data_generator.py`: Creates mock historical data for demonstration
- `model_trainer.py`: Trains the machine learning model on historical data
- `predictor.py`: Makes predictions for new appointments
- `example_data/`: Contains sample historical data

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- numpy

## Usage
1. Generate sample data:
```bash
python data_generator.py
```

2. Train the model:
```bash
python model_trainer.py
```

3. Make predictions:
```bash
python predictor.py
```

## Example Scenario
A customer calls with a leaking pipe under their sink, quoted at $150. The system analyzes:
- Job type: Pipe repair
- Initial quote: $150
- Available technicians' historical performance
- Similar past scenarios

The system might recommend Tech A over Tech B because historically:
- Tech A identifies root causes 85% of the time
- Tech A has a track record of appropriately upselling when necessary (e.g., identifying pipe corrosion requiring whole-house repiping)
- Tech A maintains a 4.8/5 customer satisfaction rating 