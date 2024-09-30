import pandas as pd
import random
from datetime import datetime, timedelta

# Constants
num_records = 10000
departments = ['Finance', 'Sales', 'IT', 'HR', 'Marketing']
process_names = ['Invoice Processing', 'Customer Onboarding', 'Software Development', 'Employee Recruitment', 'Product Launch']
tasks = [
    'Receive Invoice', 'Verify Invoice', 'Approve Invoice',
    'Initial Contact', 'Collect Documents', 'Review Documents',
    'Requirements Gathering', 'Coding', 'Testing',
    'Job Posting', 'Screen Candidates', 'Conduct Interviews',
    'Market Research', 'Product Design', 'Launch Campaign'
]

# Function to generate random dates
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Generate synthetic dataset
data = []
for _ in range(num_records):
    process_id = random.randint(1, 1000)  # Random Process ID
    process_name = random.choice(process_names)
    department = random.choice(departments)
    task_name = random.choice(tasks)
    duration_hours = random.randint(1, 20)  # Duration between 1 and 20 hours
    cost_usd = duration_hours * random.randint(10, 100)  # Random cost calculation
    responsible = f"{random.choice(['John', 'Jane', 'Alex', 'Alice', 'Bob', 'Charlie'])} {random.choice(['Doe', 'Smith', 'Brown', 'Green', 'White', 'Black'])}"
    status = random.choice(['Completed', 'In Progress', 'Pending'])
    start_date = random_date(datetime(2023, 1, 1), datetime(2023, 12, 31))
    end_date = start_date + timedelta(hours=duration_hours) if status == 'Completed' else ''
    error_rate = random.uniform(0, 10)  # Error rate between 0% to 10%
    
    data.append([
        process_id, process_name, department, task_name, duration_hours, cost_usd,
        responsible, status, start_date.date(), end_date, error_rate
    ])

# Create DataFrame
columns = ['Process_ID', 'Process_Name', 'Department', 'Task_Name', 'Duration_Hours', 
           'Cost_USD', 'Responsible', 'Status', 'Start_Date', 'End_Date', 'Error_Rate']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('D:/AI-Enhanced Business Process Optimization/data/business_process_data.csv', index=False)
print("Dataset generated: business_process_data.csv")
