import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import json

fake = Faker('id_ID')
np.random.seed(42)

def generate_financial_data(num_transactions=1000, months_back=12):
    """
    Generate synthetic financial transaction data
    
    Parameters:
    - num_transactions: jumlah transaksi yang akan digenerate
    - months_back: berapa bulan ke belakang data dibuat
    """
    
    categories = {
        'Food': {'min': 20000, 'max': 500000, 'freq': 0.25},
        'Transport': {'min': 15000, 'max': 300000, 'freq': 0.20},
        'Entertainment': {'min': 50000, 'max': 1000000, 'freq': 0.10},
        'Shopping': {'min': 100000, 'max': 2000000, 'freq': 0.15},
        'Bills': {'min': 100000, 'max': 1500000, 'freq': 0.12},
        'Health': {'min': 50000, 'max': 3000000, 'freq': 0.08},
        'Education': {'min': 500000, 'max': 5000000, 'freq': 0.05},
        'Investment': {'min': 1000000, 'max': 10000000, 'freq': 0.05}
    }
    
    income_sources = ['Salary', 'Freelance', 'Business', 'Investment Return']
    
    transactions = []
    start_date = datetime.now() - timedelta(days=months_back * 30)
    
    print(f"🔄 Generating {num_transactions} transactions...")
    
    for i in range(num_transactions):
        # Random date
        days_offset = np.random.randint(0, months_back * 30)
        trans_date = start_date + timedelta(days=days_offset)
        
        # 20% income, 80% expense
        if np.random.random() < 0.20:
            # Income transaction
            transaction = {
                'transaction_id': f'TRX{i+1:05d}',
                'date': trans_date.strftime('%Y-%m-%d'),
                'time': f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}",
                'type': 'income',
                'category': np.random.choice(income_sources),
                'amount': np.random.uniform(3000000, 15000000),
                'merchant': fake.company(),
                'description': 'Income received',
                'payment_method': 'Bank Transfer'
            }
        else:
            # Expense transaction
            category = np.random.choice(list(categories.keys()))
            cat_info = categories[category]
            
            transaction = {
                'transaction_id': f'TRX{i+1:05d}',
                'date': trans_date.strftime('%Y-%m-%d'),
                'time': f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}",
                'type': 'expense',
                'category': category,
                'amount': np.random.uniform(cat_info['min'], cat_info['max']),
                'merchant': fake.company(),
                'description': fake.sentence(nb_words=4),
                'payment_method': np.random.choice(['Cash', 'Debit Card', 'Credit Card', 'E-Wallet'])
            }
        
        transactions.append(transaction)
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add derived features
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_of_month'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    
    # Round amount
    df['amount'] = df['amount'].round(2)
    
    print(f"✅ Generated {len(df)} transactions")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n📊 Transaction Summary:")
    print(f"   Total Income: Rp {df[df['type']=='income']['amount'].sum():,.0f}")
    print(f"   Total Expense: Rp {df[df['type']=='expense']['amount'].sum():,.0f}")
    print(f"\n📈 Category Distribution:")
    print(df[df['type']=='expense']['category'].value_counts())
    
    return df


def save_data(df, format='csv'):
    """Save data to file"""
    if format == 'csv':
        filename = 'financial_transactions.csv'
        df.to_csv(filename, index=False)
        print(f"\n💾 Data saved to: {filename}")
    
    elif format == 'json':
        filename = 'financial_transactions.json'
        df.to_json(filename, orient='records', indent=2)
        print(f"\n💾 Data saved to: {filename}")
    
    elif format == 'both':
        df.to_csv('financial_transactions.csv', index=False)
        df.to_json('financial_transactions.json', orient='records', indent=2)
        print(f"\n💾 Data saved to: financial_transactions.csv and .json")
    
    return filename


if __name__ == "__main__":
    # Generate data
    df = generate_financial_data(num_transactions=1000, months_back=12)
    
    # Save to file
    save_data(df, format='both')
    
    # Display sample
    print(f"\n📋 Sample data (first 5 rows):")
    print(df.head())
    
    print(f"\n📋 Data Info:")
    print(df.info())