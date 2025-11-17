from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json

app = Flask(__name__)
app.secret_key = 'smartfinance_ai_secret_key_2024'

print("🚀 Loading SmartFinanceAI Model...")

# Global variables untuk model dan data
model_data = None
transaction_data = None

# Load trained model
def load_trained_model():
    """Load model yang sudah trained dari file .pkl"""
    global model_data
    try:
        if os.path.exists('finance_model.pkl'):
            with open('finance_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            print("✅ Trained model loaded successfully!")
            return model_data
        else:
            print("❌ Model file not found!")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load transaction data
def load_transaction_data():
    """Load data transaksi dari CSV"""
    global transaction_data
    try:
        if os.path.exists('financial_transactions.csv'):
            df = pd.read_csv('financial_transactions.csv')
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Loaded {len(df)} transactions")
            transaction_data = df
            return transaction_data
        else:
            print("❌ Transaction data not found!")
            return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Load model dan data saat startup
load_trained_model()
load_transaction_data()

def predict_next_month_spending():
    """Prediksi spending bulan depan dari model yang sudah trained"""
    if not model_data:
        return None
    
    try:
        slope = model_data['model_weights']['slope']
        intercept = model_data['model_weights']['intercept']
        n = model_data['model_weights']['n_months']
        
        next_month_spending = slope * n + intercept
        
        # Calculate confidence based on historical variance
        monthly_expenses = list(model_data['monthly_stats']['expense'].values())
        if monthly_expenses:
            historical_std = np.std([m['sum'] for m in monthly_expenses])
            confidence = max(0.6, min(0.95, 1 - (historical_std / next_month_spending)))
        else:
            historical_std = 0
            confidence = 0.7
        
        trend = 'increasing' if slope > 0 else 'decreasing'
        
        return {
            'predicted_amount': float(next_month_spending),
            'trend': trend,
            'confidence': float(confidence),
            'range': {
                'min': float(next_month_spending - historical_std),
                'max': float(next_month_spending + historical_std)
            }
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def detect_anomalies(threshold=2.5):
    """Deteksi anomalies dari data transaksi menggunakan model trained"""
    if model_data is None or transaction_data is None:
        return []
    
    try:
        anomalies = []
        
        # Check if transaction_data is DataFrame and not empty
        if isinstance(transaction_data, pd.DataFrame) and not transaction_data.empty:
            for _, trans in transaction_data.iterrows():
                if trans['type'] == 'expense':
                    category = trans['category']
                    amount = trans['amount']
                    
                    if category in model_data['category_stats']:
                        stats = model_data['category_stats'][category]
                        
                        # Calculate Z-score
                        if stats['std'] > 0:  # Avoid division by zero
                            z_score = (amount - stats['mean']) / stats['std']
                            
                            if abs(z_score) > threshold:
                                severity = 'high' if abs(z_score) > 3 else 'medium'
                                
                                anomalies.append({
                                    'date': str(trans['date']),
                                    'transaction_id': trans.get('transaction_id', 'N/A'),
                                    'category': category,
                                    'amount': float(amount),
                                    'expected_mean': float(stats['mean']),
                                    'expected_range': f"Rp {stats['q25']:,.0f} - Rp {stats['q75']:,.0f}",
                                    'z_score': float(z_score),
                                    'severity': severity,
                                    'message': f"Unusual {category} expense of Rp {amount:,.0f} (typically Rp {stats['mean']:,.0f})"
                                })
        
        return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return []

def recommend_budget():
    """Generate budget recommendations dari model trained"""
    if not model_data:
        return None
    
    try:
        recommendations = {}
        avg_monthly_income = model_data['income_stats']['avg_monthly']
        
        # Rule: 50/30/20 budget rule
        needs_budget = avg_monthly_income * 0.50
        wants_budget = avg_monthly_income * 0.30
        savings_budget = avg_monthly_income * 0.20
        
        needs_categories = ['Food', 'Transport', 'Bills', 'Health']
        wants_categories = ['Entertainment', 'Shopping']
        savings_categories = ['Investment', 'Education']
        
        for category, stats in model_data['category_stats'].items():
            if category in needs_categories:
                total_needs = sum([model_data['category_stats'][c]['total'] 
                                 for c in needs_categories if c in model_data['category_stats']])
                proportion = stats['total'] / total_needs if total_needs > 0 else 0
                recommended = needs_budget * proportion
            elif category in wants_categories:
                total_wants = sum([model_data['category_stats'][c]['total'] 
                                 for c in wants_categories if c in model_data['category_stats']])
                proportion = stats['total'] / total_wants if total_wants > 0 else 0
                recommended = wants_budget * proportion
            else:
                total_savings = sum([model_data['category_stats'][c]['total'] 
                                   for c in savings_categories if c in model_data['category_stats']])
                proportion = stats['total'] / total_savings if total_savings > 0 else 0
                recommended = savings_budget * proportion
            
            avg_actual = stats['total'] / model_data['model_weights']['n_months']
            
            recommendations[category] = {
                'recommended_monthly': float(recommended),
                'current_average': float(avg_actual),
                'difference': float(recommended - avg_actual),
                'status': 'under_budget' if avg_actual < recommended else 'over_budget'
            }
        
        return recommendations
    except Exception as e:
        print(f"Error in budget recommendation: {e}")
        return {}

def generate_insights():
    """Generate insights dari model trained"""
    if not model_data:
        return []
    
    try:
        insights = []
        
        # 1. Highest spending category
        if model_data['spending_patterns']:
            top_category = max(model_data['spending_patterns'], key=model_data['spending_patterns'].get)
            top_percentage = model_data['spending_patterns'][top_category] * 100
            
            insights.append({
                'type': 'spending_pattern',
                'priority': 'high',
                'title': 'Top Spending Category',
                'message': f"'{top_category}' accounts for {top_percentage:.1f}% of your total expenses",
                'recommendation': f"Review your {top_category} transactions for potential savings"
            })
        
        # 2. Savings rate
        if model_data['monthly_stats']['expense'] and 'avg_monthly' in model_data['income_stats']:
            avg_monthly_expense = np.mean([m['sum'] for m in model_data['monthly_stats']['expense'].values()])
            avg_monthly_income = model_data['income_stats']['avg_monthly']
            
            if avg_monthly_income > 0:
                savings_rate = ((avg_monthly_income - avg_monthly_expense) / avg_monthly_income) * 100
                
                if savings_rate < 20:
                    insights.append({
                        'type': 'savings',
                        'priority': 'high',
                        'title': 'Low Savings Rate',
                        'message': f"Your savings rate is {savings_rate:.1f}%, below the recommended 20%",
                        'recommendation': "Consider reducing discretionary spending"
                    })
                else:
                    insights.append({
                        'type': 'savings',
                        'priority': 'low',
                        'title': 'Good Savings Rate',
                        'message': f"Great job! Your savings rate is {savings_rate:.1f}%",
                        'recommendation': "Maintain this healthy financial habit"
                    })
        
        # 3. Spending trend
        prediction = predict_next_month_spending()
        if prediction and prediction['trend'] == 'increasing':
            insights.append({
                'type': 'trend',
                'priority': 'medium',
                'title': 'Increasing Spending Trend',
                'message': f"Your spending is trending upward",
                'recommendation': "Monitor your expenses closely"
            })
        
        return insights
    except Exception as e:
        print(f"Error generating insights: {e}")
        return []

def get_basic_stats():
    """Get basic statistics dari data transaksi"""
    if transaction_data is None or not isinstance(transaction_data, pd.DataFrame) or transaction_data.empty:
        return {
            'total_income': 0,
            'total_expenses': 0,
            'savings': 0,
            'category_spending': {}
        }
    
    try:
        total_income = transaction_data[transaction_data['type'] == 'income']['amount'].sum()
        total_expenses = transaction_data[transaction_data['type'] == 'expense']['amount'].sum()
        savings = total_income - total_expenses
        
        category_spending = transaction_data[transaction_data['type'] == 'expense'].groupby('category')['amount'].sum().to_dict()
        
        return {
            'total_income': float(total_income),
            'total_expenses': float(total_expenses),
            'savings': float(savings),
            'category_spending': category_spending
        }
    except Exception as e:
        print(f"Error getting basic stats: {e}")
        return {
            'total_income': 0,
            'total_expenses': 0,
            'savings': 0,
            'category_spending': {}
        }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard dengan AI insights"""
    try:
        prediction = predict_next_month_spending() or {}
        insights = generate_insights() or []
        budget_recommendations = recommend_budget() or {}
        stats = get_basic_stats()
        
        return render_template('dashboard.html',
                             prediction=prediction,
                             insights=insights,
                             budget_recommendations=budget_recommendations,
                             total_income=stats.get('total_income', 0),
                             total_expenses=stats.get('total_expenses', 0),
                             savings=stats.get('savings', 0),
                             category_spending=stats.get('category_spending', {}))
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template('dashboard.html', error=str(e))

@app.route('/anomalies')
def anomalies():
    """Anomaly detection page"""
    try:
        anomalies_list = detect_anomalies() or []
        return render_template('anomalies.html', anomalies=anomalies_list)
    except Exception as e:
        print(f"Anomalies error: {e}")
        return render_template('anomalies.html', error=str(e))

@app.route('/budget')
def budget():
    """Budget recommendations page"""
    try:
        budget_recommendations = recommend_budget() or {}
        return render_template('budget.html', budget_recommendations=budget_recommendations)
    except Exception as e:
        print(f"Budget error: {e}")
        return render_template('budget.html', error=str(e))

@app.route('/insights')
def insights():
    """AI insights page"""
    try:
        insights_list = generate_insights() or []
        return render_template('insights.html', insights=insights_list)
    except Exception as e:
        print(f"Insights error: {e}")
        return render_template('insights.html', error=str(e))

@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    """Add new transaction"""
    if request.method == 'POST':
        try:
            # Get form data
            new_transaction = {
                'transaction_id': f'T{int(datetime.now().timestamp())}',
                'date': request.form['date'],
                'type': request.form['type'],
                'category': request.form['category'],
                'amount': float(request.form['amount']),
                'description': request.form['description']
            }
            
            # Load existing data
            if os.path.exists('financial_transactions.csv'):
                existing_data = pd.read_csv('financial_transactions.csv')
                new_data = pd.DataFrame([new_transaction])
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                updated_data = pd.DataFrame([new_transaction])
            
            # Save updated data
            updated_data.to_csv('financial_transactions.csv', index=False)
            
            # Reload data setelah tambah transaksi baru
            load_transaction_data()
            
            return jsonify({
                'success': True,
                'message': 'Transaction added successfully!',
                'transaction_id': new_transaction['transaction_id']
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error adding transaction: {str(e)}'
            })
    
    return render_template('add_transaction.html')

@app.route('/api/transactions')
def api_transactions():
    """API endpoint untuk get recent transactions"""
    try:
        if transaction_data is not None and isinstance(transaction_data, pd.DataFrame) and not transaction_data.empty:
            recent_transactions = transaction_data.tail(10).to_dict('records')
            return jsonify(recent_transactions)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/stats')
def api_stats():
    """API endpoint untuk basic statistics"""
    try:
        stats = get_basic_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 SMARTFINANCEAI WEB APPLICATION")
    print("=" * 60)
    
    if model_data:
        print("✅ Model: Loaded from finance_model.pkl")
    else:
        print("❌ Model: finance_model.pkl not found")
    
    if transaction_data is not None and isinstance(transaction_data, pd.DataFrame) and not transaction_data.empty:
        print(f"✅ Data: {len(transaction_data)} transactions loaded")
    else:
        print("❌ Data: financial_transactions.csv not found or empty")
    
    print("🌐 Access: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)