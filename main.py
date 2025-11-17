import os

def create_app_py():
    """Create app.py yang langsung pakai model .pkl dan data existing"""
    content = '''from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json

app = Flask(__name__)
app.secret_key = 'smartfinance_ai_secret_key_2024'

print("🚀 Loading SmartFinanceAI Model...")

# Load trained model
def load_trained_model():
    """Load model yang sudah trained dari file .pkl"""
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
    try:
        if os.path.exists('financial_transactions.csv'):
            df = pd.read_csv('financial_transactions.csv')
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Loaded {len(df)} transactions")
            return df
        else:
            print("❌ Transaction data not found!")
            return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Load model dan data saat startup
model_data = load_trained_model()
transaction_data = load_transaction_data()

def predict_next_month_spending():
    """Prediksi spending bulan depan dari model yang sudah trained"""
    if not model_data:
        return None
    
    slope = model_data['model_weights']['slope']
    intercept = model_data['model_weights']['intercept']
    n = model_data['model_weights']['n_months']
    
    next_month_spending = slope * n + intercept
    
    # Calculate confidence based on historical variance
    monthly_expenses = list(model_data['monthly_stats']['expense'].values())
    historical_std = np.std([m['sum'] for m in monthly_expenses])
    confidence = max(0.6, min(0.95, 1 - (historical_std / next_month_spending)))
    
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

def detect_anomalies(threshold=2.5):
    """Deteksi anomalies dari data transaksi menggunakan model trained"""
    if not model_data or not transaction_data:
        return []
    
    anomalies = []
    
    for _, trans in transaction_data.iterrows():
        if trans['type'] == 'expense':
            category = trans['category']
            amount = trans['amount']
            
            if category in model_data['category_stats']:
                stats = model_data['category_stats'][category]
                
                # Calculate Z-score
                z_score = (amount - stats['mean']) / (stats['std'] + 1e-6)
                
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

def recommend_budget():
    """Generate budget recommendations dari model trained"""
    if not model_data:
        return None
    
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
            proportion = stats['total'] / sum([model_data['category_stats'][c]['total'] 
                                              for c in needs_categories if c in model_data['category_stats']])
            recommended = needs_budget * proportion
        elif category in wants_categories:
            proportion = stats['total'] / sum([model_data['category_stats'][c]['total'] 
                                              for c in wants_categories if c in model_data['category_stats']])
            recommended = wants_budget * proportion
        else:
            proportion = stats['total'] / sum([model_data['category_stats'][c]['total'] 
                                              for c in savings_categories if c in model_data['category_stats']])
            recommended = savings_budget * proportion
        
        avg_actual = stats['total'] / model_data['model_weights']['n_months']
        
        recommendations[category] = {
            'recommended_monthly': float(recommended),
            'current_average': float(avg_actual),
            'difference': float(recommended - avg_actual),
            'status': 'under_budget' if avg_actual < recommended else 'over_budget'
        }
    
    return recommendations

def generate_insights():
    """Generate insights dari model trained"""
    if not model_data:
        return []
    
    insights = []
    
    # 1. Highest spending category
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
    avg_monthly_expense = np.mean([m['sum'] for m in model_data['monthly_stats']['expense'].values()])
    avg_monthly_income = model_data['income_stats']['avg_monthly']
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

def get_basic_stats():
    """Get basic statistics dari data transaksi"""
    if not transaction_data:
        return {}
    
    total_income = transaction_data[transaction_data['type'] == 'income']['amount'].sum()
    total_expenses = transaction_data[transaction_data['type'] == 'expense']['amount'].sum()
    savings = total_income - total_expenses
    
    category_spending = transaction_data[transaction_data['type'] == 'expense'].groupby('category')['amount'].sum().to_dict()
    
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'savings': savings,
        'category_spending': category_spending
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard dengan AI insights"""
    try:
        prediction = predict_next_month_spending()
        insights = generate_insights()
        budget_recommendations = recommend_budget()
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
        return render_template('dashboard.html', error=str(e))

@app.route('/anomalies')
def anomalies():
    """Anomaly detection page"""
    try:
        anomalies = detect_anomalies()
        return render_template('anomalies.html', anomalies=anomalies)
    except Exception as e:
        return render_template('anomalies.html', error=str(e))

@app.route('/budget')
def budget():
    """Budget recommendations page"""
    try:
        budget_recommendations = recommend_budget()
        return render_template('budget.html', budget_recommendations=budget_recommendations)
    except Exception as e:
        return render_template('budget.html', error=str(e))

@app.route('/insights')
def insights():
    """AI insights page"""
    try:
        insights = generate_insights()
        return render_template('insights.html', insights=insights)
    except Exception as e:
        return render_template('insights.html', error=str(e))

@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    """Add new transaction"""
    if request.method == 'POST':
        try:
            # Get form data
            transaction_data = {
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
                new_data = pd.DataFrame([transaction_data])
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                updated_data = pd.DataFrame([transaction_data])
            
            # Save updated data
            updated_data.to_csv('financial_transactions.csv', index=False)
            
            # Reload data setelah tambah transaksi baru
            global transaction_data
            transaction_data = load_transaction_data()
            
            return jsonify({
                'success': True,
                'message': 'Transaction added successfully!',
                'transaction_id': transaction_data['transaction_id']
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
        if transaction_data is not None:
            recent_transactions = transaction_data.tail(10).to_dict('records')
            return jsonify(recent_transactions)
        else:
            return jsonify({'error': 'No transaction data available'})
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
    
    if transaction_data is not None:
        print(f"✅ Data: {len(transaction_data)} transactions loaded")
    else:
        print("❌ Data: financial_transactions.csv not found")
    
    print("🌐 Access: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: app.py")

def create_base_template():
    """Create base template"""
    content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SmartFinanceAI - Personal Financial Intelligence{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-red">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-brain"></i> SmartFinanceAI
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('index') }}">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('dashboard') }}">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('anomalies') }}">Anomalies</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('budget') }}">Budget</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('insights') }}">Insights</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link btn-add" href="{{ url_for('add_transaction') }}">
                            <i class="fas fa-plus"></i> Add Transaction
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="container my-4">
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-info alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <!-- Footer -->
    <footer class="bg-dark text-white text-center py-4 mt-5">
        <div class="container">
            <p>&copy; 2024 SmartFinanceAI. Powered by AI & Machine Learning.</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>'''
    
    with open('templates/base.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/base.html")

def create_index_template():
    """Create home page template"""
    content = '''{% extends "base.html" %}

{% block title %}Home - SmartFinanceAI{% endblock %}

{% block content %}
<div class="hero-section text-center py-5 mb-5">
    <div class="container">
        <h1 class="display-4 fw-bold text-red mb-4">
            <i class="fas fa-brain"></i> SmartFinanceAI
        </h1>
        <p class="lead mb-4">AI-Powered Personal Financial Intelligence Platform</p>
        <p class="mb-5">Get intelligent insights from your existing financial data</p>
        
        <div class="row g-4">
            <div class="col-md-3">
                <div class="feature-card">
                    <i class="fas fa-chart-line"></i>
                    <h4>Spending Prediction</h4>
                    <p>AI predicts your future expenses</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="feature-card">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h4>Anomaly Detection</h4>
                    <p>Detect unusual spending patterns</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="feature-card">
                    <i class="fas fa-lightbulb"></i>
                    <h4>Smart Budgeting</h4>
                    <p>Personalized budget recommendations</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="feature-card">
                    <i class="fas fa-robot"></i>
                    <h4>AI Insights</h4>
                    <p>Actionable financial advice</p>
                </div>
            </div>
        </div>

        <div class="mt-5">
            <a href="{{ url_for('dashboard') }}" class="btn btn-red btn-lg me-3">
                <i class="fas fa-tachometer-alt"></i> View Dashboard
            </a>
            <a href="{{ url_for('add_transaction') }}" class="btn btn-outline-red btn-lg">
                <i class="fas fa-plus"></i> Add Transaction
            </a>
        </div>
    </div>
</div>

<div class="container">
    <div class="row">
        <div class="col-md-6">
            <div class="info-card">
                <h3><i class="fas fa-database"></i> Your Data</h3>
                <ul>
                    <li>Using your existing transaction data</li>
                    <li>Pre-trained AI model</li>
                    <li>No additional training needed</li>
                    <li>Real-time analysis</li>
                </ul>
            </div>
        </div>
        <div class="col-md-6">
            <div class="info-card">
                <h3><i class="fas fa-chart-pie"></i> Features</h3>
                <ul>
                    <li>Spending prediction</li>
                    <li>Anomaly detection</li>
                    <li>Budget recommendations</li>
                    <li>Financial insights</li>
                </ul>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/index.html")

def create_dashboard_template():
    """Create dashboard template"""
    content = '''{% extends "base.html" %}

{% block title %}Dashboard - SmartFinanceAI{% endblock %}

{% block content %}
<div class="dashboard-header mb-4">
    <h1 class="text-red"><i class="fas fa-tachometer-alt"></i> Financial Dashboard</h1>
    <p class="lead">AI-powered insights into your financial health</p>
</div>

{% if error %}
<div class="alert alert-danger">
    <i class="fas fa-exclamation-triangle"></i> {{ error }}
</div>
{% else %}
<!-- Summary Cards -->
<div class="row mb-4">
    <div class="col-md-3">
        <div class="stat-card income">
            <i class="fas fa-money-bill-wave"></i>
            <h3>Rp {{ "%.0f"|format(total_income) }}</h3>
            <p>Total Income</p>
        </div>
    </div>
    <div class="col-md-3">
        <div class="stat-card expense">
            <i class="fas fa-shopping-cart"></i>
            <h3>Rp {{ "%.0f"|format(total_expenses) }}</h3>
            <p>Total Expenses</p>
        </div>
    </div>
    <div class="col-md-3">
        <div class="stat-card savings">
            <i class="fas fa-piggy-bank"></i>
            <h3>Rp {{ "%.0f"|format(savings) }}</h3>
            <p>Total Savings</p>
        </div>
    </div>
    <div class="col-md-3">
        <div class="stat-card prediction">
            <i class="fas fa-crystal-ball"></i>
            <h3>Rp {{ "%.0f"|format(prediction.predicted_amount) if prediction else 0 }}</h3>
            <p>Next Month Prediction</p>
        </div>
    </div>
</div>

<div class="row">
    <!-- Prediction Section -->
    <div class="col-md-6">
        {% if prediction %}
        <div class="card mb-4">
            <div class="card-header bg-red text-white">
                <h5 class="mb-0"><i class="fas fa-crystal-ball"></i> Next Month Prediction</h5>
            </div>
            <div class="card-body">
                <div class="prediction-details">
                    <div class="mb-3">
                        <strong>Amount:</strong> Rp {{ "%.0f"|format(prediction.predicted_amount) }}
                    </div>
                    <div class="mb-3">
                        <strong>Trend:</strong> 
                        <span class="badge bg-{{ 'success' if prediction.trend == 'decreasing' else 'warning' }}">
                            {{ prediction.trend|upper }}
                        </span>
                    </div>
                    <div class="mb-3">
                        <strong>Confidence:</strong> 
                        <span class="badge bg-{{ 'success' if prediction.confidence > 0.8 else 'warning' }}">
                            {{ "%.1f"|format(prediction.confidence * 100) }}%
                        </span>
                    </div>
                    <div class="mb-3">
                        <strong>Expected Range:</strong><br>
                        Rp {{ "%.0f"|format(prediction.range.min) }} - Rp {{ "%.0f"|format(prediction.range.max) }}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Quick Insights -->
        <div class="card">
            <div class="card-header bg-red text-white">
                <h5 class="mb-0"><i class="fas fa-bolt"></i> Quick Insights</h5>
            </div>
            <div class="card-body">
                {% if insights %}
                    {% for insight in insights[:3] %}
                    <div class="insight-item mb-3 p-2 insight-{{ insight.priority }}">
                        <strong>{{ insight.title }}</strong><br>
                        <small>{{ insight.message }}</small>
                    </div>
                    {% endfor %}
                {% else %}
                    <p class="text-muted">No insights available</p>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Category Spending -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-red text-white">
                <h5 class="mb-0"><i class="fas fa-chart-pie"></i> Spending by Category</h5>
            </div>
            <div class="card-body">
                {% if category_spending %}
                    {% for category, amount in category_spending.items() %}
                    <div class="category-item mb-2">
                        <div class="d-flex justify-content-between">
                            <span>{{ category }}</span>
                            <span>Rp {{ "%.0f"|format(amount) }}</span>
                        </div>
                        <div class="progress mb-2">
                            <div class="progress-bar bg-red" style="width: {{ (amount / total_expenses * 100) if total_expenses > 0 else 0 }}%">
                                {{ "%.1f"|format((amount / total_expenses * 100) if total_expenses > 0 else 0) }}%
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <p class="text-muted">No category data available</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endif %}
{% endblock %}'''
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/dashboard.html")

def create_anomalies_template():
    """Create anomalies template"""
    content = '''{% extends "base.html" %}

{% block title %}Anomaly Detection - SmartFinanceAI{% endblock %}

{% block content %}
<div class="anomalies-header mb-4">
    <h1 class="text-red"><i class="fas fa-exclamation-triangle"></i> Anomaly Detection</h1>
    <p class="lead">AI-detected unusual spending patterns</p>
</div>

{% if error %}
<div class="alert alert-danger">
    <i class="fas fa-exclamation-triangle"></i> {{ error }}
</div>
{% else %}
{% if anomalies %}
<div class="alert alert-warning">
    <i class="fas fa-info-circle"></i> Found {{ anomalies|length }} unusual transactions
</div>

<div class="row">
    {% for anomaly in anomalies %}
    <div class="col-md-6 mb-3">
        <div class="anomaly-card anomaly-{{ anomaly.severity }}">
            <div class="anomaly-header">
                <h5>
                    <i class="fas fa-{{ 'fire' if anomaly.severity == 'high' else 'exclamation' }}"></i>
                    {{ anomaly.category }} - {{ anomaly.date }}
                </h5>
                <span class="badge bg-{{ 'danger' if anomaly.severity == 'high' else 'warning' }}">
                    {{ anomaly.severity|upper }}
                </span>
            </div>
            <div class="anomaly-body">
                <p><strong>Amount:</strong> Rp {{ "%.0f"|format(anomaly.amount) }}</p>
                <p><strong>Expected Range:</strong> {{ anomaly.expected_range }}</p>
                <p><strong>Z-Score:</strong> {{ "%.2f"|format(anomaly.z_score) }}</p>
                <p><strong>Message:</strong> {{ anomaly.message }}</p>
            </div>
        </div>
    </div>
    {% endfor %}
</div>
{% else %}
<div class="alert alert-success">
    <i class="fas fa-check-circle"></i> No anomalies detected! All transactions appear normal.
</div>
{% endif %}
{% endif %}
{% endblock %}'''
    
    with open('templates/anomalies.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/anomalies.html")

def create_budget_template():
    """Create budget template"""
    content = '''{% extends "base.html" %}

{% block title %}Budget Recommendations - SmartFinanceAI{% endblock %}

{% block content %}
<div class="budget-header mb-4">
    <h1 class="text-red"><i class="fas fa-lightbulb"></i> Budget Recommendations</h1>
    <p class="lead">AI-powered personalized budget suggestions</p>
</div>

{% if error %}
<div class="alert alert-danger">
    <i class="fas fa-exclamation-triangle"></i> {{ error }}
</div>
{% else %}
{% if budget_recommendations %}
<div class="row">
    {% for category, recommendation in budget_recommendations.items() %}
    <div class="col-md-6 mb-4">
        <div class="budget-card {{ recommendation.status }}">
            <div class="budget-header">
                <h4>{{ category }}</h4>
                <span class="status-badge {{ recommendation.status }}">
                    {{ recommendation.status.replace('_', ' ')|title }}
                </span>
            </div>
            
            <div class="budget-details">
                <div class="budget-row">
                    <span>Recommended:</span>
                    <strong>Rp {{ "%.0f"|format(recommendation.recommended_monthly) }}/month</strong>
                </div>
                <div class="budget-row">
                    <span>Current Average:</span>
                    <span>Rp {{ "%.0f"|format(recommendation.current_average) }}/month</span>
                </div>
                <div class="budget-row">
                    <span>Difference:</span>
                    <span class="difference {{ 'positive' if recommendation.difference > 0 else 'negative' }}">
                        Rp {{ "%.0f"|format(recommendation.difference) }}
                    </span>
                </div>
            </div>
            
            <div class="budget-progress">
                <div class="progress">
                    {% set percentage = (recommendation.current_average / recommendation.recommended_monthly * 100) if recommendation.recommended_monthly > 0 else 100 %}
                    <div class="progress-bar {{ 'bg-success' if percentage <= 100 else 'bg-danger' }}" 
                         style="width: {{ min(percentage, 100) }}%">
                        {{ "%.1f"|format(percentage) }}%
                    </div>
                </div>
            </div>
            
            <div class="budget-advice">
                {% if recommendation.status == 'over_budget' %}
                <small class="text-danger">
                    <i class="fas fa-exclamation-circle"></i>
                    Consider reducing spending in this category
                </small>
                {% else %}
                <small class="text-success">
                    <i class="fas fa-check-circle"></i>
                    You're within budget for this category
                </small>
                {% endif %}
            </div>
        </div>
    </div>
    {% endfor %}
</div>
{% else %}
<div class="alert alert-info">
    <i class="fas fa-info-circle"></i> No budget recommendations available.
</div>
{% endif %}
{% endif %}
{% endblock %}'''
    
    with open('templates/budget.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/budget.html")

def create_insights_template():
    """Create insights template"""
    content = '''{% extends "base.html" %}

{% block title %}AI Insights - SmartFinanceAI{% endblock %}

{% block content %}
<div class="insights-header mb-4">
    <h1 class="text-red"><i class="fas fa-chart-line"></i> AI Insights</h1>
    <p class="lead">Actionable intelligence for better financial decisions</p>
</div>

{% if error %}
<div class="alert alert-danger">
    <i class="fas fa-exclamation-triangle"></i> {{ error }}
</div>
{% else %}
{% if insights %}
<div class="row">
    {% for insight in insights %}
    <div class="col-12 mb-4">
        <div class="insight-card insight-{{ insight.priority }}">
            <div class="insight-header">
                <h4>
                    <i class="fas fa-{{ 'exclamation-triangle' if insight.priority == 'high' else 'info-circle' if insight.priority == 'medium' else 'check-circle' }}"></i>
                    {{ insight.title }}
                </h4>
                <span class="priority-badge {{ insight.priority }}">
                    {{ insight.priority|upper }}
                </span>
            </div>
            <div class="insight-body">
                <p class="insight-message">{{ insight.message }}</p>
                <div class="insight-recommendation">
                    <strong><i class="fas fa-lightbulb"></i> Recommendation:</strong><br>
                    {{ insight.recommendation }}
                </div>
            </div>
        </div>
    </div>
    {% endfor %}
</div>
{% else %}
<div class="alert alert-info">
    <i class="fas fa-info-circle"></i> No insights available.
</div>
{% endif %}
{% endif %}
{% endblock %}'''
    
    with open('templates/insights.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/insights.html")

def create_add_transaction_template():
    """Create add transaction template"""
    content = '''{% extends "base.html" %}

{% block title %}Add Transaction - SmartFinanceAI{% endblock %}

{% block content %}
<div class="add-transaction-header mb-4">
    <h1 class="text-red"><i class="fas fa-plus"></i> Add New Transaction</h1>
    <p class="lead">Add income or expense to your financial data</p>
</div>

<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header bg-red text-white">
                <h5 class="mb-0">Transaction Details</h5>
            </div>
            <div class="card-body">
                <form id="transactionForm">
                    <div class="mb-3">
                        <label for="date" class="form-label">Date</label>
                        <input type="date" class="form-control" id="date" name="date" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="type" class="form-label">Type</label>
                        <select class="form-select" id="type" name="type" required>
                            <option value="">Select Type</option>
                            <option value="income">Income</option>
                            <option value="expense">Expense</option>
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="category" class="form-label">Category</label>
                        <select class="form-select" id="category" name="category" required>
                            <option value="">Select Category</option>
                            <optgroup label="Income">
                                <option value="Salary">Salary</option>
                                <option value="Freelance">Freelance</option>
                                <option value="Investment">Investment</option>
                            </optgroup>
                            <optgroup label="Expenses">
                                <option value="Food">Food</option>
                                <option value="Transport">Transport</option>
                                <option value="Entertainment">Entertainment</option>
                                <option value="Shopping">Shopping</option>
                                <option value="Bills">Bills</option>
                                <option value="Health">Health</option>
                                <option value="Education">Education</option>
                            </optgroup>
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="amount" class="form-label">Amount (Rp)</label>
                        <input type="number" class="form-control" id="amount" name="amount" min="1" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="description" class="form-label">Description</label>
                        <textarea class="form-control" id="description" name="description" rows="3"></textarea>
                    </div>
                    
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-red">
                            <i class="fas fa-save"></i> Add Transaction
                        </button>
                        <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                            <i class="fas fa-arrow-left"></i> Back to Dashboard
                        </a>
                    </div>
                </form>
                
                <div id="message" class="mt-3"></div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('templates/add_transaction.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: templates/add_transaction.html")

def create_css_file():
    """Create CSS file with red theme"""
    content = '''/* SmartFinanceAI Red Theme */
:root {
    --primary-red: #dc3545;
    --dark-red: #c82333;
    --light-red: #e74c3c;
    --bg-red: #dc3545;
    --text-dark: #333;
    --text-light: #666;
    --bg-light: #f8f9fa;
    --border-color: #dee2e6;
}

/* General Styles */
body {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Red Theme */
.bg-red {
    background-color: var(--bg-red) !important;
}

.text-red {
    color: var(--primary-red) !important;
}

.btn-red {
    background-color: var(--primary-red);
    border-color: var(--primary-red);
    color: white;
}

.btn-red:hover {
    background-color: var(--dark-red);
    border-color: var(--dark-red);
    color: white;
}

.btn-outline-red {
    color: var(--primary-red);
    border-color: var(--primary-red);
}

.btn-outline-red:hover {
    background-color: var(--primary-red);
    border-color: var(--primary-red);
    color: white;
}

/* Navigation */
.navbar-brand {
    font-weight: bold;
    font-size: 1.5rem;
}

.btn-add {
    background-color: #28a745;
    border-color: #28a745;
    color: white !important;
    border-radius: 20px;
    padding: 0.5rem 1rem !important;
}

.btn-add:hover {
    background-color: #218838;
    border-color: #1e7e34;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #fff 0%, #ffeaea 100%);
    border-radius: 15px;
    margin-top: 20px;
}

/* Feature Cards */
.feature-card {
    text-align: center;
    padding: 2rem 1rem;
    border-radius: 10px;
    background: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
}

.feature-card i {
    font-size: 2.5rem;
    color: var(--primary-red);
    margin-bottom: 1rem;
}

.feature-card h4 {
    color: var(--text-dark);
    margin-bottom: 1rem;
    font-size: 1.1rem;
}

/* Info Cards */
.info-card {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.info-card h3 {
    color: var(--primary-red);
    margin-bottom: 1rem;
}

.info-card ul {
    list-style: none;
    padding-left: 0;
}

.info-card li {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-color);
}

.info-card li:before {
    content: "✓";
    color: var(--primary-red);
    font-weight: bold;
    margin-right: 10px;
}

/* Stat Cards */
.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary-red);
}

.stat-card.income {
    border-left-color: #28a745;
}

.stat-card.expense {
    border-left-color: #dc3545;
}

.stat-card.savings {
    border-left-color: #17a2b8;
}

.stat-card.prediction {
    border-left-color: #ffc107;
}

.stat-card i {
    font-size: 2rem;
    margin-bottom: 1rem;
}

.stat-card.income i { color: #28a745; }
.stat-card.expense i { color: #dc3545; }
.stat-card.savings i { color: #17a2b8; }
.stat-card.prediction i { color: #ffc107; }

.stat-card h3 {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

/* Cards */
.card {
    border: none;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-radius: 10px;
    margin-bottom: 1.5rem;
}

.card-header {
    border-radius: 10px 10px 0 0 !important;
    font-weight: bold;
}

/* Anomaly Cards */
.anomaly-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 4px solid #ffc107;
}

.anomaly-card.anomaly-high {
    border-left-color: #dc3545;
}

.anomaly-card.anomaly-medium {
    border-left-color: #ffc107;
}

.anomaly-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.anomaly-header h5 {
    margin: 0;
    flex: 1;
}

/* Budget Cards */
.budget-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 4px solid #17a2b8;
}

.budget-card.under_budget {
    border-left-color: #28a745;
}

.budget-card.over_budget {
    border-left-color: #dc3545;
}

.budget-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.budget-header h4 {
    margin: 0;
    flex: 1;
}

.status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}

.status-badge.under_budget {
    background-color: #d4edda;
    color: #155724;
}

.status-badge.over_budget {
    background-color: #f8d7da;
    color: #721c24;
}

.budget-details {
    margin-bottom: 1rem;
}

.budget-row {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem 0;
    border-bottom: 1px solid var(--border-color);
}

.difference.positive {
    color: #28a745;
    font-weight: bold;
}

.difference.negative {
    color: #dc3545;
    font-weight: bold;
}

/* Insight Cards */
.insight-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 4px solid #17a2b8;
}

.insight-card.insight-high {
    border-left-color: #dc3545;
}

.insight-card.insight-medium {
    border-left-color: #ffc107;
}

.insight-card.insight-low {
    border-left-color: #28a745;
}

.insight-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.insight-header h4 {
    margin: 0;
    flex: 1;
}

.priority-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}

.priority-badge.high {
    background-color: #f8d7da;
    color: #721c24;
}

.priority-badge.medium {
    background-color: #fff3cd;
    color: #856404;
}

.priority-badge.low {
    background-color: #d4edda;
    color: #155724;
}

.insight-message {
    font-size: 1.1rem;
    margin-bottom: 1rem;
    color: var(--text-dark);
}

.insight-recommendation {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    border-left: 3px solid var(--primary-red);
}

/* Progress Bars */
.progress {
    height: 8px;
    border-radius: 4px;
}

.progress-bar {
    border-radius: 4px;
}

/* Form Styles */
.form-control, .form-select {
    border-radius: 8px;
    border: 1px solid var(--border-color);
    padding: 0.75rem;
}

.form-control:focus, .form-select:focus {
    border-color: var(--primary-red);
    box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
}

/* Responsive */
@media (max-width: 768px) {
    .hero-section {
        padding: 2rem 1rem;
    }
    
    .feature-card {
        margin-bottom: 1rem;
    }
    
    .stat-card {
        margin-bottom: 1rem;
    }
    
    .anomaly-header, .budget-header, .insight-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .anomaly-header .badge, 
    .budget-header .status-badge,
    .insight-header .priority-badge {
        margin-top: 0.5rem;
    }
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.card, .feature-card, .stat-card {
    animation: fadeIn 0.5s ease-out;
}'''
    
    with open('static/css/style.css', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: static/css/style.css")

def create_js_file():
    """Create JavaScript file"""
    content = '''// SmartFinanceAI JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Add Transaction Form Handling
    const transactionForm = document.getElementById('transactionForm');
    if (transactionForm) {
        transactionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const submitButton = this.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            
            // Show loading state
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding...';
            submitButton.disabled = true;
            
            fetch('/add_transaction', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const messageDiv = document.getElementById('message');
                if (data.success) {
                    messageDiv.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> ${data.message}
                        </div>
                    `;
                    transactionForm.reset();
                    
                    // Redirect to dashboard after 2 seconds
                    setTimeout(() => {
                        window.location.href = '/dashboard';
                    }, 2000);
                } else {
                    messageDiv.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle"></i> ${data.message}
                        </div>
                    `;
                }
            })
            .catch(error => {
                const messageDiv = document.getElementById('message');
                messageDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i> Error: ${error}
                    </div>
                `;
            })
            .finally(() => {
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
            });
        });
        
        // Set today's date as default
        const dateField = document.getElementById('date');
        if (dateField) {
            const today = new Date().toISOString().split('T')[0];
            dateField.value = today;
        }
    }
    
    // Add hover effects to cards
    document.querySelectorAll('.card, .feature-card, .stat-card').forEach(card => {
        card.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
        
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 8px 15px rgba(0,0,0,0.1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
        });
    });
});'''
    
    with open('static/js/script.js', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: static/js/script.js")

def create_requirements_file():
    """Create requirements.txt"""
    content = '''Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
'''
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: requirements.txt")

def create_readme_file():
    """Create README.md"""
    content = '''# SmartFinanceAI - Personal Financial Intelligence Platform

## 🚀 Overview
SmartFinanceAI is an AI-powered web application that provides intelligent financial analysis using your existing data and pre-trained model.

## ✨ Features
- **Spending Prediction**: AI predicts your future expenses
- **Anomaly Detection**: Detect unusual spending patterns
- **Budget Recommendations**: Personalized budget suggestions
- **AI Insights**: Actionable financial advice

## 🛠️ Quick Start

1. **Make sure you have these files in the same folder:**
   - `finance_model.pkl` (your trained model)
   - `financial_transactions.csv` (your transaction data)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

## 📊 How It Works
- Uses your existing `finance_model.pkl` file
- Loads your transaction data from `financial_transactions.csv`
- No training required - everything is pre-computed
- Real-time analysis and insights

## 🎨 Features
- **Dashboard**: Financial overview and predictions
- **Anomalies**: Unusual transaction detection
- **Budget**: Personalized recommendations
- **Insights**: AI-generated financial advice
- **Add Data**: Add new transactions

---
**Powered by your existing AI model and data**'''
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created: README.md")

def main():
    """Main function to generate complete web project"""
    print("🚀 Generating SmartFinanceAI Web Project...")
    print("=" * 60)
    
    # Create directory structure
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Create all files
    create_app_py()
    create_base_template()
    create_index_template()
    create_dashboard_template()
    create_anomalies_template()
    create_budget_template()
    create_insights_template()
    create_add_transaction_template()
    create_css_file()
    create_js_file()
    create_requirements_file()
    create_readme_file()
    
    print("=" * 60)
    print("🎉 Project generation completed!")
    print("\n📋 Next Steps:")
    print("1. Make sure you have:")
    print("   - finance_model.pkl")
    print("   - financial_transactions.csv")
    print("2. pip install -r requirements.txt")
    print("3. python app.py")
    print("4. Open http://localhost:5000")
    print("\n💡 This web app uses your EXISTING model and data - no training needed!")

if __name__ == "__main__":
    main()