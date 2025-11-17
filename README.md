# SmartFinanceAI - Personal Financial Analysis System

## Gambaran Umum

SmartFinanceAI adalah sistem analisis keuangan personal berbasis machine learning yang dirancang untuk membantu individu dalam mengelola dan menganalisis keuangan mereka. Sistem ini menggunakan teknik statistik dan algoritma sederhana untuk memberikan insights yang actionable tentang pola pengeluaran, prediksi keuangan, dan deteksi anomali.

## Fitur Utama

### 1. **Prediksi Pengeluaran Bulanan**
- Linear regression untuk memprediksi pengeluaran bulan depan
- Analisis tren pengeluaran (meningkat/menurun)
- Interval kepercayaan untuk prediksi
- Strength analysis dari tren yang terdeteksi

### 2. **Deteksi Anomali Transaksi**
- Statistical anomaly detection menggunakan Z-score
- Threshold-based detection untuk transaksi tidak biasa
- Kategorisasi severity (high/medium)
- Detail expected range vs actual amount

### 3. **Rekomendasi Budget Cerdas**
- 50/30/20 budget rule yang disesuaikan
- Kategori: Needs (50%), Wants (30%), Savings (20%)
- Personalisasi berdasarkan pola spending historis
- Status under/over budget per kategori

### 4. **Analisis Pola Pengeluaran**
- Spending distribution across categories
- Monthly trend analysis
- Category-wise statistics (mean, median, std, quartiles)
- Income source analysis

### 5. **Actionable Insights**
- Automated financial insights generation
- Priority-based recommendations
- Savings rate analysis
- Spending pattern identification

## Arsitektur Sistem

### Model Components
```
SmartFinanceAI/
├── Data Loading & Preprocessing
├── Statistical Analysis Engine
├── Linear Regression Predictor
├── Anomaly Detection System
├── Budget Recommendation Engine
└── Insights Generator
```

### Data Flow
```
Raw Transactions → Data Cleaning → Statistical Analysis → 
Model Training → Prediction & Detection → Recommendations & Insights
```

## Metodologi Analisis

### 1. **Statistical Analysis**
- Mean, median, standard deviation per kategori
- Quartile analysis (Q1, Q3) untuk normal range
- Monthly aggregation dan trend calculation
- Spending distribution patterns

### 2. **Linear Regression**
- Simple linear regression untuk spending prediction
- Time-based feature engineering (months as features)
- Confidence interval calculation
- Trend strength analysis

### 3. **Anomaly Detection**
- Z-score calculation: `(x - μ) / σ`
- Configurable threshold (default: 2.5)
- Category-specific statistical baselines
- Severity classification based on deviation

### 4. **Budget Optimization**
- Modified 50/30/20 rule application
- Historical spending pattern consideration
- Proportional distribution within categories
- Gap analysis (recommended vs actual)

## Persyaratan Sistem

### Python Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
datetime
json
pickle
warnings
```

### Struktur Data Input
File CSV harus memiliki kolom berikut:
- `date`: Tanggal transaksi (format: YYYY-MM-DD)
- `amount`: Jumlah transaksi (numeric)
- `type`: Jenis transaksi ('income' atau 'expense')
- `category`: Kategori transaksi (string)
- `transaction_id`: ID unik transaksi (opsional)

### Contoh Data
```csv
date,amount,type,category,transaction_id
2024-01-15,500000,expense,Food,T001
2024-01-20,7500000,income,Salary,T002
2024-01-25,300000,expense,Transport,T003
```

## Instalasi dan Penggunaan

### 1. **Setup Environment**
```bash
# Install dependencies
pip install pandas numpy

# Clone atau download script
# Pastikan file financial_transactions.csv tersedia
```

### 2. **Inisialisasi Model**
```python
from smart_finance_ai import SmartFinanceAI

# Initialize model
model = SmartFinanceAI()

# Load data
data = model.load_data('financial_transactions.csv')
```

### 3. **Training Model**
```python
# Train model dengan data
model.train(data)

# Print summary
model.print_summary()
```

### 4. **Menggunakan Fitur Analisis**

#### Prediksi Pengeluaran
```python
prediction = model.predict_next_month_spending()
print(f"Predicted spending: Rp {prediction['predicted_amount']:,.0f}")
```

#### Deteksi Anomali
```python
anomalies = model.detect_anomalies(data, threshold=2.5)
for anomaly in anomalies:
    print(f"Anomaly: {anomaly['message']}")
```

#### Rekomendasi Budget
```python
budget = model.recommend_budget()
for category, recommendation in budget.items():
    print(f"{category}: Rp {recommendation['recommended_monthly']:,.0f}")
```

#### Generate Insights
```python
insights = model.generate_insights()
for insight in insights:
    print(f"Insight: {insight['message']}")
```

### 5. **Menyimpan dan Memuat Model**
```python
# Save trained model
model.save_model('finance_model.pkl')

# Load model yang sudah trained
model.load_model('finance_model.pkl')
```

## Output dan Interpretasi

### 1. **Prediction Output**
```json
{
  "predicted_amount": 4500000,
  "trend": "increasing",
  "trend_strength": 0.15,
  "confidence": 0.85,
  "range": {
    "min": 3800000,
    "max": 5200000
  }
}
```

### 2. **Anomaly Detection Output**
```json
{
  "date": "2024-01-15",
  "category": "Shopping",
  "amount": 2500000,
  "expected_mean": 800000,
  "z_score": 3.2,
  "severity": "high",
  "message": "Unusual Shopping expense of Rp 2,500,000 (typically Rp 800,000)"
}
```

### 3. **Budget Recommendation Output**
```json
{
  "Food": {
    "recommended_monthly": 1500000,
    "current_average": 1200000,
    "difference": 300000,
    "status": "under_budget"
  }
}
```

### 4. **Insights Output**
```json
{
  "type": "spending_pattern",
  "priority": "high",
  "title": "Top Spending Category",
  "message": "'Food' accounts for 35.5% of your total expenses",
  "recommendation": "Review your Food transactions for potential savings opportunities"
}
```

## Customization dan Konfigurasi

### 1. **Adjusting Anomaly Threshold**
```python
# Default threshold: 2.5 (standard deviations)
anomalies = model.detect_anomalies(data, threshold=3.0)  # More strict
anomalies = model.detect_anomalies(data, threshold=2.0)  # More sensitive
```

### 2. **Budget Rule Customization**
Edit method `recommend_budget()` untuk mengubah aturan:
```python
# Current: 50/30/20 rule
needs_budget = avg_monthly_income * 0.50
wants_budget = avg_monthly_income * 0.30
savings_budget = avg_monthly_income * 0.20
```

### 3. **Category Classification**
Update kategori dalam method `recommend_budget()`:
```python
needs_categories = ['Food', 'Transport', 'Bills', 'Health', 'Housing']
wants_categories = ['Entertainment', 'Shopping', 'Dining']
savings_categories = ['Investment', 'Education', 'Emergency Fund']
```

## Contoh Use Cases

### 1. **Personal Finance Monitoring**
- Track monthly spending patterns
- Identify unusual transactions
- Get automated budget recommendations
- Receive proactive financial insights

### 2. **Financial Planning**
- Predict future spending needs
- Optimize budget allocation
- Improve savings rate
- Identify cost-saving opportunities

### 3. **Expense Audit**
- Detect potential fraud or errors
- Validate transaction patterns
- Monitor category-wise spending limits
- Generate compliance reports

## Best Practices

### 1. **Data Quality**
- Pastikan data transaksi lengkap dan akurat
- Kategorisasi yang konsisten
- Format tanggal yang standar
- Tidak ada missing values pada amount dan type

### 2. **Model Maintenance**
- Retrain model secara berkala dengan data terbaru
- Monitor prediction accuracy
- Adjust thresholds berdasarkan kebutuhan
- Validasi insights dengan real-world context

### 3. **Interpretation Guidelines**
- Confidence score < 70%: Gunakan sebagai referensi saja
- High severity anomalies: Investigasi segera
- Budget recommendations: Sesuaikan dengan kondisi personal
- Insights: Pertimbangkan konteks lifestyle

## Troubleshooting

### 1. **Common Issues**
- **Data tidak terbaca**: Pastikan format CSV sesuai
- **Prediction tidak akurat**: Data historis mungkin tidak cukup
- **Tidak ada anomalies**: Threshold mungkin terlalu tinggi
- **Budget recommendations tidak relevan**: Kategori mungkin perlu penyesuaian

### 2. **Performance Optimization**
- Untuk data besar (>10,000 transaksi), pertimbangkan sampling
- Cache model yang sudah trained
- Gunakan incremental training untuk data baru
- Optimize feature engineering untuk pattern recognition

### 3. **Accuracy Improvement**
- Collect lebih banyak data historis
- Improve data categorization accuracy
- Adjust statistical parameters berdasarkan domain knowledge
- Incorporate seasonal factors jika relevan

## Pengembangan Lanjutan

### 1. **Potential Enhancements**
- Integration dengan banking APIs
- Real-time transaction monitoring
- Advanced ML models (Random Forest, XGBoost)
- Mobile application interface
- Multi-currency support

### 2. **Advanced Features**
- Cash flow forecasting
- Investment portfolio analysis
- Debt management recommendations
- Financial goal tracking
- Risk assessment scoring

### 3. **Scalability Improvements**
- Database integration (SQL, NoSQL)
- Cloud deployment ready
- API endpoints untuk integration
- Batch processing capabilities

## Kontribusi

Kontribusi untuk pengembangan SmartFinanceAI dipersilakan. Beberapa area pengembangan potensial:

1. **Algorithm Improvements**
   - Enhanced prediction models
   - Better anomaly detection techniques
   - Advanced pattern recognition

2. **Feature Additions**
   - Visualization dashboards
   - Export capabilities (PDF, Excel)
   - Alert system integration

3. **Integration Enhancements**
   - Banking API connectors
   - Mobile app development
   - Cloud service integration

## Disclaimer

SmartFinanceAI adalah tool analisis dan tidak menggantikan konsultasi dengan professional financial advisor. Hasil prediksi dan rekomendasi harus dipertimbangkan dengan konteks personal financial situation dan professional advice.

---

**SmartFinanceAI** - Empowering personal financial intelligence through AI-driven analysis and actionable insights.
