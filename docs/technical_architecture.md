# CoinCast — Teknik Mimari Dokümanı

## 0. Genel Sistem Mimarisi

```text
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Veri Kaynakları │ ---> │  Data Pipeline    │ ---> │  Feature Store   │
│ (Binance, News,   │      │ (ingest, clean,   │      │ (Parquet/DB)     │
│  Fear&Greed, On-  │      │  resample)        │      │                  │
│  chain)           │      └──────────────────┘      └────────┬─────────┘
└─────────────────┘                                            │
                                                                 v
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Model Registry   │ <--- │  Training Pipeline│ <--- │ Feature Engine.  │
│ (model + metadata) │     │ (walk-forward CV, │      │ (indicators, NLP,│
└────────┬───────────┘     │  Optuna, ensemble)│      │  on-chain merge) │
         │                 └──────────────────┘      └─────────────────┘
         v
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Inference Service │ <--> │  FastAPI Backend  │ <--> │   Frontend (Web)  │
│ (in-memory model)  │      │ (REST + WebSocket) │      │  (React/Next.js)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         ^                          │
         │                          v
┌─────────────────┐      ┌──────────────────┐
│  Scheduler (Cron)  │      │  Redis Cache /     │
│ (APScheduler/Celery)│     │  Rate Limiter      │
└─────────────────┘      └──────────────────┘
```

Temel prensip: veri toplama, model eğitimi ve tahmin sunumu birbirinden ayrık üç süreçtir. Bunları tek bir Python betiği içinde karıştırmak küçük ölçekte çalışabilir ancak üretime taşınamaz.

---

## 1. Veri Katmanı

### 1.1 Veri Kaynakları

- Binance REST/WebSocket: OHLCV mum verisi, order book, rate limit mevcuttur.
- CoinGecko / CryptoCompare: fiyat verisi için yedek kaynak.
- Fear & Greed Index: günlük duygu verisi.
- CryptoPanic / NewsAPI: haber başlıkları.
- Glassnode / IntoTheBlock: on-chain veri (daha ileri aşama).

Öneri: İlk sürümde Binance OHLCV + Fear & Greed ile başlamak, NLP ve on-chain veriyi sonraki aşamada eklemek en mantıklı yaklaşım.

### 1.2 Ingest Katmanı

- `src/data/ingest.py`: Binance API’den klines çeker.
- Veriyi `data_raw/{symbol}_{interval}.parquet` olarak saklar.
- Idempotent olmalıdır; aynı script tekrar çalıştığında sadece eksik zaman aralığını çeker.
- Hata durumlarında exponential backoff + retry uygulanmalıdır.

### 1.3 Depolama

- Küçük ölçek için Parquet yeterli.
- Orta ölçek için TimescaleDB veya InfluxDB düşünülür.
- Şu anki proje boyutunda Parquet + klasör yapısı doğru tercih olacaktır.

---

## 2. Feature Engineering Katmanı

### 2.1 Teknik Göstergeler

- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic
- Volatilite: Bollinger Bands, ATR
- Hacim: OBV, VWAP

### 2.2 Zaman Bazlı Feature’lar

- Saat/gün/hafta içi mevsimsellik (sin/cos encoding)
- Lag feature’lar
- Rolling istatistikler

### 2.3 Alternatif Veri

- Fear & Greed Index: günlük veriyi saatlik veya dakikalık bar’lara forward-fill ile yaymak.
- Haber Duygu Skoru: FinBERT veya benzeri model ile başlık bazlı score oluşturmak.
- On-chain verisi: borsa giriş/çıkış, balina hareketleri.

### 2.4 Leakage Kontrolü

Feature’ların target ile aynı bilgiyi taşıyıp taşımadığını test etmek gerekir.

```python
# DOĞRU
# Feature'lar sadece t anında bilinen veriden türetilmeli
# target ise t+1 adımına ait getiri olmalı
```

Her yeni feature için korelasyon testi uygulanmalı; yüksek korelasyon sızıntı şüphesi yaratabilir.

---

## 3. Model Katmanı

### 3.1 Validation Stratejisi

- Time-series split veya walk-forward split kullanılmalıdır.
- Her fold’da train/validation/test mantığı korunmalıdır.
- Purge/embargo yaklaşımı finansal ML’de faydalıdır.

### 3.2 Metrikler

- Directional Accuracy: yön tahmin başarısı
- Sharpe Ratio: strateji performansı
- Maximum Drawdown
- RMSE/MAE ikincil metriklerdir

### 3.3 Model Seçimi

- XGBoost/LightGBM: tabular feature’lar için güçlü baseline
- LSTM/GRU: sırayla zaman serisi öğrenimi için
- Ensemble: iki modelin farklı hatalarını birleştirmek için faydalı

### 3.4 Hiperparametre Optimizasyonu

- Optuna ile nested cross-validation uygulanabilir.
- Hedef metrik RMSE değil, yön doğruluğu veya Sharpe gibi iş hedefi ile uyumlu olmalıdır.

### 3.5 Model Kayıt (Model Registry)

- Her eğitim çalıştırmasında model + metadata `models_saved/{symbol}/{timestamp}/` altında saklanmalıdır.
- `latest_model.json` ile inference servisi hangi modeli yükleyeceğini bilir.
- İleri aşamada MLflow kullanılabilir.

---

## 4. Backend Katmanı (`ml_backend`)

### 4.1 Servis Ayrımı

```text
ml_backend/
├── api/            # FastAPI route'ları
├── inference/      # model yükleme + tahmin mantığı
├── scheduler/      # periyodik eğitim job'ları
├── data/           # ingest + feature pipeline
└── core/           # config, logging, exception handler
```

### 4.2 Eğitim/Tahmin Ayrımı

- APScheduler ile periyodik eğitim işleri planlanabilir.
- API başlangıcında en güncel model belleğe yüklenir.
- Trafik büyüdüğünde Celery + Redis’e geçilebilir.

### 4.3 WebSocket

- Canlı fiyat akışı için WebSocket kullanılabilir.
- MVP’de REST polling yeterli olabilir.

### 4.4 Hata Yönetimi ve Rate Limiting

- Global exception handler uygulanmalıdır.
- Redis tabanlı cache ile tekrar eden istekler azaltılabilir.
- Structured logging ile request_id izlenebilir.

---

## 5. Frontend Katmanı

### 5.1 Temel Bileşenler

- Fiyat grafiği
- Tahmin kartı
- Backtest paneli
- Model şeffaflık paneli

### 5.2 Confidence Interval Hesabı

Basit bir yöntem olarak residual dağılımı üzerinden güven aralığı hesaplanabilir.

```python
residuals = y_val - y_val_pred
std_resid = residuals.std()
ci_lower = point_pred - 1.96 * std_resid
ci_upper = point_pred + 1.96 * std_resid
```

Kripto getirileri kalın kuyruklu olduğu için bootstrap tabanlı yöntem daha robust olabilir.

---

## 6. DevOps / Altyapı

### 6.1 Docker Compose

```yaml
services:
  backend:
    build: ./ml_backend
    ports: ["8000:8000"]
    depends_on: [redis]
    env_file: .env
  redis:
    image: redis:7-alpine
  scheduler:
    build: ./ml_backend
    command: python -m scheduler.run
    depends_on: [redis]
  frontend:
    build: ./coincast-pulse
    ports: ["3000:3000"]
```

### 6.2 CI/CD

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest tests/ -v
```

---

## 7. Önerilen Klasör Yapısı

```text
coincast/
├── ml_backend/
│   ├── api/
│   ├── inference/
│   ├── scheduler/
│   ├── data/
│   └── core/
├── src/
│   ├── features/
│   ├── models/
│   ├── validation/
│   └── backtest/
├── models_saved/
│   └── {symbol}/{timestamp}/
├── data_raw/
├── data_processed/
├── notebooks/
├── tests/
├── coincast-pulse/
├── docker-compose.yml
├── .github/workflows/ci.yml
└── README.md
```

---

## 8. Önerilen MVP İlerleme Sırası

### Sprint 1: Veri & Model Foundation

- Parquet tabanlı ingest
- Feature engineering (technical + cyclic time features)
- Walk-forward XGBoost baseline
- Bootstrap CI

### Sprint 2: Production Backend & Pipeline

- FastAPI inference service
- APScheduler periyodik eğitim job’u
- Redis cache ve rate limiting

### Sprint 3: Frontend & Transparency

- React / lightweight charts entegrasyonu
- Tahmin bandı overlay + confidence interval kartı
- Backtest başarım paneli

---

## 9. Ek Teknik Öneriler

### 9.1 Data Leakage Test Otomasyonu

```python
# tests/test_data_leakage.py
import numpy as np


def test_feature_target_correlation(feature_matrix, target_vector):
    for col in feature_matrix.columns:
        corr = np.corrcoef(feature_matrix[col], target_vector)[0, 1]
        assert abs(corr) < 0.90, f"KRITIK VERİ SIZINTISI: {col} feature'ı target ile çok yüksek korelasyona sahip ({corr:.2f})"
```

### 9.2 Bootstrap Tabanlı Güven Aralığı

```python
import numpy as np


def calculate_bootstrap_ci(point_pred, val_residuals, n_bootstraps=1000, ci=0.95):
    bootstrapped_preds = point_pred + np.random.choice(val_residuals, size=n_bootstraps, replace=True)
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    ci_lower = np.percentile(bootstrapped_preds, lower_percentile)
    ci_upper = np.percentile(bootstrapped_preds, upper_percentile)
    return float(ci_lower), float(ci_upper)
```

### 9.3 Backtest Paneli: Komisyon ve Slippage

Finansal net getiri şu şekilde modellenebilir:

$$
\text{Net Getiri} = \text{Brüt Getiri} - (2 \times \text{Komisyon}) - \text{Slippage}
$$

Bu, sadece yön tahminini değil, gerçek işlem maliyetlerini de dikkate alır.
