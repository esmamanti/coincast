# CoinCast

CoinCast; Binance kapanmış mumlarından coin ve zaman ufku özel tahmin üreten, risk kontrolünden geçiren ve varsayılan olarak **paper trading** yapan bir araştırma sistemidir. Gerçek para emri göndermez.

> Uyarı: Model kalite kapısı geçilmeden canlı işlem açılmamalıdır. Mevcut model kayıtlarının tamamı `model_verified=false` durumundadır; paper sonuçları yatırım tavsiyesi veya kazanç garantisi değildir.

## Özellikler

- Binance public REST üzerinden yalnızca kapanmış 1 saatlik mumlar
- 13 coin için 1, 4 ve 24 saatlik ayrı XGBoost modelleri
- Train/validation/ayrılmış test seti ve test kalite kapısı
- Residual dağılımından belirsizlik aralığı
- BUY / SELL / HOLD sinyali
- Maksimum pozisyon, günlük zarar ve drawdown kill-switch kontrolleri
- SQLite tabanlı paper cüzdanı, pozisyon ve işlem geçmişi
- Gerçekleşen paper işlemi için kısa rapor
- SMTP ile e-posta ve Twilio ile SMS bildirimi
- FastAPI ve React arayüzü

## Kurulum

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r ml_backend\requirements.txt
```

Model kayıtlarını yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe -m src.train_validated_models --horizons 1 4 24
```

Backend ve frontend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ml_backend.main:app --host 127.0.0.1 --port 8000
cd coincast-pulse
npm install
npm run dev
```

Arayüz: `http://localhost:5173` — API dokümanı: `http://localhost:8000/docs`

## Otomatik paper trading

Tüm 13 coin için 1, 4 ve 24 saatlik tahminleri paper bakiyesine dokunmadan saatlik kaydetmek ve sonuçlandırmak için:

```powershell
.\.venv\Scripts\python.exe -m src.run_forecast_tracker --every-seconds 3600
```

Spot piyasada bulunmayan HYPEUSDT ve KASUSDT için veri kaynağı otomatik olarak Binance vadeli mumlarına geçer ve arayüzde kaynak açıkça gösterilir.

Tek döngü:

```powershell
.\.venv\Scripts\python.exe -m src.run_paper_trader --symbols BTCUSDT ETHUSDT --horizon 1 --once
```

Saatlik döngü:

```powershell
.\.venv\Scripts\python.exe -m src.run_paper_trader --symbols BTCUSDT ETHUSDT --horizon 1 --every-seconds 3600
```

Docker ile isteğe bağlı paper worker:

```powershell
docker compose --profile paper up --build
```

## E-posta bildirimi

`.env` içine SMTP bilgileri yazılır. Gmail kullanılıyorsa normal hesap şifresi yerine uygulama şifresi kullanılmalıdır.

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=hesap@example.com
SMTP_PASSWORD=uygulama-sifresi
SMTP_FROM=hesap@example.com
SMTP_USE_TLS=true
ALERT_EMAIL_TO=rapor@example.com
```

## Otomatik başlangıç ve günlük rapor

- `scripts/start_coincast.ps1`: backend, arayüz ve 39 tahmin serisini izleyen worker'ı başlatır.
- `scripts/send_daily_report.ps1`: kısa paper hesap ve tahmin başarı raporunu üretir.
- Windows görevleri `CoinCast Startup` ve `CoinCast Daily Report` adlarıyla kurulabilir.
- Günlük rapor varsayılan olarak saat 20:00'de çalışır ve ayrıca `results/daily_report_latest.txt` dosyasına kaydedilir.
- Bildirim kanalları `.env` içindeki SMTP veya Twilio bilgileri tamamlandığında otomatik etkinleşir.

## SMS bildirimi

SMS için Twilio bilgileri girilir:

```dotenv
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=+1...
ALERT_SMS_TO=+90...
```

Bildirim bilgileri eksikse işlem döngüsü durmaz; rapor API ve veritabanında kalır, kanal `sent=false` döndürür.

## API

- `POST /predict`: canlı fiyat tahmini ve model metadata'sı
- `POST /signal`: BUY/SELL/HOLD ve risk kararı
- `POST /paper/run`: tek paper işlem döngüsü
- `GET /paper/account`: bakiye ve pozisyonlar
- `GET /paper/trades`: son işlemler
- `GET /health`: servis durumu
- `GET /performance`: coin ve ufuk bazında canlı tahmin başarısı
- `GET /performance/all`: 13 coinin toplu canlı tahmin başarısı

İstek örneği:

```json
{"symbol": "BTCUSDT", "horizon": 1}
```

## Teknik mimari

Ayrıntılı yol haritası ve sistem tasarımı [docs/technical_architecture.md](docs/technical_architecture.md) dosyasındadır.
