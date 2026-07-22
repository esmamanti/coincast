import { useEffect, useState } from 'react';

type Prediction = {
  symbol: string;
  horizon: number;
  interval: string;
  current_price: number;
  predicted_price: number;
  predicted_price_interval: { lower: number; upper: number };
  predicted_return: number;
  confidence_interval: { lower: number; upper: number };
  mini_chart: number[];
  data_timestamp: string;
  data_age_seconds: number;
  model_id: string;
  model_verified: boolean;
  signal_threshold: number;
  source: string;
};

type Account = {
  cash: number;
  equity: number;
  peak_equity: number;
  positions: Array<{ symbol: string; quantity: number; average_price: number; last_price: number }>;
};

type PerformanceRecord = {
  id: number;
  data_timestamp: string;
  target_timestamp: string;
  current_price: number;
  predicted_price: number;
  actual_price: number | null;
  absolute_error: number | null;
  direction_correct: boolean | null;
  interval_hit: boolean | null;
  action: string;
  status: 'pending' | 'resolved';
};

type PredictionPerformance = {
  symbol: string;
  horizon: number;
  total_predictions: number;
  resolved_predictions: number;
  pending_predictions: number;
  direction_accuracy: number | null;
  mae: number | null;
  naive_mae: number | null;
  price_improvement_ratio: number | null;
  interval_coverage: number | null;
  recent: PerformanceRecord[];
};

type AllPerformanceResponse = {
  horizon: number;
  coins: PredictionPerformance[];
};

type DailyReportPreview = {
  subject: string;
  report: string;
  generated_at: string;
  channels: Array<{ channel: string; configured: boolean }>;
};

type SignalResponse = {
  prediction: Prediction;
  action: 'BUY' | 'SELL' | 'HOLD';
  risk: { allowed: boolean; reason: string; quote_amount: number };
  account: Account;
  performance: PredictionPerformance;
};

type PaperResult = {
  status: string;
  action: string;
  report: string;
  notifications: Array<{ channel: string; sent: boolean; reason?: string }>;
  trade: null | { id: number; quantity: number; price: number; fee: number };
};

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'SOLUSDT',
  'BNBUSDT',
  'DOGEUSDT',
  'LINKUSDT',
  'AAVEUSDT',
  'HYPEUSDT',
  'INJUSDT',
  'KASUSDT',
  'ONDOUSDT',
  'RENDERUSDT',
  'SUIUSDT',
];
const HORIZONS = [1, 4, 24];

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [horizon, setHorizon] = useState(1);
  const [refreshKey, setRefreshKey] = useState(0);
  const [signal, setSignal] = useState<SignalResponse | null>(null);
  const [allPerformance, setAllPerformance] = useState<PredictionPerformance[]>([]);
  const [dailyReport, setDailyReport] = useState<DailyReportPreview | null>(null);
  const [paperResult, setPaperResult] = useState<PaperResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = window.setInterval(() => setRefreshKey((value) => value + 1), 5 * 60 * 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let active = true;
    async function loadSignal() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_URL}/signal`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: selectedSymbol, horizon }),
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || `İstek başarısız: ${response.status}`);
        if (active) setSignal(body);

        const performanceResponse = await fetch(`${API_URL}/performance/all?horizon=${horizon}&limit=5`);
        const performanceBody: AllPerformanceResponse = await performanceResponse.json();
        if (!performanceResponse.ok) throw new Error(`Toplu başarı verisi alınamadı: ${performanceResponse.status}`);
        if (active) setAllPerformance(performanceBody.coins);

        const reportResponse = await fetch(`${API_URL}/report/daily/preview`);
        const reportBody: DailyReportPreview = await reportResponse.json();
        if (!reportResponse.ok) throw new Error(`Günlük rapor önizlemesi alınamadı: ${reportResponse.status}`);
        if (active) setDailyReport(reportBody);
      } catch (caught) {
        if (active) {
          setSignal(null);
          setError(caught instanceof Error ? caught.message : 'Bilinmeyen hata');
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadSignal();
    return () => { active = false; };
  }, [selectedSymbol, horizon, refreshKey]);

  async function runPaperTrade() {
    setExecuting(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/paper/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: selectedSymbol, horizon }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `İstek başarısız: ${response.status}`);
      setPaperResult(body);
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Bilinmeyen hata');
    } finally {
      setExecuting(false);
    }
  }

  const prediction = signal?.prediction;
  const actionColor = signal?.action === 'BUY' ? '#15803d' : signal?.action === 'SELL' ? '#b91c1c' : '#a16207';

  return (
    <main style={{ fontFamily: 'Inter, system-ui, sans-serif', maxWidth: 980, margin: '2rem auto', padding: '1.5rem', color: '#0f172a' }}>
      <header style={{ marginBottom: '1.5rem' }}>
        <p style={{ textTransform: 'uppercase', letterSpacing: '0.2em', color: '#64748b', margin: 0 }}>CoinCast Pulse</p>
        <h1 style={{ margin: '0.35rem 0' }}>Risk kontrollü paper al-sat</h1>
        <p style={{ margin: 0, color: '#475569' }}>Canlı kapalı mumlar, coin/ufuk özel model, kısa rapor ve bildirim altyapısı.</p>
      </header>

      <section style={{ display: 'flex', gap: '0.75rem', alignItems: 'end', flexWrap: 'wrap', marginBottom: '1rem' }}>
        <label>Sembol<br />
          <select value={selectedSymbol} onChange={(event) => setSelectedSymbol(event.target.value)} style={{ padding: '0.65rem', minWidth: 150 }}>
            {SYMBOLS.map((symbol) => <option key={symbol}>{symbol}</option>)}
          </select>
        </label>
        <label>Ufuk<br />
          <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))} style={{ padding: '0.65rem', minWidth: 110 }}>
            {HORIZONS.map((value) => <option key={value} value={value}>{value} saat</option>)}
          </select>
        </label>
        <button onClick={() => setRefreshKey((value) => value + 1)} disabled={loading} style={{ padding: '0.72rem 1rem' }}>Yenile</button>
        <button onClick={runPaperTrade} disabled={executing || loading || !signal} style={{ padding: '0.72rem 1rem', background: '#1d4ed8', color: 'white', border: 0, borderRadius: 6 }}>
          {executing ? 'Çalışıyor…' : 'Paper işlemi çalıştır'}
        </button>
      </section>

      {loading && <p>Güncel sinyal hazırlanıyor…</p>}
      {error && <div role="alert" style={{ padding: '1rem', background: '#fef2f2', color: '#991b1b', borderRadius: 10 }}>{error}</div>}

      {signal && prediction && (
        <section style={{ border: '1px solid #dbeafe', borderRadius: 16, padding: '1.25rem', background: '#f8fafc' }}>
          {!prediction.model_verified && (
            <div style={{ padding: '0.8rem', background: '#fff7ed', color: '#9a3412', borderRadius: 8, marginBottom: '1rem' }}>
              Bu model kalite kapısını geçmedi. Paper deneme yapılabilir; gerçek para işlemi kilitlidir.
            </div>
          )}

          <div style={{ display: 'grid', gap: '0.8rem', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))' }}>
            <Metric label="Sinyal" value={signal.action} color={actionColor} />
            <Metric label="Güncel fiyat" value={formatPrice(prediction.current_price)} />
            <Metric label="Tahmini fiyat" value={formatPrice(prediction.predicted_price)} color="#1d4ed8" />
            <Metric label="Beklenen getiri" value={`%${(prediction.predicted_return * 100).toFixed(3)}`} />
            <Metric label="Paper özsermaye" value={signal.account.equity.toFixed(2)} />
          </div>

          <p style={{ color: '#475569' }}><strong>Risk kararı:</strong> {signal.risk.reason}</p>
          <p style={{ color: '#475569' }}>
            <strong>Belirsizlik bandı:</strong> %{(prediction.confidence_interval.lower * 100).toFixed(3)} – %{(prediction.confidence_interval.upper * 100).toFixed(3)}
          </p>
          <p style={{ color: '#475569' }}>
            <strong>Tahmini fiyat bandı:</strong> {formatPrice(prediction.predicted_price_interval.lower)} – {formatPrice(prediction.predicted_price_interval.upper)}
          </p>

          {prediction.mini_chart.length > 0 && <MiniChart values={prediction.mini_chart} />}

          <small style={{ color: '#64748b' }}>
            Veri: {new Date(prediction.data_timestamp).toLocaleString()} · Model: {prediction.model_id} · Kaynak: {prediction.source}
          </small>
        </section>
      )}

      {signal && (
        <PerformancePanel performance={signal.performance} />
      )}

      {signal && allPerformance.length > 0 && (
        <AllCoinsPerformance
          coins={allPerformance}
          selectedSymbol={selectedSymbol}
          horizon={horizon}
          onSelect={setSelectedSymbol}
        />
      )}

      {dailyReport && <DailyReportPanel report={dailyReport} />}

      {paperResult && (
        <section style={{ marginTop: '1rem', padding: '1rem', background: paperResult.trade ? '#f0fdf4' : '#f8fafc', borderRadius: 12 }}>
          <h2 style={{ marginTop: 0, fontSize: '1.1rem' }}>Kısa işlem raporu</h2>
          <p>{paperResult.report}</p>
          {paperResult.notifications.length > 0 && (
            <p style={{ color: '#475569' }}>
              Bildirimler: {paperResult.notifications.map((item) => `${item.channel}: ${item.sent ? 'gönderildi' : item.reason}`).join(' · ')}
            </p>
          )}
        </section>
      )}
    </main>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return <div style={{ padding: '0.9rem', background: 'white', borderRadius: 10 }}><small style={{ color: '#64748b' }}>{label}</small><div style={{ fontWeight: 750, fontSize: '1.25rem', color }}>{value}</div></div>;
}

function PerformancePanel({ performance }: { performance: PredictionPerformance }) {
  const hasResults = performance.resolved_predictions > 0;
  const ratio = performance.price_improvement_ratio;
  return (
    <section style={{ marginTop: '1rem', border: '1px solid #e2e8f0', borderRadius: 16, padding: '1.25rem', background: 'white' }}>
      <h2 style={{ margin: '0 0 0.35rem', fontSize: '1.2rem' }}>Geçmiş Tahmin Başarısı</h2>
      <p style={{ margin: '0 0 1rem', color: '#475569' }}>
        {hasResults
          ? `${performance.symbol} için ${performance.horizon} saatlik canlı tahminlerin gerçekleşen sonuçları.`
          : `Canlı ölçüm birikiyor. İlk ${performance.horizon} saatlik tahmin süresi dolunca sonuç oluşacak. Ekran 5 dakikada bir yenilenir.`}
      </p>

      <div style={{ display: 'grid', gap: '0.8rem', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
        <Metric label="Sonuçlanan / bekleyen" value={`${performance.resolved_predictions} / ${performance.pending_predictions}`} />
        <Metric label="Yön doğruluğu" value={percentOrDash(performance.direction_accuracy)} />
        <Metric label="Model fiyat hatası (MAE)" value={performance.mae === null ? '—' : formatPrice(performance.mae)} />
        <Metric label="Değişmez fiyat hatası" value={performance.naive_mae === null ? '—' : formatPrice(performance.naive_mae)} />
        <Metric label="Modele avantaj oranı" value={ratio === null ? '—' : `${ratio.toFixed(2)}×`} color={ratio === null ? undefined : ratio > 1 ? '#15803d' : '#b91c1c'} />
        <Metric label="Tahmin bandı başarısı" value={percentOrDash(performance.interval_coverage)} />
      </div>

      {hasResults && (
        <p style={{ color: '#64748b', fontSize: '0.86rem' }}>
          Avantaj oranı 1’in üzerindeyse model, “fiyat değişmez” karşılaştırmasından daha düşük hata üretmiştir.
        </p>
      )}

      {performance.recent.length > 0 && (
        <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 650 }}>
            <thead>
              <tr style={{ color: '#64748b', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>
                <th style={tableCell}>Hedef zaman</th>
                <th style={tableCell}>Başlangıç</th>
                <th style={tableCell}>Tahmin</th>
                <th style={tableCell}>Gerçekleşen</th>
                <th style={tableCell}>Yön sonucu</th>
              </tr>
            </thead>
            <tbody>
              {performance.recent.map((record) => (
                <tr key={record.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={tableCell}>{new Date(record.target_timestamp).toLocaleString()}</td>
                  <td style={tableCell}>{formatPrice(record.current_price)}</td>
                  <td style={tableCell}>{formatPrice(record.predicted_price)}</td>
                  <td style={tableCell}>{record.actual_price === null ? 'Bekliyor' : formatPrice(record.actual_price)}</td>
                  <td style={{ ...tableCell, color: record.direction_correct === null ? '#a16207' : record.direction_correct ? '#15803d' : '#b91c1c', fontWeight: 700 }}>
                    {record.direction_correct === null ? 'Bekliyor' : record.direction_correct ? 'Doğru' : 'Yanlış'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function AllCoinsPerformance({
  coins,
  selectedSymbol,
  horizon,
  onSelect,
}: {
  coins: PredictionPerformance[];
  selectedSymbol: string;
  horizon: number;
  onSelect: (symbol: string) => void;
}) {
  const resolvedTotal = coins.reduce((sum, coin) => sum + coin.resolved_predictions, 0);
  const pendingTotal = coins.reduce((sum, coin) => sum + coin.pending_predictions, 0);
  return (
    <section style={{ marginTop: '1rem', border: '1px solid #bfdbfe', borderRadius: 16, padding: '1.25rem', background: '#eff6ff' }}>
      <h2 style={{ margin: '0 0 0.35rem', fontSize: '1.2rem' }}>Bütün Coinlerin Tahmin Başarısı</h2>
      <p style={{ margin: '0 0 1rem', color: '#475569' }}>
        13 coinin {horizon} saatlik sonuçları · {resolvedTotal} sonuçlandı · {pendingTotal} bekliyor
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 790, background: 'white', borderRadius: 10 }}>
          <thead>
            <tr style={{ color: '#475569', textAlign: 'left', borderBottom: '1px solid #cbd5e1' }}>
              <th style={tableCell}>Coin</th>
              <th style={tableCell}>Sonuç / bekleyen</th>
              <th style={tableCell}>Yön doğruluğu</th>
              <th style={tableCell}>Model MAE</th>
              <th style={tableCell}>Değişmez MAE</th>
              <th style={tableCell}>Avantaj</th>
              <th style={tableCell}>Band başarısı</th>
            </tr>
          </thead>
          <tbody>
            {coins.map((coin) => {
              const selected = coin.symbol === selectedSymbol;
              const ratio = coin.price_improvement_ratio;
              return (
                <tr key={coin.symbol} style={{ borderBottom: '1px solid #e2e8f0', background: selected ? '#dbeafe' : 'white' }}>
                  <td style={tableCell}>
                    <button onClick={() => onSelect(coin.symbol)} style={{ border: 0, padding: 0, background: 'transparent', color: '#1d4ed8', fontWeight: 800, cursor: 'pointer' }}>
                      {coin.symbol.replace('USDT', '')}
                    </button>
                  </td>
                  <td style={tableCell}>{coin.resolved_predictions} / {coin.pending_predictions}</td>
                  <td style={tableCell}>{percentOrDash(coin.direction_accuracy)}</td>
                  <td style={tableCell}>{coin.mae === null ? '—' : formatPrice(coin.mae)}</td>
                  <td style={tableCell}>{coin.naive_mae === null ? '—' : formatPrice(coin.naive_mae)}</td>
                  <td style={{ ...tableCell, color: ratio === null ? '#64748b' : ratio > 1 ? '#15803d' : '#b91c1c', fontWeight: 700 }}>
                    {ratio === null ? '—' : `${ratio.toFixed(2)}×`}
                  </td>
                  <td style={tableCell}>{percentOrDash(coin.interval_coverage)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <small style={{ display: 'block', color: '#64748b', marginTop: '0.75rem' }}>
        Coin adına tıklayarak ayrıntılı tahmin ve geçmiş sonuçlarını açabilirsiniz.
      </small>
    </section>
  );
}

function DailyReportPanel({ report }: { report: DailyReportPreview }) {
  return (
    <section style={{ marginTop: '1rem', border: '1px solid #d1fae5', borderRadius: 16, padding: '1.25rem', background: '#f0fdf4' }}>
      <h2 style={{ margin: '0 0 0.35rem', fontSize: '1.2rem' }}>Otomatik Günlük Rapor</h2>
      <p style={{ margin: '0 0 0.8rem', color: '#475569' }}>
        Her gün saat 20:00’de oluşturulur. Bilgisayar açıldığında CoinCast servisleri otomatik başlar.
      </p>
      <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', marginBottom: '0.8rem' }}>
        {report.channels.map((channel) => (
          <span key={channel.channel} style={{ padding: '0.35rem 0.6rem', borderRadius: 999, background: channel.configured ? '#dcfce7' : '#ffedd5', color: channel.configured ? '#166534' : '#9a3412', fontWeight: 700, fontSize: '0.85rem' }}>
            {channel.channel === 'EmailNotifier' ? 'E-posta' : 'SMS'}: {channel.configured ? 'hazır' : 'ayar gerekli'}
          </span>
        ))}
      </div>
      <details>
        <summary style={{ cursor: 'pointer', color: '#166534', fontWeight: 700 }}>Son rapor önizlemesini göster</summary>
        <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', lineHeight: 1.55, padding: '0.8rem', background: 'white', borderRadius: 10 }}>{report.report}</pre>
      </details>
    </section>
  );
}

const tableCell = { padding: '0.65rem 0.5rem', whiteSpace: 'nowrap' as const };

function percentOrDash(value: number | null) {
  return value === null ? '—' : `%${(value * 100).toFixed(1)}`;
}

function formatPrice(value: number) {
  const maximumFractionDigits = value >= 1000 ? 2 : value >= 1 ? 4 : 8;
  return value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits });
}

function MiniChart({ values }: { values: number[] }) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const points = values.map((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * 90 + 5;
    const normalized = max === min ? 20 : ((value - min) / (max - min)) * 30 + 5;
    return `${x},${40 - normalized}`;
  }).join(' ');
  return <div style={{ padding: '0.7rem', background: 'white', borderRadius: 10, marginBottom: '0.8rem' }}><svg viewBox="0 0 100 40" style={{ width: '100%', height: 130 }}><polyline fill="none" stroke="#2563eb" strokeWidth="2" points={points} /></svg></div>;
}

export default App;
