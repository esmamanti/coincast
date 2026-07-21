import { useEffect, useState } from 'react';

type ConfidenceInterval = {
  lower: number;
  upper: number;
};

type PredictionResponse = {
  symbol: string;
  horizon: number;
  predicted_return: number;
  confidence_interval: ConfidenceInterval;
  generated_at: string;
  source: string;
};

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 'LINKUSDT', 'HYPEUSDT'];

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');

  useEffect(() => {
    let isMounted = true;

    async function loadPrediction(symbol: string) {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, horizon: 1 }),
        });

        if (!response.ok) {
          throw new Error(`Request failed with ${response.status}`);
        }

        const data = await response.json();
        if (isMounted) {
          setPrediction(data);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
          setPrediction(null);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    void loadPrediction(selectedSymbol);

    return () => {
      isMounted = false;
    };
  }, [selectedSymbol]);

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', maxWidth: 800, margin: '3rem auto', padding: '2rem', color: '#0f172a' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <p style={{ textTransform: 'uppercase', letterSpacing: '0.24em', color: '#64748b', marginBottom: '0.35rem' }}>CoinCast Pulse</p>
        <h1 style={{ margin: '0 0 0.4rem', fontSize: '2rem' }}>Live market signal</h1>
        <p style={{ margin: 0, color: '#475569' }}>Choose a symbol and inspect the latest model prediction with an estimated confidence range.</p>
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        <label htmlFor="symbol-select" style={{ fontWeight: 600 }}>Symbol</label>
        <select
          id="symbol-select"
          value={selectedSymbol}
          onChange={(event) => setSelectedSymbol(event.target.value)}
          style={{ padding: '0.6rem 0.8rem', borderRadius: 8, border: '1px solid #cbd5e1', minWidth: 160 }}
        >
          {SYMBOLS.map((symbol) => (
            <option key={symbol} value={symbol}>
              {symbol}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setSelectedSymbol((current) => current)}
          style={{ padding: '0.6rem 0.8rem', borderRadius: 8, border: '1px solid #2563eb', background: '#eff6ff', color: '#1d4ed8', cursor: 'pointer' }}
        >
          Refresh
        </button>
      </div>

      {loading && (
        <div style={{ padding: '1rem', borderRadius: 12, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
          <strong>Generating prediction…</strong>
          <p style={{ margin: '0.3rem 0 0', color: '#64748b' }}>The backend is preparing the latest signal for {selectedSymbol}.</p>
        </div>
      )}

      {error && (
        <div role="alert" style={{ padding: '1rem', borderRadius: 12, background: '#fef2f2', border: '1px solid #fecaca', color: '#b91c1c' }}>
          <strong>Prediction unavailable.</strong>
          <div>{error}</div>
        </div>
      )}

      {prediction && (
        <div style={{ border: '1px solid #dbeafe', borderRadius: 16, padding: '1.25rem', background: 'linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%)', boxShadow: '0 8px 24px rgba(15, 23, 42, 0.06)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
            <div>
              <p style={{ margin: 0, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.24em', fontSize: '0.8rem' }}>Prediction</p>
              <h2 style={{ margin: '0.2rem 0', fontSize: '1.4rem' }}>{prediction.symbol}</h2>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>Horizon</p>
              <p style={{ margin: '0.2rem 0 0', fontWeight: 700 }}>{prediction.horizon} bar</p>
            </div>
          </div>

          <div style={{ marginTop: '1rem', display: 'grid', gap: '0.8rem', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
            <div style={{ padding: '0.9rem', borderRadius: 12, background: 'white' }}>
              <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>Expected return</p>
              <p style={{ margin: '0.3rem 0 0', fontSize: '1.35rem', fontWeight: 700 }}>{prediction.predicted_return.toFixed(6)}</p>
            </div>
            <div style={{ padding: '0.9rem', borderRadius: 12, background: 'white' }}>
              <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>Confidence interval</p>
              <p style={{ margin: '0.3rem 0 0', fontSize: '1.05rem', fontWeight: 600 }}>
                [{prediction.confidence_interval.lower.toFixed(6)}, {prediction.confidence_interval.upper.toFixed(6)}]
              </p>
            </div>
            <div style={{ padding: '0.9rem', borderRadius: 12, background: 'white' }}>
              <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>Source</p>
              <p style={{ margin: '0.3rem 0 0', fontWeight: 600 }}>{prediction.source}</p>
            </div>
          </div>

          <p style={{ marginTop: '1rem', color: '#475569' }}>
            Generated at {new Date(prediction.generated_at).toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
