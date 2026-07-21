import { useEffect, useState } from 'react';

type PredictionResponse = {
  symbol: string;
  horizon: number;
  predicted_return: number;
  generated_at: string;
  source: string;
};

const API_URL = 'http://127.0.0.1:8001';

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    async function loadPrediction() {
      try {
        setLoading(true);
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: 'BTCUSDT', horizon: 1 }),
        });

        if (!response.ok) {
          throw new Error(`Request failed with ${response.status}`);
        }

        const data = await response.json();
        setPrediction(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    loadPrediction();
  }, []);

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', maxWidth: 760, margin: '3rem auto', padding: '2rem' }}>
      <h1>CoinCast Pulse</h1>
      <p>Live ML prediction from the backend.</p>

      {loading && <p>Loading prediction...</p>}
      {error && <p style={{ color: 'crimson' }}>Error: {error}</p>}
      {prediction && (
        <div style={{ border: '1px solid #ddd', borderRadius: 12, padding: '1rem' }}>
          <p><strong>Symbol:</strong> {prediction.symbol}</p>
          <p><strong>Predicted return:</strong> {prediction.predicted_return.toFixed(6)}</p>
          <p><strong>Source:</strong> {prediction.source}</p>
          <p><strong>Generated at:</strong> {prediction.generated_at}</p>
        </div>
      )}
    </div>
  );
}

export default App;
