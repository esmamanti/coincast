import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_backend.main import app


class BackendPredictTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_endpoint_uses_model_output(self):
        response = self.client.post("/predict", json={"symbol": "BTCUSDT", "horizon": 1})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertIn("predicted_return", payload)
        self.assertNotEqual(payload.get("source"), "mock_model")


if __name__ == "__main__":
    unittest.main()
