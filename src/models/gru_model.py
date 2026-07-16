import torch
import torch.nn as nn
import numpy as np


class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)      # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]        # sadece son zaman adımını al
        return self.fc(out)


def create_sequences(X: np.ndarray, y: np.ndarray, window: int = 60):
    """
    Son `window` kadar mumu kullanarak bir sonraki değeri tahmin edecek
    şekilde veriyi (samples, window, features) formatına çevirir.
    """
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i - window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_gru(X_train_seq, y_train_seq, X_test_seq, y_test_seq,
              input_size, epochs=30, batch_size=64, lr=1e-3, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    model = GRUPredictor(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1).to(device)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0

        for i in range(0, X_train_t.size(0), batch_size):
            idx = permutation[i:i + batch_size]
            batch_X, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()

        train_losses.append(epoch_loss / (X_train_t.size(0) / batch_size))
        test_losses.append(test_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.6f} - Test Loss: {test_loss:.6f}")

    return model, train_losses, test_losses