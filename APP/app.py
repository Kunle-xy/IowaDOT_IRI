from fastapi import FastAPI
import torch
import torch.nn as nn
import numpy as np
import joblib

app = FastAPI()


# Load the trained ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(14, 127),  # 14 input features
            nn.ReLU(),
            nn.Dropout(0.34297),
            nn.Linear(127, 53),
            nn.ReLU(),
            nn.Linear(53, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model and scaler
device = torch.device("cpu")
model = ANN().to(device)
model.load_state_dict(torch.load("MRI_PREDICTOR_MODEL.pth", map_location=device, weights_only=True))
model.eval()
# scaler = joblib.load("scaler.pkl")

# Generate random dataset for next 5 years
def generate_features(prev_iri):
    data = []
    for _ in range(5):
        features = np.random.rand(14)  # Random features
        features[0] = prev_iri  # Set the previous year IRI
        prev_iri = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).item()
        data.append(prev_iri)
    return data

@app.post("/predict")
def predict(initial_iri: float):
    iri_predictions = generate_features(initial_iri)
    return {"predicted_iris": iri_predictions}
