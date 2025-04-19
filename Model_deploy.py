import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Define the SAME model architecture
# ------------------------------
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# ------------------------------
# Load Model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANNModel().to(device)
model.load_state_dict(torch.load('churn_model.pth'))
model.eval()
print("✅ Model loaded!")

# ------------------------------
# Load scaler parameters
# ------------------------------
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# ------------------------------
# Predict for a new customer
# ------------------------------

input_data = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                        0, 0, 1, 0, 0, 1, 1, 0, 70.35, 1397.45,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                        1, 0, 0, 1, 0, 0, 1, 0, 0, 1
                        ]], dtype=np.float32)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()

print(f"🔍 Churn Probability: {prob:.4f}")
if prob > 0.5:
    print("🚨 Customer likely to churn.")
else:
    print("✅ Customer likely to stay.")
