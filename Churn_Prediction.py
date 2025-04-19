import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = './Churn_data.csv'
# load the dataset
dataset = pd.read_csv(path)
lst = dataset.columns.tolist()

# encoding
dataset['Churn'] = dataset['Churn'].map({'Yes': 1, 'No': 0})
dataset['gender'] = dataset['gender'].map({'Male': 0, 'Female': 1})
dataset['Partner'] = dataset['Partner'].map({'Yes': 1, 'No': 0})
dataset['Dependents'] = dataset['Dependents'].map({'Yes': 1, 'No': 0})
dataset['PhoneService'] = dataset['PhoneService'].map({'Yes': 1, 'No': 0})
dataset['PaperlessBilling'] = dataset['PaperlessBilling'].map({'Yes': 1, 'No': 0})

# Convert TotalCharges to float after handling blank strings
dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)
dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)
dataset['TotalCharges'] = dataset['TotalCharges'].fillna(dataset['TotalCharges'].mean())

categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaymentMethod', 
                    ]

dataset = pd.get_dummies(dataset, columns=categorical_cols)
dataset = dataset.astype({col: int for col in dataset.select_dtypes('bool').columns})

# X and y
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save scaler parameters
np.save('scaler_mean.npy', sc.mean_)
np.save('scaler_scale.npy', sc.scale_)
print("✅ Scaler parameters saved!")


# COnvert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# create dataset for models
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print('Pre-processing DONE!')

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)  # No sigmoid here because we’ll use BCEWithLogitsLoss

#training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ANNModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(50):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")


#save weights
torch.save(model.state_dict(), 'churn_model.pth')
print("✅ Model saved successfully!")

# evaluate the model
print("Evaluating the MODEL....")
model.eval()
correct = 0
total = 0

with  torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        output = model(x)
        predictions = torch.sigmoid(output)
        predicted_classes = (predictions > 0.5).float()
        correct += (predicted_classes == y).sum().item()
        total += y.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")