# Customer Churn Prediction

A deep learning project that predicts customer churn using an Artificial Neural Network (ANN) built with PyTorch. The model is trained on telecom customer data and achieves strong classification performance.

## Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their service. This project:

- Preprocesses and encodes telecom customer data
- Trains a 3-layer ANN using PyTorch
- Saves the trained model weights and scaler parameters
- Provides a deployment script to run predictions on new customers

## Project Structure

```
Customer_Churn/
├── Churn_data.csv          # Telecom customer dataset
├── Churn_Prediction.py     # Data preprocessing, model training & evaluation
├── Model_deploy.py         # Load model and predict on new customer data
├── checkgpu.py             # Utility to check GPU availability
├── churn_model.pth         # Saved model weights (generated after training)
├── scaler_mean.npy         # Saved scaler mean (generated after training)
└── scaler_scale.npy        # Saved scaler scale (generated after training)
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- pandas
- scikit-learn

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn
```

## Usage

### Train the Model

Run the training script to preprocess the dataset, train the ANN, and save the model:

```bash
python Churn_Prediction.py
```

This will:
1. Load and preprocess `Churn_data.csv`
2. Train the ANN for 50 epochs
3. Save `churn_model.pth`, `scaler_mean.npy`, and `scaler_scale.npy`
4. Print the test accuracy

### Run Predictions

After training, use the deployment script to predict churn probability for a new customer:

```bash
python Model_deploy.py
```

Edit the `input_data` array in `Model_deploy.py` to provide a new customer's feature values.

## Model Architecture

| Layer    | Input → Output |
|----------|----------------|
| FC1      | 40 → 64        |
| FC2      | 64 → 32        |
| Dropout  | 0.3            |
| FC3      | 32 → 1         |

- **Loss function:** BCEWithLogitsLoss  
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)  
- **Epochs:** 50  
- **Batch size:** 32

## Dataset

The dataset contains telecom customer attributes including demographics, account information, and service usage. The target column is `Churn` (Yes/No).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
