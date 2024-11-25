#測試model
import dill
import torch
import numpy as np

# Load the saved model
with open('full_model.pkl', 'rb') as f:
    model_data = dill.load(f)

# Extract individual models and scalers
lstm_model = model_data['lstm_model']
lstm_optimizer_state_dict = model_data['lstm_optimizer_state_dict']
lgbm_model = model_data['lgbm_model']
ensemble_model = model_data['ensemble_model']
scaler_X = model_data['scaler_X']
scaler_y = model_data['scaler_y']

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model.to(device)
lstm_model.eval()  # Set to evaluation mode

# Assume you have new data X_new
# Preprocess new input data
X_new_scaled = scaler_X.transform(X_new)

# Convert new data to Tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

# Predict using the LSTM model
with torch.no_grad():
    y_pred_lstm_scaled = lstm_model(X_new_tensor).cpu().numpy()

# Predict using the LightGBM model
y_pred_lgbm_scaled = lgbm_model.predict(X_new_scaled).reshape(-1, 1)

# Stack the LSTM and LightGBM predictions as features for the ensemble model
ensemble_features = np.hstack((y_pred_lstm_scaled, y_pred_lgbm_scaled))

# Make final predictions using the Ensemble linear regression model
y_pred_ensemble_scaled = ensemble_model.predict(ensemble_features)

# Inverse transform the predictions
y_pred_rescaled = scaler_y.inverse_transform(y_pred_ensemble_scaled.reshape(-1, 1))

print("Predicted results:", y_pred_rescaled)
