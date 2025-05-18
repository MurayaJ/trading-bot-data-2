import json  # Added to support json.dumps in buy_contract
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from db_utils import update_trading_status
from utils import commit_and_push
from trading_algorithm import TradingAlgorithm  # Assuming this is the base class module

class DigitUnder5(TradingAlgorithm):
    """Implementation of the DIGIT_UNDER_5 trading algorithm."""
    def __init__(self, app_id, token, target_profit, session_id, model_paths):
        super().__init__(app_id, token, target_profit, session_id, model_paths)
        self.alpha = 0.1  # Learning rate for Markov models
        self.markov_p1 = np.full((10, 10), 0.1)  # First-order Markov model
        self.markov_p2 = np.full((100, 10), 0.1)  # Second-order Markov model
        self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.expected_features = 9  # Features: 3 digits + P_under + 5 indicators
        self.training_data = []
        self.training_targets = []
        self.training_samples = 0
        self.load_models()

    def load_models(self):
        """Load models from the repository; reinitialize if missing or incompatible."""
        try:
            self.markov_p1 = joblib.load(self.model_paths["markov_p1"])
            self.markov_p2 = joblib.load(self.model_paths["markov_p2"])
            self.rf_digit_predictor = joblib.load(self.model_paths["rf_digit_predictor"])
            self.feature_scaler = joblib.load(self.model_paths["feature_scaler"])
            if self.markov_p1.shape != (10, 10) or self.markov_p2.shape != (100, 10):
                self.output.append("Markov model shapes incompatible. Reinitializing.")
                self.markov_p1 = np.full((10, 10), 0.1)
                self.markov_p2 = np.full((100, 10), 0.1)
            if hasattr(self.rf_digit_predictor, 'n_features_in_') and self.rf_digit_predictor.n_features_in_ != self.expected_features:
                self.output.append(f"RF predictor expects {self.rf_digit_predictor.n_features_in_} features, but {self.expected_features} required. Reinitializing.")
                self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            if hasattr(self.feature_scaler, 'n_features_in_') and self.feature_scaler.n_features_in_ != self.expected_features:
                self.output.append(f"Scaler expects {self.feature_scaler.n_features_in_} features, but {self.expected_features} required. Reinitializing.")
                self.feature_scaler = StandardScaler()
            self.output.append("Models loaded successfully.")
        except Exception as e:
            self.output.append(f"Error loading models: {e}. Initializing new models.")
            self.markov_p1 = np.full((10, 10), 0.1)
            self.markov_p2 = np.full((100, 10), 0.1)
            self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_scaler = StandardScaler()
            self.save_models()

    def save_models(self):
        """Save models to the repository and commit changes."""
        os.makedirs(os.path.dirname(self.model_paths["markov_p1"]), exist_ok=True)
        joblib.dump(self.markov_p1, self.model_paths["markov_p1"])
        joblib.dump(self.markov_p2, self.model_paths["markov_p2"])
        joblib.dump(self.rf_digit_predictor, self.model_paths["rf_digit_predictor"])
        joblib.dump(self.feature_scaler, self.model_paths["feature_scaler"])
        self.output.append(f"Models saved to {self.model_paths['markov_p1'].split('/')[0]}")
        commit_and_push()

    def get_features(self):
        """Extract features for prediction: 3 recent digits, P_under, and 5 indicators."""
        if len(self.digit_history) < 3 or len(self.df) < 20:
            return None
        recent_digits = self.digit_history[-3:]
        d_t = self.digit_history[-1]
        d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else 0
        P_under = self.predict_one_step(d_t, d_t_minus_1)
        indicators = self.df.iloc[-1][["MA_6", "RSI", "Volatility", "Hour_sin", "Hour_cos"]].values
        features = list(recent_digits) + [P_under] + list(indicators)
        if len(features) != self.expected_features:
            self.output.append(f"Feature mismatch: expected {self.expected_features}, got {len(features)}")
            return None
        return features

    def predict_one_step(self, d_t, d_t_minus_1=None):
        """Predict the probability that the next digit is under 5."""
        if d_t_minus_1 is not None:
            p_next = 0.5 * self.markov_p1[d_t, :] + 0.5 * self.markov_p2[10 * d_t_minus_1 + d_t, :]
        else:
            p_next = self.markov_p1[d_t, :]
        P_under = np.sum(p_next[:5])  # Sum probabilities for digits 0-4
        return P_under

    def update_training_data(self, last_digit):
        """Update training data with each new tick."""
        if len(self.digit_history) >= 3:
            features = self.get_features()
            if features:
                target = 1 if last_digit < 5 else 0  # 1 for digit < 5, 0 otherwise
                self.training_data.append(features)
                self.training_targets.append(target)
                if len(self.training_data) > 1000:
                    self.training_data.pop(0)
                    self.training_targets.pop(0)
                self.training_samples += 1
                if self.training_samples >= 100 and self.training_samples % 100 == 0:
                    self.train_rf_predictor()
                    self.save_models()

    def get_rf_prediction(self, features):
        """Get Random Forest prediction probabilities."""
        if features is None or not hasattr(self.feature_scaler, 'n_features_in_'):
            return [0.5, 0.5]
        features_scaled = self.feature_scaler.transform([features])
        pred_proba = self.rf_digit_predictor.predict_proba(features_scaled)[0]
        return pred_proba if len(pred_proba) == 2 else [0.5, 0.5]

    def get_dynamic_threshold(self):
        """Calculate a dynamic threshold for trade decisions."""
        base_threshold = 0.3
        loss_penalty = min(0.2, self.consecutive_losses * 0.05)
        sample_adjustment = (self.training_samples - 100) / 10000 if self.training_samples > 100 else 0
        return min(0.6, base_threshold + sample_adjustment + loss_penalty)

    def should_place_trade(self):
        """Determine if a trade should be placed based on predictions."""
        features = self.get_features()
        if features:
            rf_pred = self.get_rf_prediction(features)
            d_t = self.digit_history[-1]
            d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else None
            P_under = self.predict_one_step(d_t, d_t_minus_1)
            dynamic_threshold = self.get_dynamic_threshold()
            if P_under > dynamic_threshold and rf_pred[1] > 0.55:
                return True
        return False

    def decide_contract_type(self):
        """Specify the contract type for this algorithm."""
        return "DIGITUNDER"

    def is_win(self, contract_type, last_digit):
        """Check if the trade resulted in a win."""
        return contract_type == "DIGITUNDER" and last_digit < 5

    def update_models_after_trade(self, last_digit, is_win):
        """Update models after a trade is completed."""
        if self.entry_tick is not None and self.last_features is not None:
            self.update_markov_models(self.entry_tick, last_digit, self.digit_history[-2] if len(self.digit_history) >= 2 else None)
            target = 1 if last_digit < 5 else 0
            self.training_data.append(self.last_features)
            self.training_targets.append(target)
            self.train_rf_predictor()

    def update_markov_models(self, d_t, d_t_plus_1, d_t_minus_1=None):
        """Update Markov models based on digit transitions."""
        self.markov_p1[d_t, d_t_plus_1] = (1 - self.alpha) * self.markov_p1[d_t, d_t_plus_1] + self.alpha
        for k in range(10):
            if k != d_t_plus_1:
                self.markov_p1[d_t, k] = (1 - self.alpha) * self.markov_p1[d_t, k]
        self.markov_p1[d_t, :] /= np.sum(self.markov_p1[d_t, :])
        if d_t_minus_1 is not None:
            state = 10 * d_t_minus_1 + d_t
            self.markov_p2[state, d_t_plus_1] = (1 - self.alpha) * self.markov_p2[state, d_t_plus_1] + self.alpha
            for k in range(10):
                if k != d_t_plus_1:
                    self.markov_p2[state, k] = (1 - self.alpha) * self.markov_p2[state, k]
            self.markov_p2[state, :] /= np.sum(self.markov_p2[state, :])

    def train_rf_predictor(self):
        """Train the Random Forest predictor with collected data."""
        if len(self.training_data) < 100:
            return
        X = np.array(self.training_data)
        y = np.array(self.training_targets)
        if X.shape[1] != self.expected_features:
            self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_scaler = StandardScaler()
            return
        self.feature_scaler.fit(X)
        X_scaled = self.feature_scaler.transform(X)
        self.rf_digit_predictor.fit(X_scaled, y)

    def buy_contract(self, ws, contract_type):
        """Override buy_contract to include barrier for DIGITUNDER."""
        if contract_type != "DIGITUNDER":
            raise ValueError("This algorithm only supports DIGITUNDER contracts.")
        if self.account_balance < self.amount:
            self.output.append("Insufficient balance to place trade.")
            return
        self.last_features = self.get_features()
        self.prediction_tick = self.digit_history[-2] if len(self.digit_history) >= 2 else None
        self.entry_tick = self.digit_history[-1]
        self.is_trading = True
        self.last_trade_time = time.time()
        parameters = {
            "amount": round(self.amount, 2),
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": 1,
            "duration_unit": "t",
            "symbol": "R_100",
            "barrier": "5"  # Required for DIGITUNDER
        }
        json_data = json.dumps({
            "buy": 1,
            "subscribe": 1,
            "price": round(self.price, 2),
            "parameters": parameters
        })
        self.output.append(f"Trade placed: {contract_type}, Entry: {self.entry_tick}, Amount: {self.amount:.2f}")
        ws.send(json_data)
