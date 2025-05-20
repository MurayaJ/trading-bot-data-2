import numpy as np
import pandas as pd
import talib
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from trading_algorithm import TradingAlgorithm
import json
import time
import logging

class DigitUnder5(TradingAlgorithm):
    def __init__(self, app_id, token, target_profit, session_id, model_paths):
        super().__init__(app_id, token, target_profit, session_id, model_paths)
        self.expected_features = 10
        self.training_data = []
        self.training_targets = []
        self.training_weights = []
        self.alpha_win = 0.15
        self.alpha_loss = 0.05
        self.loss_multiplier = 2.2
        self.load_models()

    def load_models(self):
        for path in self.model_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.markov_p1 = joblib.load(self.model_paths["markov_p1"])
            self.markov_p2 = joblib.load(self.model_paths["markov_p2"])
            self.rf_model = joblib.load(self.model_paths["rf_digit_predictor"])
            self.scaler = joblib.load(self.model_paths["feature_scaler"])
            if self.markov_p1.shape != (10, 10) or self.markov_p2.shape != (100, 10):
                self.initialize_default_models()
            elif hasattr(self.rf_model, 'n_features_in_') and self.rf_model.n_features_in_ != self.expected_features:
                self.initialize_default_models()
            elif hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.expected_features:
                self.initialize_default_models()
            else:
                self.output.append("Models loaded successfully.")
        except Exception as e:
            self.output.append(f"Error loading models: {e}. Initializing defaults.")
            self.initialize_default_models()

    def initialize_default_models(self):
        self.markov_p1 = np.full((10, 10), 0.1)
        self.markov_p2 = np.full((100, 10), 0.1)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def save_models(self):
        try:
            joblib.dump(self.markov_p1, self.model_paths["markov_p1"])
            joblib.dump(self.markov_p2, self.model_paths["markov_p2"])
            joblib.dump(self.rf_model, self.model_paths["rf_digit_predictor"])
            joblib.dump(self.scaler, self.model_paths["feature_scaler"])
            self.output.append(f"Models saved at {self.training_samples} samples.")
        except Exception as e:
            self.output.append(f"Error saving models: {e}")

    def get_last_digit(self, tick):
        tick_rounded = round(float(tick), 2)
        return int(f"{tick_rounded:.2f}"[-1])

    def update_dataframe(self, tick, timestamp):
        tick = float(tick)
        last_digit = self.get_last_digit(tick)
        new_row = {
            'Time': timestamp, 'Tick': tick, 'Last_Digit': last_digit,
            'Hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'Hour_cos': np.cos(2 * np.pi * timestamp.hour / 24)
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        if len(self.df) >= 6:
            self.df['MA_6'] = talib.SMA(self.df['Tick'].values, timeperiod=6)
        if len(self.df) >= 14:
            self.df['RSI'] = talib.RSI(self.df['Tick'].values, timeperiod=14)
        if len(self.df) >= 20:
            self.df['Volatility'] = self.df['Tick'].rolling(window=20).std()
        self.df = self.df.tail(100)

    def update_markov_models(self, d_t, d_t_plus_1, d_t_minus_1=None, is_win=None):
        alpha = self.alpha_win if is_win else self.alpha_loss if is_win is not None else 0.1
        adjustment = 1.5 if is_win else 0.5 if is_win is not None else 1.0
        self.markov_p1[d_t, d_t_plus_1] = (1 - alpha) * self.markov_p1[d_t, d_t_plus_1] + alpha * adjustment
        for k in range(10):
            if k != d_t_plus_1:
                self.markov_p1[d_t, k] = (1 - alpha) * self.markov_p1[d_t, k]
        self.markov_p1[d_t, :] /= np.sum(self.markov_p1[d_t, :])
        if d_t_minus_1 is not None:
            state = 10 * d_t_minus_1 + d_t
            self.markov_p2[state, d_t_plus_1] = (1 - alpha) * self.markov_p2[state, d_t_plus_1] + alpha * adjustment
            for k in range(10):
                if k != d_t_plus_1:
                    self.markov_p2[state, k] = (1 - alpha) * self.markov_p2[state, k]
            self.markov_p2[state, :] /= np.sum(self.markov_p2[state, :])

    def get_features(self):
        if len(self.digit_history) < 3 or len(self.df) < 20:
            return None
        recent_digits = self.digit_history[-3:]
        d_t = self.digit_history[-1]
        d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else 0
        P_under, P_not_under = self.predict_one_step(d_t, d_t_minus_1)
        indicators = self.df.iloc[-1][['MA_6', 'RSI', 'Volatility', 'Hour_sin', 'Hour_cos']].values
        return list(recent_digits) + [P_under, P_not_under] + list(indicators)

    def predict_one_step(self, d_t, d_t_minus_1=None):
        if d_t_minus_1 is not None:
            p_next = 0.5 * self.markov_p1[d_t, :] + 0.5 * self.markov_p2[10 * d_t_minus_1 + d_t, :]
        else:
            p_next = self.markov_p1[d_t, :]
        P_under = np.sum(p_next[:5])  # Probability for digits 0-4
        P_not_under = 1 - P_under
        return P_under, P_not_under

    def train_rf_predictor(self):
        if len(self.training_data) < 100:
            return
        X = np.array(self.training_data)
        y = np.array(self.training_targets)
        weights = np.array(self.training_weights)
        if X.shape[1] != self.expected_features:
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            return
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.rf_model.fit(X_scaled, y, sample_weight=weights)

    def get_rf_prediction(self, features):
        if features is None or not hasattr(self.scaler, 'n_features_in_') or not hasattr(self.rf_model, 'classes_'):
            return [0.5, 0.5]
        features_array = np.array([features])
        features_scaled = self.scaler.transform(features_array)
        pred_proba = self.rf_model.predict_proba(features_scaled)[0]
        return pred_proba

    def get_rf_threshold(self):
        base_threshold = 0.55
        loss_increment = 0.02
        max_threshold = 0.75
        threshold = base_threshold + loss_increment * min(self.consecutive_losses, 10)
        return min(threshold, max_threshold)

    def calculate_profit(self, is_win):
        return round(self.amount * 0.8857, 2) if is_win else -self.amount

    def adjust_amount(self, is_win):
        if is_win:
            self.amount = self.initial_amount
            self.consecutive_losses = 0
        else:
            self.amount = min(round(self.amount * self.loss_multiplier, 2), self.account_balance * 0.9)
            self.consecutive_losses += 1
        self.price = self.amount

    def buy_contract(self, ws, contract_type):
        self.last_features = self.get_features()
        self.prediction_tick = self.digit_history[-2] if len(self.digit_history) >= 2 else None
        self.entry_tick = self.digit_history[-1]
        self.is_trading = True
        self.last_trade_time = time.time()
        json_data = json.dumps({
            "buy": 1, "subscribe": 1, "price": round(self.price, 2),
            "parameters": {
                "amount": round(self.amount, 2), "basis": "stake", "contract_type": contract_type,
                "currency": "USD", "duration": 1, "duration_unit": "t", "symbol": "R_100",
                "barrier": "5"
            }
        })
        ws.send(json_data)
        self.output.append(f"Trade placed: {contract_type}, Amount: {self.amount:.2f}, Entry Tick: {self.entry_tick}")

    def process_message(self, ws, data):
        if 'error' in data:
            self.output.append(f"API Error: {data['error']['message']}")
            return

        if data.get('msg_type') == 'authorize':
            ws.send(json.dumps({"ticks": "R_100", "subscribe": 1}))
        elif data.get('msg_type') == 'tick':
            tick = float(data['tick']['quote'])
            timestamp = datetime.fromtimestamp(data['tick']['epoch'])
            last_digit = self.get_last_digit(tick)

            if len(self.digit_history) >= 3:
                features = self.get_features()
                if features:
                    target = 1 if last_digit < 5 else 0
                    self.training_data.append(features)
                    self.training_targets.append(target)
                    self.training_weights.append(1.0)
                    if len(self.training_data) > 1000:
                        self.training_data.pop(0)
                        self.training_targets.pop(0)
                        self.training_weights.pop(0)
                    # Train as soon as we have enough data
                    if len(self.training_data) >= 100 and not hasattr(self.rf_model, 'classes_'):
                        self.train_rf_predictor()

            self.digit_history.append(last_digit)
            if len(self.digit_history) > 100:
                self.digit_history.pop(0)

            self.update_dataframe(tick, timestamp)

            if len(self.digit_history) >= 2:
                d_t = self.digit_history[-2]
                d_t_plus_1 = self.digit_history[-1]
                d_t_minus_1 = self.digit_history[-3] if len(self.digit_history) >= 3 else None
                self.update_markov_models(d_t, d_t_plus_1, d_t_minus_1)
                self.training_samples += 1
                if self.training_samples % 100 == 0:
                    self.save_models()
                    self.train_rf_predictor()

            if self.training_samples >= 100 and not self.is_trading and (time.time() - self.last_trade_time) >= self.trade_cooldown and self.cumulative_profit < self.target_profit:
                features = self.get_features()
                if features:
                    rf_pred = self.get_rf_prediction(features)
                    d_t = self.digit_history[-1]
                    d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else None
                    P_under, P_not_under = self.predict_one_step(d_t, d_t_minus_1)
                    rf_threshold = self.get_rf_threshold()
                    if rf_pred[1] > rf_threshold and P_under > 0.60:
                        self.buy_contract(ws, "DIGITUNDER")
        elif 'proposal_open_contract' in data:
            contract = data['proposal_open_contract']
            if contract.get('is_sold', False):
                exit_tick = float(contract['exit_tick'])
                last_digit = self.get_last_digit(exit_tick)
                contract_type = contract['contract_type']
                barrier = int(contract['barrier'])
                is_win = (contract_type == "DIGITUNDER" and last_digit < barrier)
                profit = self.calculate_profit(is_win)
                self.adjust_amount(is_win)
                self.account_balance += profit
                self.cumulative_profit += profit
                self.recent_trades.append((contract_type, is_win))
                if len(self.recent_trades) > 50:
                    self.recent_trades.pop(0)

                if self.prediction_tick is not None and self.entry_tick is not None and self.last_features is not None:
                    self.update_markov_models(self.entry_tick, last_digit, self.prediction_tick, is_win=is_win)
                    target = 1 if last_digit < 5 else 0
                    weight = 2.0 if is_win else 0.5
                    self.training_data.append(self.last_features)
                    self.training_targets.append(target)
                    self.training_weights.append(weight)
                    self.train_rf_predictor()

                self.output.append(f"Trade Result: {contract_type}, Entry Tick: {self.entry_tick}, Exit Tick: {exit_tick}, {'Win' if is_win else 'Loss'}, Profit: {profit:.2f}, Balance: {self.account_balance:.2f}")
                self.is_trading = False
                self.entry_tick = None
                self.last_features = None
                self.prediction_tick = None

                if self.cumulative_profit >= self.target_profit or self.account_balance <= 0:
                    self.save_models()
                    self.stop_trading = True
