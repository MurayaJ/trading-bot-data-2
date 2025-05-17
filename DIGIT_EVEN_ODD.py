#This file defines the trading algorithms, with a base class and the DIGIT_EVEN_ODD implementation
import threading
import time
import websocket
import json
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

class TradingAlgorithm:
    """Base class for trading algorithms."""
    def __init__(self, app_id, token, target_profit, session_id, model_paths):
        self.app_id = app_id
        self.token = token
        self.target_profit = float(target_profit)
        self.session_id = session_id
        self.account_balance = 1000.00
        self.initial_amount = 0.35
        self.amount = self.initial_amount
        self.price = self.initial_amount
        self.cumulative_profit = 0.00
        self.is_trading = False
        self.last_trade_time = 0
        self.trade_cooldown = 5
        self.entry_tick = None
        self.last_features = None
        self.prediction_tick = None
        self.consecutive_losses = 0
        self.recent_trades = []
        self.digit_history = []
        self.training_samples = 0
        self.output = []
        self.stop_trading = False
        self.model_paths = model_paths
        self.df = pd.DataFrame(columns=["Time", "Tick", "Last_Digit", "MA_6", "RSI", "Volatility", "Hour_sin", "Hour_cos"]).astype({
            "Time": "datetime64[ns]", "Tick": "float64", "Last_Digit": "int64", "MA_6": "float64",
            "RSI": "float64", "Volatility": "float64", "Hour_sin": "float64", "Hour_cos": "float64"
        })

    def get_last_digit(self, tick):
        tick_rounded = round(float(tick), 2)
        return int(f"{tick_rounded:.2f}"[-1])

    def calculate_profit(self, is_win):
        return round(self.amount * 0.8857, 2) if is_win else -round(self.amount, 2)

    def adjust_amount(self, is_win):
        if is_win:
            self.amount = self.initial_amount
            self.consecutive_losses = 0
        else:
            self.amount = min(round(self.amount * 2.2, 2), max(self.account_balance * 0.9, 0))
            if self.amount <= 0:
                self.amount = self.initial_amount
            self.consecutive_losses += 1
        self.price = self.amount

    def update_dataframe(self, tick, timestamp):
        tick = float(tick)
        last_digit = self.get_last_digit(tick)
        new_row = {
            "Time": timestamp, "Tick": tick, "Last_Digit": last_digit,
            "Hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
            "Hour_cos": np.cos(2 * np.pi * timestamp.hour / 24)
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        if len(self.df) >= 6:
            self.df["MA_6"] = talib.SMA(self.df["Tick"].values, timeperiod=6)
        if len(self.df) >= 14:
            self.df["RSI"] = talib.RSI(self.df["Tick"].values, timeperiod=14)
        if len(self.df) >= 20:
            self.df["Volatility"] = self.df["Tick"].rolling(window=20).std()
        self.df = self.df.tail(100)

    def buy_contract(self, ws, contract_type):
        if self.account_balance < self.amount:
            self.output.append("Insufficient balance to place trade.")
            return
        self.last_features = self.get_features()
        self.prediction_tick = self.digit_history[-2] if len(self.digit_history) >= 2 else None
        self.entry_tick = self.digit_history[-1]
        self.is_trading = True
        self.last_trade_time = time.time()
        json_data = json.dumps({
            "buy": 1, "subscribe": 1, "price": round(self.price, 2),
            "parameters": {
                "amount": round(self.amount, 2), "basis": "stake", "contract_type": contract_type,
                "currency": "USD", "duration": 1, "duration_unit": "t", "symbol": "R_100"
            }
        })
        self.output.append(f"Trade placed: {contract_type}, Entry: {self.entry_tick}, Amount: {self.amount:.2f}")
        ws.send(json_data)

    def on_open(self, ws):
        ws.send(json.dumps({"authorize": self.token}))

    def on_message(self, ws, message):
        data = json.loads(message)
        if "error" in data:
            self.output.append(f"API Error: {data['error']['message']}")
            return
        if data["msg_type"] == "authorize":
            ws.send(json.dumps({"ticks": "R_100", "subscribe": 1}))
        elif data["msg_type"] == "tick":
            tick = float(data["tick"]["quote"])
            timestamp = datetime.fromtimestamp(data["tick"]["epoch"])
            last_digit = self.get_last_digit(tick)
            self.digit_history.append(last_digit)
            self.update_dataframe(tick, timestamp)
            self.update_training_data(last_digit)
            if (self.training_samples >= 100 and not self.is_trading and
                (time.time() - self.last_trade_time) >= self.trade_cooldown and
                self.cumulative_profit < self.target_profit and not self.stop_trading and
                self.account_balance >= self.amount):
                if self.should_place_trade():
                    contract_type = self.decide_contract_type()
                    self.buy_contract(ws, contract_type)
        elif "proposal_open_contract" in data:
            contract = data["proposal_open_contract"]
            if contract.get("is_sold", False):
                exit_tick = float(contract["exit_tick"])
                last_digit = self.get_last_digit(exit_tick)
                contract_type = contract["contract_type"]
                is_win = self.is_win(contract_type, last_digit)
                profit = self.calculate_profit(is_win)
                self.adjust_amount(is_win)
                self.account_balance = max(self.account_balance + profit, 0)
                self.cumulative_profit += profit
                self.recent_trades.append((contract_type, is_win))
                if len(self.recent_trades) > 50:
                    self.recent_trades.pop(0)
                self.update_models_after_trade(last_digit, is_win)
                self.output.append(f"Result: {contract_type}, Entry: {self.entry_tick}, Exit: {exit_tick:.2f}, {'Win' if is_win else 'Loss'}, Profit: {profit:.2f}, Balance: {self.account_balance:.2f}")
                self.is_trading = False
                self.entry_tick = None
                self.last_features = None
                self.prediction_tick = None
                if self.cumulative_profit >= self.target_profit or self.account_balance <= 0:
                    self.save_models()
                    self.stop_trading = True
                    update_trading_status(self.session_id, 'inactive')
                    ws.close()

    def on_error(self, ws, error):
        self.output.append(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.output.append(f"WebSocket closed: {close_msg}")
        update_trading_status(self.session_id, 'inactive')

    def run(self):
        update_trading_status(self.session_id, 'active')
        while not self.stop_trading and self.cumulative_profit < self.target_profit and self.account_balance > 0:
            api_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
            ws = websocket.WebSocketApp(
                api_url, on_message=self.on_message, on_open=self.on_open,
                on_error=self.on_error, on_close=self.on_close
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if self.stop_trading or self.cumulative_profit >= self.target_profit:
                self.save_models()
                break
            self.output.append("Reconnecting in 5 seconds...")
            time.sleep(5)

    def load_models(self):
        raise NotImplementedError("Subclasses must implement load_models")

    def save_models(self):
        raise NotImplementedError("Subclasses must implement save_models")

    def update_training_data(self, last_digit):
        raise NotImplementedError("Subclasses must implement update_training_data")

    def should_place_trade(self):
        raise NotImplementedError("Subclasses must implement should_place_trade")

    def decide_contract_type(self):
        raise NotImplementedError("Subclasses must implement decide_contract_type")

    def is_win(self, contract_type, last_digit):
        raise NotImplementedError("Subclasses must implement is_win")

    def update_models_after_trade(self, last_digit, is_win):
        raise NotImplementedError("Subclasses must implement update_models_after_trade")

    def get_features(self):
        raise NotImplementedError("Subclasses must implement get_features")

class DigitEvenOdd(TradingAlgorithm):
    """Implementation of the DIGIT_EVEN_ODD algorithm."""
    def __init__(self, app_id, token, target_profit, session_id, model_paths):
        super().__init__(app_id, token, target_profit, session_id, model_paths)
        self.alpha_win = 0.15
        self.alpha_loss = 0.05
        self.markov_p1 = np.full((10, 10), 0.1)
        self.markov_p2 = np.full((100, 10), 0.1)
        self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.expected_features = 10
        self.training_data = []
        self.training_targets = []
        self.training_weights = []
        self.load_models()

    def load_models(self):
        """Load models from the repo, initialize if missing."""
        try:
            self.markov_p1 = joblib.load(self.model_paths["markov_p1"])
            self.markov_p2 = joblib.load(self.model_paths["markov_p2"])
            self.rf_digit_predictor = joblib.load(self.model_paths["rf_digit_predictor"])
            self.feature_scaler = joblib.load(self.model_paths["feature_scaler"])
            self.output.append("Models loaded successfully.")
        except Exception as e:
            self.output.append(f"Error loading models: {e}. Initializing new models.")
            self.markov_p1 = np.full((10, 10), 0.1)
            self.markov_p2 = np.full((100, 10), 0.1)
            self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_scaler = StandardScaler()
            self.save_models()

    def save_models(self):
        """Save models to the repo and commit."""
        os.makedirs(os.path.dirname(self.model_paths["markov_p1"]), exist_ok=True)
        joblib.dump(self.markov_p1, self.model_paths["markov_p1"])
        joblib.dump(self.markov_p2, self.model_paths["markov_p2"])
        joblib.dump(self.rf_digit_predictor, self.model_paths["rf_digit_predictor"])
        joblib.dump(self.feature_scaler, self.model_paths["feature_scaler"])
        self.output.append(f"Models saved to {self.model_paths['markov_p1'].split('/')[0]}")
        commit_and_push()  # Sync model updates to repo

    def get_features(self):
        if len(self.digit_history) < 3 or len(self.df) < 20:
            return None
        recent_digits = self.digit_history[-3:]
        d_t = self.digit_history[-1]
        d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else 0
        P_even, P_odd = self.predict_one_step(d_t, d_t_minus_1)
        indicators = self.df.iloc[-1][["MA_6", "RSI", "Volatility", "Hour_sin", "Hour_cos"]].values
        features = list(recent_digits) + [P_even, P_odd] + list(indicators)
        return features if len(features) == self.expected_features else None

    def update_training_data(self, last_digit):
        if len(self.digit_history) >= 3:
            features = self.get_features()
            if features:
                target = 0 if last_digit % 2 == 0 else 1
                self.training_data.append(features)
                self.training_targets.append(target)
                self.training_weights.append(1.0)
                if len(self.training_data) > 1000:
                    self.training_data.pop(0)
                    self.training_targets.pop(0)
                    self.training_weights.pop(0)
                self.training_samples += 1
                if self.training_samples >= 100 and self.training_samples % 100 == 0:
                    self.train_rf_predictor()
                    self.save_models()

    def predict_one_step(self, d_t, d_t_minus_1=None):
        if d_t_minus_1 is not None:
            p_next = 0.5 * self.markov_p1[d_t, :] + 0.5 * self.markov_p2[10 * d_t_minus_1 + d_t, :]
        else:
            p_next = self.markov_p1[d_t, :]
        P_even = np.sum(p_next[[0, 2, 4, 6, 8]])
        P_odd = np.sum(p_next[[1, 3, 5, 7, 9]])
        return P_even, P_odd

    def get_rf_prediction(self, features):
        if features is None or not hasattr(self.feature_scaler, "n_features_in_"):
            return [0.5, 0.5]
        features_scaled = self.feature_scaler.transform([features])
        pred_proba = self.rf_digit_predictor.predict_proba(features_scaled)[0]
        return pred_proba if len(pred_proba) == 2 else [0.5, 0.5]

    def get_rf_threshold(self):
        base_threshold = 0.55
        loss_increment = 0.02
        max_threshold = 0.75
        threshold = base_threshold + loss_increment * min(self.consecutive_losses, 3) + (0.05 * max(self.consecutive_losses - 3, 0))
        return min(threshold, max_threshold)

    def should_place_trade(self):
        features = self.get_features()
        if features:
            rf_pred = self.get_rf_prediction(features)
            d_t = self.digit_history[-1]
            d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else None
            P_even, P_odd = self.predict_one_step(d_t, d_t_minus_1)
            rf_threshold = self.get_rf_threshold()
            if rf_pred[0] > rf_threshold and P_even > 0.60:
                return True
            elif rf_pred[1] > rf_threshold and P_odd > 0.60:
                return True
        return False

    def decide_contract_type(self):
        features = self.get_features()
        if features:
            rf_pred = self.get_rf_prediction(features)
            if rf_pred[0] > rf_pred[1]:
                return "DIGITEVEN"
            else:
                return "DIGITODD"
        return "DIGITEVEN"  # Default fallback

    def is_win(self, contract_type, last_digit):
        return (contract_type == "DIGITEVEN" and last_digit % 2 == 0) or \
               (contract_type == "DIGITODD" and last_digit % 2 != 0)

    def update_models_after_trade(self, last_digit, is_win):
        if self.prediction_tick is not None and self.entry_tick is not None and self.last_features:
            self.update_markov_models(self.prediction_tick, self.entry_tick, self.digit_history[-3] if len(self.digit_history) >= 3 else None)
            self.update_markov_models(self.entry_tick, last_digit, self.prediction_tick, is_win)
            target = 0 if last_digit % 2 == 0 else 1
            weight = 2.0 if is_win else 0.5
            self.training_data.append(self.last_features)
            self.training_targets.append(target)
            self.training_weights.append(weight)
            self.train_rf_predictor()

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

    def train_rf_predictor(self):
        if len(self.training_data) < 100:
            return
        X = np.array(self.training_data)
        y = np.array(self.training_targets)
        weights = np.array(self.training_weights)
        if X.shape[1] != self.expected_features:
            self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_scaler = StandardScaler()
            return
        self.feature_scaler.fit(X)
        X_scaled = self.feature_scaler.transform(X)
        self.rf_digit_predictor.fit(X_scaled, y, sample_weight=weights)
