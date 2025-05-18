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
from trading_algorithm import TradingAlgorithm

class DigitUnder5(TradingAlgorithm):
    """Trading algorithm for predicting digits under 5."""
    def __init__(self, app_id, token, target_profit, session_id, model_paths):
        super().__init__(app_id, token, target_profit, session_id, model_paths)
        self.load_models()

    def load_models(self):
        try:
            self.markov_p1 = joblib.load(self.model_paths["markov_p1"])
            self.markov_p2 = joblib.load(self.model_paths["markov_p2"])
            self.rf_model = joblib.load(self.model_paths["rf_digit_predictor"])
            self.scaler = joblib.load(self.model_paths["feature_scaler"])
            self.output.append("Models loaded successfully.")
        except Exception as e:
            self.output.append(f"Error loading models: {e}")
            self.stop_trading = True

    def process_message(self, data):
        if "tick" in data:
            tick_data = data["tick"]
            current_time = datetime.fromtimestamp(tick_data["epoch"])
            tick_value = tick_data["quote"]
            last_digit = int(str(tick_value)[-1])

            self.digit_history.append(last_digit)
            if len(self.digit_history) > 100:
                self.digit_history.pop(0)

            self.df = pd.concat([self.df, pd.DataFrame([{
                "Time": current_time,
                "Tick": tick_value,
                "Last_Digit": last_digit,
                "MA_6": np.nan,
                "RSI": np.nan,
                "Volatility": np.nan,
                "Hour_sin": np.sin(2 * np.pi * current_time.hour / 24),
                "Hour_cos": np.cos(2 * np.pi * current_time.hour / 24)
            }])], ignore_index=True)

            if len(self.df) >= 6:
                self.df["MA_6"] = self.df["Tick"].rolling(window=6).mean()
                self.df["RSI"] = talib.RSI(self.df["Tick"], timeperiod=14)
                self.df["Volatility"] = self.df["Tick"].rolling(window=20).std()

            if len(self.df) > 20 and not self.df.iloc[-1].isna().any():
                features = self.df[["MA_6", "RSI", "Volatility", "Hour_sin", "Hour_cos"]].iloc[-1].values.reshape(1, -1)
                scaled_features = self.scaler.transform(features)
                prediction = self.rf_model.predict(scaled_features)[0]
                self.output.append(f"Prediction made: {prediction}")

                if time.time() - self.last_trade_time >= self.trade_cooldown and not self.stop_trading:
                    self.last_trade_time = time.time()
                    trade_result = "win" if (prediction < 5 and last_digit < 5) or (prediction >= 5 and last_digit >= 5) else "loss"
                    profit = self.amount * 0.95 if trade_result == "win" else -self.amount
                    self.cumulative_profit += profit
                    self.account_balance += profit
                    self.output.append(f"Trade: {trade_result}, Profit: {profit:.2f}, Balance: {self.account_balance:.2f}")

                    if trade_result == "loss":
                        self.consecutive_losses += 1
                        self.amount *= 2
                    else:
                        self.consecutive_losses = 0
                        self.amount = self.initial_amount

                    if self.cumulative_profit >= self.target_profit or self.account_balance <= 0:
                        self.stop_trading = True
                        update_trading_status(self.session_id, 'inactive')
                        self.output.append("Trading stopped: Target reached or balance depleted.")
