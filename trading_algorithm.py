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

    def on_open(self, ws):
        auth_message = {"authorize": self.token}
        ws.send(json.dumps(auth_message))
        self.output.append("WebSocket opened and authorization sent.")

    def on_message(self, ws, message):
        data = json.loads(message)
        self.process_message(ws, data)

    def on_error(self, ws, error):
        self.output.append(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.output.append(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_trading = False

    def process_message(self, ws, data):
        # Placeholder for message processing; to be overridden by subclasses
        pass

    def run(self):
        while not self.stop_trading:
            ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if not self.stop_trading:
                self.output.append("Connection closed. Reconnecting in 5 seconds...")
                time.sleep(5)
