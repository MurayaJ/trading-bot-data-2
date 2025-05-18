import json
import time
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from db_utils import update_trading_status
from trading_algorithm import TradingAlgorithm
import websocket

class DigitOver4(TradingAlgorithm):
    """Trading algorithm for predicting digits over 4, adapted from the original code."""
    
    def __init__(self, app_id, token, target_profit, session_id, model_paths):
        super().__init__(app_id, token, target_profit, session_id, model_paths)
        # Trading parameters
        self.account_balance = 10000
        self.initial_amount = 0.35
        self.amount = self.initial_amount
        self.price = self.initial_amount
        self.loss_multiplier = 2.2
        self.trade_cooldown = 5  # Seconds
        self.expected_features = 9  # 3 digits + P_over + 5 indicators
        
        # Machine learning parameters
        self.alpha = 0.1  # Learning rate for Markov models
        self.initial_threshold = 0.3
        
        # State variables
        self.digit_history = []
        self.training_data = []
        self.training_targets = []  # 0: no trade, 1: trade DIGITOVER
        self.training_samples = 0
        self.recent_trades = []  # (contract_type, is_win) tuples
        self.consecutive_losses = 0
        self.cumulative_profit = 0
        self.is_trading = False
        self.last_trade_time = 0
        self.entry_tick = None
        self.last_features = None
        
        # DataFrame for indicators
        self.df = pd.DataFrame(columns=['Time', 'Tick', 'Last_Digit', 'MA_6', 'RSI', 'Volatility', 'Hour_sin', 'Hour_cos']).astype({
            'Time': 'datetime64[ns]', 'Tick': 'float64', 'Last_Digit': 'int64', 'MA_6': 'float64',
            'RSI': 'float64', 'Volatility': 'float64', 'Hour_sin': 'float64', 'Hour_cos': 'float64'
        })
        
        # Load models on initialization
        self.load_models()

    def load_models(self):
        """Load models from DIGIT_OVER_4_models folder, reinitialize if mismatched."""
        try:
            self.markov_p1 = joblib.load(self.model_paths["27markov_p1.joblib"])
            self.markov_p2 = joblib.load(self.model_paths["27markov_p2.joblib"])
            self.rf_model = joblib.load(self.model_paths["27rf_digit_predictor.joblib"])
            self.scaler = joblib.load(self.model_paths["27feature_scaler.joblib"])
            
            # Validate model shapes and features
            if self.markov_p1.shape != (10, 10) or self.markov_p2.shape != (100, 10):
                self.output.append("Markov model shapes incompatible. Reinitializing.")
                self.markov_p1 = np.full((10, 10), 0.1)
                self.markov_p2 = np.full((100, 10), 0.1)
            if hasattr(self.rf_model, 'n_features_in_') and self.rf_model.n_features_in_ != self.expected_features:
                self.output.append(f"RF model expects {self.rf_model.n_features_in_} features, but {self.expected_features} required. Reinitializing.")
                self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.expected_features:
                self.output.append(f"Scaler expects {self.scaler.n_features_in_} features, but {self.expected_features} required. Reinitializing.")
                self.scaler = StandardScaler()
            self.output.append("Models loaded successfully from DIGIT_OVER_4_models.")
        except Exception as e:
            self.output.append(f"Error loading models: {e}. Initializing defaults.")
            self.markov_p1 = np.full((10, 10), 0.1)
            self.markov_p2 = np.full((100, 10), 0.1)
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()

    def save_models(self):
        """Save models to DIGIT_OVER_4_models folder."""
        try:
            joblib.dump(self.markov_p1, self.model_paths["27markov_p1.joblib"])
            joblib.dump(self.markov_p2, self.model_paths["27markov_p2.joblib"])
            joblib.dump(self.rf_model, self.model_paths["27rf_digit_predictor.joblib"])
            joblib.dump(self.scaler, self.model_paths["27feature_scaler.joblib"])
            self.output.append(f"Models saved at {self.training_samples} samples.")
        except Exception as e:
            self.output.append(f"Error saving models: {e}")

    def get_last_digit(self, tick):
        """Extract last digit from tick value."""
        return int(str(float(tick))[-1])

    def calculate_profit(self, is_win):
        """Calculate profit/loss with accurate payout (0.8857 net)."""
        if is_win:
            return self.amount * 0.8857  # Net profit after stake
        return -self.amount  # Loss of stake

    def adjust_amount(self, is_win):
        """Adjust trade amount using Martingale strategy for loss recovery."""
        if is_win:
            self.amount = self.initial_amount
            self.consecutive_losses = 0
        else:
            self.amount = min(round(self.amount * self.loss_multiplier, 2), self.account_balance * 0.1)
            self.consecutive_losses += 1
        self.price = self.amount

    def update_dataframe(self, tick, timestamp):
        """Update DataFrame with tick data and technical indicators."""
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

    def update_markov_models(self, d_t, d_t_plus_1, d_t_minus_1=None):
        """Update Markov models with digit transitions."""
        # First-order Markov update
        self.markov_p1[d_t, d_t_plus_1] = (1 - self.alpha) * self.markov_p1[d_t, d_t_plus_1] + self.alpha
        for k in range(10):
            if k != d_t_plus_1:
                self.markov_p1[d_t, k] = (1 - self.alpha) * self.markov_p1[d_t, k]
        self.markov_p1[d_t, :] /= np.sum(self.markov_p1[d_t, :])
        
        # Second-order Markov update
        if d_t_minus_1 is not None:
            state = 10 * d_t_minus_1 + d_t
            self.markov_p2[state, d_t_plus_1] = (1 - self.alpha) * self.markov_p2[state, d_t_plus_1] + self.alpha
            for k in range(10):
                if k != d_t_plus_1:
                    self.markov_p2[state, k] = (1 - self.alpha) * self.markov_p2[state, k]
            self.markov_p2[state, :] /= np.sum(self.markov_p2[state, :])

    def get_features(self):
        """Extract 9 features for prediction, including P_over."""
        if len(self.digit_history) < 3 or len(self.df) < 20:
            return None
        recent_digits = self.digit_history[-3:]
        d_t = self.digit_history[-1]
        d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else 0
        P_over = self.predict_one_step(d_t, d_t_minus_1)
        indicators = self.df.iloc[-1][['MA_6', 'RSI', 'Volatility', 'Hour_sin', 'Hour_cos']].values
        features = list(recent_digits) + [P_over] + list(indicators)
        if len(features) != self.expected_features:
            self.output.append(f"Feature mismatch: expected {self.expected_features}, got {len(features)}")
            return None
        return features

    def predict_one_step(self, d_t, d_t_minus_1=None):
        """Predict probability of next digit being over 4 (5-9)."""
        if d_t_minus_1 is not None:
            p_next = 0.5 * self.markov_p1[d_t, :] + 0.5 * self.markov_p2[10 * d_t_minus_1 + d_t, :]
        else:
            p_next = self.markov_p1[d_t, :]
        return np.sum(p_next[5:])  # Probability for digits 5-9

    def train_rf_predictor(self):
        """Train Random Forest predictor for DIGITOVER."""
        if len(self.training_data) < 100:
            self.output.append("Insufficient data for training. Need at least 100 samples.")
            return
        X = np.array(self.training_data)
        y = np.array(self.training_targets)
        if X.shape[1] != self.expected_features:
            self.output.append(f"Training data has {X.shape[1]} features, expected {self.expected_features}. Reinitializing.")
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            return
        try:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.rf_model.fit(X_scaled, y)
            self.output.append("Random Forest predictor trained.")
        except Exception as e:
            self.output.append(f"Error training RF predictor: {e}")

    def get_rf_prediction(self, features):
        """Get Random Forest prediction probabilities."""
        if features is None or not hasattr(self.scaler, 'n_features_in_'):
            return [0.5, 0.5]
        try:
            features_array = np.array([features])
            if features_array.shape[1] != self.expected_features:
                self.output.append(f"Prediction mismatch: expected {self.expected_features}, got {features_array.shape[1]}. Returning defaults.")
                return [0.5, 0.5]
            features_scaled = self.scaler.transform(features_array)
            pred_proba = self.rf_model.predict_proba(features_scaled)[0]
            return pred_proba  # [P(no trade), P(trade DIGITOVER)]
        except Exception as e:
            self.output.append(f"Prediction error: {e}. Returning default probabilities.")
            return [0.5, 0.5]

    def get_dynamic_threshold(self):
        """Calculate dynamic threshold based on losses and samples."""
        loss_penalty = min(0.2, self.consecutive_losses * 0.05)
        return min(0.6, self.initial_threshold + (self.training_samples - 100) / 10000 + loss_penalty)

    def buy_contract(self, ws, contract_type, barrier):
        """Place a DIGITOVER trade."""
        self.last_features = self.get_features()
        self.is_trading = True
        self.last_trade_time = time.time()
        self.entry_tick = self.digit_history[-1]
        json_data = json.dumps({
            "buy": 1, "subscribe": 1, "price": round(self.price, 2),
            "parameters": {
                "amount": round(self.amount, 2), "basis": "stake", "contract_type": contract_type,
                "currency": "USD", "duration": 1, "duration_unit": "t", "symbol": "R_100",
                "barrier": barrier
            }
        })
        ws.send(json_data)
        self.output.append(f"Trade placed: {contract_type}, Entry Tick: {self.entry_tick}, Amount: {self.amount:.2f}")

    def process_message(self, ws, data):
        """Handle WebSocket messages, implementing trading logic for DIGITOVER."""
        if 'error' in data:
            self.output.append(f"API Error: {data['error']['message']}")
            return

        if data.get('msg_type') == 'authorize':
            ws.send(json.dumps({"ticks": "R_100", "subscribe": 1}))
        
        elif data.get('msg_type') == 'tick':
            tick = data['tick']['quote']
            timestamp = datetime.fromtimestamp(data['tick']['epoch'])
            last_digit = self.get_last_digit(tick)
            
            # Train with previous features and current outcome
            if len(self.digit_history) >= 3:
                features = self.get_features()
                if features:
                    target = 1 if last_digit > 4 else 0  # Target for DIGITOVER
                    self.training_data.append(features)
                    self.training_targets.append(target)
                    if len(self.training_data) > 1000:
                        self.training_data.pop(0)
                        self.training_targets.pop(0)
            
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
                self.output.append(f"Trained: {d_t} -> {d_t_plus_1}, Samples: {self.training_samples}")
            
            # Handle trade timeout
            if self.is_trading and (time.time() - self.last_trade_time) > 10:
                self.is_trading = False
                self.output.append("Trade timed out, resetting is_trading.")
            
            # Trading logic
            if (self.training_samples >= 100 and not self.is_trading and 
                (time.time() - self.last_trade_time) >= self.trade_cooldown and 
                self.account_balance > 0 and self.account_balance < self.target_profit):
                features = self.get_features()
                if features:
                    rf_pred = self.get_rf_prediction(features)
                    d_t = self.digit_history[-1]
                    d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else None
                    P_over = self.predict_one_step(d_t, d_t_minus_1)
                    dynamic_threshold = self.get_dynamic_threshold()
                    if P_over > dynamic_threshold and rf_pred[1] > 0.55:
                        self.buy_contract(ws, "DIGITOVER", "4")
        
        elif 'proposal_open_contract' in data:
            contract = data['proposal_open_contract']
            if contract.get('is_sold', False):
                exit_tick = float(contract['exit_tick'])
                last_digit = self.get_last_digit(exit_tick)
                contract_type = contract['contract_type']
                barrier = int(contract['barrier'])
                is_win = (contract_type == "DIGITOVER" and last_digit > barrier)
                profit = self.calculate_profit(is_win)
                self.account_balance += profit
                self.cumulative_profit += profit
                self.adjust_amount(is_win)
                self.recent_trades.append((contract_type, is_win))
                if len(self.recent_trades) > 50:
                    self.recent_trades.pop(0)
                
                # Update models with trade outcome
                if self.entry_tick is not None and self.last_features is not None:
                    self.update_markov_models(self.entry_tick, last_digit, 
                                           self.digit_history[-2] if len(self.digit_history) >= 2 else None)
                    target = 1 if last_digit > 4 else 0  # Target for DIGITOVER
                    self.training_data.append(self.last_features)
                    self.training_targets.append(target)
                    self.train_rf_predictor()
                
                self.output.append(f"Trade Result: {contract_type}, Entry Tick: {self.entry_tick}, "
                                 f"Exit Tick: {exit_tick}, {'Win' if is_win else 'Loss'}, "
                                 f"Profit: {profit:.2f}, Balance: {self.account_balance:.2f}")
                self.is_trading = False
                self.entry_tick = None
                self.last_features = None
                
                # Stop trading when target profit reached or balance depleted
                if self.account_balance >= self.target_profit or self.account_balance <= 0:
                    self.save_models()
                    self.stop_trading = True
                    update_trading_status(self.session_id, 'inactive')
                    self.output.append("Trading stopped: Target profit reached or balance depleted.")

    def on_open(self, ws):
        """Authorize on WebSocket open."""
        ws.send(json.dumps({'authorize': self.token}))

    def on_error(self, ws, error):
        """Handle WebSocket errors and trigger reconnection."""
        self.output.append(f"WebSocket error: {error}")
        if not self.stop_trading:
            self.output.append("Reconnecting in 5 seconds due to error...")
            time.sleep(5)
            self.run()

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure and trigger reconnection."""
        self.output.append(f"WebSocket closed: {close_status_code} - {close_msg}")
        if not self.stop_trading:
            self.output.append("Reconnecting in 5 seconds due to closure...")
            time.sleep(5)
            self.run()

    def run(self):
        """Run the WebSocket connection with reconnection logic."""
        while not self.stop_trading:
            try:
                self.output.append("Starting bot...")
                self.load_models()
                api_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
                ws = websocket.WebSocketApp(
                    api_url,
                    on_message=lambda ws, msg: self.process_message(ws, json.loads(msg)),
                    on_open=self.on_open,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                self.output.append(f"Unexpected error: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            if self.stop_trading:
                break

if __name__ == "__main__":
    # Example usage (typically instantiated via app.py)
    model_paths = {
        "27markov_p1.joblib": os.path.join("DIGIT_OVER_4_models", "27markov_p1.joblib"),
        "27markov_p2.joblib": os.path.join("DIGIT_OVER_4_models", "27markov_p2.joblib"),
        "27rf_digit_predictor.joblib": os.path.join("DIGIT_OVER_4_models", "27rf_digit_predictor.joblib"),
        "27feature_scaler.joblib": os.path.join("DIGIT_OVER_4_models", "27feature_scaler.joblib")
    }
    bot = DigitOver4(
        app_id="64136",
        token="b4jTWTdQZ3mS4wB",
        target_profit=10010,
        session_id="example_session",
        model_paths=model_paths
    )
    bot.run()
