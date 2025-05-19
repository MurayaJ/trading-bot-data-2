import streamlit as st
import threading
import time
import os
import logging
import importlib
from db_utils import *
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Algorithm Definitions
ALGORITHMS = {
    "DIGIT_EVEN_ODD": {
        "module": "DIGIT_EVEN_ODD",
        "class": "DigitEvenOdd",
        "model_paths": {
            "markov_p1": "DIGIT_EVEN_ODD_models/4markov_p1.joblib",
            "markov_p2": "DIGIT_EVEN_ODD_models/4markov_p2.joblib",
            "rf_digit_predictor": "DIGIT_EVEN_ODD_models/4rf_digit_predictor.joblib",
            "feature_scaler": "DIGIT_EVEN_ODD_models/4feature_scaler.joblib"
        },
        "implemented": True
    },
    "DIGIT_OVER_4": {
        "module": "DIGIT_OVER_4",
        "class": "DigitOver4",
        "model_paths": {
            "markov_p1": "DIGIT_OVER_4_models/27markov_p1.joblib",
            "markov_p2": "DIGIT_OVER_4_models/27markov_p2.joblib",
            "rf_digit_predictor": "DIGIT_OVER_4_models/27rf_digit_predictor.joblib",
            "feature_scaler": "DIGIT_OVER_4_models/27feature_scaler.joblib"
        },
        "implemented": True
    },
    "DIGIT_UNDER_5": {
        "module": "DIGIT_UNDER_5",
        "class": "DigitUnder5",
        "model_paths": {
            "markov_p1": "DIGIT_UNDER_5_models/17markov_p1.joblib",
            "markov_p2": "DIGIT_UNDER_5_models/17markov_p2.joblib",
            "rf_digit_predictor": "DIGIT_UNDER_5_models/17rf_digit_predictor.joblib",
            "feature_scaler": "DIGIT_UNDER_5_models/17feature_scaler.joblib"
        },
        "implemented": True
    }
}

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            padding: 10px; border-radius: 5px; border: 1px solid #ccc;
        }
        .stMetric { font-size: 18px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # Initialize app
    init_db()  # Database schema is now ensured to be correct

    # Session state initialization
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "email" not in st.session_state:
        st.session_state["email"] = None
    if "trading_active" not in st.session_state:
        st.session_state["trading_active"] = False
    if "last_activity" not in st.session_state:
        st.session_state["last_activity"] = time.time()

    if not st.session_state["logged_in"]:
        st.title("Trading Bot")
        option = st.selectbox("Choose an option", ["Login", "Register", "Forgot Password"])

        if option == "Register":
            with st.form("register_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Register")
                if submit:
                    with st.spinner("Registering..."):
                        if name and email and password:
                            # Basic input validation
                            if "@" in email and len(password) >= 6:
                                if register_user(name, email, password):
                                    st.success("Registered successfully! Please log in.")
                                else:
                                    st.error("Email already exists.")
                            else:
                                st.error("Invalid email or password (min 6 chars).")
                        else:
                            st.error("Please fill in all fields.")

        elif option == "Login":
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    with st.spinner("Logging in..."):
                        if email and password:
                            login_result = login_user(email, password)
                            if login_result["success"]:
                                user_data = check_subscription_status(email)
                                if user_data["subscription_status"] in ["expired", "deactivated"]:
                                    st.error("Your subscription is inactive. Contact admin at t.me/atlanettrading.")
                                elif user_data["subscription_status"] == "blocked":
                                    st.error("Your account is blocked.")
                                else:
                                    st.session_state["logged_in"] = True
                                    st.session_state["email"] = email.lower()
                                    st.session_state["trading_active"] = user_data["trading_status"] == "active"
                                    if st.session_state["trading_active"]:
                                        saved_state = get_trading_state(email)
                                        if saved_state and "bot" not in st.session_state:
                                            # Resume trading
                                            available_algorithms = [alg for alg, info in ALGORITHMS.items() if info["implemented"]]
                                            algo = saved_state.get("algorithm", available_algorithms[0])
                                            try:
                                                module = importlib.import_module(ALGORITHMS[algo]["module"])
                                                bot_class = getattr(module, ALGORITHMS[algo]["class"])
                                                bot = bot_class(saved_state["app_id"], saved_state["token"], saved_state["target_profit"], email, ALGORITHMS[algo]["model_paths"], saved_state)
                                                st.session_state["bot"] = bot
                                                thread = threading.Thread(target=bot.run)
                                                st.session_state["thread"] = thread
                                                thread.start()
                                                st.info("Resumed active trading session.")
                                            except Exception as e:
                                                logging.error(f"Error loading bot: {e}")
                                                st.error(f"Failed to resume trading: {e}")
                                    st.success("Logged in successfully!")
                                    st.rerun()
                            else:
                                st.error("Invalid email or password.")
                        else:
                            st.error("Please enter email and password.")

        elif option == "Forgot Password":
            st.write("Contact admin via Telegram at [t.me/atlanettrading](https://t.me/atlanettrading) to reset your password.")
            with st.form("reset_password_form"):
                email = st.text_input("Registered Email")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Save New Password")
                if submit:
                    with st.spinner("Saving new password..."):
                        if email and new_password and confirm_password:
                            if new_password == confirm_password:
                                user_data = get_user_data(email)
                                if user_data:
                                    reset_user_password(email, new_password)
                                    st.success("Password updated successfully! Please log in.")
                                else:
                                    st.error("Email not registered.")
                            else:
                                st.error("Passwords do not match.")
                        else:
                            st.error("Please fill in all fields.")

    else:
        user_data = check_subscription_status(st.session_state["email"])
        if not user_data or user_data["subscription_status"] == "blocked":
            st.session_state["logged_in"] = False
            st.error("Account issue detected. Logging out.")
            return

        st.title("Trading Dashboard")
        st.write(f"Welcome, {user_data['name']} ({st.session_state['email']})")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Trading Controls")
            st.write(f"**Trading Status:** {'Active' if st.session_state['trading_active'] else 'Inactive'}")
            if not st.session_state["trading_active"]:
                available_algorithms = [alg for alg, info in ALGORITHMS.items() if info["implemented"]]
                selected_algorithm = st.selectbox("Select Algorithm", available_algorithms)
                algorithm_info = ALGORITHMS[selected_algorithm]
                try:
                    module = importlib.import_module(algorithm_info["module"])
                    bot_class = getattr(module, algorithm_info["class"])
                except (ImportError, AttributeError) as e:
                    st.error(f"Error loading algorithm {selected_algorithm}: {e}")
                    return
                app_id = st.text_input("App ID", key=f"app_id_{st.session_state['email']}")
                token = st.text_input("Token", type="password", key=f"token_{st.session_state['email']}")
                target_profit = st.number_input("Target Profit ($)", min_value=0.01, value=10.0, step=0.1)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if not st.session_state["trading_active"]:
                    if st.button("Start Trading"):
                        with st.spinner("Starting trading..."):
                            if app_id and token and target_profit > 0:
                                session_id = st.session_state["email"]
                                try:
                                    bot = bot_class(app_id, token, target_profit, session_id, algorithm_info["model_paths"])
                                    st.session_state["bot"] = bot
                                    st.session_state["trading_active"] = True
                                    update_trading_status(session_id, "active")
                                    save_trading_state(session_id, {"algorithm": selected_algorithm, "app_id": app_id, "token": token, "target_profit": target_profit})
                                    thread = threading.Thread(target=bot.run)
                                    st.session_state["thread"] = thread
                                    thread.start()
                                    st.success("Trading started!")
                                except Exception as e:
                                    logging.error(f"Error starting bot: {e}")
                                    st.error(f"Failed to start trading: {e}")
                            else:
                                st.error("Please enter App ID, Token, and a valid Target Profit.")
            with col_btn2:
                if st.session_state["trading_active"]:
                    if st.button("Stop Trading"):
                        with st.spinner("Stopping trading..."):
                            if "bot" in st.session_state:
                                st.session_state["bot"].stop_trading = True
                                st.session_state["thread"].join()
                                st.session_state["trading_active"] = False
                                update_trading_status(st.session_state["email"], "inactive")
                                clear_trading_state(st.session_state["email"])
                                st.session_state.pop("bot", None)
                                st.session_state.pop("thread", None)
                                st.success("Trading stopped.")
                            else:
                                st.session_state["trading_active"] = False
                                update_trading_status(st.session_state["email"], "inactive")
                                st.warning("Bot not found. Trading stopped.")

        with col2:
            st.subheader("Account Status")
            st.write(f"Subscription Status: {user_data['subscription_status']}")
            if "bot" in st.session_state:
                bot = st.session_state["bot"]
                st.metric("Balance", f"${bot.account_balance:.2f}")
                st.metric("Profit", f"${bot.cumulative_profit:.2f}")
            else:
                st.metric("Balance", "$1000.00")
                st.metric("Profit", "$0.00")
            if st.session_state["trading_active"]:
                st.write("Logout disabled during trading.")
            if st.button("Logout", disabled=st.session_state["trading_active"]):
                with st.spinner("Logging out..."):
                    if "bot" in st.session_state:
                        st.session_state["bot"].stop_trading = True
                        st.session_state["thread"].join()
                    st.session_state["logged_in"] = False
                    st.session_state["email"] = None
                    st.session_state["trading_active"] = False
                    update_trading_status(st.session_state["email"], "inactive")
                    st.session_state.pop("bot", None)
                    st.session_state.pop("thread", None)
                    st.success("Logged out successfully!")
                    st.rerun()

        st.subheader("Trade Output")
        if "bot" in st.session_state:
            bot = st.session_state["bot"]
            # Filter output for trade-related messages
            for log in bot.output:
                if "Trade placed" in log or "Trade Result" in log:
                    st.write(log)
            if bot.cumulative_profit >= bot.target_profit:
                bot.stop_trading = True
                st.session_state["thread"].join()
                st.session_state["trading_active"] = False
                update_trading_status(st.session_state["email"], "inactive")
                clear_trading_state(st.session_state["email"])
                st.success(f"Target profit of ${bot.target_profit:.2f} reached!")
                st.session_state.pop("bot", None)
                st.session_state.pop("thread", None)

if __name__ == "__main__":
    main()
