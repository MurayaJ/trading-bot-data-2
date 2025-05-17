# The refactored dashboard with algorithm selection.
import streamlit as st
import threading
import time
import os
import subprocess
import shutil
import logging
import importlib
from db_utils import *
from utils import *
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define GitHub URLs
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")
PUBLIC_GITHUB_REPO_URL = "https://github.com/MurayaJ/trading-bot-data.git"
AUTH_GITHUB_REPO_URL = f"https://MurayaJ:{GITHUB_PAT}@github.com/MurayaJ/trading-bot-data.git" if GITHUB_PAT else PUBLIC_GITHUB_REPO_URL
BASE_DIR = os.getcwd()  # Use Render's working directory (/app)

# Algorithm definitions with dynamic imports
ALGORITHMS = {
    "DIGIT_EVEN_ODD": {
        "module": "DIGIT_EVEN_ODD",
        "class": "DigitEvenOdd",
        "model_paths": {
            "markov_p1": "DIGIT_EVEN_ODD_models/4markov_p1.joblib",
            "markov_p2": "DIGIT_EVEN_ODD_models/4markov_p2.joblib",
            "rf_digit_predictor": "DIGIT_EVEN_ODD_models/4rf_digit_predictor.joblib",
            "feature_scaler": "DIGIT_EVEN_ODD_models/4feature_scaler.joblib"
        }
    },
    "DIGIT_OVER_4": {
        "module": "DIGIT_OVER_4",
        "class": "DigitOver4",  # Placeholder, to be implemented
        "model_paths": {
            "markov_p1": "DIGIT_OVER_4_models/27markov_p1.joblib",
            "markov_p2": "DIGIT_OVER_4_models/27markov_p2.joblib",
            "rf_digit_predictor": "DIGIT_OVER_4_models/27rf_digit_predictor.joblib",
            "feature_scaler": "DIGIT_OVER_4_models/27feature_scaler.joblib"
        }
    },
    "DIGIT_UNDER_5": {
        "module": "DIGIT_UNDER_5",
        "class": "DigitUnder5",  # Placeholder, to be implemented
        "model_paths": {
            "markov_p1": "DIGIT_UNDER_5_models/17markov_p1.joblib",
            "markov_p2": "DIGIT_UNDER_5_models/17markov_p2.joblib",
            "rf_digit_predictor": "DIGIT_UNDER_5_models/17rf_digit_predictor.joblib",
            "feature_scaler": "DIGIT_UNDER_5_models/17feature_scaler.joblib"
        }
    }
}

def is_valid_git_repo(path):
    """Check if the directory is a valid Git repository."""
    try:
        subprocess.run(["git", "status"], cwd=path, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def init_github_repo():
    """Set up the existing Git repository with authentication, no cloning."""
    if not is_valid_git_repo(BASE_DIR):
        logging.error("Not a valid git repository. Ensure Render deployment is correct.")
        raise RuntimeError("Deployment directory is not a valid Git repository.")
    try:
        subprocess.run(["git", "remote", "set-url", "origin", AUTH_GITHUB_REPO_URL], cwd=BASE_DIR, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "bot@tradingbot.com"], cwd=BASE_DIR, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Trading Bot"], cwd=BASE_DIR, check=True, capture_output=True, text=True)
        logging.info("Git repository configured with authentication.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to configure Git repository: {e.stderr}")
        raise

def sync_with_github():
    """Pull latest changes from GitHub."""
    try:
        result = subprocess.run(["git", "pull", "origin", "main"], cwd=BASE_DIR, capture_output=True, text=True, check=True)
        logging.info("Successfully pulled latest changes from GitHub.")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logging.error(f"Git pull failed: {e.stderr}")
        return False

def send_reset_email(email, reset_code):
    """Send a password reset code to the user's email."""
    sender_email = "virtual101assistance@gmail.com"
    sender_password = "john0705168"  # Replace with app-specific password if 2FA is enabled
    message = MIMEMultipart("alternative")
    message["Subject"] = "Password Reset Code"
    message["From"] = sender_email
    message["To"] = email
    text = f"Your reset code is: {reset_code}"
    part = MIMEText(text, "plain")
    message.attach(part)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        logging.info(f"Reset code sent to {email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send reset email: {e}")
        return False

def main():
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .stMetric {
            font-size: 18px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize app components
    init_github_repo()
    sync_with_github()
    init_db()

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "email" not in st.session_state:
        st.session_state["email"] = None
    if "last_activity" not in st.session_state:
        st.session_state["last_activity"] = time.time()
    if "trading_active" not in st.session_state:
        st.session_state["trading_active"] = False

    # Login/Register/Forgot Password Section
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
                            if register_user(name, email, password):
                                st.success("Registered successfully! Please log in.")
                                commit_and_push()  # Ensure registration is synced
                            else:
                                st.error("Email already exists.")
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
                                if user_data["subscription_status"] == "expired":
                                    st.error("Your subscription has expired. Please renew to continue.")
                                    st.markdown("[Pay 70 GBP via Skrill](https://skrill.me/rq/John/70/GBP?key=p56pcU69FeFTB70NHi9Qh3Q2RQ8)")
                                    st.write("After payment, contact the administrator via Telegram at t.me/atlanettrading or email virtual101assistance@gmail.com for activation.")
                                elif user_data["subscription_status"] == "blocked":
                                    st.error("Your account is blocked.")
                                else:
                                    st.session_state["logged_in"] = True
                                    st.session_state["email"] = email.lower()
                                    st.session_state["last_activity"] = time.time()
                                    st.session_state["trading_active"] = user_data['trading_status'] == 'active'
                                    if st.session_state["trading_active"] and "bot" not in st.session_state:
                                        update_trading_status(email, 'inactive')
                                        st.warning("Trading was active but stopped due to downtime. Please restart.")
                                        st.session_state["trading_active"] = False
                                    st.success("Logged in successfully! Dashboard is now available below.")
                                    st.rerun()
                            else:
                                st.error("Invalid email or password.")
                        else:
                            st.error("Please enter email and password.")

        elif option == "Forgot Password":
            with st.form("forgot_password_form"):
                email = st.text_input("Email")
                submit = st.form_submit_button("Send Reset Code")
                if submit:
                    with st.spinner("Sending reset code..."):
                        if email:
                            reset_code = generate_reset_code(email)
                            if reset_code:
                                if send_reset_email(email, reset_code):
                                    st.success("A reset code has been sent to your email.")
                                else:
                                    st.error("Failed to send reset email. Please try again later.")
                            else:
                                st.success("If the email exists, a reset code has been sent.")
                        else:
                            st.error("Please enter your email.")

            with st.form("reset_password_form"):
                email = st.text_input("Email for Reset")
                reset_code = st.text_input("Reset Code")
                new_password = st.text_input("New Password", type="password")
                submit = st.form_submit_button("Reset Password")
                if submit:
                    with st.spinner("Resetting password..."):
                        if email and reset_code and new_password:
                            if verify_reset_code(email, reset_code):
                                reset_password(email, new_password)
                                commit_and_push()  # Ensure password reset is synced
                                st.success("Password reset successfully! Please log in.")
                            else:
                                st.error("Invalid reset code.")
                        else:
                            st.error("Please fill in all fields.")
    else:
        # Check user status
        user_data = check_subscription_status(st.session_state["email"])
        if not user_data:
            st.error("User not found.")
            st.session_state["logged_in"] = False
            return
        if user_data["subscription_status"] == "blocked":
            st.error("Your account is blocked.")
            st.session_state["logged_in"] = False
            return
        elif user_data["subscription_status"] == "expired" and not st.session_state["trading_active"]:
            st.error("Your subscription has expired. Please renew to continue.")
            st.markdown("[Pay 70 GBP via Skrill](https://skrill.me/rq/John/70/GBP?key=p56pcU69FeFTB70NHi9Qh3Q2RQ8)")
            st.write("After payment, contact the administrator via Telegram at t.me/atlanettrading or email virtual101assistance@gmail.com for activation.")
            st.session_state["logged_in"] = False
            return
        # Inactivity logout
        if not st.session_state["trading_active"] and time.time() - st.session_state["last_activity"] > 1200:
            email = st.session_state["email"]
            if "bot" in st.session_state:
                st.session_state["bot"].stop_trading = True
            st.session_state["logged_in"] = False
            st.session_state["email"] = None
            st.session_state.pop("bot", None)
            st.session_state.pop("thread", None)
            st.session_state["trading_active"] = False
            if email is not None:
                update_trading_status(email, 'inactive')
            st.warning("Logged out due to inactivity.")
            return

        # Dashboard
        st.title("Trading Dashboard")
        st.write(f"Welcome, {user_data['name']} ({st.session_state['email']})")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Trading Controls")
            st.write(f"**Trading Status:** {'Active' if st.session_state['trading_active'] else 'Inactive'}")
            selected_algorithm = st.selectbox("Select Algorithm", list(ALGORITHMS.keys()))
            algorithm_info = ALGORITHMS[selected_algorithm]
            try:
                module = importlib.import_module(algorithm_info["module"])
                bot_class = getattr(module, algorithm_info["class"])
            except (ImportError, AttributeError) as e:
                st.error(f"Error loading algorithm {selected_algorithm}: {e}. Please ensure the algorithm file is implemented.")
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
                                bot = bot_class(app_id, token, target_profit, session_id, algorithm_info["model_paths"])
                                st.session_state["bot"] = bot
                                st.session_state[f"output_{session_id}"] = []
                                st.session_state["trading_active"] = True
                                update_trading_status(session_id, 'active')
                                thread = threading.Thread(target=bot.run)
                                st.session_state["thread"] = thread
                                thread.start()
                                st.success("Trading started!")
                            else:
                                st.error("Please enter App ID, Token, and a valid Target Profit.")
            with col_btn2:
                if st.session_state["trading_active"]:
                    if st.button("Stop Trading"):
                        with st.spinner("Stopping trading..."):
                            if "bot" in st.session_state:
                                st.session_state["bot"].stop_trading = True
                                st.session_state["trading_active"] = False
                                update_trading_status(st.session_state["email"], 'inactive')
                                st.success("Trading stopped.")
                            else:
                                st.session_state["trading_active"] = False
                                update_trading_status(st.session_state["email"], 'inactive')
                                st.warning("Bot not found. Trading status reset.")

        with col2:
            st.subheader("Account Status")
            st.write(f"Subscription Status: {user_data['subscription_status']}")
            if user_data['subscription_status'] == 'trial':
                trial_start = datetime.fromisoformat(user_data['trial_start_date'])
                days_left = 7 - (datetime.now() - trial_start).days
                st.write(f"Trial days left: {max(days_left, 0)}")
            elif user_data['subscription_status'] == 'active':
                if user_data['last_payment_date']:
                    last_payment = datetime.fromisoformat(user_data['last_payment_date'])
                    next_payment = last_payment + timedelta(days=30)
                    days_until_next = (next_payment - datetime.now()).days
                    st.write(f"Next payment due in: {max(days_until_next, 0)} days")
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
                email = st.session_state["email"]
                with st.spinner("Logging out..."):
                    if "bot" in st.session_state:
                        st.session_state["bot"].stop_trading = True
                    st.session_state["logged_in"] = False
                    st.session_state["email"] = None
                    st.session_state.pop("bot", None)
                    st.session_state.pop("thread", None)
                    st.session_state["trading_active"] = False
                    if email is not None:
                        update_trading_status(email, 'inactive')
                    st.success("Logged out successfully!")
                    st.rerun()

        # Trade Output Section
        st.subheader("Trade Output")
        session_id = st.session_state["email"]
        output_container = st.empty()
        if st.button("Refresh Output"):
            with st.spinner("Refreshing output..."):
                if "bot" in st.session_state:
                    with output_container.container():
                        for line in st.session_state["bot"].output:
                            st.write(line)
                else:
                    st.write("No trading session active.")
        st.info("Click 'Refresh Output' to see the latest trading updates.")
        if "bot" in st.session_state and st.session_state["bot"].cumulative_profit >= st.session_state["bot"].target_profit:
            st.success(f"Target profit of ${st.session_state['bot'].target_profit:.2f} reached! Start a new session.")

        st.info("Disclaimer: Before trading, ensure your account has $1000. Set target profit to $10 per session. Take a 3-hour break between sessions. Safe trading!")
        st.session_state["last_activity"] = time.time()

if __name__ == "__main__":
    main()
