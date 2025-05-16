# The refactored dashboard with algorithm selection.

import streamlit as st
import threading
import time
import os
import subprocess
import shutil
import logging
from algorithms import DigitEvenOdd
from db_utils import *
from utils import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define GitHub URLs
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")
PUBLIC_GITHUB_REPO_URL = "https://github.com/MurayaJ/trading-bot-data.git"
AUTH_GITHUB_REPO_URL = f"https://MurayaJ:{GITHUB_PAT}@github.com/MurayaJ/trading-bot-data.git" if GITHUB_PAT else PUBLIC_GITHUB_REPO_URL
BASE_DIR = "trading_bot_data"

# Algorithm definitions
ALGORITHMS = {
    "DIGIT_EVEN_ODD": {
        "class": DigitEvenOdd,
        "model_paths": {
            "markov_p1": "4markov_p1.joblib",
            "markov_p2": "4markov_p2.joblib",
            "rf_digit_predictor": "4rf_digit_predictor.joblib",
            "feature_scaler": "4feature_scaler.joblib"
        }
    }
    # Add "DIGIT_UNDER_5" and "DIGIT_OVER_4" later with their model paths
}

def is_valid_git_repo(path):
    """Check if the directory is a valid Git repository."""
    try:
        subprocess.run(["git", "status"], cwd=path, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def init_github_repo():
    """Initialize the GitHub repository by cloning or ensuring it's a Git repo."""
    try:
        if os.path.exists(BASE_DIR):
            if os.path.exists(os.path.join(BASE_DIR, ".git")) and is_valid_git_repo(BASE_DIR):
                logging.info("Existing Git repository found. Skipping clone.")
                return
            else:
                logging.info("Removing invalid or non-Git directory.")
                shutil.rmtree(BASE_DIR)
        logging.info(f"Cloning repository from {AUTH_GITHUB_REPO_URL}")
        subprocess.run(["git", "clone", AUTH_GITHUB_REPO_URL, BASE_DIR], check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "bot@tradingbot.com"], cwd=BASE_DIR, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Trading Bot"], cwd=BASE_DIR, check=True, capture_output=True, text=True)
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)
        if not os.path.exists(os.path.join(BASE_DIR, "db", "trading_users.db")):
            open(os.path.join(BASE_DIR, "db", "trading_users.db"), 'a').close()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to initialize GitHub repository: {e.stderr}")
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)
        if not os.path.exists(os.path.join(BASE_DIR, "db", "trading_users.db")):
            open(os.path.join(BASE_DIR, "db", "trading_users.db"), 'a').close()
    except Exception as e:
        logging.error(f"Unexpected error during Git setup: {e}")

def sync_with_github():
    """Pull latest changes from GitHub."""
    try:
        result = subprocess.run(["git", "pull", "origin", "main"], cwd=BASE_DIR, capture_output=True, text=True, check=True)
        logging.info("Successfully pulled latest changes from GitHub.")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logging.error(f"Git pull failed: {e.stderr}")
        return False

def main():
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

    init_github_repo()
    sync_with_github()
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "email" not in st.session_state:
        st.session_state["email"] = None
    if "last_activity" not in st.session_state:
        st.session_state["last_activity"] = time.time()
    if "trading_active" not in st.session_state:
        st.session_state["trading_active"] = False

    if not st.session_state["logged_in"]:
        st.title("Trading Bot")
        option = st.selectbox("Choose an option", ["Login", "Register"])
        if option == "Register":
            with st.form("register_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Register")
                if submit:
                    if name and email and password:
                        if register_user(name, email, password):
                            st.success("Registered successfully! Please log in.")
                            commit_and_push()
                        else:
                            st.error("Email already exists.")
                    else:
                        st.error("Please fill in all fields.")
        else:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    if email and password:
                        login_result = login_user(email, password)
                        if login_result["success"]:
                            user_data = check_subscription_status(email)
                            if user_data["subscription_status"] == "expired":
                                st.error("Your subscription has expired. Please renew to continue.")
                                st.markdown("[Pay 70 GBP via Skrill](https://skrill.me/rq/John/70/GBP?key=p56pcU69FeFTB70NHi9Qh3Q2RQ8)")
                                st.write("After payment, contact the administrator to activate your account.")
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
                                st.success("Logged in successfully!")
                        else:
                            st.error("Invalid email or password.")
                    else:
                        st.error("Please enter email and password.")
    else:
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
            st.write("After payment, contact the administrator to activate your account.")
            st.session_state["logged_in"] = False
            return
        if not st.session_state["trading_active"] and time.time() - st.session_state["last_activity"] > 1200:
            if "bot" in st.session_state:
                st.session_state["bot"].stop_trading = True
            st.session_state["logged_in"] = False
            st.session_state["email"] = None
            st.session_state.pop("bot", None)
            st.session_state.pop("thread", None)
            st.session_state["trading_active"] = False
            update_trading_status(st.session_state["email"], 'inactive')
            st.warning("Logged out due to inactivity.")
            return

        st.title("Trading Dashboard")
        st.write(f"Welcome, {user_data['name']} ({st.session_state['email']})")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Trading Controls")
            algorithm_options = list(ALGORITHMS.keys())
            selected_algorithm = st.selectbox("Select Algorithm", algorithm_options)
            app_id = st.text_input("App ID", key=f"app_id_{st.session_state['email']}")
            token = st.text_input("Token", type="password", key=f"token_{st.session_state['email']}")
            target_profit = st.number_input("Target Profit ($)", min_value=0.01, value=10.0, step=0.1)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if not st.session_state["trading_active"]:
                    if st.button("Start Trading"):
                        if app_id and token and target_profit > 0:
                            session_id = st.session_state["email"]
                            algorithm_info = ALGORITHMS[selected_algorithm]
                            bot_class = algorithm_info["class"]
                            model_paths = algorithm_info["model_paths"]
                            bot = bot_class(app_id, token, target_profit, session_id, model_paths)
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
                if "bot" in st.session_state:
                    st.session_state["bot"].stop_trading = True
                st.session_state["logged_in"] = False
                st.session_state["email"] = None
                st.session_state.pop("bot", None)
                st.session_state.pop("thread", None)
                st.session_state["trading_active"] = False
                update_trading_status(st.session_state["email"], 'inactive')
                st.success("Logged out successfully!")

        st.subheader("Account Management")
        col_manage1, col_manage2 = st.columns(2)
        with col_manage1:
            with st.form("change_password_form"):
                old_password = st.text_input("Old Password", type="password")
                new_password = st.text_input("New Password", type="password")
                submit = st.form_submit_button("Change Password")
                if submit:
                    if old_password and new_password:
                        if change_password(st.session_state["email"], old_password, new_password):
                            st.success("Password changed successfully!")
                            commit_and_push()
                        else:
                            st.error("Incorrect old password.")
                    else:
                        st.error("Please fill in both fields.")
        with col_manage2:
            if st.button("Delete Account", disabled=st.session_state["trading_active"]):
                if st.checkbox("Are you sure you want to delete your account? This action cannot be undone."):
                    if st.button("Confirm Delete"):
                        delete_user(st.session_state["email"])
                        st.success("Account deleted successfully.")
                        st.session_state["logged_in"] = False
                        st.session_state["email"] = None
                        st.session_state.pop("bot", None)
                        st.session_state.pop("thread", None)
                        st.session_state["trading_active"] = False
                        commit_and_push()

        st.subheader("Trade Output")
        session_id = st.session_state["email"]
        output_container = st.empty()
        if st.button("Refresh Output"):
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
