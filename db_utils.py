import sqlite3
import bcrypt
from datetime import datetime, timedelta
import os
import json

DB_PATH = "db/trading_users.db"

def init_db():
    if not os.path.exists("db"):
        os.makedirs("db")
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            trial_start_date TEXT,
            subscription_status TEXT DEFAULT 'trial',
            last_payment_date TEXT,
            trading_status TEXT DEFAULT 'inactive',
            trading_state TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_user(name, email, password):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode('utf-8')
        trial_start_date = datetime.now().isoformat()
        c.execute("""
            INSERT INTO users (name, email, password_hash, trial_start_date, subscription_status, trading_status)
            VALUES (?, ?, ?, ?, 'trial', 'inactive')
        """, (name, email, password_hash, trial_start_date))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        result = c.fetchone()
        if result and bcrypt.checkpw(password.encode("utf-8"), result[0].encode('utf-8')):
            return {"success": True, "message": "Login successful"}
        return {"success": False, "message": "Invalid credentials"}
    finally:
        conn.close()

def get_user_data(email):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = c.fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "email": row[2], "password_hash": row[3],
                "trial_start_date": row[4], "subscription_status": row[5], "last_payment_date": row[6],
                "trading_status": row[7], "trading_state": row[8]
            }
        return None
    finally:
        conn.close()

def check_subscription_status(email):
    email = email.lower()
    user_data = get_user_data(email)
    if not user_data:
        return None
    now = datetime.now()
    if user_data["subscription_status"] == "trial" and user_data["trial_start_date"]:
        trial_start = datetime.fromisoformat(user_data["trial_start_date"])
        if (now - trial_start) > timedelta(days=7):
            deactivate_user(email)
            user_data["subscription_status"] = "deactivated"
    elif user_data["subscription_status"] == "active" and user_data["last_payment_date"]:
        last_payment = datetime.fromisoformat(user_data["last_payment_date"])
        if (now - last_payment).days > 30:
            deactivate_user(email)
            user_data["subscription_status"] = "deactivated"
    return user_data

def update_trading_status(email, status):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET trading_status = ? WHERE email = ?", (status, email))
        conn.commit()
    finally:
        conn.close()

def save_trading_state(email, state):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        state_json = json.dumps(state)
        c.execute("UPDATE users SET trading_state = ? WHERE email = ?", (state_json, email))
        conn.commit()
    finally:
        conn.close()

def get_trading_state(email):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("SELECT trading_state FROM users WHERE email = ?", (email,))
        result = c.fetchone()
        if result and result[0]:
            return json.loads(result[0])
        return None
    finally:
        conn.close()

def clear_trading_state(email):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET trading_state = NULL WHERE email = ?", (email,))
        conn.commit()
    finally:
        conn.close()

def deactivate_user(email):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET subscription_status = 'deactivated' WHERE email = ?", (email,))
        conn.commit()
    finally:
        conn.close()

def activate_user(email, is_trial=False):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        status = "trial" if is_trial else "active"
        date_field = "trial_start_date" if is_trial else "last_payment_date"
        c.execute(f"UPDATE users SET subscription_status = ?, {date_field} = ? WHERE email = ?", 
                  (status, datetime.now().isoformat(), email))
        conn.commit()
    finally:
        conn.close()

def reset_user_password(email, new_password):
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        if get_user_data(email):
            new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode('utf-8')
            c.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email))
            conn.commit()
            return True
        return False
    finally:
        conn.close()
