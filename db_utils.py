import sqlite3
import bcrypt
from datetime import datetime, timedelta
import os

DB_PATH = "db/trading_users.db"

def init_db():
    """Initialize the database, ensuring it exists."""
    if not os.path.exists(DB_PATH):
        raise ValueError(f"Error: Database file '{DB_PATH}' does not exist!")
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            trial_start_date TEXT,
            subscription_status TEXT,
            last_payment_date TEXT,
            trading_status TEXT DEFAULT 'inactive'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reset_codes (
            email TEXT PRIMARY KEY,
            reset_code TEXT
        )
    """)
    c.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in c.fetchall()]
    for col in ["password_hash", "trial_start_date", "subscription_status", "last_payment_date", "trading_status"]:
        if col not in columns:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT" + (" DEFAULT 'inactive'" if col == "trading_status" else ""))
    conn.commit()
    conn.close()

def register_user(name, email, password):
    """Register a new user."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode('utf-8')
        trial_start_date = datetime.now().isoformat()
        subscription_status = 'trial'
        trading_status = 'inactive'
        c.execute("""
            INSERT INTO users (name, email, password_hash, trial_start_date, subscription_status, trading_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, email, password_hash, trial_start_date, subscription_status, trading_status))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    """Authenticate a user."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("SELECT email, password_hash FROM users WHERE email = ?", (email,))
        result = c.fetchone()
        if result:
            stored_hash = result[1]
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode('utf-8')
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
                return {"success": True, "message": "Login successful"}
            else:
                return {"success": False, "message": "Incorrect password"}
        return {"success": False, "message": "User not found"}
    finally:
        conn.close()

def get_user_data(email):
    """Retrieve user data."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT id, name, email, password_hash, trial_start_date, subscription_status, last_payment_date, trading_status
            FROM users WHERE email = ?
        """, (email,))
        row = c.fetchone()
        if row:
            return {
                'id': row[0], 'name': row[1], 'email': row[2], 'password_hash': row[3],
                'trial_start_date': row[4], 'subscription_status': row[5], 'last_payment_date': row[6],
                'trading_status': row[7]
            }
        return None
    finally:
        conn.close()

def check_subscription_status(email):
    """Check and update subscription status."""
    email = email.lower()
    user_data = get_user_data(email)
    if not user_data:
        return None
    if user_data['subscription_status'] == 'trial':
        trial_start = datetime.fromisoformat(user_data['trial_start_date'])
        if (datetime.now() - trial_start) > timedelta(days=7):
            update_user_status(email, 'expired')
            user_data['subscription_status'] = 'expired'
    elif user_data['subscription_status'] == 'active':
        if user_data['last_payment_date']:
            last_payment = datetime.fromisoformat(user_data['last_payment_date'])
            if (datetime.now() - last_payment).days > 30:
                update_user_status(email, 'expired')
                user_data['subscription_status'] = 'expired'
    return user_data

def update_user_status(email, status):
    """Update user subscription status."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET subscription_status = ? WHERE email = ?", (status, email))
        conn.commit()
    finally:
        conn.close()

def update_trading_status(email, status):
    """Update user trading status."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET trading_status = ? WHERE email = ?", (status, email))
        conn.commit()
    finally:
        conn.close()

def save_reset_code(email, reset_code):
    """Save a reset code for the user."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO reset_codes (email, reset_code) VALUES (?, ?)", (email, reset_code))
    conn.commit()
    conn.close()

def verify_reset_code(email, reset_code):
    """Verify a reset code."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT reset_code FROM reset_codes WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == reset_code

def reset_password(email, new_password):
    """Reset user password."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode('utf-8')
    c.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email))
    conn.commit()
    conn.close()
