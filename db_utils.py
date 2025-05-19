import sqlite3
import os
from datetime import datetime, timedelta
import bcrypt
import json

# Database path
DB_PATH = "db/trading_users.db"

def init_db():
    """Initialize the database, ensuring all required columns exist."""
    # Create db directory if it doesn't exist
    if not os.path.exists("db"):
        os.makedirs("db")
    
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    
    # Create the users table if it doesn't exist
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
    
    # Define all required columns with their SQL definitions
    required_columns = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT NOT NULL",
        "email": "TEXT UNIQUE NOT NULL",
        "password_hash": "TEXT NOT NULL",
        "trial_start_date": "TEXT",
        "subscription_status": "TEXT DEFAULT 'trial'",
        "last_payment_date": "TEXT",
        "trading_status": "TEXT DEFAULT 'inactive'",
        "trading_state": "TEXT"
    }
    
    # Get existing columns from the table
    c.execute("PRAGMA table_info(users)")
    existing_columns = [col[1] for col in c.fetchall()]
    
    # Add any missing columns
    for col, col_type in required_columns.items():
        if col not in existing_columns:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
    
    conn.commit()
    conn.close()

def get_user_data(email):
    """Retrieve user data with explicit column selection and error handling."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        # Explicitly select all 9 columns in the exact order
        c.execute("""
            SELECT id, name, email, password_hash, trial_start_date, 
                   subscription_status, last_payment_date, trading_status, trading_state
            FROM users WHERE email = ?
        """, (email,))
        row = c.fetchone()
        
        # If no row is found, return None
        if not row:
            return None
        
        # Ensure row has all expected columns (should be 9)
        # This is a safety net; the table structure should already be correct
        if len(row) < 9:
            # Log this issue if it occurs (in a real app, use logging)
            print(f"Warning: Row for email {email} has {len(row)} columns, expected 9")
            # Fill missing values with defaults
            row = list(row) + [None] * (9 - len(row))
        
        # Construct and return user data dictionary
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "password_hash": row[3],
            "trial_start_date": row[4],
            "subscription_status": row[5] if row[5] is not None else "trial",
            "last_payment_date": row[6],
            "trading_status": row[7] if row[7] is not None else "inactive",
            "trading_state": row[8]
        }
    except sqlite3.Error as e:
        print(f"Database error in get_user_data: {e}")
        return None
    finally:
        conn.close()

def register_user(name, email, password):
    """Register a new user with hashed password."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        trial_start = datetime.now().isoformat()
        c.execute("""
            INSERT INTO users (name, email, password_hash, trial_start_date)
            VALUES (?, ?, ?, ?)
        """, (name, email, password_hash, trial_start))
        conn.commit()
        return {"success": True, "message": "Registration successful"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Email already exists"}
    finally:
        conn.close()

def login_user(email, password):
    """Authenticate a user."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        result = c.fetchone()
        if result and bcrypt.checkpw(password.encode("utf-8"), result[0].encode("utf-8")):
            return {"success": True, "message": "Login successful"}
        return {"success": False, "message": "Invalid email or password"}
    finally:
        conn.close()

def check_subscription_status(email):
    """Check and update user's subscription status."""
    user_data = get_user_data(email)
    if not user_data:
        return None
    
    trial_start = user_data.get("trial_start_date")
    subscription_status = user_data.get("subscription_status", "trial")
    
    if subscription_status == "trial" and trial_start:
        trial_start_date = datetime.fromisoformat(trial_start)
        if datetime.now() > trial_start_date + timedelta(days=7):
            update_subscription_status(email, "expired")
            user_data["subscription_status"] = "expired"
    
    return user_data

def update_subscription_status(email, status):
    """Update user's subscription status."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET subscription_status = ? WHERE email = ?", (status, email))
        conn.commit()
    finally:
        conn.close()

def update_trading_status(email, status):
    """Update user's trading status."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET trading_status = ? WHERE email = ?", (status, email))
        conn.commit()
    finally:
        conn.close()

def save_trading_state(email, state):
    """Save user's trading state as JSON."""
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
    """Retrieve user's trading state."""
    user_data = get_user_data(email)
    if user_data and user_data["trading_state"]:
        return json.loads(user_data["trading_state"])
    return None

def clear_trading_state(email):
    """Clear user's trading state."""
    email = email.lower()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET trading_state = NULL WHERE email = ?", (email,))
        conn.commit()
    finally:
        conn.close()

# Initialize the database on module load
init_db()
