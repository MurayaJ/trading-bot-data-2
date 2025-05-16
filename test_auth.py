import sqlite3

DB_PATH = r"C:\Users\admin\trading-bot-data\db\trading_users.db"  # Correct database file

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT password_hash FROM users LIMIT 1")
sample_hash = c.fetchone()

if sample_hash:  # Ensure there's data before accessing it
    print(type(sample_hash[0]))  # Should print <class 'str'>
else:
    print("No data found in password_hash column")

conn.close()
