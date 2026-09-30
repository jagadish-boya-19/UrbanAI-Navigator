import sqlite3
import hashlib
import os

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/users.db")
cur  = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT    UNIQUE NOT NULL,
        password TEXT    NOT NULL,
        name     TEXT    NOT NULL,
        role     TEXT    NOT NULL,
        created  TEXT    DEFAULT CURRENT_TIMESTAMP
    )
""")

users = [
    ("admin",    hash_pw("admin123"),    "Admin",           "Administrator"),
    ("jagadish", hash_pw("urbanai2025"), "Jagadish S Boya", "Team Lead"),
    ("omkar",    hash_pw("urbanai2025"), "Omkar Shanbhag",  "Developer"),
    ("rohit",    hash_pw("urbanai2025"), "Rohit K Savant",  "Developer"),
    ("guest",    hash_pw("guest123"),    "Guest User",      "Viewer"),
]

cur.executemany("""
    INSERT OR IGNORE INTO users (username, password, name, role)
    VALUES (?, ?, ?, ?)
""", users)

conn.commit()
conn.close()

print("=" * 45)
print("  Database created: data/users.db")
print(f"  Users added: {len(users)}")
print("=" * 45)
print("\n  Login credentials:")
print("  admin    / admin123")
print("  jagadish / urbanai2025")
print("  omkar    / urbanai2025")
print("  rohit    / urbanai2025")
print("  guest    / guest123")