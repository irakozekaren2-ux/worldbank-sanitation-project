from database import get_connection

try:
    conn = get_connection()
    print("✅ Connection to PostgreSQL was successful!")

    conn.close()

except Exception as e:
    print("❌ Connection failed!")
    print(e)