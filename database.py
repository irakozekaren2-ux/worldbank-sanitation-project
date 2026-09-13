import psycopg2
from config import *

def get_connection():
    """
    Creates and returns a connection to the PostgreSQL database.
    """

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    return conn