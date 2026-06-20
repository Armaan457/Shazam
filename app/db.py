from psycopg2.pool import ThreadedConnectionPool
import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

db_pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=20,
    dsn=DATABASE_URL,
)

def get_connection():
    return db_pool.getconn()

def release_connection(conn):
    db_pool.putconn(conn)