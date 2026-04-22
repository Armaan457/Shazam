import os
import sys
from dotenv import load_dotenv
import psycopg2

load_dotenv()
    
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("DATABASE_URL not found in .env file")
    sys.exit(1)

try:
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    print("Creating tables...")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT,
            duration FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("Songs table created")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash BIGINT NOT NULL,
            song_id INT NOT NULL,
            time_offset INT NOT NULL,
            anchor_freq INT NOT NULL,
            FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
        );
    """)
    print("Fingerprints table created")
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprint_hash
        ON fingerprints(hash);
    """)
    print("Fingerprint hash index created")
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprint_hash_anchor
        ON fingerprints(hash, anchor_freq);
    """)
    print("Fingerprint hash-anchor index created")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("\nDatabase setup completed successfully!")
    
except psycopg2.OperationalError as e:
    print(f"Database connection error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during database setup: {e}")
    sys.exit(1)
