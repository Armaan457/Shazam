# Shazam

A FastAPI-based service that identifies songs from audio clips using audio fingerprinting, inspired by the research paper [*An Industrial-Strength Audio Search Algorithm* by Avery Li-Chun Wang (Shazam, 2003)](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf).

---

##  How It Works

* Audio is processed into a spectrogram
* Key frequency peaks are extracted
* Peaks are converted into compact hashes (fingerprints)
* Fingerprints are stored in a database
* For a new query, matching hashes are fetched
* A voting mechanism based on time alignment selects the best match

---

## Key Highlights

* Efficient hash-based matching (not raw audio comparison)
* Scalable design using database indexing
* Accurate results using offset-based voting
* Lightweight and modular FastAPI architecture
* Redis-backed caching for faster repeated searches

---

## Results

| Metric                    | Result                   |
| ------------------------- | ------------------------ |
| Dataset                   | 8,000 Tracks             |
| Top-1 Accuracy            | **99.17%**               |
| Top-5 Accuracy            | **99.82%**               |
| Coverage                  | **99.85%**               |
| Average Retrieval Latency | **0.1545s**             |
| P95 Retrieval Latency     | **0.284s**              |

### Under Distortions

| Condition            | Top-1 Accuracy |
| -------------------- | -------------: |
| Phone Recording      | **98.41%**     |
| Noise (Low)          | **98.47%**     |
| Noise (High)         | **98.77%**     |
| Realistic Distortion | **98.69%**     |
| Reverb               | **99.02%**     |

### Strengths and Limitations

The system maintains high recognition accuracy on clean audio as well as under common real-world distortions such as environmental noise, phone-recording artifacts, and reverberation. Performance degrades under pitch-shifting and significant time-stretching due to the frequency- and time-dependent nature of Shazam-style audio fingerprints.

### Evaluation

To explore the benchmarking pipeline, dataset preparation, and performance evaluation scripts, check out the `evals` directory.

---

## Tech Stack

* **Backend:** FastAPI
* **Audio Processing:** Librosa, NumPy, SciPy
* **Database:** PostgreSQL (web application) and SQLite (local evaluation)
* **Caching Layer:** Redis
---

## Setup Instructions

### Prerequisites

* Python 3.10 or higher
* PostgreSQL database
* pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/Armaan457/Shazam.git
```

### 2. Start PostgreSQL and Redis

Start the PostgreSQL database and Redis using Docker Compose inside `app` directory:

```bash
cd app
docker compose up -d
cd ..
```

Alternatively, you may use managed PostgreSQL and Redis services

### 3. Create a Virtual Environment

Activate virtual environment using the python version specified in `.python-version` file:

- **macOS/Linux:**

    ```bash
    python -m venv env
    source env/bin/activate
    ```
 - **Windows:**
     ```bash
     python -m venv env
     env\Scripts\activate
     ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file using `.env.example` and add the respective values (PostgreSQL and Redis connection URL):

   - **macOS/Linux:**
        ```bash
        cp .env.example .env   
        ```
   - **Windows:**
     ```bash
     copy .env.example .env
     ```


### 6. Set Up the Database

Run the database setup script to create tables and indexes:

```bash
cd app
python setup_db.py
```

### 7. Run the App

Start the FastAPI server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```


