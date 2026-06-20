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

### Evaluation

To explore the benchmarking pipeline, dataset preparation, and performance evaluation scripts, check out the `evals` directory.

---

## Tech Stack

* **Backend:** FastAPI
* **Audio Processing:** Librosa, NumPy, SciPy
* **Database:** PostgreSQL (web application) and SQLite (local evaluation)
---

## Setup Instructions

### Prerequisites

* Python 3.10 or higher
* PostgreSQL database
* pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/Armaan457/Shazam.git
cd shazam
```

### 2. Create a Virtual Environment

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using `.env.example` and add the respective values (PostgreSQL database URL):

   - **macOS/Linux:**
        ```bash
        cp .env.example .env   
        ```
   - **Windows:**
     ```bash
     copy .env.example .env
     ```


### 5. Set Up the Database

Run the database setup script to create tables and indexes:

```bash
python setup_db.py
```

### 6. Run the App

Start the FastAPI server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```


