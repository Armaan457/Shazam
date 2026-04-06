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

## Tech Stack

* **Backend:** FastAPI
* **Audio Processing:** librosa, NumPy, SciPy
* **Database:** PostgreSQL

