from fastapi import FastAPI
from app.routes import router, mount_static_files

app = FastAPI(title="Shazam API", version="1.0.0")

app.include_router(router)
mount_static_files(app)
