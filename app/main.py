import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import api_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AIVLE School Team20 Big Project")

app.include_router(api_router, prefix="/api")
