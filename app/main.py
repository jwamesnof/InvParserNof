from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.database import init_db
from app.controllers.invoice_controller import router as invoice_router
from app.controllers.extract_controller import router as extract_router
from app.controllers.health_controller import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="InvParser API", lifespan=lifespan)

app.include_router(invoice_router)
app.include_router(extract_router)
app.include_router(health_router)

