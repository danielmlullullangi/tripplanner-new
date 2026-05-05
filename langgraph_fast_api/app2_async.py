#App2.py yang sudah disesuaikan dengan Guvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import time, os
from contextlib import asynccontextmanager

# Import agent dan state TypedDict dari file yang sudah ada (tidak diubah)
from embedding_lanjutan_3_async import agent, init_db_pool, get_db_pool, _db_pool, close_db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Worker PID {os.getpid()}] Starting up...")
    await init_db_pool()
    yield
    

    await close_db_pool()
    print("Menutup _db_pool ...")
    print(f"[Worker PID {os.getpid()}] Shutting down...")

app = FastAPI(
    title="Trip Planner AI API",
    description="API untuk perencanaan itinerary perjalanan dengan semantic searching",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== PYDANTIC MODELS ==================
class PlanRequest(BaseModel):
    destination_name: str
    num_days: int
    price_max: int
    user_prompt: str
    travel_whom: str
    destination_type: List[str]

class TripPlannerResponse(BaseModel):
    final_rekomendasi: List[Dict[str, Any]]

# ================== ENDPOINT ==================
@app.get("/")
def root():
    return {"message": "Trip Planner AI API is running"}

@app.get("/hello")
def hello():
    return {"Hello World!"}

@app.post("/plan", response_model=TripPlannerResponse)
async def plan_trip(request: PlanRequest):
    """
    Menerima preferensi pengguna dan mengembalikan rencana perjalanan.
    """
    # Konversi request ke dictionary (hanya field input)
    input_dict = request.model_dump()

    # Buat state awal dengan menambahkan field default yang diperlukan graph
    initial_state = {
        **input_dict,
    }

    # Setel pool database ke dalam state agar bisa diakses oleh node
    # (dengan asumsi node fetch_filter_dan_sort_data menggunakan pool dari global)
    # Anda perlu menyesuaikan node tersebut untuk menggunakan db_pool global.

    try:
        start = time.time()
        # Jalankan graph secara async
        result = await agent.ainvoke(initial_state)
        end = time.time()
        print(f"Waktu proses: {end - start:.2f} detik")

        final_data = result.get("rekomendasi_destinasi_paraph", [])
        return TripPlannerResponse(final_rekomendasi=final_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

# if __name__ == "__main__":
#     # Untuk development, jalankan langsung dengan uvicorn (single worker)
#     uvicorn.run("app2_async:app", host="0.0.0.0", port=6006, reload=True)