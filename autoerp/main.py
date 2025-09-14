from fastapi import FastAPI

app = FastAPI(title="AutoERP API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Bienvenue dans AutoERP 🚀"}
