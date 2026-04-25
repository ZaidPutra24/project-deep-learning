from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline.generation import ResponseGenerator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = ResponseGenerator()
bot.setup_engine(
    api_key="AIzaSyBeqi4QGhN_lFpT4F3YnalsddC-Mxt_urQ",
    emb_path="embedding/stunting_embeddings.npz",
    meta_path="embedding/stunting_embeddings_metadata.json"
)

@app.post("/chat")
def chat(req: dict):
    query = req.get("query", "")
    response = bot.generate(query)
    return {"response": response}