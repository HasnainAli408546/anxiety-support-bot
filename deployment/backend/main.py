# deployment/backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add project root so that "src" is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.main_pipeline import AnxietyBotPipeline


app = FastAPI(title="Anxiety Support Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize pipeline once
pipeline = AnxietyBotPipeline()


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
async def chat_endpoint(chat: ChatRequest):
    """
    POST /chat
    Body:
      - user_id: string
      - message: string

    Returns the full pipeline result, including:
    - clinical_response.message
    - intent_scores
    - emotion_scores
    - metadata, safety, session, analytics, etc.
    """
    try:
        # Continue existing flow if active, otherwise start/process new message
        if chat.user_id in pipeline.clinical_flow_manager.active_flows:
            result = pipeline.continue_conversation(chat.user_id, chat.message)
        else:
            result = pipeline.process_message(chat.message, chat.user_id)

        # Directly return full result dict
        return result
    except Exception as e:
        # Error fallback structure
        return {
            "error": str(e),
            "clinical_response": {
                "message": "I'm experiencing a technical difficulty, but I still want to help you. "
                           "How are you feeling right now?",
                "flow_type": "error_support",
                "requires_input": True,
                "suggested_responses": [],
                "step_info": {"current_step": 1, "total_steps": 1, "intervention_type": "error_recovery"},
            },
        }


@app.get("/")
def health_check():
    return {"status": "Anxiety Support Bot API is running."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
