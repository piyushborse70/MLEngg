from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.predict import SingletonModelLoader  # Assuming SingletonModelLoader is in the 'predict.py' file

# Create FastAPI Router
router = APIRouter()

# Define input format using Pydantic models
class PredictRequest(BaseModel):
    context: str

# Load the model using Singleton pattern
model_path = "model/next_word_model.pkl"
model_loader = SingletonModelLoader(model_path)

# Define the route for predicting the next word
@router.post("/predict/")
async def predict_next_word(request: PredictRequest):
    try:
        context = request.context
        prediction = model_loader.predict(context)
        return {"context": context, "next_word": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")