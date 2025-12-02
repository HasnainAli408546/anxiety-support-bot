import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionPredictor:
    def __init__(self, model_path):
        """Load your trained emotion detection model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer from folder
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Emotion labels - MUST match your Kaggle training `target_columns` in order!
        self.target_columns = [
            'custom_panic', 'custom_anxiety', 'custom_loneliness', 
            'custom_frustration', 'custom_calm'
        ]
    
    def predict_emotions(self, text: str, threshold: float = 0.5) -> dict:
        """Predict emotions from text and return scores for each target"""
        # Tokenize, transfer to device
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=64
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            # For multi-label (sigmoid activation): use sigmoid
            predictions = torch.sigmoid(outputs.logits)
        
        # Assemble dictionary mapping each label to prediction score
        emotion_scores = {}
        for i, emotion in enumerate(self.target_columns):
            score = float(predictions[0][i])
            emotion_scores[emotion] = score
        return emotion_scores
