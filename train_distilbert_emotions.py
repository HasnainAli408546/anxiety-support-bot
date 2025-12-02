"""
DistilBERT Multi-Label Emotion Classification Training Script
Fine-tunes DistilBERT on GoEmotions-derived anxiety-related emotions
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import Dataset
from torch import nn

print("=== STARTING DISTILBERT EMOTION CLASSIFICATION TRAINING ===")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check if GPU is available (though we'll use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# 1. LOAD SPLIT DATA AND METADATA
# =============================================================================
print("\n=== LOADING TRAINING DATA ===")

# Load training and validation data
train_df = pd.read_csv(r'data\training\splits\train.csv')
val_df = pd.read_csv(r'data\training\splits\val.csv')

# Load training metadata
with open(r'data\training\splits\training_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Training samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")

# Extract target columns and class weights
target_cols = metadata['target_columns']
class_weights = torch.tensor(metadata['class_weights'], dtype=torch.float32)

print(f"Target columns: {target_cols}")
print(f"Class weights: {class_weights.tolist()}")

# Prepare X and Y for both sets
X_train = train_df['text'].astype(str).tolist()
Y_train = train_df[target_cols].astype(int).values
X_val = val_df['text'].astype(str).tolist()
Y_val = val_df[target_cols].astype(int).values

# =============================================================================
# 2. TOKENIZATION
# =============================================================================
print("\n=== TOKENIZING DATA ===")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_texts(texts, tokenizer, max_len=64):
    """Tokenize texts for DistilBERT"""
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )

print("Tokenizing training data...")
train_encodings = tokenize_texts(X_train, tokenizer)
print("Tokenizing validation data...")
val_encodings = tokenize_texts(X_val, tokenizer)
print("✅ Tokenization complete!")

# =============================================================================
# 3. PYTORCH DATASET CLASS
# =============================================================================
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = EmotionDataset(train_encodings, Y_train)
val_dataset = EmotionDataset(val_encodings, Y_val)

print(f"Training dataset size: {len(train_dataset):,}")
print(f"Validation dataset size: {len(val_dataset):,}")

# =============================================================================
# 4. MODEL SETUP
# =============================================================================
print("\n=== SETTING UP DISTILBERT MODEL ===")

num_labels = len(target_cols)
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

print(f"Model loaded with {num_labels} output labels")

# =============================================================================
# 5. EVALUATION METRICS
# =============================================================================
def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification"""
    predictions, labels = eval_pred
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Convert to binary predictions using 0.5 threshold
    y_pred = (probs.numpy() >= 0.5).astype(int)
    y_true = labels.astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Accuracy (exact match for multi-label)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'accuracy': accuracy
    }
    
    # Add per-class F1 scores
    for i, col in enumerate(target_cols):
        metrics[f'f1_{col}'] = per_class_f1[i]
    
    return metrics

# =============================================================================
# 6. CUSTOM TRAINER WITH WEIGHTED LOSS
# =============================================================================
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kgargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted BCE loss for class imbalance
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 7. TRAINING ARGUMENTS
# =============================================================================
print("\n=== SETTING UP TRAINING ARGUMENTS ===")

output_dir = r'distilbert_emotion_model'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,              # Increased from 2 for better results
    per_device_train_batch_size=8,   # Small batch size for CPU
    per_device_eval_batch_size=16,   # Larger eval batch size
    eval_strategy="steps",
    eval_steps=500,                  # Evaluate every 500 steps
    save_steps=1000,                 # Save every 1000 steps
    logging_steps=100,               # Log every 100 steps
    save_total_limit=3,              # Keep only 3 best models
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    use_cpu=True,                    # Force CPU usage
    dataloader_num_workers=0,        # Important for Windows CPU training
    report_to=[],                    # Disable wandb/tensorboard
    seed=42
)

print(f"Output directory: {output_dir}")
print(f"Training epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")

# =============================================================================
# 8. CREATE TRAINER AND START TRAINING
# =============================================================================
print("\n=== CREATING TRAINER ===")

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("✅ Trainer created successfully!")

print("\n=== STARTING TRAINING ===")
print("This may take several hours on CPU...")
print("Monitor the logs for progress...")

start_time = datetime.now()
trainer.train()
end_time = datetime.now()

training_duration = end_time - start_time
print(f"\n🎉 TRAINING COMPLETED!")
print(f"Training duration: {training_duration}")

# =============================================================================
# 9. FINAL EVALUATION
# =============================================================================
print("\n=== FINAL EVALUATION ===")

final_metrics = trainer.evaluate()
print("Final validation metrics:")
for key, value in final_metrics.items():
    if key.startswith(('eval_macro_f1', 'eval_micro_f1', 'eval_accuracy')) or key.startswith('eval_f1_'):
        print(f"  {key}: {value:.4f}")

# =============================================================================
# 10. SAVE MODEL AND RESULTS
# =============================================================================
print("\n=== SAVING MODEL AND RESULTS ===")

# Save the model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(output_dir)

# Save training results
results = {
    'training_completed_at': end_time.isoformat(),
    'training_duration_seconds': training_duration.total_seconds(),
    'final_metrics': final_metrics,
    'model_config': {
        'base_model': 'distilbert-base-uncased',
        'num_labels': num_labels,
        'target_columns': target_cols,
        'max_sequence_length': 64
    },
    'training_config': {
        'epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
        'class_weights': class_weights.tolist()
    }
}

with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Model saved to: {output_dir}")
print(f"✅ Tokenizer saved to: {output_dir}")
print(f"✅ Training results saved to: {output_dir}/training_results.json")

# =============================================================================
# 11. TEST INFERENCE
# =============================================================================
print("\n=== TESTING INFERENCE ===")

def test_emotion_prediction(text, model, tokenizer, target_cols, threshold=0.5):
    """Test the trained model on a sample text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        
    predicted_emotions = []
    for i, prob in enumerate(predictions[0]):
        if prob > threshold:
            predicted_emotions.append({
                'emotion': target_cols[i],
                'confidence': float(prob)
            })
    
    return predicted_emotions

# Test with sample texts
test_texts = [
    "I'm so scared about tomorrow's presentation",
    "I feel completely alone and nobody understands me",
    "This is so annoying, nothing is working right!",
    "Everything is going well, I feel at peace",
    "I can't breathe, my heart is racing, I'm panicking"
]

print("Sample predictions:")
for text in test_texts:
    emotions = test_emotion_prediction(text, model, tokenizer, target_cols)
    print(f"Text: {text}")
    if emotions:
        for emotion in emotions:
            print(f"  - {emotion['emotion']}: {emotion['confidence']:.3f}")
    else:
        print("  - No emotions detected")
    print()

print("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY! ===")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
