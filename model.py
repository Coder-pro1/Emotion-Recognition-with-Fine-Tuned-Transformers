
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("emotion-finetune-distilbert").to(device)
tokenizer = AutoTokenizer.from_pretrained("emotion-finetune-distilbert")
model.eval()

# Emotion classes
classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def predict_emotion(text):
    """Predict emotion for given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**inputs)
    predicted_class = output.logits.argmax(-1).item()
    return classes[predicted_class]

if __name__ == "__main__":
    print("Emotion Classifier")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Enter text: ")
        if text.lower() == "quit":
            break
        emotion = predict_emotion(text)
        print(f"Predicted emotion: {emotion}\n")