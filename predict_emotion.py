from apify_client import ApifyClient
import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

# Emotion labels from your training
EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

class EmotionPredictor:
    def __init__(self, model_path="emotion-finetune-distilbert"):
        """Initialize the emotion prediction model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
    def predict_emotion(self, text):
        """Predict emotion from text"""
        if not text or text.strip() == "":
            return None, None
            
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = predictions.argmax(-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return EMOTIONS[predicted_class], confidence

class InstagramScraper:
    def __init__(self):
        """Initialize the Apify client"""
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment variables")
        self.client = ApifyClient(api_token)
    
    def scrape_post(self, instagram_url):
        """Scrape Instagram post caption"""
        run_input = {
            "username": [instagram_url],
            "resultsLimit": 1,
        }
        
        print(f"Scraping Instagram post: {instagram_url}")
        
        # Run the Actor and wait for it to finish
        run = self.client.actor("nH2AHrwxeTRJoN5hX").call(run_input=run_input)
        
        # Fetch results
        captions = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            if 'caption' in item:
                captions.append(item['caption'])
        
        return captions[0] if captions else None

def main():
    # Example Instagram post URL
    instagram_url = input("Enter Instagram post URL: ")
    
    # Initialize scraper and predictor
    scraper = InstagramScraper()
    predictor = EmotionPredictor()
    
    try:
        # Scrape the caption
        caption = scraper.scrape_post(instagram_url)
        
        if caption:
            print(f"Caption: {caption}")
  
            # Predict emotion
            emotion, confidence = predictor.predict_emotion(caption)
            
            if emotion:
                print(f"Predicted Emotion: {emotion}")
                print(f"Confidence: {confidence:.2%}")
            else:
                print("Could not predict emotion (empty caption)")
        else:
            print("No caption found in the post")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()