from transformers import pipeline

# Load pretrained sentiment model
sentiment_analyzer = pipeline("sentiment-analysis")

# Test text
text = "I love this project, it is very easy and fun!"

# Run prediction
result = sentiment_analyzer(text)

print("Input Text:", text)
print("Prediction:", result)
