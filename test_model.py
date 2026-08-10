from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_PATH = "./roberta_deployment"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Tokenizer loaded!")

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

print("Model loaded!")

print("Number of labels:", model.config.num_labels)
print("Labels:", model.config.id2label)