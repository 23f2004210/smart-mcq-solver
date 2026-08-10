import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./roberta_deployment"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict(question, options):

    combined_text = (
        f"Question: {question} "
        f"A: {options['A']} "
        f"B: {options['B']} "
        f"C: {options['C']} "
        f"D: {options['D']} "
        f"E: {options['E']}"
    )

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    top3 = torch.topk(
        probabilities,
        k=3,
        dim=1
    ).indices[0]

    id2label = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }

    return [
        id2label[int(idx)]
        for idx in top3
    ]


# Test MCQ
question = "What is the capital of France?"

options = {
    "A": "London",
    "B": "Paris",
    "C": "Berlin",
    "D": "Rome",
    "E": "Madrid"
}

prediction = predict(question, options)

print("Top 3 predictions:", prediction)