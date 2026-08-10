import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Hugging Face model
MODEL_PATH = "ilesha06/smart-mcq-roberta"


# Load model only when needed
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    return tokenizer, model


# Label mapping
id2label = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}


# Prediction function
def predict(question, options):

    # Load model here
    tokenizer, model = load_model()

    combined_text = (
        question
        + " "
        + " ".join(
            f"{label}: {option}"
            for label, option in options.items()
        )
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

    predictions = [
        id2label[int(index)]
        for index in top3
    ]

    return predictions


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Smart MCQ Solver")

st.write(
    "Enter an MCQ and its five options. "
    "The model will predict the top 3 answers."
)


question = st.text_area(
    "Question",
    placeholder="Enter the question here..."
)

option_a = st.text_input("Option A")
option_b = st.text_input("Option B")
option_c = st.text_input("Option C")
option_d = st.text_input("Option D")
option_e = st.text_input("Option E")


if st.button("Solve MCQ"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif not all([
        option_a.strip(),
        option_b.strip(),
        option_c.strip(),
        option_d.strip(),
        option_e.strip()
    ]):

        st.warning("Please enter all five options.")

    else:

        options = {
            "A": option_a,
            "B": option_b,
            "C": option_c,
            "D": option_d,
            "E": option_e
        }

        with st.spinner("Loading model and generating prediction..."):

            predictions = predict(
                question,
                options
            )

        st.subheader("Top 3 Predictions")

        st.write(f"1. **{predictions[0]}**")
        st.write(f"2. **{predictions[1]}**")
        st.write(f"3. **{predictions[2]}**")
