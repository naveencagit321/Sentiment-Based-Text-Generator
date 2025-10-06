import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="Sentiment-Based Text Generator",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("✍️ Sentiment-Based Text Generator")
st.write(
    "Enter a topic, sentence, or idea. The application will first analyze the sentiment "
    "(Positive, Negative, or Neutral) of your input and then generate a short paragraph "
    "that matches the detected emotional tone. Use the sidebar to control the output length and creativity."
)

# Streamlit's caching
@st.cache_resource
def load_models():
    """
    Loads and returns the sentiment analysis and text generation models.
    This function is cached to improve performance.
    """
    try:
        # Loaded a pre-trained model for sentiment analysis.
        sentiment_analyzer = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        # Loaded a larger, more capable pre-trained model for text generation (GPT-2 Medium).
        # This model provides more coherent and contextually relevant text.
        text_generator = pipeline("text-generation", model="gpt2-medium")
        return sentiment_analyzer, text_generator
    except Exception as e:
        st.error(f"Error loading models: {e}. Please check your internet connection and dependencies.")
        return None, None

with st.spinner("Loading AI models... This may take a moment (the text generator is larger now)."):
    sentiment_analyzer, text_generator = load_models()

# UI for Generation Settings (Sidebar)
st.sidebar.header("Generation Settings")
max_length = st.sidebar.slider(
    "Maximum Length (tokens)",
    min_value=100,
    max_value=1024,
    value=400,
    help="Controls the maximum number of tokens in the generated text. Longer text takes more time to generate."
)
temperature = st.sidebar.slider(
    "Temperature (Creativity)",
    min_value=0.5,
    max_value=1.5,
    value=0.9,
    step=0.1,
    help="Controls the randomness of the output. Lower values make the text more predictable, while higher values make it more creative and surprising."
)


if sentiment_analyzer and text_generator:
    # Create a text area for user input.
    user_input = st.text_area(
        "Enter your prompt here:",
        "Energetic Goa beach vibes along with best friends for pleasant evening",
        height=150,
    )

    if st.button("Generate Text"):
        if user_input:
            with st.spinner("Analyzing sentiment and generating text..."):
                try:
                    # Analyze the sentiment of the user's input.
                    sentiment_result = sentiment_analyzer(user_input)
                    sentiment = sentiment_result[0]['label']
                    score = sentiment_result[0]['score']

                    st.subheader("Sentiment Analysis Result")
                    
                    if sentiment == "POSITIVE":
                        st.success(f"Detected Sentiment: **{sentiment}** (Confidence: {score:.2f})")
                        # Creation of a sentiment-aware prompt for the generator.
                        prompt = f"Write a joyful, vivid, and optimistic essay about the following idea: {user_input}"
                    elif sentiment == "NEGATIVE":
                        st.error(f"Detected Sentiment: **{sentiment}** (Confidence: {score:.2f})")
                        prompt = f"Write a thoughtful and somber essay exploring the following concern: {user_input}"
                    else:  # NEUTRAL
                        st.info(f"Detected Sentiment: **{sentiment}** (Confidence: {score:.2f})")
                        prompt = f"Write a neutral and informative essay about the following topic: {user_input}"

                    # Generate text based on the new prompt and user settings.
                    generated_text_result = text_generator(
                        prompt,
                        max_length=max_length,
                        num_return_sequences=1,
                        truncation=True,
                        temperature=temperature
                    )
                    generated_text = generated_text_result[0]['generated_text']

                    st.subheader("Generated Text")
                    # Clean the output by removing the instructional prompt more reliably.
                    cleaned_text = generated_text[len(prompt):].strip()
                    st.write(cleaned_text)

                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")
        else:
            st.warning("Please enter some text in the prompt box.")
else:
    st.error("The AI models could not be loaded. The application cannot proceed.")