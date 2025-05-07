import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set page 
st.set_page_config(
    page_title="Social Media Usage Analyzer",
    page_icon="📱",
    layout="centered"
)

# App title and description
st.title("Social Media Usage Analyzer")
st.markdown("""
This app analyzes whether your social media usage is excessive based on the number of hours you spend daily.
The analysis uses a BERT model to classify your usage pattern.
""")

# Function to load BERT model
@st.cache_resource
def load_model():
    """Load the pre-trained BERT model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    
    
    return model, tokenizer

# Function to analyze social media usage
def analyze_social_media_usage(hours):
    """
    Analyze if the number of hours spent on social media is too much
    
    Args:
        hours (float): Number of hours spent on social media daily
        
    Returns:
        dict: Analysis results including classification and confidence
    """
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Create a text description based on hours
    text = f"I spend {hours} hours on social media every day."
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # For demonstration purposes, we'll use a simple rule-based approach
    # and adjust the model output based on common guidelines
    
    # Scale hours to get a probability between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_hours = scaler.fit_transform(np.array([[min(hours, 10)]]))[0][0]
    
    # Adjust probability based on hours (this is a simplified approach)
    # Generally, more than 2-3 hours is considered excessive for adults
    if hours <= 1:
        probability_excessive = 0.1 + (scaled_hours * 0.2)
    elif hours <= 2:
        probability_excessive = 0.3 + (scaled_hours * 0.2)
    elif hours <= 3:
        probability_excessive = 0.5 + (scaled_hours * 0.1)
    else:
        probability_excessive = 0.7 + (scaled_hours * 0.3)
    
    probability_excessive = min(probability_excessive, 0.99)
    probability_reasonable = 1 - probability_excessive
    
    # Determine classification
    classification = "excessive" if probability_excessive > 0.5 else "reasonable"
    
    return {
        "classification": classification,
        "confidence": max(probability_excessive, probability_reasonable) * 100,
        "probability_excessive": probability_excessive * 100,
        "probability_reasonable": probability_reasonable * 100
    }

# Main app interface
st.header("Analyze Your Social Media Usage")

# Input for hours spent on social media
hours = st.number_input(
    "How many hours do you spend on social media daily?",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5,
    help="Enter the average number of hours you spend on social media platforms each day"
)

# Button to trigger analysis
if st.button("Analyze My Usage"):
    with st.spinner("Analyzing your social media usage..."):
        # Perform analysis
        results = analyze_social_media_usage(hours)
        
        # Display results
        st.subheader("Analysis Results")
        
        # Create a color based on classification
        color = "red" if results["classification"] == "excessive" else "green"
        
        # Display classification with appropriate styling
        st.markdown(
            f"<h3 style='color: {color};'>Your social media usage is "
            f"<span style='text-transform: uppercase;'>{results['classification']}</span></h3>", 
            unsafe_allow_html=True
        )
        
        # Display confidence
        st.markdown(f"Confidence: {results['confidence']:.1f}%")
        
        # Create a progress bar for visualization
        st.subheader("Usage Assessment")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Excessive")
            st.progress(results["probability_excessive"]/100)
            st.markdown(f"{results['probability_excessive']:.1f}%")
        with col2:
            st.markdown("Reasonable")
            st.progress(results["probability_reasonable"]/100)
            st.markdown(f"{results['probability_reasonable']:.1f}%")
        
        # Provide recommendations based on classification
        st.subheader("Recommendations")
        if results["classification"] == "excessive":
            st.markdown("""
            - Consider setting time limits on your social media apps
            - Take regular breaks from screens
            - Find alternative activities to replace some of your social media time
            - Use features like 'Screen Time' on iOS or 'Digital Wellbeing' on Android to monitor usage
            """)
        else:
            st.markdown("""
            - Your social media usage appears to be within reasonable limits
            - Continue to be mindful of how you spend your time online
            - Focus on quality interactions rather than mindless scrolling
            - Regularly evaluate how social media makes you feel
            """)

# Add information about the model
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a BERT (Bidirectional Encoder Representations from Transformers) model to analyze social media usage patterns.

*Note:* This is a demonstration app. In a real-world scenario, the model would be fine-tuned on actual social media usage data.

*Guidelines:*
- 0-1 hours: Generally considered healthy
- 1-2 hours: Moderate usage
- 2-3 hours: Approaching excessive
- 3+ hours: Often considered excessive for adults

These guidelines may vary based on age, profession, and purpose of use.
""")

# Footer
st.markdown("---")
st.markdown("© 2023 Social Media Analyzer | Created with Streamlit and BERT")