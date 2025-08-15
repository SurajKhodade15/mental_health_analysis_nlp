import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.sparse import hstack
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Mental Health Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stAlert {
    margin-top: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-result {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    text-align: center;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except:
        return False

# Load models and preprocessing components
@st.cache_resource
def load_models():
    """Load the best trained model and preprocessing components"""
    try:
        # Load the TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load the label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load preprocessing pipeline
        with open('models/preprocessing_pipeline.pkl', 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        
        # Load model information
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Load the best trained model
        with open('models/training_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Extract preprocessing components
        stemmer = preprocessing_pipeline['stemmer']
        lemmatizer = preprocessing_pipeline['lemmatizer']
        stopwords = preprocessing_pipeline['stopwords']
        
        # Define text_preprocessing function using loaded components
        def text_preprocessing(text):
            """
            Basic text preprocessing: lowercase, remove non-alphabetic, remove stopwords.
            """
            # Lowercase
            text = text.lower()
            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords
            filtered_tokens = [token for token in tokens if token not in stopwords]
            # Join back to string
            return ' '.join(filtered_tokens)
        
        return {
            'vectorizer': vectorizer,
            'label_encoder': label_encoder,
            'text_preprocessing': text_preprocessing,
            'stemmer': stemmer,
            'lemmatizer': lemmatizer,
            'stopwords': stopwords,
            'best_model': best_model,
            'model_info': model_info
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def predict_mental_health(text, components):
    """Complete prediction pipeline for mental health classification"""
    try:
        # Extract components
        vectorizer = components['vectorizer']
        label_encoder = components['label_encoder']
        text_preprocessing = components['text_preprocessing']
        stemmer = components['stemmer']
        lemmatizer = components['lemmatizer']
        best_model = components['best_model']
        model_info = components['model_info']
        
        # Step 1: Text preprocessing
        cleaned_text = text_preprocessing(text)
        
        # Step 2: Lemmatization
        lemmatized_text = lemmatizer.lemmatize(cleaned_text)
        
        # Step 3: Tokenization
        tokens = word_tokenize(lemmatized_text)
        
        # Step 4: Stemming
        stemmed_tokens = ' '.join([stemmer.stem(str(token)) for token in tokens])
        
        # Step 5: Calculate numerical features
        num_characters = len(text)
        num_sentences = len(nltk.sent_tokenize(text))
        
        # Step 6: TF-IDF transformation
        text_features = vectorizer.transform([stemmed_tokens])
        
        # Step 7: Combine features
        numerical_features = np.array([[num_characters, num_sentences]])
        combined_features = hstack([text_features, numerical_features])
        
        # Step 8: Make prediction using the best model
        prediction = best_model.predict(combined_features)[0]
        
        # Get prediction probabilities if available
        probabilities = None
        confidence = None
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(combined_features)[0]
            confidence = np.max(probabilities)
            # Create probability dictionary
            prob_dict = {label_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}
        else:
            prob_dict = None
        
        # Decode the prediction
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'model_used': model_info['model_name'],
            'processed_text': stemmed_tokens[:100] + '...' if len(stemmed_tokens) > 100 else stemmed_tokens,
            'features': {
                'num_characters': num_characters,
                'num_sentences': num_sentences,
                'num_tokens': len(tokens)
            },
            'success': True
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def create_probability_chart(probabilities):
    """Create a bar chart for class probabilities"""
    if probabilities is None:
        return None
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = px.bar(
        x=classes, 
        y=probs,
        title="Prediction Probabilities by Mental Health Category",
        labels={'x': 'Mental Health Category', 'y': 'Probability'},
        color=probs,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_feature_chart(features):
    """Create a chart showing text features"""
    feature_names = list(features.keys())
    feature_values = list(features.values())
    
    fig = px.bar(
        x=feature_names,
        y=feature_values,
        title="Text Feature Analysis",
        labels={'x': 'Features', 'y': 'Count'},
        color=feature_values,
        color_continuous_scale='blues'
    )
    
    fig.update_layout(
        showlegend=False,
        height=300
    )
    
    return fig

# Main app
def main():
    # Title and description
    st.title("🧠 Mental Health Text Analyzer")
    st.markdown("""
    This application uses Natural Language Processing (NLP) and Machine Learning to analyze text 
    and predict mental health conditions. Enter any text below to get predictions from multiple AI models.
    """)
    
    # Initialize NLTK
    if not download_nltk_data():
        st.error("Failed to download required NLTK data. Please check your internet connection.")
        return
    
    # Load models
    components = load_models()
    if components is None:
        st.error("Failed to load models. Please ensure all model files are present in the 'models' folder.")
        return
    
    # Sidebar
    st.sidebar.header("🔧 Model Information")
    
    # Model information
    model_name = components['model_info']['model_name']
    model_accuracy = components['model_info']['accuracy']
    
    st.sidebar.success(f"**Best Model:** {model_name}")
    st.sidebar.metric("Model Accuracy", f"{model_accuracy:.1%}")
    
    # Show all model performances
    st.sidebar.markdown("### 📊 All Model Performances")
    all_accuracies = components['model_info']['all_accuracies']
    for name, acc in sorted(all_accuracies.items(), key=lambda x: x[1], reverse=True):
        icon = "🏆" if name == model_name else "📈"
        st.sidebar.write(f"{icon} {name}: {acc:.3f}")
    
    # Available classes
    st.sidebar.markdown("### 🏷️ Mental Health Categories")
    classes = components['label_encoder'].classes_
    for cls in classes:
        st.sidebar.write(f"• {cls}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type your own text", "Use example texts"],
            horizontal=True
        )
        
        if input_method == "Use example texts":
            example_texts = {
                "Depression": (
                    "I've been feeling really sad and hopeless lately. Nothing seems to bring me joy anymore and I just want to stay in bed all day. "
                    "I struggle to find motivation for even the simplest tasks, and my energy levels are always low. "
                    "Sometimes I feel empty inside and have trouble concentrating. I often feel worthless and guilty about not being able to do more. "
                    "Sleep is difficult, and I wake up feeling tired no matter how long I rest."
                ),
                "Anxiety": "My heart races and I can't breathe properly when I'm in crowded places. I'm constantly worried about having a panic attack.",
                "Normal": "I feel great today! The weather is beautiful and I'm excited about my new project at work. Life is good.",
                "Stress": "I feel overwhelmed with work and can't seem to manage my stress levels. Everything feels too much.",
                "Insomnia": (
                    "I've been having trouble sleeping for weeks now. I just can't seem to get a good night's rest no matter what I try. "
                    "I lie awake for hours, my mind racing with thoughts, and when I finally fall asleep, I wake up repeatedly throughout the night. "
                    "During the day, I feel exhausted and unable to focus. My mood is affected and I feel irritable because of the lack of sleep."
                )
            }
            
            selected_example = st.selectbox("Choose an example:", list(example_texts.keys()))
            user_text = st.text_area(
                "Example text (you can edit this):",
                value=example_texts[selected_example],
                height=150
            )
        else:
            user_text = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste your text here for mental health analysis...",
                height=150
            )
        
        # Prediction button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner(f"Analyzing with {model_name}..."):
                    result = predict_mental_health(user_text, components)
                
                if result['success']:
                    # Main prediction result
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Prediction: {result['predicted_class']}</h2>
                        <p>Model: {result['model_used']}</p>
                        {f"<p>Confidence: {result['confidence']:.1%}</p>" if result['confidence'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    st.header("📊 Detailed Analysis")
                    
                    # Create columns for metrics
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.metric(
                            "Characters", 
                            result['features']['num_characters'],
                            help="Total number of characters in the text"
                        )
                    
                    with metric_cols[1]:
                        st.metric(
                            "Sentences", 
                            result['features']['num_sentences'],
                            help="Number of sentences detected"
                        )
                    
                    with metric_cols[2]:
                        st.metric(
                            "Tokens", 
                            result['features']['num_tokens'],
                            help="Number of words after tokenization"
                        )
                    
                    # Charts
                    chart_cols = st.columns(2)
                    
                    with chart_cols[0]:
                        if result['probabilities']:
                            prob_chart = create_probability_chart(result['probabilities'])
                            st.plotly_chart(prob_chart, use_container_width=True)
                    
                    with chart_cols[1]:
                        feature_chart = create_feature_chart(result['features'])
                        st.plotly_chart(feature_chart, use_container_width=True)
                    
                    # Processed text
                    with st.expander("🔍 View Processed Text"):
                        st.text(result['processed_text'])
                    
                    # Probability details
                    if result['probabilities']:
                        with st.expander("📈 Detailed Probabilities"):
                            prob_df = pd.DataFrame(
                                list(result['probabilities'].items()),
                                columns=['Category', 'Probability']
                            ).sort_values('Probability', ascending=False)
                            st.dataframe(prob_df, use_container_width=True)
                
                else:
                    st.error(f"Error in prediction: {result['error']}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("ℹ️ About This App")
        st.markdown("""
        ### How it works:
        1. **Text Preprocessing**: Cleans and normalizes the input text
        2. **Feature Extraction**: Converts text to numerical features using TF-IDF
        3. **Model Prediction**: Uses the best trained ML model to classify mental health condition
        4. **Results**: Shows prediction with confidence scores and analysis
        
        ### Features:
        - 🤖 Best performing AI model
        - 📊 Confidence scores
        - 📈 Feature analysis
        - 🎯 Real-time predictions
        
        ### Mental Health Categories:
        The model can identify various mental health conditions including depression, anxiety, stress, and normal states.
        
        ⚠️ **Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
        """)
        
        # Model performance
        st.markdown("### 🏆 Model Performance")
        st.info(f"Using {model_name} with {model_accuracy:.1%} accuracy on test data.")
        
        # Performance comparison chart
        if len(all_accuracies) > 1:
            st.markdown("#### Model Comparison")
            perf_df = pd.DataFrame(list(all_accuracies.items()), columns=['Model', 'Accuracy'])
            st.bar_chart(perf_df.set_index('Model'))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit | Mental Health Text Analysis | 
        <a href='https://github.com/SurajKhodade15/mental_health_analysis_nlp' target='_blank'>View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
