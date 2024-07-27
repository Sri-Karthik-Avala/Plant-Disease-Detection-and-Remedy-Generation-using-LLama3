import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import ollama

# Initialize the embeddings models
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant clients
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="planty")
# Initialize the Ollama model (use appropriate initialization if different)

# Load the model for classification
def load_model():
    model_path = r"part1/model/plantking2.h5"
    model = tf.keras.models.load_model(model_path)
    return model

def plant_disease_classifier_app():
    # Define class labels
    class_labels = ['Apple__black_rot', 'Apple__healthy', 'Apple__rust', 'Apple__scab', 'Cassava__bacterial_blight', 
                    'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease', 
                    'Cherry__healthy', 'Cherry__powdery_mildew', 'Chili__healthy', 'Chili__leaf curl', 'Chili__leaf spot', 
                    'Chili__whitefly', 'Chili__yellowish', 'Coffee__cercospora_leaf_spot', 'Coffee__healthy', 
                    'Coffee__red_spider_mite', 'Coffee__rust', 'Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 
                    'Corn__northern_leaf_blight', 'Cucumber__diseased', 'Cucumber__healthy', 'Gauva__diseased', 
                    'Gauva__healthy', 'Grape__black_measles', 'Grape__black_rot', 'Grape__healthy', 
                    'Grape__leaf_blight_(isariopsis_leaf_spot)', 'Jamun__diseased', 'Jamun__healthy', 'Lemon__diseased', 
                    'Lemon__healthy', 'Mango__diseased', 'Mango__healthy', 'Peach__bacterial_spot', 'Peach__healthy', 
                    'Pepper_bell__bacterial_spot', 'Pepper_bell__healthy', 'Pomegranate__diseased', 'Pomegranate__healthy', 
                    'Potato__early_blight', 'Potato__healthy', 'Potato__late_blight', 'Rice__brown_spot', 'Rice__healthy', 
                    'Rice__hispa', 'Rice__leaf_blast', 'Rice__neck_blast', 'Soybean__bacterial_blight', 'Soybean__caterpillar', 
                    'Soybean__diabrotica_speciosa', 'Soybean__downy_mildew', 'Soybean__healthy', 'Soybean__mosaic_virus', 
                    'Soybean__powdery_mildew', 'Soybean__rust', 'Soybean__southern_blight', 'Strawberry___leaf_scorch', 
                    'Strawberry__healthy', 'Sugarcane__bacterial_blight', 'Sugarcane__healthy', 'Sugarcane__red_rot', 
                    'Sugarcane__red_stripe', 'Sugarcane__rust', 'Tea__algal_leaf', 'Tea__anthracnose', 'Tea__bird_eye_spot', 
                    'Tea__brown_blight', 'Tea__healthy', 'Tea__red_leaf_spot', 'Tomato__bacterial_spot', 'Tomato__early_blight', 
                    'Tomato__healthy', 'Tomato__late_blight', 'Tomato__leaf_mold', 'Tomato__mosaic_virus', 
                    'Tomato__septoria_leaf_spot', 'Tomato__spider_mites_(two_spotted_spider_mite)', 'Tomato__target_spot', 
                    'Tomato__yellow_leaf_curl_virus', 'Wheat__brown_rust', 'Wheat__healthy', 'Wheat__septoria', 'Wheat__yellow_rust']

    # Customize Streamlit layout
    st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Classifier & Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Add a sidebar for additional information
    st.sidebar.markdown("## Information")
    st.sidebar.markdown("This app classifies plant diseases and provides remedies via chatbot.")

    # Image Classification Section
    st.subheader("Plant Disease Classifier")
    uploaded_image = st.file_uploader("Upload a plant image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.markdown("## Uploaded Image")
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Plant Image", use_column_width=True)

        if st.button("Predict"):
            # Convert the image to a NumPy array
            img = np.array(image)

            # Ensure the image has three channels (RGB)
            if img.ndim == 2 or img.shape[-1] != 3:
                img = np.repeat(img[..., np.newaxis], 3, axis=-1)

            # Resize the image to match the model's input size (224x224)
            img = tf.image.resize(img, (224, 224))

            # Normalize pixel values to [0, 1]
            img = img / 255.0

            # Make a prediction
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            model = load_model()  # Load the model
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            tf.keras.backend.clear_session()  # Clear GPU memory

            # Display the prediction result
            st.markdown("## Prediction Result")
            st.markdown(f"**Predicted Class:** {class_labels[predicted_class]}")
            st.markdown(f"**Confidence:** {prediction[0][predicted_class]:.2f}")

            # Use the predicted class to get remedies
            search_query = f"remedies for {class_labels[predicted_class]}"
            
            # Perform semantic search in the database
            docs = db.similarity_search_with_score(query=search_query, k=5)
            
            # Prepare the context from the search results
            context = ""
            for doc, score in docs:
                context += doc.page_content + " "
            
            # Define prompt
            prompt = f"You are a recommendation system for giving remedies for plant diseases. Using the following context, write about {search_query}:\n{context}"
            
            # Generate response using Ollama's chat method
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])

            # Display the remedies
            st.markdown("## Remedies")
            st.markdown(response['message']['content'])

    # Chatbot Section
    st.subheader("Chatbot for Plant Remedies")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    # Display all messages
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.text_area(f"User-{index}", value=message["content"], height=50, disabled=True)
        elif message["role"] == "assistant":
            st.text_area(f"Assistant-{index}", value=message["content"], height=50, disabled=True)

    # Chat input
    prompt = st.text_input("Say something")

    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            # Perform a semantic search
            docs = db.similarity_search_with_score(query=prompt, k=5)

            # Prepare the context for the Llama3 model
            context = "\n".join([doc.page_content for doc, score in docs])

            # Define prompt
            prompt = f"You are a recommendation system for giving remedies for plant diseases. Using the following context, write about {prompt}:\n{context}"
            
            # Generate response using Ollama's chat method
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])

            # Build full response
            full_response = response['message']['content']

        # Display assistant's response with a unique key
                # Display assistant's response with a unique key
        st.text_area(f"Assistant-{len(st.session_state.messages)}", value=full_response, height=50, disabled=True)
        
        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    plant_disease_classifier_app()
