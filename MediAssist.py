import streamlit as st
import google.generativeai as genai
import requests
import json
import os
from datetime import datetime

# Function to initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = False
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = st.secrets["GEMINI_API_KEY"]
    if "serper_api_key" not in st.session_state:
        st.session_state.serper_api_key = st.secrets["SERPER_API_KEY"]

# Function to configure Gemini API
def configure_gemini():
    genai.configure(api_key=st.session_state.gemini_api_key)
    
    # Set up the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return model

# Function to perform web search using Serper API
def search_web(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query + " medical information",  # Add medical context to search
        "num": 5  # Number of search results to return
    })
    headers = {
        'X-API-KEY': st.session_state.serper_api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Function to format search results for the model
def format_search_results(results):
    if "error" in results:
        return f"Error performing search: {results['error']}"
    
    formatted_results = "Medical Search Results:\n\n"
    
    if "organic" in results:
        for i, result in enumerate(results["organic"][:5], 1):
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "No description")
            formatted_results += f"{i}. {title}\n   URL: {link}\n   Description: {snippet}\n\n"
    
    if "answerBox" in results:
        answer_box = results["answerBox"]
        title = answer_box.get("title", "")
        answer = answer_box.get("answer", "")
        snippet = answer_box.get("snippet", "")
        formatted_results += f"Featured Medical Answer: {title}\n{answer}\n{snippet}\n\n"
    
    if "knowledgeGraph" in results:
        kg = results["knowledgeGraph"]
        title = kg.get("title", "")
        description = kg.get("description", "")
        formatted_results += f"Medical Knowledge: {title}\n{description}\n\n"
    
    return formatted_results

# Function to get response from Gemini
def get_gemini_response(model, prompt, with_search=False, query=None):
    try:
        # Medical context for the AI
        medical_context = """
        You are MediAssist, a helpful and compassionate medical AI assistant. Your purpose is to provide
        information about medical conditions, treatments, and general health advice. 
        
        Important guidelines:
        1. Always clarify that you're an AI and not a doctor
        2. Recommend consulting healthcare professionals for diagnosis and treatment
        3. Provide factual, evidence-based information
        4. Be empathetic and supportive in your tone
        5. Never make definitive diagnoses
        6. Emphasize the importance of seeking proper medical care
        7. Use clear, understandable language without excessive medical jargon
        """
        
        # If search mode is enabled and we have a query
        search_context = ""
        if with_search and query:
            search_results = search_web(query)
            search_context = format_search_results(search_results)
            
            # Prepare prompt with search results and medical context
            full_prompt = f"""
            {medical_context}
            
            The user query is: {prompt}
            
            Here are relevant medical search results from the web:
            {search_context}
            
            Please provide a helpful response based on these search results and your knowledge. 
            If the search results are relevant, incorporate that information. 
            Always cite the sources if you use information from the search results.
            Remember to follow the medical guidelines provided above.
            """
        else:
            # Standard chat mode without search but with medical context
            full_prompt = f"""
            {medical_context}
            
            The user query is: {prompt}
            
            Please provide a helpful response based on your medical knowledge.
            Remember to follow the medical guidelines provided above.
            """
        
        # Generate response
        chat = model.start_chat(history=st.session_state.chat_history)
        response = chat.send_message(full_prompt)
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "parts": [prompt]})
        st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
        
        return response.text
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main app function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="MediAssist - Medical AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Sidebar for settings (without API keys)
    with st.sidebar:
        st.title("🏥 MediAssist Settings")
        
        # Search mode toggle
        st.subheader("Web Search")
        search_toggle = st.toggle("Enable Medical Search", value=st.session_state.search_mode)
        
        if search_toggle != st.session_state.search_mode:
            st.session_state.search_mode = search_toggle
            st.success("Medical search " + ("enabled" if search_toggle else "disabled"))
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        # Medical disclaimer
        st.markdown("---")
        st.caption("""
        **Medical Disclaimer**
        
        This AI assistant provides general information only and is not a substitute for professional medical advice, 
        diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any 
        questions you may have regarding a medical condition.
        """)
    
    # Main chat interface
    st.title("🏥 MediAssist - Your Medical AI Assistant")
    
    # Display search mode status
    search_status = "🔍 Medical Search: " + ("Enabled" if st.session_state.search_mode else "Disabled")
    st.caption(search_status)
    
    # Introduction message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm MediAssist, your medical AI assistant. I can help answer your health questions and provide general medical information. How may I assist you today? Remember, I'm not a doctor, and my responses shouldn't replace professional medical advice."
        })
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # User input
    user_prompt = st.chat_input("Ask your health question...")
    
    # Process user input
    if user_prompt:
        # Configure Gemini
        model = configure_gemini()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Researching medical information...")
            
            # Get response from Gemini (with or without search)
            full_response = get_gemini_response(
                model, 
                user_prompt,
                with_search=st.session_state.search_mode,
                query=user_prompt if st.session_state.search_mode else None
            )
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
