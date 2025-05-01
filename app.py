import streamlit as st  
from langchain_google_genai import GoogleGenerativeAI  

# Title  
st.title("🤖 Medical Chatbot")  

# Input  
user_question = st.text_input("Ask your medical question:")  

# Process question  
if user_question:  
    # Initialize Gemini  
    llm = GoogleGenerativeAI(  
        model="gemini-2.0-flash",  
        google_api_key=st.secrets["GEMINI_API_KEY"]  # Will add secret later  
    )  
      
    # Get response  
    response = llm.invoke(f"Answer this medical question: {user_question}")  
      
    # Display  
    st.write("**Answer:**")  
    st.write(response)  