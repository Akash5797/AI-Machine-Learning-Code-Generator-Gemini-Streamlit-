import os
import logging
import streamlit as st
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import litellm

# --- 1. SILENCE TERMINAL OUTPUT ---
# We set logging to ERROR only, to keep your terminal clean
logging.basicConfig(level=logging.ERROR)
logging.getLogger("lite_llm").setLevel(logging.ERROR)
logging.getLogger("smolagents").setLevel(logging.ERROR)

# --- 2. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="ML Code Generator (Gemini)", page_icon="⚡", layout="wide")
load_dotenv()

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")
    
    # Check for GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Google Gemini API Key", type="password")
        st.caption("Get your key here: https://aistudio.google.com/app/apikey")
    
    # Model Selection (Gemini 1.5 Flash is efficient & cheap)
    model_choice = st.selectbox(
        "Select Model", 
        ["gemini/gemini-1.5-flash", "gemini/gemini-2.5-flash", "gemini/gemini-2.0-flash-exp"]
    )
    
    st.info("Using Google Gemini via LiteLLM.")

# --- 4. AGENT INITIALIZATION ---
def get_agent(api_key, model_id):
    if not api_key:
        return None
    
    # Initialize the model using LiteLLM wrapper for Gemini
    model = LiteLLMModel(
        model_id=model_id,
        api_key=api_key
    )

    # Initialize the agent with extra allowed imports to prevent crashes
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        additional_authorized_imports=[
            "pandas", "numpy", "sklearn", "matplotlib", "seaborn",
            "random", "math", "datetime", "re", "io"
        ],
        model=model
    )
    return agent

# --- 5. MAIN UI LOGIC ---
st.title("⚡ AI Machine Learning Engineer")
st.markdown("Ask me to create a Machine Learning pipeline using **Google Gemini**.")

# Initialize Session State to keep the output visible
if "generated_result" not in st.session_state:
    st.session_state.generated_result = None

# User Input
user_request = st.text_area("Describe your ML task:", placeholder="e.g., Train a Random Forest on the Iris dataset...", height=150)

# Buttons
col1, col2 = st.columns([1, 5])
with col1:
    generate_btn = st.button("Generate Code", type="primary")
with col2:
    if st.button("Clear Output"):
        st.session_state.generated_result = None
        st.rerun()

if generate_btn:
    if not user_request:
        st.warning("Please enter a task description.")
    elif not api_key:
        st.error("Gemini API Key is missing. Please add it in the sidebar or .env file.")
    else:
        # Prompt Engineering: Force the agent to RETURN code, not just print it
        system_instruction = (
            f"You are an expert Machine Learning engineer.\n"
            f"USER TASK: {user_request}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Generate a robust Python script to solve the task.\n"
            f"2. IMPORTANT: Do NOT just print the code internally. You MUST 'return' the full Python code as your final answer string.\n"
            f"3. Wrap the code inside Markdown code blocks (```python ... ```).\n"
            f"4. Use pandas, sklearn, matplotlib, seaborn where needed.\n"
        )

        with st.status("⚡ Agent is working (Gemini)...", expanded=True) as status:
            try:
                st.write("Initializing agent...")
                ml_code_agent = get_agent(api_key, model_choice)
                
                st.write("Generating solution...")
                
                # Run the agent
                response = ml_code_agent.run(system_instruction)
                
                # Convert response to string
                final_answer = str(response)

                st.write("Formatting response...")
                st.session_state.generated_result = final_answer
                
                status.update(label="Generation Complete!", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status.update(label="Error occurred", state="error")

# --- 6. OUTPUT WINDOW ---
if st.session_state.generated_result:
    st.markdown("### Generated Code")
    
    # Container to make it look like an editor window
    with st.container(border=True):
        # We clean the output slightly to ensure st.code displays it well
        content = st.session_state.generated_result
        
        # If the agent wrapped it in ```python ... ```, we extract it for cleaner display
        if "```python" in content:
            # Simple logic to strip markdown tags if needed, 
            # but st.markdown usually handles it well. 
            # We will use st.markdown for maximum compatibility.
            st.markdown(content)
        else:
            # If it's raw code, use st.code
            st.code(content, language='python')