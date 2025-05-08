import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.3, model="gpt-4")

# --- TOOL DEFINITIONS ---

def generate_code(prompt: str) -> str:
    return llm.predict(f"Write a Python function for: {prompt}")

def explain_code(code: str) -> str:
    return llm.predict(f"Explain this Python code:\n{code}")

def generate_docstring(code: str) -> str:
    return llm.predict(f"Add a Python docstring to this function:\n{code}")

def generate_tests(code: str) -> str:
    return llm.predict(f"Write 2-3 unit tests for this function:\n{code}")

def optimize_code(code: str) -> str:
    return llm.predict(f"Suggest optimizations for this code:\n{code}")

def run_code(code: str) -> str:
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return f"✅ Executed successfully. Variables: {list(local_vars.keys())}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Define LangChain tools
tools = [
    Tool(name="GenerateCode", func=generate_code, description="Generate Python code from a description."),
    Tool(name="ExplainCode", func=explain_code, description="Explain a given Python code block."),
    Tool(name="GenerateDocstring", func=generate_docstring, description="Add docstring to a Python function."),
    Tool(name="GenerateTests", func=generate_tests, description="Generate unit tests for Python code."),
    Tool(name="OptimizeCode", func=optimize_code, description="Optimize and improve Python code."),
    Tool(name="RunCode", func=run_code, description="Execute Python code and return output.")
]

# Agent with function calling (tool-calling)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Code Agent", layout="centered")
st.title("🧠 AI Code Agent with Tool Calling")

prompt = st.text_area("💬 Describe the code or task you need:", height=150)

if st.button("🔍 Run Agent"):
    if not prompt.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("🧠 Agent thinking..."):
            try:
                result = agent.run(prompt)
                st.success("Done!")
                st.code(result, language="python")
            except Exception as e:
                st.error(f"Agent Error: {str(e)}")

st.markdown("---")
st.caption("Built with OpenAI + LangChain Agent Tool Calling 🔧")
st.markdown("Shreya Alajangi 😼")    
