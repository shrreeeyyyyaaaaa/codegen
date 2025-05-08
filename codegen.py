# Code Companion with Tool Calling (LangChain + Streamlit)

import streamlit as st
import os
import ast
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM setup
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

# Supported Languages
languages_supported = ["Python", "Java", "C", "C++", "Rust", "Go", "R"]

# --- TOOL FUNCTIONS ---

def generate_code_from_prompt(prompt: str, language: str = "Python") -> str:
    return llm.predict(f"Write a complete and executable {language} function for: {prompt}")

def explain_code(code: str, language: str = "Python") -> str:
    return llm.predict(f"Explain what this {language} code does:\n{code}")

def generate_docstring(code: str, language: str = "Python") -> str:
    return llm.predict(f"Add a proper {language} docstring to this function:\n{code}")

def generate_tests(code: str, language: str = "Python") -> str:
    return llm.predict(f"Write 2-3 unit tests for this {language} function:\n{code}")

def optimize_code(code: str, language: str = "Python") -> str:
    return llm.predict(f"Suggest improvements or optimizations for this {language} code:\n{code}")

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def run_code(code: str) -> str:
    if not is_valid_python(code):
        return "❌ Error: The generated code is not valid Python."
    try:
        local_env = {}
        exec(code, {}, local_env)
        return "✅ Code executed successfully."
    except Exception as e:
        return f"❌ Runtime Error: {str(e)}"

# --- LANGCHAIN TOOLS ---

tools = [
    Tool(name="GenerateCodeTool", func=lambda prompt: generate_code_from_prompt(prompt), description="Generate code from description."),
    Tool(name="ExplainCodeTool", func=lambda code: explain_code(code), description="Explain code."),
    Tool(name="GenerateDocstring", func=lambda code: generate_docstring(code), description="Add a docstring."),
    Tool(name="AddTestsTool", func=lambda code: generate_tests(code), description="Generate unit tests."),
    Tool(name="OptimizeCodeTool", func=lambda code: optimize_code(code), description="Optimize code."),
    Tool(name="RunCodeTool", func=run_code, description="Run Python code only."),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


st.set_page_config(page_title="AI Code Companion", layout="wide")
st.title("🤖 Code Companion with Tool Calling")

prompt = st.text_input("💬 Describe the code you need:")
language = st.selectbox("📌 Choose Programming Language:", languages_supported)

if st.button("Generate and Analyze"):
    if not prompt:
        st.warning("Please enter a code prompt.")
    else:
        with st.spinner("🔧 Generating code and analysis..."):
            code = generate_code_from_prompt(prompt, language)
            explanation = explain_code(code, language)
            docstring_code = generate_docstring(code, language)
            tests = generate_tests(code, language)
            optimization = optimize_code(code, language)
            run_result = run_code(code) if language == "Python" else f"⚠️ Execution not supported for {language}."

        st.subheader("🧾 Generated Code")
        st.code(code, language=language.lower())

        with st.expander("📘 Explanation"):
            st.write(explanation)

        with st.expander("📄 With Docstring"):
            st.code(docstring_code, language=language.lower())

        with st.expander("🧪 Unit Tests"):
            st.code(tests, language=language.lower())

        with st.expander("🚀 Optimization Suggestions"):
            st.write(optimization)

        with st.expander("🖥️ Execution Result"):
            st.write(run_result)

st.markdown("---")
st.markdown("Built with LangChain + OpenAI ")
st.markdown("Shreya Alajangi 😼")    
