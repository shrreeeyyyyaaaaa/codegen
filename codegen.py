import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os

# For environment variables
from dotenv import load_dotenv
load_dotenv()

# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-4")

# --- TOOL DEFINITIONS ---

def generate_code_from_prompt(prompt: str, language: str = "Python") -> str:
    return llm.predict(f"Write a complete, valid {language} function for: {prompt}")

def explain_code(code: str, language: str = "Python") -> str:
    return llm.predict(f"Explain what this {language} code does:\n{code}")

def generate_docstring(code: str, language: str = "Python") -> str:
    return llm.predict(f"Add a {language} docstring to this function:\n{code}")

def generate_tests(code: str, language: str = "Python") -> str:
    return llm.predict(f"Write 2-3 unit tests for this {language} function:\n{code}")

def optimize_code(code: str, language: str = "Python") -> str:
    return llm.predict(f"Suggest any improvements or optimizations for this {language} code:\n{code}")

def run_code(code: str) -> str:
    try:
        compiled = compile(code, "<string>", "exec")  # Validate syntax first
        local_env = {}
        exec(compiled, {}, local_env)
        return "✅ Code executed successfully."
    except SyntaxError as e:
        return f"❌ SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return f"❌ Runtime Error: {str(e)}"

# LangChain Tool Wrappers
languages_supported = ["Python", "Java", "C", "C++", "Rust", "Go", "R"]

tools = [
    Tool(name="GenerateCodeTool", func=lambda prompt: generate_code_from_prompt(prompt), description="Generate code from description."),
    Tool(name="ExplainCodeTool", func=lambda code: explain_code(code), description="Explain code."),
    Tool(name="GenerateDocstring", func=lambda code: generate_docstring(code), description="Add a docstring to a function."),
    Tool(name="AddTestsTool", func=lambda code: generate_tests(code), description="Generate unit tests."),
    Tool(name="OptimizeCodeTool", func=lambda code: optimize_code(code), description="Optimize code for performance."),
    Tool(name="RunCodeTool", func=run_code, description="Run the given Python code."),  # Still limited to Python
]

# Agent Initialization
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Code Companion", layout="wide")
st.title("🤖 Code Companion with Tool Calling")

prompt = st.text_input("💬 Describe the code you need:")
language = st.selectbox("📌 Choose Programming Language:", languages_supported)

if st.button("Generate and Analyze"):
    if not prompt:
        st.warning("Please enter a code prompt.")
    else:
        with st.spinner("🔧 Generating code and tools..."):
            code = generate_code_from_prompt(prompt, language)
            explanation = explain_code(code, language)
            docstring_code = generate_docstring(code, language)
            tests = generate_tests(code, language)
            optimization = optimize_code(code, language)
            run_result = run_code(code) if language == "Python" else "⚠️ Execution supported only for Python."

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
st.markdown("Built with OpenAI and LangChain 🔧")
st.markdown("Shreya Alajangi 😼")    
