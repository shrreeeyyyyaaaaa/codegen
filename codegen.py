import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.tools.python.tool import PythonREPLTool
import os

# For environment variables
from dotenv import load_dotenv
load_dotenv()

# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-4")

# --- TOOL DEFINITIONS ---

def generate_code_from_prompt(prompt: str) -> str:
    return llm.predict(f"Write a Python function for: {prompt}")

def explain_code(code: str) -> str:
    return llm.predict(f"Explain what this code does:\n{code}")

def generate_docstring(code: str) -> str:
    return llm.predict(f"Add a Python docstring to this function:\n{code}")

def generate_tests(code: str) -> str:
    return llm.predict(f"Write 2-3 unit tests for this Python function:\n{code}")

def optimize_code(code: str) -> str:
    return llm.predict(f"Suggest any improvements or optimizations for this code:\n{code}")

def run_code(code: str) -> str:
    try:
        local_env = {}
        exec(code, {}, local_env)
        return "Code executed successfully."
    except Exception as e:
        return f"Error during execution: {str(e)}"

# LangChain Tool Wrappers
tools = [
    Tool(name="GenerateCodeTool", func=generate_code_from_prompt, description="Generate Python code from description."),
    Tool(name="ExplainCodeTool", func=explain_code, description="Explain Python code."),
    Tool(name="GenerateDocstring", func=generate_docstring, description="Add a docstring to a function."),
    Tool(name="AddTestsTool", func=generate_tests, description="Generate unit tests."),
    Tool(name="OptimizeCodeTool", func=optimize_code, description="Optimize code for performance."),
    Tool(name="RunCodeTool", func=run_code, description="Run the given Python code."),
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

if st.button("Generate and Analyze"):
    if not prompt:
        st.warning("Please enter a code prompt.")
    else:
        with st.spinner("🔧 Generating code and tools..."):
            # Call each tool manually (or you can route via agent.run(prompt))
            code = generate_code_from_prompt(prompt)
            explanation = explain_code(code)
            docstring_code = generate_docstring(code)
            tests = generate_tests(code)
            optimization = optimize_code(code)
            run_result = run_code(code)

        st.subheader("🧾 Generated Code")
        st.code(code, language="python")

        with st.expander("📘 Explanation"):
            st.write(explanation)

        with st.expander("📄 With Docstring"):
            st.code(docstring_code, language="python")

        with st.expander("🧪 Unit Tests"):
            st.code(tests, language="python")

        with st.expander("🚀 Optimization Suggestions"):
            st.write(optimization)

        with st.expander("🖥️ Execution Result"):
            st.write(run_result)

st.markdown("---")
st.markdown("Built OpenAI & Langchain 🔧")
