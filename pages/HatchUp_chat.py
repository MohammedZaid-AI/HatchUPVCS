import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from mcp_use import MCPClient
from my_random import get_random_user_display

# Load .env first
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------
# CRITICAL: Inject Streamlit Secrets into os.environ
# This ensures that tools running as subprocesses (MCP servers)
# can access keys like REDDIT_CLIENT_ID from Streamlit Cloud Secrets.
# ---------------------------------------------------------
try:
    if hasattr(st, "secrets"):
        # 1. Flatten specific keys if they exist in general sections
        secret_keys = [
            "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "USER_AGENT",
            "GOOGLE_API_KEY", "SEARCH_ENGINE_ID", "GROQ_API_KEY"
        ]
        
        # Helper to find a key in secrets (handles [general] or top-level)
        def get_secret(key):
            if key in st.secrets:
                return str(st.secrets[key])
            # Check common sections
            for section in ["general", "env", "credentials"]:
                if section in st.secrets and key in st.secrets[section]:
                    return str(st.secrets[section][key])
            return None

        for key in secret_keys:
            val = get_secret(key)
            if val:
                os.environ[key] = val
                # print(f"Loaded {key} from secrets") # Debug only

except Exception as e:
    print(f"Error loading secrets: {e}")
# ---------------------------------------------------------

# Initialize Chat Model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3, # Slightly higher for conversational flow
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# Flexible Prompt for Chat mode
chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are HatchUp Chat, a smart VC research assistant.
        
        BEHAVIOR:
        1. If the user says "Hello" or engages in small talk, reply conversationally and politely. Do NOT generate a business report for greetings.
        2. If the user asks a specific question or topic (e.g., "AI Agents", "Market for EV batteries"), use the provided [Context] to generate a structured analysis:
           - 🎯 Key Insights
           - 📈 Market Signals
           - ⚠️ Risks
        3. IGNORE error messages in the context (e.g., "MCP Error", "Tool not found"). 
        4. Do NOT hallucinate acronyms or facts. If the context is empty or irrelevant, say so.
        
        Your goal is to be a helpful, chatty partner who offers deep research when asked.
        """
    ),
    (
        "human",
        """
        [Context from Live Tools]
        {context}

        [Conversation History]
        {history}

        [Current User Input]
        {question}
        """
    )
])

# Initialize MCP Client
# Initialize MCP Client with Dynamic Config
# We use sys.executable to ensure the sub-processes use the same python environment (with installed deps)
# We use absolute paths to ensure the scripts are found regardless of CWD
import sys

base_dir = Path(__file__).parent.parent.resolve() # Start from pages/, go up to root
reddit_script = base_dir / "mcp_reddit" / "server.py"
wiki_script = base_dir / "mcp_wiki" / "server.py"
google_script = base_dir / "mcp_google" / "server.py"
medium_script = base_dir / "mcp_medium" / "server.py"

server_config = {
    "mcpServers": {
        "@echolab/mcp-reddit": {
            "command": sys.executable,
            "args": [str(reddit_script)]
        },
        "@echolab/mcp-wikipedia": {
            "command": sys.executable,
            "args": [str(wiki_script)]
        },
        "@echolab/mcp-google": {
            "command": sys.executable,
            "args": [str(google_script)]
        },
        "@echolab/mcp-medium": {
            "command": sys.executable,
            "args": [str(medium_script)]
        }
    }
}

# Use the dynamic config directly
# Note: MCPClient might need an update if it doesn't support dict init directly, 
# but usually from_config_file just reads json. 
# Looking at standard implementations, we can likely pass the dict or write a temp file.
# Assuming MCPClient has a constructor that accepts the config dict or we monkeypatch.
# Let's check imports. from mcp_use import MCPClient.
# If MCPClient doesn't accept a dict, we'll write a temp config.
# Safest bet: Write a temp absoluted config file.

import json
temp_config_path = base_dir / "config_dynamic.json"
with open(temp_config_path, "w") as f:
    json.dump(server_config, f, indent=2)

client = MCPClient.from_config_file(str(temp_config_path))

async def run_searches(query: str):
    """
    Runs live searches using MCP tools. Returns a dictionary of results.
    """
    # Ensure session exists
    if "mcp_sessions" not in st.session_state:
        st.session_state.mcp_sessions = await client.create_all_sessions()

    sessions = st.session_state.mcp_sessions

    def fail(name, e):
        return f"[{name} MCP Error: {str(e)}]"

    # 1. Reddit (Community Sentiment) - Limit 1 to save tokens
    try:
        reddit = await sessions["@echolab/mcp-reddit"].call_tool(
            "fetch_reddit_posts_with_comments", 
            {"subreddit": "startups", "limit": 1} 
        )
    except Exception as e:
        reddit = fail("Reddit", e)

    # 2. Wikipedia (Definitions/Background)
    try:
        wiki = await sessions["@echolab/mcp-wikipedia"].call_tool(
            "search", {"query": query}
        )
    except Exception as e:
        wiki = fail("Wikipedia", e)

    # 3. Google (News & Competitors)
    try:
        google = await sessions["@echolab/mcp-google"].call_tool(
            "google_search", {"query": query}
        )
    except Exception as e:
        google = fail("Google", e)

    # 4. Medium (Thought Leadership)
    try:
        medium = await sessions["@echolab/mcp-medium"].call_tool(
            "search_medium", {"query": query}
        )
    except Exception as e:
        medium = fail("Medium", e)

    return {"reddit": reddit, "wiki": wiki, "google": google, "medium": medium}


def build_context_string(results: dict) -> str:
    """
    Formats the search results into a context string, truncating to avoid token overflow.
    """
    def truncate(content, limit=2000):
        s = str(content)
        return s[:limit] + "... [TRUNCATED]" if len(s) > limit else s

    return f"""
    --- SEARCH RESULTS ---
    [Reddit]: {truncate(results.get("reddit"))}
    [Wikipedia]: {truncate(results.get("wiki"))}
    [Google]: {truncate(results.get("google"))}
    [Medium]: {truncate(results.get("medium"))}
    ----------------------
    """

async def main():
    st.set_page_config(page_title="HatchUp Chat", page_icon="💬")
    st.title("💬 HatchUp Chat")
    
    # Initialize Chat History
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your research assistant. Ask me about a market, startup, or trend, and I'll find live data for you."}
        ]

    # Display History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask a question (e.g., 'Competitors to Airbnb')..."):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 2. Assistant Helper Logic
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔎 *Researching live sources...*")
            
            try:
                # A. Run Research (Always run to capture context if needed)
                # If query is short greeting, we might skip, but LLM handles it best.
                results = await run_searches(prompt)
                context_str = build_context_string(results)
                
                # B. Prepare Prompt
                history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_messages[-5:]]) # Last 5 turns
                
                messages = chat_prompt.format_messages(
                    context=context_str,
                    history=history_text,
                    question=prompt
                )
                
                # C. Generate Answer
                full_response = llm.invoke(messages).content
                
                # D. Display Final Answer
                message_placeholder.markdown(full_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"⚠️ An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                # Don't append error to history to avoid pollution, or append as system note?
                # We'll validly append it so user knows.
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    asyncio.run(main())
