import streamlit as st
import html
import os
import datetime
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(
    page_title="AI Chatbot - UrbanAI Navigator",
    layout="wide",
    page_icon="🤖"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; }
    .stButton > button {
        background-color: #1F4E79; color: white;
        border-radius: 8px; border: none;
        font-weight: bold;
    }
    .stButton > button:hover { background-color: #2E86AB; }
    .chat-header {
        background: linear-gradient(135deg, #1F4E79, #2E86AB);
        padding: 16px 20px; border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(31,78,121,0.3);
    }
    .info-card {
        background: white; padding: 14px;
        border-radius: 10px; margin: 6px 0;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        font-size: 13px;
    }
    .suggestion-chip {
        display: inline-block;
        background: #EBF5FB; color: #1F4E79;
        border-radius: 16px; padding: 5px 12px;
        font-size: 12px; font-weight: bold;
        margin: 3px; cursor: pointer;
        border: 1px solid #AED6F1;
    }
    .stat-box {
        background: white; padding: 10px 14px;
        border-radius: 8px; text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border-top: 3px solid #1F4E79;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── System prompt for the chatbot ────────────────────────────
SYSTEM_PROMPT = """
You are UrbanAI Navigator, a helpful smart-city assistant for Indian cities.

You can explain traffic, public transport, AQI, weather, routes, events, and general smart-city topics.
Important: this chatbot does NOT have live traffic, AQI, weather, map, or transport data unless that data is explicitly provided in the conversation. Never claim that you checked live/current data. If the user asks for live information, clearly say that this chat cannot verify it and give a useful general answer instead.

Response style:
- Start with a one-line **Short answer:**.
- Then use 2-4 short bullets under a useful heading when explanation is needed.
- Finish with one **💡 Tip:** when useful.
- Keep normal answers under about 120 words.
- Use simple English and Indian city context.
- Use Markdown headings, bold text, bullets, and short paragraphs.
- Do not repeat the user's question.
- Do not invent statistics, exact traffic levels, routes, timings, or live conditions.
- If the question is simple, answer simply rather than forcing many sections.
"""

# ── Groq API call ─────────────────────────────────────────────

def format_chat_response(text):
    """Normalize model output for compact, readable Streamlit chat."""
    if not text:
        return "**Short answer:** I couldn't generate a response. Please try again."
    text = str(text).strip().replace("\r\n", "\n")
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned, blank = [], False
    for line in lines:
        if not line.strip():
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(line)
            blank = False
    return "\n".join(cleaned).strip()

def ask_groq(messages):
    try:
        from groq import Groq
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role":"system","content":SYSTEM_PROMPT}] + messages,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content, True
    except ImportError:
        return "Groq library not installed. Run: pip install groq", False
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return "Invalid API key. Check your GROQ_API_KEY in .env file.", False
        elif "429" in err:
            return "Rate limit reached. Please wait a moment and try again.", False
        else:
            return f"Error: {err}", False

# ── Suggested questions ───────────────────────────────────────
SUGGESTIONS = {
    "Traffic": [
        "Why is MG Road usually busy during morning peak hours?",
        "What is the best time to travel from Hebbal to Electronic City?",
        "How does rain affect traffic in Bangalore?",
        "Which types of roads are usually busiest on Friday evenings?",
    ],
    "Transport": [
        "How can I reduce crowding when using BMTC at 9 AM?",
        "When is Bangalore Metro usually less crowded on weekends?",
        "Best time to take bus from Silk Board to Majestic?",
        "How crowded is Andheri metro at peak hours?",
    ],
    "Health & AQI": [
        "How does poor AQI affect jogging and outdoor activities?",
        "Why do some Delhi areas have worse AQI than others?",
        "Should elderly people go outside when AQI is 150?",
        "How does vehicle traffic affect air quality?",
    ],
    "General": [
        "What are peak traffic hours in Mumbai?",
        "How to avoid traffic during IPL match days?",
        "Best route from Koramangala to Whitefield?",
        "What is Urban Health Score?",
    ],
}

# ── Initialize session state ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages    = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.datetime.now()

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h2 style="color:white;margin:0;font-size:20px">
        🤖 UrbanAI City Chatbot
    </h2>
    <p style="color:#AED6F1;margin:4px 0 0;font-size:13px">
        Powered by Groq Llama 3.3 70B &nbsp;|&nbsp;
        Ask anything about Indian city conditions
    </p>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ────────────────────────────────────────────────────
chat_col, info_col = st.columns([2.2, 1])

# ════════════════════════════════════════
# LEFT — CHAT WINDOW
# ════════════════════════════════════════
with chat_col:

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:30px 20px;
                        background:white;border-radius:12px;
                        box-shadow:0 1px 6px rgba(0,0,0,0.06)">
                <div style="font-size:40px">🏙️</div>
                <h3 style="color:#1F4E79;margin:8px 0">
                    Welcome to UrbanAI City Chatbot
                </h3>
                <p style="color:#888;font-size:13px">
                    Ask me anything about traffic, transport,<br>
                    air quality, or city conditions across Indian cities.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"],
                    avatar="🧑" if msg["role"]=="user" else "🤖"):
                    st.markdown(msg["content"])
                    if msg.get("time"):
                        st.caption(msg["time"])

    st.markdown("---")

    # ── SUGGESTED QUESTIONS ───────────────────────────────────
    st.markdown("**Quick Questions — Click to Ask:**")
    tab_t, tab_tr, tab_h, tab_g = st.tabs([
        "🚦 Traffic", "🚌 Transport",
        "🌫️ Health & AQI", "💡 General"
    ])

    selected_question = None

    with tab_t:
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTIONS["Traffic"]):
            if cols[i%2].button(q, key=f"t_{i}", use_container_width=True):
                selected_question = q

    with tab_tr:
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTIONS["Transport"]):
            if cols[i%2].button(q, key=f"tr_{i}", use_container_width=True):
                selected_question = q

    with tab_h:
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTIONS["Health & AQI"]):
            if cols[i%2].button(q, key=f"h_{i}", use_container_width=True):
                selected_question = q

    with tab_g:
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTIONS["General"]):
            if cols[i%2].button(q, key=f"g_{i}", use_container_width=True):
                selected_question = q

    st.markdown("---")

    # ── CHAT INPUT ────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about traffic, transport, AQI, routes..."
    )

    # Handle suggested question click OR typed input
    question = selected_question or user_input

    if question:
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found in .env file. Add it and restart the app.")
        else:
            # Add user message
            now_str = datetime.datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                "role":    "user",
                "content": question,
                "time":    now_str,
            })
            st.session_state.total_queries += 1

            # Get response
            with st.spinner("UrbanAI is thinking..."):
                # Send only recent turns to keep prompts compact and responsive.
                recent_messages = st.session_state.messages[-12:]
                api_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in recent_messages
                ]
                answer, success = ask_groq(api_messages)
                answer = format_chat_response(answer)

            # Add bot response
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "time":    datetime.datetime.now().strftime("%H:%M"),
            })

            st.rerun()

    # ── CLEAR CHAT ────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.total_queries = 0
            st.rerun()

# ════════════════════════════════════════
# RIGHT — INFO PANEL
# ════════════════════════════════════════
with info_col:

    # Session stats
    st.markdown("**Session Stats**")
    elapsed = datetime.datetime.now() - st.session_state.session_start
    mins    = int(elapsed.total_seconds() // 60)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size:22px;font-weight:bold;color:#1F4E79">
                {st.session_state.total_queries}
            </div>
            <div style="font-size:10px;color:#888">Queries</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size:22px;font-weight:bold;color:#1F4E79">
                {mins}m
            </div>
            <div style="font-size:10px;color:#888">Session Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Model info
    st.markdown("**Model Information**")
    for label, value in [
        ("Model",     "Groq-powered LLM"),
        ("Provider",  "Groq API"),
        ("Response",  "Concise + structured"),
        ("Context",   "Recent chat history"),
        ("Language",  "English + Hindi"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    font-size:12px;margin-bottom:5px;
                    padding:4px 0;border-bottom:1px solid #f0f0f0">
            <span style="color:#888">{label}</span>
            <span style="font-weight:bold;color:#1F4E79">{value}</span>
        </div>""", unsafe_allow_html=True)

    # API status
    st.markdown("---")
    if GROQ_API_KEY:
        st.success("Groq API Connected")
    else:
        st.error("API Key Missing\nAdd GROQ_API_KEY to .env")
        st.code("GROQ_API_KEY=your_key_here", language="bash")
        st.markdown("[Get free key →](https://console.groq.com)",
                    unsafe_allow_html=False)

    st.markdown("---")

    # What the chatbot can answer
    st.markdown("**What I Can Help With**")
    topics = [
        ("🚦", "Traffic Congestion",    "Peak hours, best routes"),
        ("🚌", "Transport Crowd",       "Bus/metro crowd levels"),
        ("🌫️", "Air Quality / AQI",    "Health & pollution advice"),
        ("🎪", "Event Impact",          "Match/concert traffic"),
        ("🗺️", "Route Planning",       "Best path suggestions"),
        ("🌦️", "Weather Impact",       "Rain effect on city"),
    ]
    for icon, title, desc in topics:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;
                    gap:8px;margin-bottom:8px;padding:6px;
                    background:white;border-radius:6px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.05)">
            <span style="font-size:16px">{icon}</span>
            <div>
                <div style="font-size:11px;font-weight:bold;
                            color:#1F4E79">{title}</div>
                <div style="font-size:10px;color:#888">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Chat history summary
    if st.session_state.messages:
        st.markdown("**Recent Questions**")
        user_msgs = [m for m in st.session_state.messages
                     if m["role"] == "user"][-3:]
        for m in reversed(user_msgs):
            safe_q = html.escape(str(m["content"])[:55])
            suffix = "..." if len(str(m["content"])) > 55 else ""
            st.markdown(f"""
            <div style="font-size:11px;color:#555;padding:5px 8px;
                        background:#F8F9FA;border-radius:6px;
                        margin-bottom:4px;border-left:3px solid #2E86AB">
                {safe_q}{suffix}
            </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#888'>UrbanAI Navigator | AI City Chatbot | "
    "Powered by Groq API | "
    "VTU Final Year Project 2025-26 | GEC Karwar</small></center>",
    unsafe_allow_html=True
)
