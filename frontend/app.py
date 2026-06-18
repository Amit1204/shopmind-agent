"""
ShopMind Streamlit Frontend
Run: streamlit run frontend/app.py
Connects to FastAPI backend at http://localhost:8000
"""
import streamlit as st
import requests
from PIL import Image
import io

API_BASE = "http://localhost:8000/api"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopMind Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 ShopMind")
    st.caption("Multimodal AI Shopping Assistant")
    st.divider()

    user_id = st.text_input("User ID (for personalization)", value="demo_user")
    st.divider()

    st.subheader("📊 Session Stats")
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    st.metric("Messages sent", st.session_state.message_count)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🛒 ShopMind — AI Shopping Agent")
st.caption("Ask anything about products. Upload an image to find similar items.")

# Init conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("metadata"):
            meta = msg["metadata"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{meta.get('confidence', 0):.0%}")
            col2.metric("Tools used", len(meta.get('tools_used', [])))
            col3.metric("Safety", "✅" if meta.get('safety_passed') else "⚠️")
            if meta.get("tools_used"):
                st.caption(f"Tools: {', '.join(meta['tools_used'])}")

st.divider()

# ── Input area ────────────────────────────────────────────────────────────────
col_text, col_image = st.columns([3, 1])

with col_text:
    user_input = st.chat_input("Ask about products, compare prices, find by image...")

with col_image:
    uploaded_image = st.file_uploader(
        "📷 Upload image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_image:
        st.image(Image.open(uploaded_image), width=80)

# ── Handle user input ─────────────────────────────────────────────────────────
if user_input or uploaded_image:
    query = user_input or "Find products similar to this image"

    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.message_count += 1

    with st.chat_message("user"):
        st.write(query)
        if uploaded_image:
            st.image(Image.open(uploaded_image), width=120)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("ShopMind is thinking..."):
            try:
                if uploaded_image:
                    # Visual search endpoint
                    files = {"image": (uploaded_image.name, uploaded_image.getvalue())}
                    response = requests.post(f"{API_BASE}/visual-search", files=files, timeout=60)
                else:
                    # Chat endpoint
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    payload = {
                        "message": query,
                        "user_id": user_id,
                        "conversation_history": history
                    }
                    response = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response")
                    confidence = data.get("confidence", 0)
                    tools_used = data.get("tools_used", [])
                    safety_passed = data.get("safety_passed", True)

                    st.write(answer)

                    # Metrics row
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence", f"{confidence:.0%}")
                    col2.metric("Tools used", len(tools_used))
                    col3.metric("Safety", "✅ Passed" if safety_passed else "⚠️ Warning")

                    if tools_used:
                        st.caption(f"🔧 Tools: {', '.join(tools_used)}")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "confidence": confidence,
                            "tools_used": tools_used,
                            "safety_passed": safety_passed
                        }
                    })
                else:
                    st.error(f"API error {response.status_code}: {response.text}")

            except requests.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure backend is running:\n```\nuvicorn api.app:app --reload\n```")
            except Exception as e:
                st.error(f"Error: {e}")
