import streamlit as st
import os
import hashlib
from io import BytesIO
from audio_recorder_streamlit import audio_recorder

from agents import chain
from input_loader import upload_file, upload_image, record_voice
from rag.ragapp import update_kb
from langgraph.types import Command


# PAGE CONFIG
st.set_page_config(
    page_title="AI Math Mentor",
    layout="wide"
)

st.title("🧠 AI Math Mentor")


# DIRECTORIES
DATA_DIR = "RAG/data/uploaded_data"
os.makedirs(DATA_DIR, exist_ok=True)


# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "kb_hashes" not in st.session_state:
    st.session_state.kb_hashes = {}

if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

if "kb_checked" not in st.session_state:
    st.session_state.kb_checked = False


thread_config = {
    "configurable": {"thread_id": "math-session"}
}



# HASH CHECK
def get_file_hashes(directory):

    hashes = {}

    for file in os.listdir(directory):

        if file.endswith(".pdf"):

            path = os.path.join(directory, file)

            with open(path, "rb") as f:
                content = f.read()

            hashes[file] = hashlib.md5(content).hexdigest()

    return hashes


# SIDEBAR
st.sidebar.title("📚 Knowledge Base")


# Uploaded History
st.sidebar.subheader("Uploaded Files")

pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

if pdfs:
    for p in pdfs:
        st.sidebar.write(f"📄 {p}")
else:
    st.sidebar.write("No uploads yet")



# Upload Files
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        save_path = os.path.join(DATA_DIR, file.name)

        with open(save_path, "wb") as f:
            f.write(file.read())

        st.session_state.upload_history.append(file.name)

    st.sidebar.success("Files uploaded")


# KB UPDATE BUTTON
if st.sidebar.button("Build / Update Knowledge Base"):

    with st.spinner("Building vector database..."):
        update_kb(DATA_DIR)

    st.session_state.kb_hashes = get_file_hashes(DATA_DIR)

    st.sidebar.success("Knowledge Base Updated")


# GRAPH VIEW
st.sidebar.subheader("Agent Workflow")

try:
    graph = chain.get_graph().draw_mermaid_png()
    st.sidebar.image(graph)
except:
    st.sidebar.write("Graph unavailable")



# ASK USER TO BUILD KB ON START
pdf_count = len(pdfs)

if pdf_count > 0 and not st.session_state.kb_checked:

    st.info(
        "📚 Documents detected in knowledge base.\n\n"
        "Click **Build Knowledge Base** to activate retrieval."
    )

    st.session_state.kb_checked = True


# DISPLAY CHAT HISTORY
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# INPUT SECTION (BOTTOM STYLE)
st.divider()

col1, col2 = st.columns([4,1])


with col2:

    mode = st.selectbox(
        "Input",
        ["Text", "Image", "PDF", "Voice"]
    )


user_input = None



# TEXT 
if mode == "Text":

    user_input = st.chat_input("Ask a math question...")


# IMAGE 
elif mode == "Image":

    image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if image:

        extracted = upload_image(image)

        user_input = st.text_area(
            "Edit extracted text",
            value=extracted
        )


# PDF 
elif mode == "PDF":

    pdf = st.file_uploader("Upload problem PDF", type=["pdf"])

    if pdf:

        extracted = upload_file(pdf)

        user_input = st.text_area(
            "Edit extracted text",
            value=extracted
        )


# VOICE 
elif mode == "Voice":

    st.write("🎤 Record voice")

    audio_bytes = audio_recorder()

    if audio_bytes:

        st.audio(audio_bytes)

        buffer = BytesIO(audio_bytes)

        extracted = record_voice(buffer)

        user_input = st.text_area(
            "Edit transcription",
            value=extracted
        )


# SOLVE PROBLEM
if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # AUTO UPDATE KB
    current_hashes = get_file_hashes(DATA_DIR)

    if current_hashes != st.session_state.kb_hashes:

        with st.spinner("Updating knowledge base..."):

            update_kb(DATA_DIR)

        st.session_state.kb_hashes = current_hashes


    # RUN AGENT
    with st.spinner("Solving..."):

        result = chain.invoke(
            {
                "data": user_input,
                "history": st.session_state.history
            },
            config=thread_config
        )


    # INTERRUPT HANDLING
    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value

        with st.chat_message("assistant"):

            st.warning(interrupt_data["message"])

            edited_solution = st.text_area(
                "Edit Solution",
                value=interrupt_data["solution"],
                height=200
            )

        if st.button("Submit Correction"):

            result = chain.invoke(
                Command(resume=edited_solution),
                config=thread_config
            )


    # DISPLAY RESULT
    with st.chat_message("assistant"):

        if "solution" in result:

            st.markdown("### ✅ Solution")

            try:
                st.latex(result["solution"])
            except:
                st.markdown(result["solution"])


        if "explanation" in result:

            st.markdown("### 📖 Explanation")
            st.markdown(result["explanation"])


        # SOURCES BUTTON
        if result.get("rag_sources"):

            with st.expander("📚 Sources"):

                for src in result["rag_sources"]:
                    st.write(f"- {src}")


    # SAVE RESPONSE
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("explanation", "")
        }
    )


    # MEMORY UPDATE
    if result.get("history"):
        st.session_state.history = result["history"]