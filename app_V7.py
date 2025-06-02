# --- Imports ---
import os
import base64
import time
from datetime import datetime
import streamlit as st
import openai
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# --- API Keys from secrets ---
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")
app_password = st.secrets.get("APP_PASSWORD")

# --- Configure Gemini + OpenAI ---
genai.configure(api_key=gemini_api_key)
openai.api_key = openai_api_key

# --- Password gate ---
st.title("🔐 Secure PDF Topic App")
password = st.text_input("Enter app password:", type="password")
if password != app_password:
    st.warning("❌ Incorrect password.")
    st.stop()

# --- UI Title ---
st.title("📘 PDF Topic Extractor & Lecture Generator with Visuals + Feedback")

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file:
    text = uploaded_file.read().decode(errors="ignore")

    # --- Topic Input ---
    topic = st.text_input("Enter topic to extract:")
    if topic:
        # --- Gemini Text Extraction ---
        with st.spinner("🔍 Extracting relevant topic text using Gemini..."):
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"Extract detailed content for the topic '{topic}' from the given text:\n\n{text}"
            response = model.generate_content(prompt)
            extracted_text = response.text.strip()

        st.subheader("📄 Extracted Text")
        st.text_area("Extracted text:", extracted_text, height=200)

        # --- Feedback 1: Text Extraction ---
        st.subheader("✅ Feedback on Extracted Text")
        relevant = st.radio("Is the text relevant?", ["Yes", "No"])
        comment = st.text_area("Comment:")
        if st.button("Continue to Generate Lecture"):
            st.session_state["continue_gen"] = True
            st.session_state["feedback_text"] = {"relevant": relevant, "comment": comment}

if st.session_state.get("continue_gen"):

    # --- GPT-4o Lecture Generation ---
    st.subheader("🧠 Generating Lecture with GPT-4o")
    with st.spinner("Generating lecture text..."):
        progress_bar = st.progress(0)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=5000)
        template = """
        Use this extracted text to generate a structured lecture on the topic '{topic}'.
        Add visual cues only when they improve understanding. Format cues as:
        [Draw: simple, colorful diagram for ...]
        Text:
        {extracted_text}
        """
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["topic", "extracted_text"], template=template)
        )
        lecture_text = chain.run({"topic": topic, "extracted_text": extracted_text})
        progress_bar.progress(100)

    # --- Image Cue Extraction + Generation ---
    st.subheader("🖼 Generating Visuals")
    cue_to_image = {}
    image_prompts = [line for line in lecture_text.splitlines() if line.strip().startswith("[Draw:")]
    progress_bar = st.progress(0)
    for i, cue in enumerate(image_prompts):
        try:
            image_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": cue}],
                tools=[{"type": "image_generation"}],
                tool_choice={"type": "image_generation"},
            )
            blocks = getattr(image_response, "output", [])
            for block in blocks:
                if getattr(block, "type", None) == "image_generation_call":
                    image_data = base64.b64decode(block.result)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"image_{topic}_{i+1}_{timestamp}.png"
                    with open(filename, "wb") as f:
                        f.write(image_data)
                    cue_to_image[cue] = filename
        except Exception as e:
            st.warning(f"Image failed for {cue}: {e}")
        progress_bar.progress(int((i+1) / len(image_prompts) * 100))

    # --- Render Lecture with Images ---
    st.subheader("🧾 Final Lecture with Visuals")
    for line in lecture_text.splitlines():
        if line.strip().startswith("[Draw:"):
            img_path = cue_to_image.get(line.strip())
            if img_path:
                st.image(img_path, caption=line, use_container_width=True, width=350)
        else:
            st.markdown(line)

    # --- Feedback 2: Full Generation ---
    st.subheader("✅ Feedback on Generated Lecture")
    f2_relevant = st.radio("Was the teaching material relevant?", ["Yes", "No"])
    f2_acc = st.radio("Was the content accurate?", ["Yes", "No"])
    f2_img_rel = st.radio("Were images relevant?", ["Yes", "No"])
    f2_img_acc = st.radio("Were images accurate?", ["Yes", "No"])
    f2_comment = st.text_area("Final feedback or anything unnecessary?")
    if st.button("Continue to Submit Feedback"):
        st.session_state["feedback_gen"] = {
            "relevant": f2_relevant,
            "accuracy": f2_acc,
            "images_relevant": f2_img_rel,
            "images_accurate": f2_img_acc,
            "comment": f2_comment
        }

# --- Final Log ---
if st.session_state.get("feedback_gen"):
    st.success("🎉 Thanks for your feedback! You may now enter a new topic or end the session.")

    log_path = f"feedback/{topic}_fdb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("feedback", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("Feedback on Extracted Text:\n")
        f.write(str(st.session_state["feedback_text"]) + "\n\n")
        f.write("Feedback on Generated Lecture:\n")
        f.write(str(st.session_state["feedback_gen"]) + "\n")
    with open(log_path, "rb") as f:
        st.download_button("📥 Download Feedback Log", f, file_name=os.path.basename(log_path))

