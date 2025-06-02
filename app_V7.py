import streamlit as st
import os
import re
import base64
from datetime import datetime
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
#from google import genai
#from google.genai import types
import google.generativeai as genai
from google.generativeai import types

import openai

# --- Load API Keys ---
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not gemini_api_key or not openai_api_key:
    st.error("Missing API keys. Set GEMINI_API_KEY and OPENAI_API_KEY in your .env file.")
    st.stop()

# --- Initialize clients ---
gemini_client = genai.Client(api_key=gemini_api_key)
openai_client = openai.Client(api_key=openai_api_key)

# --- App Title ---
st.title("📘 PDF Topic Extractor & Lecture Generator with Visuals + Feedback")

# --- Memory ---
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="input")

# --- Upload PDF ---
uploaded_file = st.file_uploader("📎 Upload your PDF", type="pdf")

# --- Cache PDF bytes for reuse ---
if uploaded_file:
    if 'pdf_bytes' not in st.session_state or st.session_state.get('last_file_id') != uploaded_file.name:
        st.session_state.pdf_bytes = uploaded_file.read()
        st.session_state.last_file_id = uploaded_file.name
        st.session_state.memory.clear()

# --- Topic Input ---
topic = st.text_input("🎯 Enter the topic to extract and teach:")

# --- Gemini Extraction ---
if uploaded_file and topic:
    pdf_bytes = st.session_state.pdf_bytes

    prompt_text_gemini = (
        f"From the provided document, extract detailed and meaningful information strictly related to the topic: '{topic}'. "
        "Do not add any content beyond what is present in the document. Do not generate new explanations or insert external knowledge. "
        "Reorganize, clean up, and structure the extracted content to remove redundancy and improve clarity. "
        "Preserve important definitions, formulas, or labeled steps as they appear. Avoid copying repeated or irrelevant information."
    )

    contents = [
        types.Content(role="user", parts=[
            types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
            types.Part.from_text(text=prompt_text_gemini)
        ]),
    ]

    config = types.GenerateContentConfig(temperature=0.2, response_mime_type="text/plain")
    st.header("🔍 Extracting Topic from PDF...")
    extracted_text = ""

    try:
        extracted_text_chunks = []
        for chunk in gemini_client.models.generate_content_stream(model="gemini-2.0-flash", contents=contents, config=config):
            extracted_text_chunks.append(chunk.text)
        extracted_text = "".join(extracted_text_chunks)
        st.subheader(f"📘 Extracted Text for '{topic}'")
        st.markdown(extracted_text if extracted_text.strip() else "*No topic-specific info found*")
    except Exception as e:
        st.error(f"Gemini extraction failed: {e}")
        extracted_text = ""

    # --- Feedback on Extracted Text ---
    st.subheader("📝 Feedback on Extracted Text")
    relevance_text = st.radio("Is the extracted text relevant to the topic?", ["Yes", "No"], key="relevance_text")
    comment_text = st.text_area("Any comments on the extracted text?", key="comment_text")

    # 👉 Continue button to proceed to Lecture Generation
    continue_1 = st.button("Continue to Lecture Generation")
    if not continue_1:
        st.stop()

    # --- Lecture Generation ---
    if extracted_text.strip():
        st.header("📚 Generating Lecture with Visuals")
        lecture_duration = "2-hour" if len(extracted_text.split()) > 1000 else "1-hour"

        lecture_prompt = PromptTemplate(
            input_variables=["extracted_text", "lecture_duration", "topic"],
            template=(
                "Based on the following extracted text, create a detailed {lecture_duration} lecture on the topic '{topic}'. "
                "Only include visual cues where helpful, formatted like this:\n[Draw: simple, colorful diagram for ...]\n"
                "Do not add external knowledge.\n\nExtracted Text:\n{extracted_text}"
            )
        )

        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.7, max_tokens=5000)

        with st.spinner("🧠 Generating lecture content with GPT-4o..."):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            lecture_chain = LLMChain(llm=llm, prompt=lecture_prompt, memory=st.session_state.memory, output_key="lecture_material")
            result = lecture_chain({
                "input": topic,
                "extracted_text": extracted_text,
                "lecture_duration": lecture_duration,
                "topic": topic
            })
            lecture_material = result["lecture_material"]
            progress_text.markdown("✅ Lecture content generated.")
            progress_bar.progress(100)

        # --- Generate Images ---
        image_prompts = re.findall(r'\[Draw:(.*?)\]', lecture_material, flags=re.DOTALL)
        cue_to_image = {}
        sanitized_topic = re.sub(r'\W+', '_', topic.lower())
        os.makedirs("generated_images", exist_ok=True)

        if image_prompts:
            progress_text = st.empty()
            progress_bar = st.progress(0)

        for i, cue in enumerate(image_prompts):
            cue = cue.strip()
            progress_text.markdown(f"🖼 Generating image {i+1} of {len(image_prompts)}...")
            try:
                response = openai_client.responses.create(
                    model="gpt-4o",
                    input=cue,
                    tools=[{"type": "image_generation"}],
                    tool_choice={"type": "image_generation"},
                )
                blocks = getattr(response, "output", [])
                for block in blocks:
                    if getattr(block, "type", "") == "image_generation_call":
                        image_data = base64.b64decode(block.result)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"image_{sanitized_topic}_{i+1}_{timestamp}.png"
                        filepath = os.path.join("generated_images", filename)
                        with open(filepath, "wb") as f:
                            f.write(image_data)
                        cue_to_image[cue] = filepath
                        break
            except Exception as e:
                cue_to_image[cue] = ""
            if image_prompts:
                progress_bar.progress(int((i+1)/len(image_prompts)*100))

        if image_prompts:
            progress_text.markdown("✅ All images processed.")

        # --- Render Lecture ---
        def render_lecture_blocks(lecture_text, cue_to_image):
            parts = re.split(r'\[Draw:(.*?)\]', lecture_text, flags=re.DOTALL)
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part.strip():
                        math_blocks = re.findall(r'\$\$(.*?)\$\$', part, re.DOTALL)
                        if math_blocks:
                            for math_expr in math_blocks:
                                st.latex(math_expr.strip())
                        else:
                            st.markdown(part.strip())
                else:
                    cue = part.strip()
                    img_path = cue_to_image.get(cue, "")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=cue, width=500)
                    else:
                        st.warning(f"Image missing for: {cue}")

        st.subheader(f"📖 {lecture_duration} Lecture for '{topic}'")
        render_lecture_blocks(lecture_material, cue_to_image)

        # --- Feedback on Lecture & Images ---
        st.subheader("📝 Feedback on Lecture and Visuals")
        relevance_lecture = st.radio("Is the teaching material relevant?", ["Yes", "No"], key="relevance_lecture")
        accuracy_lecture = st.radio("Is the lecture accurate?", ["Yes", "No"], key="accuracy_lecture")
        relevance_images = st.radio("Are the images relevant?", ["Yes", "No"], key="relevance_images")
        accuracy_images = st.radio("Are the images accurate?", ["Yes", "No"], key="accuracy_images")
        comment_overall = st.text_area("Any other comments?", key="comment_overall")

        # 👉 Continue before submitting feedback
        continue_2 = st.button("Continue to Submit Feedback")
        if not continue_2:
            st.stop()

        # ✅ Final message
        st.success("🎉 Thanks for your feedback! You may now enter a new topic for generation or end the session.")

        # --- Save Feedback ---
        if st.button("Submit Feedback"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback/{sanitized_topic}_fdb_{timestamp}.txt"
            os.makedirs("feedback", exist_ok=True)
            with open(filename, "w") as fdb:
                fdb.write(f"Topic: {topic}\n\n")
                fdb.write(f"Text Relevant: {relevance_text}\nComment: {comment_text}\n\n")
                fdb.write(f"Lecture Relevant: {relevance_lecture}\nLecture Accurate: {accuracy_lecture}\n\n")
                fdb.write(f"Images Relevant: {relevance_images}\nImages Accurate: {accuracy_images}\n\n")
                fdb.write(f"Other Comments: {comment_overall}\n")
            st.success(f"✅ Feedback saved to {filename}")
