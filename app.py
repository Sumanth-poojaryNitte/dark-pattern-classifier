import streamlit as st
import joblib
import easyocr
import re
import html


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dark Pattern Classifier",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

        /* Main title */
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #777777;
            font-size: 17px;
            margin-bottom: 30px;
        }

        /* Result box */
        .result-box {
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            text-align: center;
        }

        /* Deceptive result */
        .deceptive {
            background-color: #ffe5e5;
            border: 2px solid #ff4d4d;
        }

        /* Non-deceptive result */
        .not-deceptive {
            background-color: #e5ffe9;
            border: 2px solid #28a745;
        }

        /* Confidence */
        .confidence {
            font-size: 24px;
            font-weight: bold;
        }

        /* Confidence level */
        .level {
            font-size: 18px;
            font-weight: 600;
        }

        /* OCR notification box */
        .notification-box {
            background-color: #f5f5f5;
            border: 1px solid #cccccc;
            border-radius: 10px;
            padding: 15px;
            margin-top: 8px;
            margin-bottom: 15px;
            color: #222222;
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        /* Small OCR information */
        .ocr-info {
            color: #666666;
            font-size: 14px;
            margin-top: 5px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("dark_pattern_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


try:

    model, vectorizer = load_model()

except Exception as e:

    st.error("Could not load the model files.")

    st.write(
        "Make sure these files are present in the same folder as app.py:"
    )

    st.code(
        """
dark_pattern_model.pkl
tfidf_vectorizer.pkl
        """
    )

    st.stop()


# ============================================================
# LOAD OCR
# ============================================================

@st.cache_resource
def load_ocr():

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return reader


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text_from_image(uploaded_file):

    reader = load_ocr()

    # Read uploaded image bytes
    image_bytes = uploaded_file.getvalue()

    # OCR
    results = reader.readtext(
        image_bytes,
        detail=1,
        paragraph=False
    )

    detected_lines = []

    for result in results:

        if len(result) < 3:
            continue

        text = result[1]
        confidence = result[2]

        text = text.strip()

        if not text:
            continue

        detected_lines.append(
            {
                "text": text,
                "confidence": confidence
            }
        )

    return detected_lines


# ============================================================
# NOTIFICATION / MESSAGE FILTER
# ============================================================

def filter_notification_text(detected_lines):

    """
    Attempts to keep notification/message-like text
    and remove common UI elements such as:

    - Time
    - Navigation
    - Buttons
    - Prices
    - Locations
    - Generic website labels
    """

    if not detected_lines:
        return ""

    # Common UI words that are usually not useful
    ignored_exact = {
        "home",
        "help",
        "login",
        "sign in",
        "sign up",
        "register",
        "menu",
        "search",
        "settings",
        "profile",
        "account",
        "back",
        "next",
        "previous",
        "submit",
        "continue",
        "select",
        "cancel",
        "close",
        "open",
        "save",
        "edit",
        "delete",
        "share",
        "download",
        "upload",
        "flights",
        "hotels",
        "packages",
        "my trips",
        "from",
        "depart",
        "travellers",
        "sort",
        "recommended",
        "economy",
        "non-stop",
        "per traveller",
        "fare includes",
    }

    filtered = []

    for item in detected_lines:

        text = item["text"].strip()
        confidence = item["confidence"]

        lower_text = text.lower()

        # Ignore very low confidence OCR
        if confidence < 0.40:
            continue

        # Ignore exact UI words
        if lower_text in ignored_exact:
            continue

        # Ignore extremely short text
        if len(text) <= 2:
            continue

        # Ignore pure numbers
        if re.fullmatch(r"[\d\s.,:+₹$€£/-]+", text):
            continue

        # Ignore time formats
        if re.fullmatch(
            r"\d{1,2}[:.]\d{2}(\s?[ap]m)?",
            lower_text
        ):
            continue

        # Ignore dates
        if re.fullmatch(
            r"\d{1,2}\s?[a-zA-Z]{3,9}",
            text
        ):
            continue

        # Ignore common price-only text
        if re.fullmatch(
            r"[₹$€£]?\s?[\d,]+(\.\d+)?",
            text
        ):
            continue

        # Ignore simple flight codes
        if re.fullmatch(
            r"(flight\s*)?[A-Z]{1,3}\s?\d{2,5}",
            text,
            re.IGNORECASE
        ):
            continue

        # Ignore common navigation labels
        if lower_text in {
            "skyferry",
            "bengaluru",
            "goa",
            "blr",
            "goi",
            "indair",
            "gojet",
            "vistablue"
        }:
            continue

        filtered.append(text)

    return "\n".join(filtered)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔍 DARK PATTERN CLASSIFIER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered website text and screenshot analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT METHOD
# ============================================================

st.subheader("Choose Analysis Method")

input_method = st.radio(
    "Select an option:",
    [
        "📝 Enter Text",
        "🖼️ Upload Screenshot"
    ],
    horizontal=True
)


# ============================================================
# TEXT INPUT
# ============================================================

if input_method == "📝 Enter Text":

    st.subheader("Enter website text")

    user_text = st.text_area(
        "Enter the text you want to analyze:",
        height=160,
        placeholder=(
            "Example: Hurry! Only 2 items left. "
            "Order now before it's too late!"
        )
    )

    # --------------------------------------------------------
    # TEXT PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔎 PREDICT",
        use_container_width=True
    ):

        if not user_text.strip():

            st.warning(
                "Please enter some text before clicking Predict."
            )

        else:

            # Convert text into TF-IDF
            text_tfidf = vectorizer.transform(
                [user_text]
            )

            # Prediction
            prediction = model.predict(
                text_tfidf
            )[0]

            # Probability
            probabilities = model.predict_proba(
                text_tfidf
            )[0]

            # Confidence
            confidence = max(probabilities) * 100

            # Confidence level
            if confidence >= 80:

                confidence_level = "HIGH"

            elif confidence >= 60:

                confidence_level = "MEDIUM"

            else:

                confidence_level = "LOW"

            # Result
            if prediction == 1:

                result = "DECEPTIVE PATTERN"
                result_class = "deceptive"

            else:

                result = "NOT DECEPTIVE"
                result_class = "not-deceptive"

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="result-box {result_class}">

                    <h2>Prediction</h2>

                    <h3>{result}</h3>

                    <p class="confidence">
                        Confidence: {confidence:.2f}%
                    </p>

                    <p class="level">
                        Confidence Level: {confidence_level}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# IMAGE INPUT
# ============================================================

else:

    st.subheader("Upload Website Screenshot")

    uploaded_image = st.file_uploader(
        "Upload a screenshot containing a notification or message:",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ]
    )

    if uploaded_image is not None:

        # Show uploaded screenshot
        st.image(
            uploaded_image,
            caption="Uploaded Screenshot",
            use_container_width=True
        )

        st.markdown("---")

        # ----------------------------------------------------
        # OCR BUTTON
        # ----------------------------------------------------

        if st.button(
            "🔍 EXTRACT & ANALYZE",
            use_container_width=True
        ):

            with st.spinner(
                "Reading screenshot and detecting notification..."
            ):

                try:

                    # OCR
                    detected_lines = extract_text_from_image(
                        uploaded_image
                    )

                    if not detected_lines:

                        st.warning(
                            "No readable text was detected in the image."
                        )

                    else:

                        # Filter notification-like content
                        extracted_text = filter_notification_text(
                            detected_lines
                        )

                        # ------------------------------------------------
                        # OCR RESULT
                        # ------------------------------------------------

                        st.markdown(
                            "### 📩 Detected Notification / Message Text"
                        )

                        if extracted_text.strip():

                            # Escape HTML characters safely
                            safe_text = html.escape(
                                extracted_text
                            )

                            # Visible grey notification box
                            st.markdown(
                                f"""
                                <div class="notification-box">
                                    {safe_text}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                """
                                <div class="ocr-info">
                                    Only relevant notification/message-like
                                    text is sent to the AI classifier.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # ------------------------------------------------
                            # PREDICTION
                            # ------------------------------------------------

                            text_tfidf = vectorizer.transform(
                                [extracted_text]
                            )

                            prediction = model.predict(
                                text_tfidf
                            )[0]

                            probabilities = model.predict_proba(
                                text_tfidf
                            )[0]

                            confidence = (
                                max(probabilities) * 100
                            )

                            # Confidence level
                            if confidence >= 80:

                                confidence_level = "HIGH"

                            elif confidence >= 60:

                                confidence_level = "MEDIUM"

                            else:

                                confidence_level = "LOW"

                            # Result
                            if prediction == 1:

                                result = "DECEPTIVE PATTERN"
                                result_class = "deceptive"

                            else:

                                result = "NOT DECEPTIVE"
                                result_class = "not-deceptive"

                            # ------------------------------------------------
                            # DISPLAY RESULT
                            # ------------------------------------------------

                            st.markdown(
                                f"""
                                <div class="result-box {result_class}">

                                    <h2>🤖 AI Prediction</h2>

                                    <h3>{result}</h3>

                                    <p class="confidence">
                                        Confidence:
                                        {confidence:.2f}%
                                    </p>

                                    <p class="level">
                                        Confidence Level:
                                        {confidence_level}
                                    </p>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.warning(
                                "Text was detected, but no relevant "
                                "notification/message text could be "
                                "identified."
                            )

                except Exception as e:

                    st.error(
                        "An error occurred while processing the image."
                    )

                    st.exception(e)


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.caption(
    "This application uses EasyOCR to extract text from screenshots "
    "and TF-IDF with Logistic Regression to classify potential "
    "dark-pattern text."
)