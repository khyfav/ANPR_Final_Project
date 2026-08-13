import tempfile
from pathlib import Path

import cv2
import streamlit as st

from pipeline import load_model, read_plate


st.set_page_config(
    page_title="Meridian ANPR Prototype",
    page_icon="🚘",
    layout="wide",
)

st.title("Meridian Access Systems — ANPR Prototype")
st.caption(
    "Educational prototype: cropped plate image → segmentation → "
    "CNN recognition → confidence routing"
)

MODEL_PATH = "models/emnist_cnn_36.pt"

st.sidebar.subheader("Model")
st.sidebar.caption("36-class EMNIST CNN")
st.sidebar.code(MODEL_PATH, language=None)

accept = st.sidebar.slider(
    "Auto-accept threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.95,
    step=0.01,
)

review = st.sidebar.slider(
    "Human-review threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.80,
    step=0.01,
)

if review >= accept:
    st.sidebar.warning(
        "Human-review threshold should be lower than "
        "the auto-accept threshold."
    )


@st.cache_resource
def get_model(model_path):
    return load_model(model_path)


uploaded = st.file_uploader(
    "Upload an unseen cropped plate image",
    type=["png", "jpg", "jpeg"],
)

if uploaded:
    suffix = Path(uploaded.name).suffix or ".png"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as f:
        f.write(uploaded.getbuffer())
        image_path = f.name

    try:
        model, device = get_model(MODEL_PATH)
        result = read_plate(image_path, model, device)

        confidence = result["plate_confidence"]

        if confidence >= accept:
            decision = "AUTO-ACCEPT"
        elif confidence >= review:
            decision = "HUMAN-REVIEW"
        else:
            decision = "MANUAL-ENTRY"

        detected = result["image"].copy()

        for x, y, w, h in result["boxes"]:
            cv2.rectangle(
                detected,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        detected = cv2.cvtColor(
            detected,
            cv2.COLOR_BGR2RGB,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                detected,
                caption="Detected character regions",
                width="stretch",
            )

        with col2:
            st.image(
                result["binary"],
                caption="Thresholded segmentation image",
                width="stretch",
            )

        predicted_text = (
            result["text"]
            if result["text"]
            else "(no characters detected)"
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Predicted plate", predicted_text)

        with m2:
            st.metric("Plate confidence", f"{confidence:.1%}")

        with m3:
            st.metric("Routing decision", decision)

        if decision == "AUTO-ACCEPT":
            st.success(
                "High-confidence read: eligible for automatic processing."
            )
        elif decision == "HUMAN-REVIEW":
            st.warning(
                "Moderate-confidence read: route to a human reviewer."
            )
        else:
            st.error(
                "Low-confidence read: use manual entry rather than trusting "
                "the automated result."
            )

        st.subheader("Character confidences")

        confidences = result["character_confidences"]

        if confidences and result["text"]:
            confidence_rows = []

            for i, (character, char_conf) in enumerate(
                zip(result["text"], confidences),
                start=1,
            ):
                confidence_rows.append(
                    {
                        "Position": i,
                        "Character": character,
                        "Confidence": f"{char_conf:.1%}",
                    }
                )

            st.dataframe(
                confidence_rows,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "No character confidence values are available because "
                "no characters were detected."
            )

        with st.expander("How to interpret this result"):
            st.write(
                "The prototype segments the cropped plate into character "
                "regions, classifies each character with the 36-class CNN, "
                "uses the weakest character confidence as the plate-level "
                "confidence, and then applies the routing thresholds shown "
                "in the sidebar."
            )

    except Exception as e:
        st.error(f"Demo error: {e}")
