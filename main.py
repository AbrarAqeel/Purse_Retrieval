"""
Streamlit entry point for the Purse Retrieval System.

Run with:
    streamlit run main.py
"""

from PIL import Image
import streamlit as st

import config
from core.embedding import load_embedding_model
from core.indexer import load_database, index_exists
from core.search import search_purse

st.set_page_config(page_title="Purse Retrieval System", page_icon="👜", layout="wide")


@st.cache_resource
def get_embedding_model():
    """Load the DINOv2 model once and cache it across reruns/sessions."""
    return load_embedding_model()


@st.cache_resource
def get_database():
    """Load the FAISS index + metadata once and cache it across reruns/sessions."""
    return load_database()


def main() -> None:
    st.title("Purse Retrieval System")
    st.write(
        "Upload a photo. Choose **Detect Purse** to highlight the purse directly in your "
        "uploaded photo, or **Find Model with Purse** to search for matching items in the database."
    )

    if not index_exists():
        st.error(
            "No index found yet. Run `python build_index.py` first to build it "
            "from the photos in `dataset/women/`, then reload this page."
        )
        return

    embed_model, embed_processor = get_embedding_model()
    faiss_index, metadata = get_database()

    col_input, col_results = st.columns([1, 2])

    query_image = None
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Purse or Model Image", type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Uploaded Image", use_container_width=True)

        # Separate Action Buttons side by side
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            detect_clicked = st.button("Detect Purse", use_container_width=True)
        with col_btn2:
            search_clicked = st.button("Find Model with Purse", type="primary", use_container_width=True)

    with col_results:
        # Option A: Detect Purse (Draw Bounding Box on upload)
        if detect_clicked:
            if query_image is None:
                st.warning("Please upload an image first.")
            else:
                with st.spinner("Loading YOLO and detecting purse..."):
                    from core.detection import load_yolo_model, detect_purse_bbox
                    from PIL import ImageDraw
                    import tempfile
                    from pathlib import Path
                    import os

                    @st.cache_resource
                    def get_yolo_model_cached():
                        return load_yolo_model()

                    yolo_model = get_yolo_model_cached()

                    # Put image context to an isolated temp path for compatibility with detect_purse_bbox
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        query_image.save(tmp.name)
                        tmp_path = Path(tmp.name)

                    try:
                        # Uses the updated hardcoded value from config.py natively
                        bbox = detect_purse_bbox(tmp_path, yolo_model)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                    if bbox is None:
                        st.info(
                            f"No purse detected at confidence threshold {config.DETECTION_CONF_THRESHOLD}. "
                        )
                    else:
                        x1, y1, x2, y2, confidence = bbox
                        
                        annotated_image = query_image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=5)
                        
                        st.success(f"Purse isolated successfully! (Confidence: {confidence:.2f})")
                        st.image(
                            annotated_image, 
                            caption=f"Detected Purse Bounding Box (conf={confidence:.2f})", 
                            use_container_width=True
                        )

        # Option B: Find Model with Purse (Database Index Search)
        elif search_clicked:
            if query_image is None:
                st.warning("Upload a purse image first.")
            else:
                with st.spinner("Searching database index..."):
                    results = search_purse(
                        query_image, faiss_index, metadata, embed_model, embed_processor
                    )

                if not results:
                    st.info("No confident match found in the indexed dataset.")
                else:
                    st.success(f"Found {len(results)} matching photo(s).")
                    result_cols = st.columns(min(len(results), 3))
                    for i, r in enumerate(results):
                        with result_cols[i % len(result_cols)]:
                            st.image(
                                r["path"],
                                caption=f"Rank {i + 1} — similarity {r['similarity']:.3f}",
                                use_container_width=True,
                            )


if __name__ == "__main__":
    main()