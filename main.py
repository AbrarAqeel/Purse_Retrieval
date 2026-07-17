"""
Streamlit entry point for the Purse Retrieval System.

Run with:
    streamlit run main.py

This ONLY loads a previously built index and serves the search UI -- it
never runs YOLO detection itself. Build or update the index separately:
    python build_index.py
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
    st.title("👜 Purse Retrieval System")
    st.write(
        "Upload a photo of **just a purse**. The system searches the indexed "
        "database and returns the original photo(s) of the model(s) carrying "
        "the closest-matching purse."
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
            "Upload Purse Image", type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Query", use_container_width=True)

        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    with col_results:
        if search_clicked:
            if query_image is None:
                st.warning("Upload a purse image first.")
            else:
                with st.spinner("Searching..."):
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
