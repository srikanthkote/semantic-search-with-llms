import streamlit as st
from pathlib import Path
import asyncio
import tempfile
from semantic_search import RAGPipeline


def main():
    st.title("Semantic Search with RAG")
    st.write(
        "Upload your PDF documents, ask a question, and get answers from the content."
    )

    # Get HuggingFace API token
    hf_token = st.text_input("Enter your HuggingFace API token:", type="password")

    # Upload PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    # Ask a question
    question = st.text_area("Ask a question about the documents:")

    if st.button("Get Answer"):
        if not hf_token:
            st.error("Please enter your HuggingFace API token.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF file.")
        elif not question:
            st.error("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                try:
                    # Create a temporary directory to store uploaded files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        for uploaded_file in uploaded_files:
                            with open(temp_path / uploaded_file.name, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                        # Initialize and build the RAG pipeline
                        rag_pipeline = RAGPipeline(dir=temp_path, hf_token=hf_token)

                        # Run the async build method
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(rag_pipeline.build())

                        # Query the pipeline
                        response = rag_pipeline.query(question)

                    if response and 'result' in response:
                        st.subheader("Answer")
                        st.write(response["result"])                
                                
                        if "source_documents" in response and response["source_documents"]:
                            st.subheader("Source Documents")
                            for doc in response["source_documents"]:
                                with st.expander(
                                    f"Source: {doc.metadata.get('source', 'N/A')}"
                                ):
                                    st.write(doc.page_content)
                    else:
                        st.error("Failed to get an answer from the pipeline.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
