import os
from typing import List, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain.chains import RetrievalQA
from getpass import getpass
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Semantic Search Pipeline for PDF Documents
# This script loads PDF documents from a specified directory, splits them into chunks,
# and prepares them for semantic search using LangChain.


# DocumentLoader class is responsible for loading PDF documents from a directory
class DocumentLoader:
    def __init__(self, dir: Path, ext: str = ".pdf"):
        self.dir = dir
        self.ext = ext

    async def load_content(self) -> Path:
        try:
            pages = []
            # Iterate over files in directory
            for file in os.listdir(self.dir):
                if file.endswith(self.ext):
                    print(f"Processing file: {file}")

                    loader = PyPDFLoader(str(self.dir) + "/" + file, mode="single")
                    async for page in loader.alazy_load():
                        pages.append(page)

            print(f"Loaded {len(pages)} pages from {self.dir}")
            return pages
        except Exception as e:
            print(f"Error loading content: {str(e)}")
            return []


# The DocumentChunker class splits documents into smaller chunks using LangChain’s RecursiveCharacterTextSplitter.
# This allows for better processing by creating manageable text pieces with overlap for context preservation.
class DocumentChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")

        return chunks


# The HuggingFaceEmbeddings class uses HuggingFace’s API to convert text into vector embeddings,
# capturing semantic meaning for effective similarity-based search.
# Embedding models are ML algorithms that transform data (like text, images, or audio) into numerical representations called embeddings.
# These embeddings capture semantic relationships within the data, allowing machines to understand and compare different data points effectively.
# Essentially, they convert complex information into a format that computers can easily process and analyze
class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Get HuggingFace token if not already set
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            hf_token = getpass("Enter your HuggingFace API token: ")
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=model_name,
            task="feature-extraction",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

    def get_embeddings(self):
        return self.embeddings


# The VectorStore class stores embeddings created above in Chroma, enabling efficient querying by creating a searchable vector store from the documents.
class VectorStore:
    def __init__(
        self,
        embeddings: HuggingFaceEndpointEmbeddings,
        collection_name: str = "semantic_search_collection",
    ):
        self.embeddings = embeddings
        self.vectorstore = None
        self.collection_name = collection_name

    def create_store(self, documents: List[Document]) -> Chroma:
        print(f"Creating vector store with {len(documents)} documents")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="semantic_search_collection",
        )

        return self.vectorstore


class SimilaritySearch:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def similarity_search_with_score(self, query: str):
        results = self.vectorstore.similarity_search_with_score(query, k=4)
        doc1, score1 = results[0]
        doc2, score2 = results[1]
        print(
            tabulate(
                [
                    ["Document Content", "Metadata", "Score"],
                    [doc1.page_content, doc1.metadata, score1],
                    [doc2.page_content, doc2.metadata, score2],
                ],
                maxcolwidths=[40, 120, None],
                tablefmt="grid",
            )
        )


# Responsible for fetching relevant documents from a knowledge base (often a vector store) based on a query.
# The Retriever class uses Chroma to retrieve relevant documents based on a query,
# applying Maximum Marginal Relevance (MMR) to optimize for relevance and diversity.
class Retriever:
    def __init__(self, vectorstore: Chroma, k: int = 3):
        self.retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k}  # Maximum Marginal Relevance
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.retriever.invoke(query)
        print(
            tabulate(
                [
                    ["Document Content", "Metadata"],
                    [results[0].page_content, results[0].metadata],
                    [results[1].page_content, results[1].metadata],
                ],
                maxcolwidths=[40, 120],
                tablefmt="grid",
            )
        )

        return self.retriever.invoke(query)


# The PromptManager class creates prompts in the Zephyr format,
# providing context to the model and guiding it to generate accurate responses.
class PromptManager:
    @staticmethod
    def create_zephyr_prompt(query: str, context: str = "") -> str:
        return f"""
<|system|>
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in context
</s>
<|user|>
Context: {context}

Question: {query}
</s>
<|assistant|>
"""


# LLM (Large Language Model): Generates the answer based on the retrieved documents and the query.
# Chain Type: Determines how the retrieved documents are combined and passed to the LLM (e.g., "stuff" chain for concatenating documents, "map_reduce" for processing documents in batches).
#   The ResponseGenerator class integrates the HuggingFace model with the RetrievalQA chain,
class ResponseGenerator:
    def __init__(self, model_id: str = "microsoft/DialoGPT-medium"):
        self.qa_chain = None

    def setup_qa_chain(self, retriever, model_name="microsoft/DialoGPT-medium"):
        """Setup QA chain with Hugging Face model"""

        try:
            # Setup tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1,
            )

            # Create HuggingFace LLM
            llm = HuggingFacePipeline(pipeline=pipe)

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            return True

        except Exception as e:
            print(f"Error setting up QA chain: {str(e)}")
            return False

    def ask_question(self, question):
        """Ask a question using the QA chain"""
        if self.qa_chain is None:
            return None

        print(f"Asking question: {question}")
        try:
            result = self.qa_chain.invoke(question)
            return {
                "answer": result["result"],
                "source_documents": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in result["source_documents"]
                ],
            }
        except Exception as e:
            return None

    def ask_question_from_prompt(self, prompt: str):
        """Ask a question using the QA chain"""
        if self.qa_chain is None:
            return None

        print(f"Asking question: {prompt}")
        try:
            result = self.qa_chain.invoke(prompt)
            return {
                "answer": result["result"],
                "source_documents": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in result["source_documents"]
                ],
            }
        except Exception as e:
            return None


# The RAGPipeline class integrates all components - loading, chunking, embeddings, retrieval, prompt creation, and model generation -
# into a unified pipeline that processes queries and generates responses based on relevant documents.
class RAGPipeline:
    def __init__(self, dir: Path):
        self.loader = DocumentLoader(dir)
        self.chunker = DocumentChunker()
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.generator = None
        self.similaritysearch = None
        self.retriever_component = None

    async def build(self):
        documents = await self.loader.load_content()
        chunks = self.chunker.create_chunks(documents)
        vector_store = VectorStore(self.embeddings.get_embeddings())
        self.vectorstore = vector_store.create_store(chunks)
        self.similaritysearch = SimilaritySearch(self.vectorstore)
        self.retriever_component = Retriever(self.vectorstore)
        self.retriever = self.retriever_component.retriever
        self.generator = ResponseGenerator()
        self.qa_chain = self.generator.setup_qa_chain(self.retriever)

    def query(self, question: str) -> str:
        # Perform similarity search on vector store to find relevant documents
        self.similaritysearch.similarity_search_with_score(question)

        # Retrieve relevant documents using VectorStoreRetriever
        self.retriever_component.get_relevant_documents(question)

        response = self.generator.ask_question(
            "What are some of the challenges of industry 5.0?"
        )
        print(
            tabulate(
                [
                    [response["answer"]],
                    [response["source_documents"][0].get("content", "")],
                    [response["source_documents"][0].get("metadata", "")],
                    [response["source_documents"][1].get("content", "")],
                    [response["source_documents"][1].get("metadata", "")],
                ],
                maxcolwidths=[60, 60, 60],
                tablefmt="grid",
            )
        )

        # Create a Zephyr prompt for the question
        prompt = PromptManager.create_zephyr_prompt(question)
        response = self.generator.ask_question_from_prompt(prompt)

        return response


import asyncio


async def main():

    directory = os.path.expanduser("~") + "/Downloads/search-sources/"
    print(f"Building RAG pipeline for: {directory}")

    pipeline = RAGPipeline(Path(directory))
    await pipeline.build()

    query = "What are some of the challenges of industry 5.0?"
    response = pipeline.query(query)


if __name__ == "__main__":
    asyncio.run(main())
