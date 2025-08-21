import os
from typing import List, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain.chains import RetrievalQA
from getpass import getpass
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
import torch


# Semantic Search Pipeline for PDF Documents
# This script loads PDF documents from a specified directory, splits them into chunks,
# and prepares them for semantic search using LangChain.


# Helper function for printing docs
def pretty_print_docs(docs):
    doc0 = docs[0]
    doc1 = docs[1]
    doc2 = docs[2]

    print(
        tabulate(
            [
                [doc0.page_content, doc0.metadata],
                [doc1.page_content, doc1.metadata],
                [doc2.page_content, doc2.metadata],
            ],
            maxcolwidths=[40, 100],
            tablefmt="grid",
        )
    )


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
        doc0, score0 = results[0]
        doc1, score1 = results[1]
        doc2, score2 = results[2]
        pretty_print_docs([doc0, doc1, doc2])


# Responsible for fetching relevant documents from a knowledge base (often a vector store) based on a query.
# The Retriever class uses Chroma to retrieve relevant documents based on a query,
# applying Maximum Marginal Relevance (MMR) to optimize for relevance and diversity.
class Retriever:
    def __init__(self, vectorstore: Chroma, k: int = 20):

        # Fetch more documents for the MMR algorithm to consider
        # But only return the top 5
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 50},  # Maximum Marginal Relevance
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.retriever.invoke(query)
        pretty_print_docs(results)
        return results


class PromptManager:
    def __init__(self):
        self.prompt_template = """You are a story teller, answering questions in an excited, insightful, and empathetic way. Answer the question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {question}"""

    def create_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt_template,
            input_variables=[
                "context",
                "question",
            ],
        )


# LLM (Large Language Model): Generates the answer based on the retrieved documents and the query.
# Chain Type: Determines how the retrieved documents are combined and passed to the LLM (e.g., "stuff" chain for concatenating documents, "map_reduce" for processing documents in batches).
#   The ResponseGenerator class integrates the HuggingFace model with the RetrievalQA chain,
class ResponseGenerator:
    def __init__(self, model_id: str = "microsoft/DialoGPT-medium"):
        self.retriever = None
        self.qa_chain = None

    def setup_qa_chain(self, retriever, model_name="microsoft/DialoGPT-medium"):
        """Setup QA chain with Hugging Face model"""

        try:
            self.retriever = retriever
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

            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            compressor = CrossEncoderReranker(model=model, top_n=3)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.retriever
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PromptManager().create_prompt()},
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
            print(f"Answer: {result['result']}")
            pretty_print_docs(result["source_documents"])
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
        self.reranker = None

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

        # Ask the question using the ResponseGenerator
        response = self.generator.ask_question(question)

        return response


import asyncio


async def main():

    directory = os.path.expanduser("~") + "/Downloads/search-sources/"
    print(f"Building RAG pipeline for: {directory}")

    pipeline = RAGPipeline(Path(directory))
    await pipeline.build()

    query = "What is the business model of WazirX ?"
    response = pipeline.query(query)


if __name__ == "__main__":
    asyncio.run(main())
