import os
import logging
from typing import List
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
from tabulate import tabulate
from transformers import pipeline
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate


# Semantic Search Pipeline for PDF Documents
# This script loads PDF documents from a specified directory, splits them into chunks,
# and prepares them for semantic search using LangChain.

def _setup_logger(self) -> logging.Logger:
    """Set up a dedicated logger for the agent"""
    logger = logging.getLogger(f"{self.name}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Helper function for printing docs
def pretty_print_docs(docs):

    if not docs:
        print("No documents found.")
        return

    print(f"Found {len(docs)} results")

    table_data = [[doc.page_content, doc.metadata["source"]] for doc in docs]
    print(
        tabulate(
            table_data,
            headers=["Page Content", "Metadata"],
            maxcolwidths=[70, 70],
            tablefmt="grid",
        )
    )


# DocumentLoader class is responsible for loading PDF documents from a directory
class DocumentLoader:
    def __init__(self, dir: Path, ext: str = ".pdf"):
        self.dir = dir
        self.ext = ext
        self.name = "DocumentLoader"
        self.logger = _setup_logger(self)

    async def load_content(self) -> Path:
        try:
            pages = []
            # Iterate over files in directory
            for file in os.listdir(self.dir):
                if file.endswith(self.ext):
                    #print(f"Processing file: {file}")

                    loader = PyPDFLoader(str(self.dir) + "/" + file, mode="single")
                    async for page in loader.alazy_load():
                        pages.append(page)

            self.logger.info(f"Loaded {len(pages)} pages from {self.dir}")
            return pages
        except Exception as e:
            self.logger.error(f"Error loading content: {str(e)}")
            return []


# The DocumentChunker class splits documents into smaller chunks using LangChain’s RecursiveCharacterTextSplitter.
# This allows for better processing by creating manageable text pieces with overlap for context preservation.
class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.name = "DocumentChunker"
        self.logger = _setup_logger(self)

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        try:    
            chunks = self.splitter.split_documents(documents)
            self.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            return []

# The HuggingFaceEmbeddings class uses HuggingFace’s API to convert text into vector embeddings,
# capturing semantic meaning for effective similarity-based search.
# Embedding models are ML algorithms that transform data (like text, images, or audio) into numerical representations called embeddings.
# These embeddings capture semantic relationships within the data, allowing machines to understand and compare different data points effectively.
# Essentially, they convert complex information into a format that computers can easily process and analyze
class HuggingFaceEmbeddings:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_token: str = None,
    ):
        if api_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        elif "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            raise ValueError(
                "HuggingFace API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable or pass it as an argument."
            )

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
        self.name = "VectorStore",
        self.logger = _setup_logger(self)


    def create_store(self, documents: List[Document]) -> Chroma:

        self.logger.info(f"Creating semantic_search_collection with {len(documents)} documents")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="semantic_search_collection",
        )
        self.logger.info("semantic_search_collection created in vector store")

        return self.vectorstore

class SimilaritySearch:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def similarity_search_with_score(self, query: str):
        results = self.vectorstore.similarity_search_with_score(query, k=4)
        #pretty_print_docs([doc for doc, score in results])


# Responsible for fetching relevant documents from a knowledge base (often a vector store) based on a query.
# The Retriever class uses Chroma to retrieve relevant documents based on a query,
# applying Maximum Marginal Relevance (MMR) to optimize for relevance and diversity.
class Retriever:
    def __init__(self, vectorstore: Chroma, k: int = 10):

        # Fetch more documents for the MMR algorithm to consider
        # But only return the top 10
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 50},  # Maximum Marginal Relevance
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.retriever.invoke(query)
        #pretty_print_docs(results)
        return results

class PromptManager:
    def __init__(self):
        self.prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use atleast five sentences minimum to answer the question. Always say "thanks for asking!" at the end of the answer.

        <context>
        {context}
        </context>

        Question: {question}

        Answer:"""

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
    def __init__(self):
        self.retriever = None
        self.qa_chain = None
        self.name = "ResponseGenerator"
        self.logger = _setup_logger(self)

    def setup_qa_chain(self, retriever, model_name="google/flan-t5-small"):
        """Setup QA chain with DialoGPT for text generation"""

        try:
            self.retriever = retriever
            self.logger.info(f"Loading tokenizer and model: {model_name}")
            
            # Load tokenizer and model for text generation (use fast tokenizer)
            from transformers import T5ForConditionalGeneration, T5TokenizerFast
            tokenizer = T5TokenizerFast.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            self.logger.info("Model loaded successfully. Creating text generation pipeline...")
            # Create text generation pipeline (compatible with T5)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=256,
                temperature=0.7,
                do_sample=True,
            )

            # Create HuggingFace LLM
            llm = HuggingFacePipeline(pipeline=pipe)

            # Create CrossEncoderReranker for document compression
            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
            compressor = CrossEncoderReranker(model=model, top_n=3)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.retriever
            )

            # Create a prompt with the expected input variables
            prompt = PromptManager().create_prompt()
            
            # Create a simpler chain that combines retrieval and generation
            def get_answer(input_dict):
                # Get relevant documents using the new invoke method
                docs = compression_retriever.invoke(input_dict["question"])
                
                # Format the context
                context = "\n\n".join(doc.page_content for doc in docs)
                # Format the prompt
                formatted_prompt = prompt.format(
                    context=context,
                    question=input_dict["question"]
                )
                #self.logger.info("Formatted prompt: ", formatted_prompt)
                print("Formatted prompt: ", formatted_prompt)
                
                # Ensure input length does not exceed model limits by truncating via tokenizer
                try:
                    # Some tokenizers set an extremely large default; cap to 512 for FLAN-T5
                    max_len = getattr(tokenizer, "model_max_length", 512) or 512
                    if isinstance(max_len, int) and max_len > 4096:
                        max_len = 512
                    tokenized = tokenizer(
                        formatted_prompt,
                        truncation=True,
                        max_length=max_len,
                        return_tensors=None,
                    )
                    # Decode back the (possibly truncated) prompt
                    input_ids = tokenized.get("input_ids", [])
                    # Handle both [ids] and [[ids]]
                    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                        input_ids = input_ids[0]
                    truncated_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
                except Exception as e:
                    # Fallback to original prompt if tokenization fails
                    truncated_prompt = formatted_prompt
                    print(f"Warning: tokenizer truncation failed: {e}")

                # Invoke the LLM with the truncated prompt
                response = llm.invoke(truncated_prompt)

                # Return the response with source documents
                return {
                    "result": response.content if hasattr(response, 'content') else str(response),
                    "source_documents": docs
                }
                
            # Create a simple chain that just calls our function
            self.qa_chain = get_answer

        except Exception as e:
            self.logger.error(f"Error setting up QA chain: {str(e)}")
            return False

    def ask_question(self, question):
        """Ask a question using the QA chain"""
        if self.qa_chain is None:
            return {"result": "QA chain not properly initialized", "source_documents": []}

        try:
            # Prepare the input format expected by the chain
            input_data = {"question": question}
            
            # Debug: Print input data
            result = self.qa_chain(input_data)
            self.logger.info("QA Chain Response:", result)  # Debug print
            
            # Ensure we have the expected structure
            if not isinstance(result, dict):
                return {
                    "result": str(result) if result else "No answer generated.", 
                    "source_documents": []
                }
                
            # Ensure we have both result and source_documents
            response = {
                "result": result.get("result", "No answer generated."),
                "source_documents": result.get("source_documents", [])
            }
            
            return response
            
        except Exception as e:
            error_msg = f"Error asking question: {str(e)}"
            self.logger.error(f"Error asking question: {str(e)}", exc_info=True)
            #print(error_msg)
            import traceback
            traceback.print_exc()
            return {"result": error_msg, "source_documents": []}


# The RAGPipeline class integrates all components - loading, chunking, embeddings, retrieval, prompt creation, and model generation -
# into a unified pipeline that processes queries and generates responses based on relevant documents.
class RAGPipeline:

    def __init__(self, dir: Path, hf_token: str = None):
        self.loader = DocumentLoader(dir)
        self.chunker = DocumentChunker()
        self.embeddings = HuggingFaceEmbeddings(api_token=hf_token)
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

    # For command-line, we can use an environment variable for the token.
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
    if not hf_token:
        print("HuggingFace API token not found in environment variables.")
        # fallback to asking the user, useful for local testing
        try:
            from getpass import getpass

            hf_token = getpass("Enter your HuggingFace API token: ")
        except ImportError:
            print(
                "getpass not available. Please set the HUGGINGFACEHUB_API_TOKEN environment variable."
            )
            return

    pipeline = RAGPipeline(Path(directory), hf_token=hf_token)
    await pipeline.build()

    #query = "What is the business model of WazirX ?"
    query = "What is Industry 5.0 ?"
    response = pipeline.query(query)

    if response:
        print("\n\n--- Answer ---")
        print(response["result"])
        #print("\n--- Source Documents ---")
        #pretty_print_docs(response["source_documents"])


if __name__ == "__main__":
    asyncio.run(main())
