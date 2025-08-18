import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import asyncio

# Add the parent directory to the path so we can import semantic_search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_search import (
    DocumentLoader,
    DocumentChunker,
    HuggingFaceEmbeddings,
    VectorStore,
    SimilaritySearch,
    Retriever,
    PromptManager,
    ResponseGenerator,
    RAGPipeline,
)
from langchain.schema import Document


class TestDocumentLoader(unittest.TestCase):
    @patch("semantic_search.os.listdir")
    @patch("semantic_search.PyPDFLoader")
    def test_load_content_successfully(self, mock_pypdf_loader, mock_listdir):
        # Arrange
        mock_listdir.return_value = ["test.pdf"]
        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_loader_instance = MagicMock()

        async def mock_alazy_load():
            yield mock_page

        mock_loader_instance.alazy_load.return_value = mock_alazy_load()
        mock_pypdf_loader.return_value = mock_loader_instance

        # Act
        async def run_test():
            loader = DocumentLoader(dir=MagicMock())
            pages = await loader.load_content()
            return pages

        pages = asyncio.run(run_test())

        # Assert
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_content, "This is a test page.")

    @patch("semantic_search.os.listdir")
    def test_load_content_with_non_pdf_files(self, mock_listdir):
        # Arrange
        mock_listdir.return_value = ["test.txt", "document.docx"]

        # Act
        async def run_test():
            loader = DocumentLoader(dir=MagicMock())
            pages = await loader.load_content()
            return pages

        pages = asyncio.run(run_test())

        # Assert
        self.assertEqual(len(pages), 0)

    @patch("semantic_search.os.listdir", side_effect=Exception("Test error"))
    def test_load_content_with_os_error(self, mock_listdir):
        # Act
        async def run_test():
            loader = DocumentLoader(dir=MagicMock())
            pages = await loader.load_content()
            return pages

        pages = asyncio.run(run_test())

        # Assert
        self.assertEqual(len(pages), 0)


class TestDocumentChunker(unittest.TestCase):
    def test_create_chunks_successfully(self):
        # Arrange
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        mock_doc = MagicMock()
        mock_doc.page_content = "This is a test document."
        mock_doc.metadata = {"source": "test.pdf"}
        documents = [mock_doc]

        # Act
        chunks = chunker.create_chunks(documents)

        # Assert
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Document)


class TestHuggingFaceEmbeddings(unittest.TestCase):
    @patch("semantic_search.getpass")
    @patch("semantic_search.HuggingFaceEndpointEmbeddings")
    def test_get_embeddings_successfully(self, mock_hf_embeddings, mock_getpass):
        # Arrange
        mock_getpass.return_value = "test_token"
        mock_embeddings_instance = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings_instance

        # Act
        hf_embeddings = HuggingFaceEmbeddings()
        embeddings = hf_embeddings.get_embeddings()

        # Assert
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings, mock_embeddings_instance)


class TestVectorStore(unittest.TestCase):
    @patch("semantic_search.Chroma")
    def test_create_store_successfully(self, mock_chroma):
        # Arrange
        mock_embeddings = MagicMock()
        mock_vector_store_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_vector_store_instance
        documents = [MagicMock()]

        # Act
        vector_store = VectorStore(embeddings=mock_embeddings)
        store = vector_store.create_store(documents)

        # Assert
        self.assertIsNotNone(store)
        self.assertEqual(store, mock_vector_store_instance)


class TestSimilaritySearch(unittest.TestCase):
    def test_similarity_search_with_score(self):
        # Arrange
        mock_vector_store = MagicMock()
        mock_document = MagicMock()
        mock_document.page_content = "Test content"
        mock_document.metadata = {"source": "test.pdf"}
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_document, 0.9),
            (mock_document, 0.8),
        ]
        search = SimilaritySearch(vectorstore=mock_vector_store)
        query = "test query"

        # Act
        search.similarity_search_with_score(query)

        # Assert
        mock_vector_store.similarity_search_with_score.assert_called_once_with(query, k=4)


class TestRetriever(unittest.TestCase):
    def test_get_relevant_documents(self):
        # Arrange
        mock_vector_store = MagicMock()
        mock_retriever_instance = MagicMock()
        mock_document = MagicMock()
        mock_document.page_content = "Test content"
        mock_document.metadata = {"source": "test.pdf"}
        mock_retriever_instance.invoke.return_value = [mock_document, mock_document]
        mock_vector_store.as_retriever.return_value = mock_retriever_instance
        retriever = Retriever(vectorstore=mock_vector_store)
        query = "test query"

        # Act
        documents = retriever.get_relevant_documents(query)

        # Assert
        self.assertIsNotNone(documents)
        self.assertEqual(len(documents), 2)
        mock_vector_store.as_retriever.assert_called_once()
        mock_retriever_instance.invoke.assert_called_with(query)


class TestPromptManager(unittest.TestCase):
    def test_create_zephyr_prompt(self):
        # Arrange
        query = "test query"
        context = "test context"

        # Act
        prompt = PromptManager.create_zephyr_prompt(query, context)

        # Assert
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)


class TestResponseGenerator(unittest.TestCase):
    @patch("semantic_search.AutoTokenizer")
    @patch("semantic_search.AutoModelForCausalLM")
    @patch("semantic_search.pipeline")
    @patch("semantic_search.RetrievalQA")
    @patch("semantic_search.HuggingFacePipeline")
    def test_setup_qa_chain_and_ask_question(
        self,
        mock_hf_pipeline,
        mock_retrieval_qa,
        mock_pipeline,
        mock_model,
        mock_tokenizer,
    ):
        # Arrange
        mock_retriever = MagicMock()
        generator = ResponseGenerator()
        mock_hf_pipeline.return_value = MagicMock()

        # Act
        success = generator.setup_qa_chain(mock_retriever)

        # Assert
        self.assertTrue(success)
        self.assertIsNotNone(generator.qa_chain)

        # Arrange for ask_question
        question = "test question"
        expected_result = {
            "result": "test answer",
            "source_documents": [MagicMock(page_content="doc1", metadata={})],
        }
        generator.qa_chain.invoke.return_value = expected_result

        # Act
        result = generator.ask_question(question)

        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], "test answer")
        generator.qa_chain.invoke.assert_called_with(question)

        # Act for ask_question_from_prompt
        prompt = "test prompt"
        result_prompt = generator.ask_question_from_prompt(prompt)

        # Assert
        self.assertIsNotNone(result_prompt)
        generator.qa_chain.invoke.assert_called_with(prompt)


@patch("semantic_search.ResponseGenerator")
@patch("semantic_search.Retriever")
@patch("semantic_search.SimilaritySearch")
@patch("semantic_search.VectorStore")
@patch("semantic_search.HuggingFaceEmbeddings")
@patch("semantic_search.DocumentChunker")
@patch("semantic_search.DocumentLoader")
class TestRAGPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_build_and_query_pipeline(
        self,
        mock_loader,
        mock_chunker,
        mock_embeddings,
        mock_vector_store,
        mock_similarity_search,
        mock_retriever,
        mock_generator,
    ):
        # Arrange
        dir_path = MagicMock()
        pipeline = RAGPipeline(dir_path)

        # Mock instances and their methods
        mock_loader_instance = mock_loader.return_value
        future = asyncio.Future()
        future.set_result([MagicMock()])
        mock_loader_instance.load_content.return_value = future

        mock_chunker_instance = mock_chunker.return_value
        mock_chunker_instance.create_chunks.return_value = [MagicMock()]

        mock_embeddings_instance = mock_embeddings.return_value
        mock_embeddings_instance.get_embeddings.return_value = MagicMock()

        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.create_store.return_value = MagicMock()

        mock_retriever_instance = mock_retriever.return_value
        mock_retriever_instance.retriever = MagicMock()

        mock_generator_instance = mock_generator.return_value
        mock_generator_instance.setup_qa_chain.return_value = True

        # Act
        await pipeline.build()

        # Assert build process
        mock_loader.assert_called_with(dir_path)
        mock_chunker.assert_called()
        mock_embeddings.assert_called()
        mock_vector_store.assert_called()
        mock_similarity_search.assert_called()
        mock_retriever.assert_called()
        mock_generator.assert_called()

        # Arrange for query
        query = "test query"
        mock_response = {
            "answer": "test answer",
            "source_documents": [
                {"content": "doc1", "metadata": {}},
                {"content": "doc2", "metadata": {}},
            ],
        }
        mock_generator_instance.ask_question.return_value = mock_response
        mock_generator_instance.ask_question_from_prompt.return_value = mock_response

        # Act
        pipeline.query(query)

        # Assert query process
        mock_similarity_search.return_value.similarity_search_with_score.assert_called_with(
            query
        )
        mock_retriever.return_value.get_relevant_documents.assert_called_with(query)
        mock_generator_instance.ask_question.assert_called()
        mock_generator_instance.ask_question_from_prompt.assert_called()


if __name__ == "__main__":
    unittest.main()
