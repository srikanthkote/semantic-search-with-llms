import os
import asyncio
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from semantic_search import RAGPipeline


def create_dummy_pdf(filepath: Path, content: str):
    c = canvas.Canvas(str(filepath), pagesize=letter)
    width, height = letter
    text = c.beginText(100, height - 100)
    text.setFont("Helvetica", 12)
    for line in content.split("\n"):
        text.textLine(line)
    c.drawText(text)
    c.save()
    print(f"Created dummy PDF at {filepath}")


async def main():
    # --- Setup ---
    test_dir = Path("test_docs")
    test_dir.mkdir(exist_ok=True)
    pdf_path = test_dir / "dummy.pdf"
    pdf_content = "The quick brown fox jumps over the lazy dog. This is a test document for the RAG pipeline. The business model of WazirX is based on transaction fees."
    create_dummy_pdf(pdf_path, pdf_content)

    # --- Get Token ---
    # The user provided the token: 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
    hf_token = "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"

    # --- Run Pipeline ---
    try:
        pipeline = RAGPipeline(dir=test_dir, hf_token=hf_token)
        await pipeline.build()

        query = "What is the business model of WazirX?"
        response = pipeline.query(query)

        # --- Verify ---
        if response and response.get("result"):
            print("\n--- Test Passed ---")
            print(f"Query: {query}")
            print(f"Answer: {response}")
        else:
            print("\n--- Test Failed ---")
            print("Did not receive a valid response from the pipeline.")
            if response:
                print(f"Full response: {response}")

    except Exception as e:
        print(f"\n--- Test Failed with Exception ---")
        print(e)
    finally:
        # --- Teardown ---
        os.remove(pdf_path)
        os.rmdir(test_dir)
        print("\nCleaned up test files.")


if __name__ == "__main__":
    asyncio.run(main())
