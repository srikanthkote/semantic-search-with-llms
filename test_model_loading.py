from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def test_model_loading():
    try:
        print("Testing model loading...")
        model_name = "microsoft/DialoGPT-medium"
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded successfully!")
        
        # Test a simple text generation
        print("\nTesting text generation...")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=50)
        result = pipe("Hello, how are you?")
        print("\nGeneration result:", result[0]['generated_text'])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
