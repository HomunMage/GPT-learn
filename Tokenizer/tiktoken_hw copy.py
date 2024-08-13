from transformers import AutoTokenizer

# Load a tokenizer for a specific model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # You can use different models

# Sample text
text = "Hello, world!"

# Tokenize the text
tokens = tokenizer.encode(text, add_special_tokens=False)
print("Tokens:", tokens)

# Decode the tokens back to text
decoded_text = tokenizer.decode(tokens)
print("Decoded Text:", decoded_text)
