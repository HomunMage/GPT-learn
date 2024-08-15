import tiktoken

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")  # You can use different encoding if needed

# Sample text
text = "Hello, world!"

# Tokenize the text
tokens = enc.encode(text)
print("Tokens:", tokens)

# Decode the tokens back to text
decoded_text = enc.decode(tokens)
print("Decoded Text:", decoded_text)