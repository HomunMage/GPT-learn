from transformers import AutoTokenizer

# Load a tokenizer for a specific model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Sample text
text = "Beautiful, there stood a mysterious and towering tree known as the Whispering Oak. And there is a beauty girl. "

# 1. Tokenize the text to get token IDs
token_ids = tokenizer.encode(text, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Convert token IDs to tokens
print("Tokens:", tokens)
print("Original Token IDs:", token_ids)

# 2. Decode the token IDs back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)

# 3. Re-encode the decoded text back to token IDs
reencoded_token_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
print("Re-encoded Token IDs:", reencoded_token_ids)

# Check if the original token IDs match the re-encoded token IDs
print("Do the original and re-encoded token IDs match?", token_ids == reencoded_token_ids)
