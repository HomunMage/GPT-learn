from collections import Counter
from transformers import AutoTokenizer

# Load data from file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load input text and tokenize
input_txt = load_data('input.txt')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer.encode(input_txt)

# Count frequency of each token
token_counts = Counter(input_ids)

# Sort tokens by frequency (most frequent first)
sorted_tokens = token_counts.most_common()

# Decode top 1000 most frequent tokens
top_1000_tokens = sorted_tokens[:1000]
top_1000_decoded = [(token_id, tokenizer.decode([token_id]), freq) for token_id, freq in top_1000_tokens]

# Display the results
for token_id, token, freq in top_1000_decoded:
    print(f"Token ID: {token_id} | Token: '{token}' | Frequency: {freq}")
