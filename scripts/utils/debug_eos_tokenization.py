from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

print(f"Loading tokenizer from {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

text_pointer = "Here is the JSON<|im_end|>"
encoded = tokenizer(text_pointer, add_special_tokens=False)

print(f"Text: '{text_pointer}'")
print(f"IDs: {encoded['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")

# Check if <|im_end|> is a single token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"Expected ID for <|im_end|>: {im_end_id}")

if im_end_id in encoded['input_ids']:
    print("✅ <|im_end|> is tokenized as a SINGLE token.")
else:
    print("❌ <|im_end|> is SPLIT into multiple tokens!")
    print("This explains why the model learns to output the text string but not the EOS token.")
