from transformers import pipeline, set_seed

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

# Take input from user
input = input(" Enter a topic or sentence to generate text: ")

# Generating paragraph
output = generator(input, max_length=100, num_return_sequences=1)

# To display result
print("\n Generated Text:\n")
print(output[0]['generated_text'])
