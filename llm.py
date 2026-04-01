from huggingface_hub import InferenceClient

# Initialize the client with your token
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct", token=""  # Add your Key
)


def llm(text, tokens=200):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": text}], max_tokens=tokens
    )
    return response.choices[0].message.content
