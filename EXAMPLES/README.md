# dLLM Examples

This directory contains example code for using dLLM with its OpenAI-compatible API.

## Quick Start Examples

### 1. Basic Chat Completion

```python
# basic_chat.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)
print(response.choices[0].message.content)
```

### 2. Streaming Responses

```python
# streaming.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

stream = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### 3. Embeddings

```python
# embeddings.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.embeddings.create(
    model="bert-base",
    input=["Hello world", "How are you?", "Goodbye"]
)

for i, embedding in enumerate(response.data):
    print(f"Embedding {i}: dimension={len(embedding.embedding)}")
```

### 4. Text Generation

```python
# text_generation.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="gpt2-small",
    prompt="The quick brown fox",
    max_tokens=50,
    temperature=0.7
)
print(response.choices[0].text)
```

## Advanced Examples

### 5. Async/Await

```python
# async_client.py
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1")
    
    response = await client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### 6. Batch Processing

```python
# batch_processing.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

responses = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "user", "content": "Question 1"},
        {"role": "user", "content": "Question 2"}
    ],
    n=2
)
```

### 7. Error Handling

```python
# error_handling.py
from openai import OpenAI, APIError, RateLimitError

client = OpenAI(base_url="http://localhost:8000/v1")

try:
    response = client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIError as e:
    print(f"API error: {e}")
```

### 8. Custom Parameters

```python
# custom_params.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
    presence_penalty=0.1,
    frequency_penalty=0.1
)
```

## Running the Examples

```bash
# Install dependencies
pip install openai numpy

# Run an example
python basic_chat.py