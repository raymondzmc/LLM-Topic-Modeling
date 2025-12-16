import os, json
from openai import OpenAI
from settings import settings


def get_openai_embedding(texts, model="text-embedding-3-small", batch_size=100):
    client = OpenAI(api_key=settings.openai_api_key)
    embeddings = {}
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(input=texts[i:i+batch_size], model=model)
        embeddings.update({text: embedding.embedding for text, embedding in zip(texts[i:i+batch_size], response.data)})
    return embeddings
