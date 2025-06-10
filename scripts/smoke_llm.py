import asyncio
from mrce.llm import ollama_client

async def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    reply = await ollama_client.chat(messages)
    print("RAW reply repr:", repr(reply))   #  <-- add
    print("Model replied:", reply)

    emb = await ollama_client.embed("The quick brown fox jumps over the lazy dog.")
    print("Embedding length:", len(emb))

if __name__ == "__main__":
    asyncio.run(main())
