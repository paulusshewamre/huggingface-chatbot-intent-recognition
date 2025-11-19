# Chatbot with Intent Recognition and Memory

This project is a simple chatbot that uses sentence embeddings for intent detection, Qdrant for memory storage, and TinyLlama for generating fallback responses.

## Features
- Intent recognition using MiniLM
- Memory storage and retrieval with Qdrant
- Simple rule-based responses
- TinyLlama fallback replies
- Interactive terminal chat loop

## How It Works
1. User input is embedded and compared with stored intent examples.
2. If an intent is confidently recognized, the bot responds with a predefined or rule-based answer.
3. If not, relevant memories are retrieved and added to the TinyLlama prompt.
4. User input is stored in Qdrant as memory for future context.
