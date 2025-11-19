Chatbot with Intent Recognition + Memory

A lightweight chatbot that uses sentence embeddings for intent detection, Qdrant for memory, and TinyLlama for fallback responses.

Features

Intent recognition using MiniLM embeddings

Memory storage + retrieval with Qdrant

Simple rule-based responses

TinyLlama-generated replies when intent is unclear

Small interactive command-line chat loop

How It Works

Embed user input and compare with example intent embeddings.

If intent is confident → return predefined or rule-based response.

Otherwise → retrieve relevant memories and send to TinyLlama.

Store user messages in Qdrant for future context.
