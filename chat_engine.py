from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch, random

logging.set_verbosity_error()

MODEL_NAME = "microsoft/DialoGPT-medium"  # much better for chat
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

INTENTS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "later"],
    "ask_weather": ["weather", "temperature", "forecast"],
    "programming_help": ["python", "code", "bug", "error", "program"],
    "general_chat": []
}

def recognize_intent(msg):
    m = msg.lower()
    for intent, words in INTENTS.items():
        if any(w in m for w in words):
            return intent
    return "general_chat"

conversation_history = []
MAX_HISTORY = 6

def query_model(prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # only keep text after the last "Assistant:"
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text

def generate_response(intent, message):
    # give each intent a small “example pre-prompt” instead of an instruction
    intent_prefix = {
        "greeting": "User: hello\nAssistant: Hey there! How’s it going?\n",
        "farewell": "User: bye\nAssistant: Goodbye! Have a great day!\n",
        "ask_weather": "User: what's the weather like?\nAssistant: Looks nice outside today!\n",
        "programming_help": "User: I have a Python bug\nAssistant: Sure, tell me the error, and I'll help debug it.\n",
        "general_chat": ""
    }.get(intent, "")

    conversation_history.append(f"User: {message}")
    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history[:] = conversation_history[-MAX_HISTORY * 2:]

    # build prompt like a real dialogue
    full_prompt = intent_prefix + "\n".join(conversation_history) + "\nAssistant:"
    response = query_model(full_prompt)
    conversation_history.append(f"Assistant: {response}")
    return response

def chat():
    print("🤖 Chatbot ready! Type 'exit' to quit.\n")
    while True:
        msg = input("You: ")
        if msg.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! 👋")
            break
        intent = recognize_intent(msg)
        print(f"(Detected intent: {intent})")
        ans = generate_response(intent, msg)
        print(f"Chatbot: {ans}\n")

if __name__ == "__main__":
    chat()
