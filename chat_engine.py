from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch

# -------------------------
# Suppress Transformers warnings
# -------------------------
logging.set_verbosity_error()  # Hide warnings like pad_token_id

# -------------------------
# Model Setup
# -------------------------
MODEL_NAME = "gpt2"  # CPU-friendly; swap for larger GPU models if desired
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Safety fix

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------------
# Intent Recognition Setup
# -------------------------
INTENTS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "later"],
    "ask_weather": ["weather", "temperature", "forecast"],
    "programming_help": ["python", "code", "bug", "error", "program"],
    "general_chat": []
}

def recognize_intent(message: str):
    message_lower = message.lower()
    for intent, keywords in INTENTS.items():
        if any(keyword in message_lower for keyword in keywords):
            return intent
    return "general_chat"

# -------------------------
# Conversation history
# -------------------------
conversation_history = []
MAX_HISTORY = 6  # last 6 exchanges (user + assistant)

# -------------------------
# Query Hugging Face Model
# -------------------------
def query_hf_model(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,        # Enable randomness
        top_p=0.9,             # Nucleus sampling
        temperature=0.7        # Creativity
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# Generate Response
# -------------------------
def generate_response(intent: str, message: str):
    if intent == "greeting":
        system_prompt = "You are a friendly chatbot. Respond warmly to greetings."
    elif intent == "farewell":
        system_prompt = "You are a polite chatbot. Say goodbye nicely."
    elif intent == "ask_weather":
        system_prompt = "You are a helpful chatbot. The user is asking about the weather."
    elif intent == "programming_help":
        system_prompt = "You are a programming tutor. Explain coding topics clearly and simply."
    else:
        system_prompt = "You are a general-purpose chat assistant."

    # Add user message to history
    conversation_history.append(f"User: {message}")

    # Keep only the last MAX_HISTORY exchanges
    history_to_use = conversation_history[-MAX_HISTORY*2:]  # *2 because user+assistant

    # Build full prompt
    full_prompt = system_prompt + "\n" + "\n".join(history_to_use) + "\nAssistant:"

    # Get model response
    response = query_hf_model(full_prompt)

    # Add assistant response to history
    conversation_history.append(f"Assistant: {response}")

    return response

# -------------------------
# Chat Loop
# -------------------------
def chat():
    print("🤖 Chat Engine (Hugging Face) started! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! 👋")
            break

        intent = recognize_intent(user_input)
        print(f"(Detected intent: {intent})")
        answer = generate_response(intent, user_input)
        print(f"Chatbot: {answer}\n")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    chat()
