import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, Style

# Ask the user for the model size
model_size = input(f"{Fore.RED}Enter the model size (1 for 1 billion parameters or 3 for 3 billion parameters): {Style.RESET_ALL}").strip().lower()

# Determine the model path based on the input
if model_size in {"1", "1b", "1bn"}:
    model_id = "/home/dowens/projects/llama/meta-llama-3.2-1b-hf"
elif model_size in {"3", "3b", "3bn"}:
    model_id = "/home/dowens/projects/llama/meta-llama-3.2-3b-hf"
else:
    print(f"{Fore.RED}Invalid model size. Please restart the script and enter a valid model size.{Style.RESET_ALL}")
    exit()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Set pad_token_id to eos_token_id if not already set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Initialize chat messages with system prompt
sys_prompt = """
    You are Domchi, an AI assistant designed to provide helpful, accurate, and conversational responses.
    Your primary goal is to assist the user by answering questions, generating ideas, solving problems, and providing clear explanations.
    
    You should always: 
    Be Polite and Professional: Maintain a friendly yet professional tone in all interactions.
    Adapt to User Needs: Adjust the complexity of explanations based on the user's expertise level, which may range from beginner to advanced.
    Stay Accurate and Up-to-Date: Provide accurate, fact-based information. If unsure, indicate uncertainty and offer ways to verify.
    Provide Context and Examples: When explaining concepts, include context or examples to enhance understanding, unless the user specifies otherwise.
    Focus on the Task: Stay on-topic and concise unless the user invites elaboration.

    When generating responses:
    Use simple, clear language unless the user prefers technical details or specialized terminology.
    Break complex topics into digestible parts.
    Clarify ambiguities by asking follow-up questions.

    Your role is to empower the user by facilitating learning, creativity, and problem-solving."""
    
messages = [{"role": "system", "content": sys_prompt}]

# Define the function for generating responses
def generate_response(messages, max_new_tokens=256):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"[System] {content}\n"
        elif role == "user":
            prompt += f"[User] {content}\n"
        elif role == "assistant":
            prompt += f"[Assistant] {content}\n"
    prompt += "[Assistant]"

    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to("cuda")
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Add attention mask here
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
        temperature=0.4,  # Lower temperature reduces randomness
        #top_k=50,  # Limit to the top 50 tokens by probability
        #top_p=0.9,  # Nucleus sampling: sum of probabilities is <= 0.9
    )
    
    # Decode and extract assistant's response
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Extract only the assistant's new response
    response = decoded_output.split("[Assistant]")[-1].strip()
    return response

# Interactive session
print(f"{Fore.RED}\n\nChat with Domchi! Type 'exit' to end the chat.\n\n{Style.RESET_ALL}")
while True:
    user_input = input(f"{Fore.BLUE}{Style.BRIGHT}").strip()
    if user_input.lower() == "exit":
        print(f"{Fore.GREEN}Bye-bye! See ya!{Style.RESET_ALL}")
        break
    # Add user message to messages
    messages.append({"role": "user", "content": user_input})
    print()  # Add a blank line to separate input and response
    # Generate response
    response = generate_response(messages)
    print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
    print()  # Add a blank line to separate input and response
    # Add assistant's response to messages
    messages.append({"role": "assistant", "content": response})