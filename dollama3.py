#!/usr/bin/env python3
# Dominic Owens, November 2024
# Updated to fix CUDA out of memory error by avoiding adding new tokens

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import contextlib
import os

# Model global variables for inference
model = None
tokenizer = None
device = None

def check_device():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_status = {
            "name": torch.cuda.get_device_name(0),
            "total_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        }
        print("Success: Device assigned to GPU.")
        print("GPU Status:")
        for key, value in gpu_status.items():
            print(f"  {key}: {value}")
    else:
        print("Success: Device assigned to CPU.")
    return device

def load_llama():
    global model, tokenizer, device

    # Interactively ask the user for the path to the Hugging Face formatted model
    model_path = input("Enter the path to your Hugging Face formatted llama3 model: ")

    # Check the GPU is available
    device = check_device()

    # Load the tokenizer and model, then move the model to the GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Set pad_token_id to an existing token ID
    if tokenizer.pad_token_id is None:
        # Option 1: Use unk_token_id if available and different from eos_token_id
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id != tokenizer.eos_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        # Option 2: Use token ID 0 if it's not eos_token_id
        elif tokenizer.eos_token_id != 0:
            tokenizer.pad_token_id = 0
        else:
            # As a last resort, set pad_token_id to eos_token_id (may still cause warnings)
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check special tokens
    print("Special Tokens:")
    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

    # # If necessary, define missing tokens
    # if tokenizer.eos_token is None:
    #     tokenizer.eos_token = '</s>'
    # if tokenizer.bos_token is None:
    #     tokenizer.bos_token = '<s>'
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = '<pad>'

def chat_with_llama():
    global model, tokenizer, device

    print("Welcome to the Llama Chat! Type 'exit' to quit.")

    while True:
        # User input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Format the prompt
        input_prompt = f"""### Instruction:
You are a helpful AI assistant called domchi.

### User:
{user_input}

### Assistant:
"""

        # Tokenize the input and get attention mask
        inputs = tokenizer(
            input_prompt,
            return_tensors='pt',
            add_special_tokens=True,
            padding=False
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        max_new_tokens = 200  # Increased to allow complete responses

        # Generate the response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and post-process the generated text
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stop at end markers
        # EOS tokens not properly working, so this is a quick fix
        # should be improved in future!
        end_markers = ["### User:", "### Assistant:", tokenizer.eos_token]
        for marker in end_markers:
            if marker in generated_text:
                generated_text = generated_text.split(marker)[0]
                break

        print(f"Model: {generated_text.strip()}")


if __name__ == "__main__":
    load_llama()
    chat_with_llama()
