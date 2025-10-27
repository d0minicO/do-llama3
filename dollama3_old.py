#!/usr/bin/env python3
# Dominic Owens, November 2024
# Updated to include BOS token in the input prompt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
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

def load_llama(model_path):
    global model, tokenizer, device

    # Check if CUDA is available and set the device
    device = check_device()

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Set pad_token_id to an existing token ID if necessary
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id != 0:
            tokenizer.pad_token_id = 0
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure bos_token is set
    if tokenizer.bos_token is None or tokenizer.bos_token_id is None:
        tokenizer.bos_token = '<s>'
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<s>')

    # Update model configuration with special tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

def chat_with_llama():
    global model, tokenizer, device

    print("Welcome to the Llama Chat! Type 'exit' to quit.")

    while True:
        # User input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Format the prompt and prepend the BOS token
        input_prompt = f"""You are a helpful AI assistant called domchi.

User: {user_input}

Assistant:"""

        input_prompt = tokenizer.bos_token + input_prompt

        # Tokenize the input without adding special tokens again
        inputs = tokenizer(
            input_prompt,
            return_tensors='pt',
            add_special_tokens=False,
            padding=False
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Generate the response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and post-process the generated text
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stop at end markers
        end_markers = [[tokenizer.eos_token]] # ["User:", "Assistant:", tokenizer.eos_token]
        for marker in end_markers:
            if marker in generated_ids:
                generated_text = generated_text.split(marker)[0]
                break

        print(f"Model: {generated_text.strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Llama Chatbot')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/dowens/projects/llama/meta-llama-3.2-1b-hf',  # Set your default model path here
        help='Path to the Hugging Face formatted Llama model'
    )
    args = parser.parse_args()

    load_llama(args.model_path)
    chat_with_llama()
