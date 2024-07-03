#!/usr/bin/env python3
# Dominic Owens, July 2024
# Functions for loading the hugging face Llama3 model and for basic inference from a prompt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import contextlib
import os

# Model global variables for inference with
model = None
tokenizer = None
device = None

# Function to check if CUDA is available and set the device accordingly
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

# Load llama model on the GPU, this takes some time
# Interactive user input of the path to the model
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

# Suppress stdout and stderr when doing inference for anything other than the output tokens
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)

# Function to generate and print text in chunks
def generate_and_print(model, tokenizer, input_ids, chunk_size, total_length):
    generated_tokens = 0
    while generated_tokens < total_length:
        with suppress_output():
            with torch.no_grad():
                max_length = min(input_ids.shape[1] + chunk_size, total_length + input_ids.shape[1])
                output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
        
        new_token_id = output[0, input_ids.shape[1]:].unsqueeze(0)
        input_ids = torch.cat((input_ids, new_token_id), dim=1)
        generated_text = tokenizer.decode(new_token_id[0], skip_special_tokens=True)
        print(generated_text, end='', flush=True)
        
        generated_tokens += new_token_id.shape[1]

# Main inference function for llama with interactive user inputs
def infer_llama():
    global model, tokenizer, device

    # Prompt the user for inputs
    total_length = int(input("Specify how many tokens to get back from the model: "))
    chunk_size = 1 #int(input("Set the output token chunk size: ")) # uncomment here to provide chunks back greater than one token (word by word)
    input_prompt = input("Enter the input prompt to begin your inference: ")

    # Tokenize the input prompt and move the input to the GPU if available
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)

    # Print the input prompt first
    print(input_prompt, end='')

    # Generate and print text
    generate_and_print(model, tokenizer, input_ids, chunk_size, total_length)
