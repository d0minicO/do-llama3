# do-llama3

Full instructions for setting up local Llama3 inference on linux (wsl2) using Python with huggingface transformers library. 

This is using Meta Llama3 model, so requires adhering to all their licenses and terms and conditions. 

The inference is quite slow at present, will work on speeding it up in future. Not sure if this is hardware limited or whether code improvements might help too. 


## Step 1: Configure python environment

```
conda create --name llama3.2
conda activate llama3.2
conda install notebook ipykernel
```

Install necessary python packages
```
pip install transformers torch blobfile tiktoken llama-stack
```

##### Optional steps to setup environment for jupyter notebook
Add the environment to jupyter
```
python -m ipykernel install --user --name=llama3.2 --display-name "Python (llama3.2)"

# verify the environment is available in jupyter
jupyter kernelspec list
```

Ensure this environment is available within jupyter notebook by selecting it in the kernel



## Step 2: Download model
Download the desired Llama model from Meta website
https://www.llama.com/llama-downloads/

You need to register and accept the terms, then you will receive a URL containing the download key (also sent to your email).   

Download the desired model using their command.
```
llama model download --source meta --model-id  meta-llama/Llama-3.2-1B-Instruct
```

When prompted, enter your url that contains your key.   
Downloads take a while as they are large files!

```
6.0G    Llama3.2-3B-Instruct
2.4G    Llama3.2-1B-Instruct
```

**NOTE**
Alternative is to download the preconverted model from huggingface directly. You just need to wait a few mins after registering for approval. In that case, step 3 below (to convert the model for hugging face compatibility) can be skipped. 



## Step 3: Convert model weights
Need to convert model weights into a format huggingface can use (not required for -hf models directly from huggingface)

Download the conversion script from huggingface transformers github here: 
```
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
```

Run the script (make sure to specify the correct llama_version and model_size)
```
python convert_llama_weights_to_hf.py --input_dir /home/dowens/.llama/checkpoints/Llama3.2-1B-Instruct --model_size 1B --llama_version 3.2 --output_dir /home/dowens/projects/llama/meta-llama-3.2-1b-hf
```


## Step 4: Load python scripts and chat with Llama-3.2

First download the python functions from this repo
```
wget dollama3.py
```

Then enter python console and load the custom functions from the script
```
conda activate llama3.2
python
from dollama3 import load_llama, chat_with_llama
```

Load the model to the GPU (takes several minutes and uses all the RAM for larger models, 1B seems much faster!)
```
load_llama()
```

It will ask you for the path to the huggingface model, enter the full path like so
```
/home/dowens/projects/llama/meta-llama-3.2-1b-hf
```

Chat with the loaded model
```
chat_with_llama()
```
