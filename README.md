# do-llama3

Full instructions for setting up local Llama3 inference on linux (wsl2) using Python with huggingface transformers library. 

This is using Meta Llama3 model, so requires adhering to all their licenses and terms and conditions. 

The inference is quite slow at present, will work on speeding it up in future. Not sure if this is hardware limited or whether code improvements might help too. 

################################################################################
#### Step 1: Download model
Download the desired Llama model from Meta website
https://llama.meta.com/llama-downloads/

You need to register, and then will receive an email with a URL containing the download key.

Download the download.sh script from Meta's github 
```
wget https://raw.githubusercontent.com/meta-llama/llama3/main/download.sh
```

Run it
```
bash download.sh
```

When prompted, enter your url and chose the desired model(s). 
Downloads take a while as they are large files!

```
15G     Meta-Llama-3-8B-Instruct
15G     Meta-Llama-3-8B
```

**NOTE**
Alternative is to download the preconverted model from huggingface directly. You just need to wait a few mins after registering for approval. In that case, step 3 below (to convert the model for hugging face compatibility) can be skipped. 

################################################################################
#### Step 2: Configure python environment

```
conda create --name llama3
conda activate llama3
conda install notebook ipykernel
```

Install necessary python packages
```
pip install transformers torch blobfile tiktokenizer
```

##### Optional steps to setup environment for jupyter notebook
Add the environment to jupyter
```
python -m ipykernel install --user --name=llama3 --display-name "Python (llama3)"

# verify the environment is available in jupyter
jupyter kernelspec list
```

Ensure this environment is available within jupyter notebook by selecting it in the kernel

################################################################################
#### Step 3: Convert model weights
Need to convert model weights into a format huggingface can use (not required for -hf models directly from huggingface)

Download the conversion script from huggingface transformers github here: 
```
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
```

Run the script (make sure to specify the correct llama_version and model_size)
```
python convert_llama_weights_to_hf.py --llama_version 3 --input_dir /home/dowens/projects/llama/Meta-Llama-3-8B --model_size 8B --output_dir /home/dowens/projects/llama/meta-llama-3-8b-hf
```

################################################################################
#### Step 4: Load python scripts and perform inference with Llama-3-8B

First download the helpels
r functions from this repo
```
wget dollama3.py
```

Then enter python console and load the custom functions from the script
```
conda activate llama3
python
from dollama3 import load_llama, infer_llama
```

Load the model to the GPU (takes several minutes and uses all the RAM...)
```
load_llama()
```

It will ask you for the path to the huggingface model, enter the full path like so
```
/home/dowens/projects/llama/meta-llama-3-8b-hf
```

Perform inference using the loaded model
```
infer_llama()
```

It will ask you for the following:

Specify how many tokens to get back from the model: # this is the length of the response you will get back
Enter the input prompt to begin your inference: # this is the start of the prompt the model will complete