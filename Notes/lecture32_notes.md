## Loading Pre-Trained OpenAI Weights

* [GPT-2: 1.5B release](https://openai.com/index/gpt-2-1-5b-release/)
* [Kaggle.com](https://www.kaggle.com/datasets)

***

* 5:00

```python
# pip install tensorflow tqdm
```

```python
print("TensorFlow version:", version("tensorflow"))
print("tqdm version:", version("tqdm"))
```

```python
# Relative import from the gpt_download.py contained in this folder

from gpt_download import download_and_load_gpt2
# Alternatively:
# from llms_from_scratch.ch05 import download_and_load_gpt2
```

```python
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

```python
print("Settings:", settings)
```

```python
print("Parameter dictionary keys:", params.keys())
```

```python
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
```


```python
# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval();
```

```python
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

```

***

* 20:00

#### Paremeter, Dictonary Keys:
1. wte: Token Embeddings (50,257 X 768)
2. wpe: Positional Embeddings (1,024 X 768)
3. Blocks (12 Transfomer Blocks)
4. g: Final norm scale
5. b: Final norm shift

* LayerNorm-1, LayerNorm-2 (Scale + Shift)

***

#### Within Blocks we are gettign 4 things

* Attention layers in Transfomer blocks:
    * `transformer/h0/attn/c_attn/w` (c_attn, which is K,Q,W weights); (similarly for h1,...,h12)
    * `transformer/h0/attn/c_attn/b (biases)`                 (similarly for h1,...,h12)

* Feedforward NN weights in Transfomer blocks:
    * `transformer/h0/mlp/c_fc/w`    (similarly for h1,...,h12); MLP: Multi-layer perceptron
    * `transformer/h0/mlp/c_fc/b`    (similarly for h1,...,h12)
    * `transformer/h0/mlp/c_proj/w`  (similarly for h1,...,h12); Projection layer
    * `transformer/h0/mlp/c_proj/b`  (similarly for h1,...,h12)

* Output projection layers in transfomer block:
    * `transformer/h0/attn/c_proj/b` (similarly for h1,...,h12)

* Layer normalization:
    * `transformer/h0/ln_1/c_proj/g` -- layer norm scale
    * `transformer/h0/ln_1/c_proj/b` -- layer norm shift
    * `transformer/h0/ln_2/c_proj/g` -- layer norm scale
    * `transformer/h0/ln_2/c_proj/b` -- layer norm shift

***

* 30:00

```python
from gpt_download import download_and_load_gpt2
```
  
```python
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

***

* 35:00


```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
```


```python
import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
    
load_weights_into_gpt(gpt, params)
gpt.to(device);
```

***

* 40:00

#### Weight Tying
* Sam token embedding weights are used for output head layer.
* 162 Million --> 124 Million

```python
load_weights_into_gpt(gpt, params)
gpt.to(device);
```


```python
torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```


***


code has become but it's all worth it because we have learned how to build GPT from scratch now what you can do is you
Research explorations and next steps
45:26
can go ahead and do your own research so for example if you want to change the temperature value to 10 I know that this
45:33
is not good but I want to see the effect of a higher temperature value we cannot do this using chat GPT right but now
45:39
since we have built our own GPT we can explore with so many things so here you see I've increased the temperature value
45:45
to 10 and let's see the output text which is coming right now ideally it should be a bit random because
45:51
increasing the temperature increases the entropy now see the output every effort moves you towards finding an ideal new


***

* 45:00
  
set piece but only at times or for hours I was working on my first game called G
46:03
so as expected increasing the temperature has given me random outputs but you see now this opens the door to
46:09
so much more creativity hyperparameter tuning even you can do research such as small language models what if you want
46:16
to change the architecture you can now easily go ahead and change the architecture right all
46:21
you need to do is that you need to go to this GPT model uh once you have understood the the code once you have
46:27
seen the previous lectures you need to go to the GPT model which we have defined so here's the GPT model right
46:34
over here and then you can add or subtract a few layers what if you we don't need 12 Transformer blocks what if
46:40
you want to test with a smaller language model all of these experiments now remain open to you so this lecture
46:46
series is also the pathway for you to become a large language model researcher or a machine learning researcher because
46:52
now you have something which works on your local computer and runs in a fast manner you can do iterations you can
46:58
test so if you want to test the effect of top K if you want to test the effect of Maximum new tokens if you want to
47:04
vary the number of attention heads the number of Transformer blocks the optimizer size you can even vary the
47:09
optimizer step size so for example we have used Adam rate with the learning
47:15
rate of 5 into 10us 4 weight DK of 0.1 you can even change this and check the
47:22
output all of this is now accessible to you so that's why I believe that we have achieved a significant Milestone
47:28
completing this lecture if you have come to this lecture for the first time without watching the other videos in
47:34
this lecture series uh it's amazing but now please go back and try to watch all
47:40
the other videos to master your Concepts to become a very powerful llm engineer
47:46
who is creating new Norms who is really doing Cutting Edge research you need to know the nuts and bolts of how the
47:52
modeling process works and I hope that through this whiteboard approach
47:57
and through this coding approach you are getting exposed to those nuts and bolts I've not seen any other content out
48:04
there like this currently that's why it's very hard for researchers to even download or load publicly available
48:10
weights so I'm trying my level best to teach you every single thing by not making short videos but by making longer
48:17
format videos like this one and now we are confident that we have loaded the weights correctly why
48:23
because the model is producing coherent text so again let me switch the temperature to
48:28
0.1 um and let me run this again so now I'm running this and as you see earlier
48:34
when the temperature was I think it was not 0.1 it was 1.4 we need to run this again for 1.4
48:41
but if you see for 0.1 uh again the model is quite good but
48:47
even I'll do for 1.4 because we started with that condition earlier so we are confident
48:53
that the model weights are uh loaded correctly because the model can produce coherent
48:59
text at least the words make sense a tiny mistake in this process would cause the model to fail now in the next
49:05
chapters what we are going to see is that now that we have mastered pre-training we are going to look at fine tuning so the llm subject does not
49:12
end at pre pre-training after getting this model let's say if you want to build a text classifier let's say if you
49:19
want to build an educational quiz app which is a very specific application how can you use the pre-train weights how
49:25
can you fine tune these weights so that it's specific to the application which you are building we are going to learn
49:31
about this in the next lecture so all the next lectures are going to be very interesting since they are going to be
49:36
application oriented and for all of those we are going to use this GPT model which we ourselves have built not
49:42
relying on any other GPT model and that gives us a lot of confidence as a large language model or a machine learning
49:49
engineer so thanks everyone for this lecture this was a a bit of a long lecture and also a dense lecture but I
49:55
hope you understood everything which I was trying to teach uh please uh try to
50:01
reach out or comment if you have any doubts or any questions and I'll be happy to discuss this further and I'll
50:07
also be happy to see what all research you have worked on by using this code file which I'll share with you thanks a
50:13
lot everyone and I look forward to seeing you in the next lecture

***
