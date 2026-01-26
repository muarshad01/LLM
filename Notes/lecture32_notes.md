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

* 25:00
  
we'll access the weights of the query key and the value uh matrices in the attention block attention layers so
26:20
that's why we need all of these five Keys which are returned by the parameter dictionary so the whole goal
26:26
of this GP load gpt2 params from the tensorflow checkpoint is to get the
26:32
parameter values uh from this checkpoint and then convert the parameter values
26:38
into this params dictionary so here you see we Define the params dictionary which is empty currently and we first
26:44
only Define the blocks keys and then we fill the blocks keys with every attention layer the feed forward neural
26:50
network the output projection head all of these are already present in the model checkpoint but we just need to put
26:56
them in the appropriate values I'm not going to explain this part of the code but because the main learning lies in
27:02
you understanding these five keys so this is how the blocks Keys is filled up
27:08
and similarly all the other Keys wte WP uh G and B they are already present
27:14
in this uh model checkpoint path so we just augment the params dictionary with
27:20
all of those keys so when you finally execute the load gpt2 params from TF
27:26
checkpoint which is mentioned over here the params dictionary will have those five Keys which I mentioned to you on
27:32
the Whiteboard this is what is happening in this piece of code and then when you finish this function you return return
27:38
two things you return return the settings dictionary which consists of the vocabulary size context length
27:44
embeding Dimension number of attention heads and number of Transformer blocks and you also return the params which is
27:50
the params dictionary consisting of the five Keys which I just showed to you on the Whiteboard I could have just skipped
27:57
this part but then I really wanted to show you the nuts and bols of how the downloading is done if you want to do
28:03
research in large language models it is very likely that you will need to download the pre-trained weights to do
28:09
uh some testing or some training and for that you really need to understand the format in which gpt2 releases these
28:16
weights if you don't understand the format and if you don't understand how to convert this format into this
28:23
parameter dictionary it will be difficult to do novel research so I hope you have understood this this part uh
28:29
now let me move back to the go uh to the Jupiter notebook and uh until now we
28:36
have reached this stage where uh right up till here where we will now what we'll be doing is that from this GPT
Downloading gpt-2 weights into Python code
28:43
download 3py this python file we'll import the download and load gpt2 function it this
28:49
function the download and load gpt2 function and then what we are going to do is that we are just going to run this
28:55
function we have to pass two things we have to pass the model size because remember that the model size can be 355
29:01
million 774 million and 1558 million also and I encourage you to experiment
29:07
with this after today's lecture is over so we put in this model size and we specify the directory so I have
29:13
specified the directory to be gpt2 so here you can see in the folder name gpt2 all of my files have been stored and
29:20
then what you can do is that you can run this piece of code so then settings dictionary as we saw we'll get the
29:26
settings dictionary and we'll get the par dictionary when you run this piece of code now when you run this as I told
29:32
you this total size of all of this is around 500 megabytes initially when I ran this code it took a very long time
29:38
on my laptop because my laptop kept crashing it was not in a good internet
29:44
area and then I moved to another place where the internet connectivity was a bit strong so here you can see I was
29:50
getting speeds of 5 225 mb per second and then this entire loading took around
29:55
5 to 10 minutes so I encourage you to SA sit in a place with a good internet connectivity and don't restart your
30:01
session or close your laptop during this time because once this is loaded the rest of the code proceeds in a very
30:07
smooth manner this is the most time consuming part of the code until this point now let's say this code is
30:14
executed after it's executed since we have loaded this tqdm Library we'll see the progress which is happening so here
30:20
you can see that I've have reached 100% in all of the different steps uh so after this code has been completed you
30:27
can inspect things you can inspect the settings dictionary and you can inspect the parameter dictionary keys so if you
30:33
print out the settings dictionary you'll see that it has keys like n vocab nctx n embed n head and N layer now as I
30:41
mentioned this this is exactly the same as the ham. Json file here it's just
30:47
being converted into a dictionary now uh and if you print the parameter dictionary Keys you will see blocks b g
30:54
WP and wte we learned about this EXA ly on the Whiteboard where we saw that the
31:00
parameter dictionary will have these five Keys wte WP blocks G and
31:06
B awesome so I hope you have understood until this part where we have actually loaded uh the gpt2 architecture right
31:15
now we have loaded all of the parameters into our laptop and the
31:21
parameters seem to be loaded correctly what we can also do is that uh we could have printed the
31:28
um parameter weight contents but that would take a lot of screen space hence we only printed the parameter dictionary
31:34
keys not its values but we can go a step ahead and look at the params dictionary
31:39
and print out the wte which is the key corresponding to the Token embedding
31:45
vector and we saw that the dimension should be 50257 rows 768 columns let's
31:51
just see if the dimensions make sense so if you if you access the params dictionary with the key wte you get this
31:58
tensor whose shapee is 50257 and 768 at least the dimensions seem to be
32:03
making sense great so these values which you see on the screen right now they are
32:09
optimized values which means that for every token the token embedding weight
32:14
Dimension encodes some semantic meaning again we should be thankful to open a for releasing the weights publicly
32:20
because they would have spent about a million dollars or even more for this pre-training awesome so as a I told you
32:28
we could have also downloaded the 355 million 774 million or 1.5 billion parameter which is this release which
32:35
gpt2 had made and you can feel free to experiment with that but we have loaded the 124 million parameter now before
32:42
moving forward one change which we'll need to do is until now when we use the GPT configuration in this lecture series
32:49
we used a GPT we used this thing called GPT config 124 million and the
32:54
configuration was almost exactly same as what's actually used in gpt2 except that
32:59
we used a context size of 256 whereas the actual context size is 1024 so we'll
33:05
need to change that so what we are going to do is that we are going to say that the new configuration is the same as our
33:11
old configuration but we'll update the context length to be 10 to4 and the
33:16
second thing which we are going to update is the query key value bias so when we trained the attention mechanism
33:23
and when we run our own llm before we have put this query key value bias to false but in gpt2 this was actually put
33:30
to true so we are also going to put this to True uh here I have added a small note that uh bias vectors are not
33:38
commonly used in llms anymore because they don't improve the modeling performance and they are not that
33:44
necessary however since we are working with pre-trained weights we need to match the settings for consistency and
33:50
that's why what we are going to do is we are going to enable the query key value bias to be equal to true and we are
33:56
going to use the context l to be 1024 so then we uh create an instance of
34:01
the GPT model class with this new configuration I just want to show you the GPT model class which we have so
34:08
that it is on the screen in case you have you coming to this lecture for the first time we have
34:15
developed a GPT model class which looks something like this yeah this is our GPT
34:20
model class uh and now the main goal which we have is how are we going to
34:26
integrate the weights which we have downloaded with the GPT model class which we have defined so let's learn about that a bit
Integrating the gpt-2 weights with our LLM architecture
34:34
so there is a specific way in which we are going to do this integration so look at the GPT model class what we are doing
34:40
currently is that we are just initializing the token embedding matrices the positional embedding matrices the Transformer blocks weights
34:48
we are initializing them to random values but now our main goal is that the
34:53
weights which we have downloaded from gpt2 and which are currently stored in this params dictionary which we have
34:59
returned we need to somehow make sure that these weights are integrated with our GPT model class and instead of these
35:06
random initializations using NM do nn. embedding we actually make the
35:11
initializations uh from the downloaded gpt2 parameters so for that we first
35:16
need to look at the Transformer block and I want to show you a couple of things in this uh Transformer block so I
35:24
just control F here and searched for the Transformer block um yeah so here's the Transformer block
35:30
okay what we are going to do in the code is that here you can see that there is a object called attention so that's a
35:37
instance of the multi-ad attention class what we are going to do is that we are going to take this object and we are
35:42
going to make sure that when you define this at object the query key and the
35:47
value matrices are assigned to the query key and the value matrices which are obtained from the parameters
35:54
dictionary uh from this dictionary over here this the attention layers from this
36:00
dictionary similarly when we look at the feed forward neural network FF object we
36:05
are going to make sure that this feed forward neural network receives values from this feed forward neural network
36:12
weights dictionary which we have in the parameters dictionary so let me again take you back
36:19
to the current code it's a bit down below but let me scroll down below so that um you understand what's really
36:27
going on one awesome so now what we are going to do as I said is that we are going to link our GPT model class with
36:33
the downloaded weights from open AI gpt2 so the way we are going to do this is
36:39
that first let's take a look at the attention block right let's take a look at the attention block and let's take a
36:45
look at the queries the keys and the values so what we are doing here is that first let's access the queries keys and
36:52
the values downloaded from open a gpt2 and the way to access it as we have already seen is that you go to the
36:59
params dictionary you go to the blocks Keys then you go to the Transformer sub Keys the hn the ATN the C ATN and the W
37:08
this is exactly how we are accessing these weights but remember these weights are Fusion of queries keys and the
37:13
values so we are going to split these along the columns and then we'll get the queries weight Matrix the keys weight
37:19
Matrix and the values weight Matrix as I told you before we are going to get the at object remember I showed you in the
37:26
Transformer block class the at object and then in that object I'm going to assign the queries the key and the value
37:34
weight equal to the qore W the Kore W and the Vore W which has been obtained
37:41
from open a gp22 that's it it's as simple as that this right here is the assignment
37:47
step and the A and assign is the function which we have defined here what this assign does is that it takes left
37:53
and right and uh it will first check whether these two values the shape is matching and if the shape is matching we
38:01
just return the right values which means the left is just assigned the value equal to the right and then we return it
38:06
if the shape does not match we it will give us an error and that means that we are not loading the gpt2 weights
38:13
correctly and not assigning them correctly so this is the part where the trans uh where the attention block query
38:20
key and the value weight matrices are updated similarly in this part the bias
38:26
is updated so it's the same as the earlier part but then W is replaced with b um to update the bias
38:34
terms now if you look at the Transformer block there are other things also there is this output projection layer which is
38:40
accessible to trans through Transformer dh- at and- c-w so what we are going to do is that
38:48
again we are going to access this output projection layer weights and we are going to assign these weights downloaded
38:54
from open a to the at. output projection weight so we are going to look at the at
38:59
object again and then we are going to assign output projection weights equal to what we have downloaded and this is
39:05
the same for weights as well as biases right then we are going to look at the feed forward neural network what I'm
39:11
highlighting on the screen right now is for the first layer which is the fully connected layer as I've shown over here
39:17
the feed forward neural network has two layers the fully connected layer and the projection layer so in the fully
39:23
connected layer what we are doing here is that we are accessing ing the weights and the biases of the fully connected
39:29
layer from the gpt2 downloaded values and then we are assigning these weights
39:35
and biases to the FF object which we saw in the Transformer block so that way the
39:40
neural network the fully connected layer weights and biases are equal to the gpt2 downloaded weights and biases now this
39:47
same thing is done for the second layer which is the projection or the output layer of the multi-layer perceptron or
39:53
the feed forward neural network and then finally we come to the last uh puzzle or
39:58
the last building block of the Transformers rather and that is the layer normalization so there are two
40:05
layer normalization the layer normalization one and Layer normalization Two and both have scale as
40:10
well as shift right so this is what's Happening Here what I'm highlighting right now we
40:16
are accessing the uh shift and the scale parameters from the gpt2 downloaded and
40:22
then we're assigning those parameters to our GPT model class and what I'm
40:28
highlighting on the screen right now is the similar process done for the second normalization layer which comes after
40:34
the attention mechanism in the Transformer block okay now when we come
40:39
out of the Transformer block you see there is another normalization layer right the final normalization layer and
40:45
it just accessible through G and the B keys so what we are doing is that we are accessing the param G and the params B
40:52
and then we are assigning our GPT model class scale and shift values to whatever is down loed from
40:58
gpt2 now you must be thinking that okay there is if I look at the architecture closely there is this final layer there
41:05
is this linear output layer and careful readers might remember that this linear output layer also is a
41:12
neural network and where do we get the dimensions where do we get the weights of these we did not download this from
41:18
GPT right so the way gpt2 was designed is that it uses this concept of weight
41:24
tying which means that the token embedding weights are used for
41:31
constructing this output head so the same token embedding weights are used for this output head layer so we don't
41:38
have to Define any new weights for this layer we recycle the same weights what's used in the token embedding weight tying
41:44
is not used these days too much but it was used in the gpt2 architecture and so we are also using the concept of weight
41:50
tying that actually brings the total parameters from 162 million to 124 million if you don't do weight time the
41:57
number of parameters will be 164 million awesome right so what we have
42:03
done here is that until now this this piece of code over here load weights into GPT this code what it does is that
42:10
it takes two values the first it takes the instance of the GPT model class and the second it takes the params
42:16
dictionary so it takes this dictionary which essentially contains this dictionary which essentially contains
42:21
all the weights of gpt2 and then it just assigns all of these weights into the GP
42:27
G PT model that's what this load weights into GPT is doing and now if you see above we have already created an
42:33
instance of the GPT model class and if you scroll even above we have already got the we have already got the params
42:40
dictionary awesome right so now we just need to call this function and that's exactly what we are doing here we are
42:46
calling this function load weights into GPT what this function does is that it takes the params values and then it uh
42:53
which are downloaded from gpt2 and then it loads into our GPT model instance
42:59
which means that our own GPT model which we have constructed from scratch is now fully ready it's fully functional to be
43:06
tested so now let's go ahead and do the most exciting thing in this lecture which you all have been waiting for let
43:12
us test our model which we used our own architecture with the gpt2 pre-train
43:19
weights and let's see what the output is great so now let's go ahead and test our model uh so here you can see that here's
Testing the pre-trained GPT2 model
43:27
the the generate function which basically generates new tokens and we are passing the model which we have
43:32
defined and that this model now had has the weights which have been downloaded from gpt2 the input token IDs are every
43:40
effort moves you and the maximum number of new tokens is 25 I have defined the temperature not too high 1.5 and the top
43:48
K is 50 which means that 50 tokens will have the chance or the opportunity to be in the among the maximum new tokens or
43:56
to be in the generated token so before we see the output for this let me show
44:01
you our performance without using gpt2 weights and here you can see that if we do not
44:08
use gpt2 weights we were getting something like this which did not make any sense at all and now what we are
44:14
going to do is that we are going to run this right now so I'm going to run this live and just to show you that once you
44:21
load the weights the running actually this code does not take too much time because we have already
44:27
we already have pre-trained weights so here you can see with the star symbol that it's running right now and now it
44:34
generated so every effort moves you toward finding an ideal new way to
44:39
practice something what makes us want to be on top of that this is incredible right this output sentence is much more
44:46
coherent than what we had obtained earlier so right now in this lecture we have built our own GPT from scratch and
44:53
it seems to be working that's incredible all all the other students which have not been through this course or who have
45:00
not seen these lectures would be just using chat GPT but now we have built our
45:05
own GPT from scratch isn't that incredible it took us a long time to get here and the code length is also um the
45:14
code length is also pretty large so you can scroll above and see how long the
45:20
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
45:57
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











