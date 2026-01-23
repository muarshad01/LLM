#### Stage_2: Foundational Model
1. Training Loop
2. Model Evaluation
3. Load Pretrained Weights

* Gradient descent back-propogation algorithms

* $$L_{BCE}=\frac{1}{N}\sum_{i=1}^{N}y_i.\log(p(x_i))))+(1-y_i).\log(1-p(x_i))$$

***

* 5:00

#### Using GPT to generate text

* Input - predicted - trarget values

***

* 10:00

#### Outputs -> Predicted (Logits Tensors)
* Inputs - GPT model - Output token IDs

1. Tokenizer convers tokens into token IDs
2. GPT model converts token IDs into logits
3. Logits are converted back to into token IDs

***

* 20:00

#### Finding the loss between targets and outputs

```python
import torch
from previous_chapters import GPTModel
# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference
```



```python
import tiktoken
from previous_chapters import generate_text_simple

# Alternatively:
# from llms_from_scratch.ch04 import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```
***

* 25:00

```python
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]
```

```python
with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
```


```python
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
```

***

* 30:00

***

* 35:00

1. Logits
2. Probabilities
3. Target probabilities
4. Log probabilities
5. Average log probabilities
6. Nagative average log probabilities

* $$L_{BCE}=\frac{1}{N}\sum_{i=1}^{N}y_i.\log(p(x_i))))+(1-y_i).\log(1-p(x_i))$$


* $$[\log(p_{11}),\log(p_{12}),\log(p_{13}),\log(p_{21}),\log(p_{22}),\log(p_{23}),)]$$
* $$Mean = \frac{\log(p_{11}) + \log(p_{12}) + \log(p_{13})+ \log(p_{21})+\log(p_{22})+\log(p_{23})}{6}$$
* $$-Mean = -\frac{\log(p_{11}) - \log(p_{12}) - \log(p_{13})- \log(p_{21})-\log(p_{22})-\log(p_{23})}{6}$$

***

* 40:00

#### Loss between output and target
* `torch.nn.functional.cross_entropy[logits-flat, targets-flat]`
* Softmax of logits
* Negative log likelihood

***

* 45:00

***

* 50:00

#### Perplexity: 
* Measures how well the probability distribution predicted by by the model matches the actual distribution of words in the data set.
* More interpretable way of understanding model uncertainty in predicting the next token.
* Lower perplexity score = better predictions
* `Perplexity = torch.exp(loss)`

***

* 55:00
  
***
