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

loss as close to zero as possible that's the goal so when we train the large language models later the goal is to
51:03
bring this categorical cross entropy loss to as low as possible before we end the lecture I
Perplexity loss measure
51:09
want to show you one last thing there is another major of loss and uh that's like
51:15
cross entropy itself but a minor modification and that's called as perplexity so perplexity actually
51:22
measures how well the probability distribution predicted by by the model matches the actual distribution of words
51:29
in the data set and so in that sense it is more interpretable and it's more
51:34
interpretable way of understanding model uncertainty in predicting the next token and I'll show you why in a minute
51:41
remember that lower per perplexity score also means better predictions so the formula for
51:47
perplexity is just to find the exponent of the loss and surprisingly this simple uh
51:54
modification leads to a lot more intuition so if the exponent of the loss
51:59
in our case is 48725 it means that the model is roughly
52:04
as uncertain as if it had to choose the next token randomly from about 48725
52:10
tokens in the vocabulary so the number of tokens in the vocabulary is 50 uh 257
52:16
I think now what this means is that if the input is uh
52:22
every effort moves right every effort moves is the input currently our llm is
52:29
at a stage that to predict the next token that's as as if we have to choose
52:35
between 48725 tokens that's pretty pretty bad right it means that there is
52:40
a lot of uncertainty in predicting the next token if the perplexity score was equal to two that's pretty good which
52:46
means that we just need to predict between two tokens so that means our llm
52:51
is very accurate but in this case the perplexity score is 48725 48725 which means that 4 48,000 tokens
53:00
are kind of equally likely to become the next token which means that our model is not good at all uh you can see how it is
53:07
more interpretable than getting a categorical cross entropy loss of equal to 10 when I get a categorical cross
53:13
entropy loss of 10 I don't really know how it relates to my vocabulary size also but if you get a uh entropy or a
53:21
perplexity score of 48725 you can kind of relate it to your vocabulary size and make interpretable
53:28
predictions like these that the 48,000 next token 48,000 tokens in our
53:33
vocabulary of 50,000 are equally likely to be the next token and that's pretty
53:38
bad because our llm is not yet trained so this brings us to the end of today's lecture where basically what we
Recap and next steps
53:46
did is we took at we uh we first started with the inputs then we started with the True
53:53
Values which were known as the targets and then what what we did was we found the output values this was kind of a
53:59
revision we got the Logics tensor we converted it into a tensor of probabilities and then we got these
54:05
output tokens so then we Tred to find the loss between the targets and the
54:11
output using the cross entropy loss and we ultimately
54:16
saw that using one single line of code torge nn. functional. cross entropy we can find the loss between the loged
54:23
sensor and the target tensor as the lecture concluded we even looked
54:28
at another way of measuring loss which was called as perplexity which is much more intuitive and the way perplexity is
54:35
calculated it just e rais to loss which is exponent of loss and I also mentioned
54:40
the perplexity concept over here in our case the loss was 10.79 and the
54:45
perplexity value was 48725 and this is usually more
54:50
interpretable awesome in the next lecture what we are going to do is that we are going to take an actual
54:56
data set from a book which is called the verdict and first we'll tokenize this data set we'll divide it into input
55:04
output input output Pairs and then we are going to get the llm output and we are going to find the loss function or
55:11
the loss value for this entire uh data set until now in this lecture we just
55:17
looked at uh in the code we just looked at two sample inputs right we looked at
55:23
um let me yeah we looked at these two inputs in the next lecture we'll be scaling this up and look at an entire
55:29
text data set so next lecture will be a lot of fun and uh we'll run the entire


***


55:34
architecture on this um Hands-On example you can replace this example with any
55:40
book which you like Harry Potter book any other book so next lecture is going to be a lot of fun I hope you all are
55:46
liking these lectures we have now started a new module which is on U the
55:52
training large language model and we are slowly making a lot of progress on in building the large language models we
55:59
have already finished stage one now we are on stage two and rapidly moving towards completion thanks a lot everyone
56:06
and I look forward to seeing you in the next lecture

***


