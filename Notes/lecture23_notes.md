#### Transformer
* Transformer block is the fundamental building block of GPT and other llm architectures.
* The transformer block is repeated 12 times in GPT-2 small (124M parameters)

#### Trransformer Block
1. Multi-ead attention
2. Layer normalization
3. Dropout
4. Feed forward layer
5. GELU activation

***

* 15:00

***

* 20:00

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

***

* 25:00

```python
# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch03 import MultiHeadAttention

from previous_chapters import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
```

***

* 30:00

```python
torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

***

* 35:00

***


* 40:00

input sequence how every token relates to the other tokens of the input sequence that's the whole idea of the
40:14
attention mechanism which we looked about before or which you understood before so the GPT architecture is the
40:20
broad level within that there is a Transformer model within that there is a attention mechanism so these three
40:26
things power each other other it starts from the attention mechanism which is a key component of the Transformer block
40:33
the and the Transformer block is the key component of the whole GPT architecture so finally this means that
40:40
while the physical dimensions of the sequence length and feature size remain unchanged as it passes through the Transformer block the content of each
40:47
output Vector is reint is re-encoded to integrate contextual information from
40:52
across the entire input sequence so this is just saying that although the input and the output dimensions are the same
40:59
the output contains a lot more information since it also contains information about how each token relates
41:05
to the other tokens in the input that actually brings us to the end
Lecture summary and next steps
41:10
of this lecture I just want to show you one thing as we are about to conclude
41:16
um I want to show you um how what all we have learned so far relates to each
41:23
other so when we looked at the GPT architecture we saw that there are four
41:29
things which are important the layer normalization the J
41:36
activation uh let me draw that again the layer normalization which is here the J
41:44
activation which is here the feed forward neural network the feed forward neural network
41:50
component and finally the shortcut connections and we also saw today how all of these four come together together
41:57
to build the entire Transformer block and we coded out this Transformer block together now in the next lecture what we
42:04
are going to see is that how the Transformer block uh leads to the entire GPT GPT
42:11
architecture so remember what I said earlier it all starts from this
42:16
attention which is the mass multi-ad attention that forms the core of the Transformer block now that we have coded
42:23
the Transformer block our task is not over because remember that after the Transformer block there are lot of
42:29
pre-processing steps and then finally we have to use the output Vector in order to predict the next next token and we
42:37
still have to see how that is done so the last step is still remaining and the last step is
42:42
the final GPT architecture but try to understand this sequence here it all
42:49
starts from the attention block then so it all starts here the magic starts here
42:55
at the attention block then that forms the core of the Transformer
43:01
block which I'm marking over here and the Transformer block essentially forms
43:06
the core of the entire GPT
43:12
architecture I hope you have got this sequence in mind so that's why for us it was very important to first spend a lot
43:18
of time understanding attention we have covered five to six lectures on that then it was very important to spend a
43:24
lot of time on every single component of this Transformer Block in the next lecture we are finally going to see uh
43:31
how the Transformer block leads or fits in the entire GPT architecture or rather
43:37
put in other words how can we pre-process the or postprocess the output from the Transformer block so
43:43
that we can predict the next word remember the whole goal of GPT style models is that given an input given an
43:50
input let's say the input is every effort moves you how to predict the next word up till now we have just seen that
43:56
the Transformer block retains the input shape right the Transformer block output but how is this converted to next word
44:03
prediction that's what we are going to see in the next lecture okay so now with the Transformer
44:10
block implemented we have all the ammunition or building blocks needed to implement the entire GPT architecture
44:17
and here you can see the next lecture which I've already planned is coding the entire GPT
44:24
model so I hope everyone you are liking this style of lectures where I first cover intuition plus theory on the
44:30
Whiteboard and then I take you to the code please follow with me and then try
44:35
to implement the code on your own try to write things on the Whiteboard Because
44:40
unless you understand the nuts and bolts of how large language model works it will be very difficult to invent
44:46
something new in this field it will be very difficult to truly be a researcher or engineer and such kind of fundamental
44:52
understanding will help you as you transition in your career as well thanks thanks everyone and I look forward to
44:58
seeing you in the next lecture

***








