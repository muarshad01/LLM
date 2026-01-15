#### Multi-head Attention

* We are going to study two types of multi-head attention mechanisms the first type is basically by just concatenating the context Vector matrices, which is obtained from different query key and value matrices and the second approach is a more unified approach, which is more commonly implemented in modern LLMs and modern code bases. 

***

* 5:00

***

* 10:00

* What is multi-head attention?

* The term multi-head essentially refers to dividing the attention mechanism itself into multiple heads. Each head will be operating independently to give you a understanding related to code. When we get the attention scores here right now or the attention weights from one set of query key and value, we say that this is one head this is one attention head right. We are not decomposing into multiple query keys and values.

* There is only one query Matrix.
* There is only one Keys Matrix and one values Matrix.
* What multi-head attention does is that it extends the causal attention mechanism so that we have multiple heads and each of these heads will be operating independently.

* Then what we do in multi-head attention is we basically stack multiple single attention head layers together so what we'll simply do is that we will create multiple instances of the causal self attention mechanism each with its own weights and then combine their outputs.

*  It's actually very simple what the output which we had obtained before for one attention head we'll just combine them together and I'll show you how it can be done so as you might have expected this can be a bit computationally intensive but it makes llms powerful at complex pattern recognition tasks so researchers have found out that although when you stack multiple heads the computations which need to be made increase it really helps

***

* 15:00

***

* 20:00

***

* 25:00

#### 3.6.1 Stacking multiple single-head attention layers

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

***

* 30:00

#### 3.6.2 Implementing multi-head attention with weight splits

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

***

