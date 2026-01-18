## Layer Normalization 

#### Vanishing Gradient Problem
* Training DNN with many layers can be challenging due to two things:
1. It can lead to vanishing / exploding gradient problem or
2. Unstable training dynamics
3. Layer normalization improves the stability and efficiency of NN training.
4. __Main idea__: Adjuct outputs of NN to have $$\mu=0$$ and $$\sigma^2=1$$. This speeds up convergence.

***

* 5:00 

#### Gradient depends on layer output
1. If the layer output is too large-or-small, gradient magnitudes can become too large-or-small. This affects training. __Layer normalization__ keeps gradients stable.
2. As training proceeds, the inputs to each layer can change __(internal covariate change)__. This delays convergence. __Layer normalization__ prevents this.

***

* 10:00

* $$\\{x_1, x_2, x_3, x_4\\}$$

* $$\sigma^2 = \frac{1}{4}\[(x_1-\mu)^2 + (x_2-\mu)^2 + (x_3-\mu)^2 + (x_4-\mu)^2\]$$

* $$Normalized = \[\frac{(x_1-\mu)}{\sqrt{\sigma}}, \frac{(x_2-\mu)}{\sqrt{\sigma}}, \frac{(x_3-\mu)}{\sqrt{\sigma}}, \frac{(x_4-\mu)}{\sqrt{\sigma}}\]$$

***

* 15:00

5. In GPT-2 and modern Transformer architectures, layer normalization is typically applied before and after the multi-head attention module and before the final output layer.

***

* 25:00

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

***

* 30:00

```python
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
```

```python
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)
```

***

* 35:00

* Available hardware dictates batch size.

#### Layer versus Batch Normalization
* Normalize along feature dimension, independent of batch size.
* Leads to more flexibility and stability for __distributed training__ / environments which lack resources.

***

