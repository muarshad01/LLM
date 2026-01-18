## Layer Normalization 

#### Vanishing gradient problem
* Training DNN with many layers can be challenging due to two things as it can either lead to:
1. Vanishing gradient problem or it can lead to vanishing/exploding gradient problem or
2. unstable training dynamics
3. Layer normalization improves the stability and efficiency of NN training.
4. __Main idea__: Adjuct outputs of NN to have mean=0 and variance=1. This speeds up convergence.

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

* 20:00

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
* Leads to more flexibility and stability for distributed training / environments which lack resources.

the second batch and then when you get the mean and the variance of the resulting
35:18
output so output Ln is my resulting output and if you get the mean and the
35:23
variance you'll see that the mean for both the batches is zero and the standard deviation or the variance for
35:29
both the batches is equal to one so as we can clearly see based on the results the layer normalization code works as
35:36
expected and normalizes the value of each of the two inputs such that they have a mean of zero and variance of
35:43
one uh so I hope you have understood this concept I tried to explain this through through whiteboard as well as
35:50
through code and I hope that you have understood the basics as well as the coding aspect of it now let's see what
35:56
all we have learned in this lecture so far so in this lecture what we have learned is that we learned about layer
36:04
normalization and let's see how this fits into the context of the entire GPT architecture so to master the GPT
36:10
architecture we need to learn all of these building blocks in the previous lecture we learned about the GPT
36:16
Backbone in this lecture we looked at layer normalization in the next lecture we are going to look at this thing
36:22
called G activation then we'll look at feed forward Network and then we'll look at shortcut connections so in separate
36:29
lectures we're going to cover all of this and then you will see that all of this essentially comes together to teach
36:34
us about the Transformer block and only then we'll be fully equipped to understand the GPT
36:41
architecture I don't think there are any other YouTube videos out there which cover the GPT architecture from scratch
36:48
in so much depth but I believe this much understanding is necessary for you to truly understand how the llm
36:54
architecture works one last point which I need to mention is is that in this lecture U sometimes I said batch
Layer vs Batch normalization
37:01
normalization and then I corrected to layer normalization right if you also have this confusion remember that layer
37:08
and batch normalization are very different from each other in layer normalization we usually normalize along
37:14
the feature Dimension which is the columns and layer normalization does not depend on the batch size at all no
37:21
matter what the batch size is we just take the output of every layer and normalize it based on the mean and the
37:27
standard standard deviation batch normalization on the other hand we do the normalization for an entire batch so
37:33
it definitely depends on the batch size now U the main issue is that the
37:38
available Hardware typically dictates the batch size if you if you don't have too powerful of a hardware you might
37:45
need to use a lower batch size so batch normalization is not that flexible because it depends on the batch
37:51
size which depends which depends on the available Hardware on the other hand layer normalization is pretty flexible
37:58
it leads to more flexibility and training for it leads to more flexibility and stability for
38:04
distributed training so if you have if you are in environments which lack resources and Hardware capabilities are
38:11
not there which leads to low batches Etc or if you don't want to care about batch size you want the normalization to be
38:18
independent of the batch size layer normalization is much better in terms of flexibility so don't make the error or
38:25
don't make the confusion between layer normalization versus batch normalization this brings us to the end
38:32
of today's lecture as I mentioned in the subsequent lectures we'll talk about Jou we'll talk about feed forward Network
38:38
and we'll also talk about shortcut connections and then later we'll see how all of it comes together to make the GPT
38:45
architecture thank you so much everyone and I look forward to seeing you in the next lecture

***






















