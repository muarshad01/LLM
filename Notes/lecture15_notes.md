#### Self Attention Mechanism with Trainable Weights 
* (Key, Query, Value)
* Scaled dot product

* $$\text{Attention}(Q,K,V)=Y=\bigg(\frac{QK^T}{\sqrt{d}}\bigg)V$$

***

5:00

```
your journey starts with one step
```

* We'll see how these __train weight matrices are__ constructed and once these weight matrices are trained the model can learn to produce good __context Vector__ for every token.
  
* How to convert the input embeddings, which are the input vectors into (key, query, value) vectors?
* Goal here is the same as the last lecture we want to get from the __input embeddings__ to __context embeddings__ for every token.

***

10:00

#### Three Trainable Weight Matrices
1. Query weight Matrix $$W_q$$
2. Key weight Matrix $$W_k$$
3. Value weight Matrix $$Q_v$$

* transformation is not fixed.
* parameters of these weight matrices are to be optimized

***

```python
x_2 = inputs[1] # second input element (2nd row, i.e. "journey")
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
```

```python
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

```python
query_2 = x_2 @ W_query # x_2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

print(query_2)
```

***

15:00

```python
keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
```

***

20:00

```python
keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
```

***

* 25:00

#### Computing the Attention Score

***

* 30:00

****

#### Attention Weights

* 35:00

* $$Q.K^{T}$$

```python
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
```

* Problem with attion scores is that:
1. They are not interpretable. So, we do normalization!

#### Normalization serves two purposes
1. It helps make things interpretable
2. It helps when we do back-propagation

* Difference between attention scores and weights ...meaning is same ...but attention weights sum-up to one

#### Scale by $$\sqrt{d_\text{keys}}$$
* all of these values are taken and they are scaled by something which is called as __square root of the keys Dimension__
* Scale by $$\sqrt{d_\text{keys}}$$


#### Attention weight Matrix
1. We scaled by the square root of the dimension ($$\sqrt{d_\text{keys}}$$)
2. Then we implement __softmax__

***

* 40:00

#### Reason-1
* For Stability in Learning

```python
d_k = keys.shape[1]...-1???
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
```

***

#### Reason-2: 
* For Stability in Learning
* Make the variance of dot product stable
* It turns out that the dot product of Q and K increases the variance because multiplying two random numbers increase the variance. So remember that when we get to the attention scores, we are multiplying Q and K right, the query and the key,
* it turns out that if you don't divide by anything the higher the dimensions of these vectors whose dot product you are taking the variance goes on increasing that much and dividing by the square root of Dimension keeps the variance close to one.

* if the variance increases a lot it again makes the learning very unstable and we don't want that we want to keep the standard deviation of the variance closed so that the learning does not fly-off in random directions and the values the variance generally should stay to one. It __helps in the back propagation__ and that's also generally better for uh avoiding any computational issues.

*  so that's the reason why uh we want the variance to be close to one so this is the second reason why we especially use square root so uh there are two reasons the first reason is of course we want the values to be as small as possible this helps  if the values are not small the __softmax becomes peaky__ and then it starts giving preferential values to Keys, which we don't want it can make the learning unstable but why square root the reason

***

* 50:00

***

* 55:00

***

* 1:00

#### 3.4.2 Implementing a Compact Self-Attention class

```python
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```

***

* 1:05

```python
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```

***

* 1:10

***

* 1:15

***

* 1:15

* \text{Attention}(Q,K,V)=Y=\bigg(\frac{QK^T}{\sqrt{d}}\bigg)V

#### Why do we use the Terms: Key, Query, Value (K, Q, V)
* __Query__: Analogous to search query in a database. It represents the current token the model focuses on.
* __Key__: In attention mechanism, each item in input sequence has a key. Keys are used to match with query.
* __Value__: Value represents the actual content or representation of the input items. Once the model determines which keys (which part of the input) are most relevant to the query (current focus item), it retrieves the corresponding values.

#### Next
* We'll modify the self attention mechanism so that we __prevent the model from accessing future information in the sequence__
* We'll be looking at multi-head attention, which is essentially splitting the attention mechanism into multiple heads.

***





