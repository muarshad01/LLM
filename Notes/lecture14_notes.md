#### A Simplified Self Attention Mechanism withoug Trainable Weights

* In this section, we will implement a simple variant of the self-attention mechanism, free from any trainable weights.

```
your journey starts with one step
```

```python
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
```

1. pre-processing 
2. convert sentence into individual tokens GPT uses
3. bite-pair encoder which is a (word, sub-word, character)-tokenizer
4. convert tokens into token IDs
5. convert token ID into a vector representation called __vector embedding__
6. these vectors capture the __semantic meaning__ of each word 

* the main problem here ...how a particular word relates to other words in the sentence...
* we really need to know the __context__, i.e., which word in this sentence is closely related to other words
* how much attention should we pay to each each of these words when we look at "Journey" that's where attention mechanism comes into the picture
*  Goal of the attention mechanism is to take the __embedding Vector__ for word "Journey" and then transform it into another Vector which is called as the __context Vector__, which can be thought of as an __enriched embedding Vector__ because it contains much more information.
*  It not only contains the __semantic meaning__, which the embedding Vector also contains but it also contains information about how that given word relates to other words in the sentence. This contextual information really helps a lot in predicting the next word in LLM.

***

* One such context Vector will be generated from each of the embedding vectors
* The goal of all of these mechanisms is the same we need to convert the __embedding Vector__ into __context vectors__, which are then inputs to the LLMS.


* $$\\{x^{1}, x^{2}, x^{3}\\}$$

* $$z^{2}= \alpha_{21} \times x^{1} + \alpha_{22} \times x^{2} + \alpha_{23} \times x^{3}$$

* Alphas are also called "attention weights""attention scores"

* the goal of today's lecture or the goal of any attention mechanism is to basically calculate a context Vector for each element in the input and as I mentioned before the context Vector can be thought of as an enriched embedding Vector which also contains information about how that particular word relates to other words in the sentense

***


* we'll be finding a context Vector for each element
* query
* What are attention scores?


***

```python
query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)
```

* __Dot product__ is that fundamental operation because it encodes the the meaning of __how aligned or how far apart__ the vectors are.
* Higher the dot product higher is the __similarity__ and the attention scores between the two elements.

***

#### Normalization

```python
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
```

* Why do we really need normalization?
* the most important reason why we need normalization is because of interpretability

* The main goal behind the normalization is to obtain attention weights that sum up to one

* The second reason why we do normalization is because generally it's good if things are between zero and one.

* so if the summation of the attention weights is between zero and one it helps
in the training we are going to do __back propagation__ later so we need __stability during the training procedure__ and that's why normalization is integrated in many machine learning framework.

* It generally helps a lot when you are back propagating and including __gradient descent__.
 
 
* Both attention scores and attention weights represent the same thing.
* The only difference is that all the attention weights sum up to one whereas that's not the case for attention scores.

* There are better ways to do normalization many of you might have heard about __soft Max__ right.

***

#### Softmax

* $$\{x_1, x_2, x_3, x_4, x_5, x_6\}$$

* $$\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\}$$

* $$\text{sum} = e^{x_2} + e^{x_2} + e^{x_3}+e^{x_4}+e^{x_5}+e^{x_6}$$


* __softmax__ is preferred compared to let's say the normal summation especially when you consider extreme values so let me take a simple example right now and I'm going to switch the color to Black so that you can see what I'm writing on the screen

* One the reason why this is a problem is that when you get values like this they are not exactly equal to zero so when you are doing __back-propagation__ and __gradient-descent__ it confuses the optimizer and the optimizer still gives enough or weight some weightage to these values although we should not give any weightage to these values.
*  Softmax normalisation thing is achieved
* if you add add all of these elements together you'll see that they definitely sum up to one that's fine but the more important thing is when you look at these extreme cases if you look at 400 that will almost be like Infinity so this value when normalized will be very close to one and the smaller values when normalized using softmax will be very close to zero.

***

35:00

```python
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
```

#### Softmax PyTorch
* $$\{\frac{e^{x_1-\text{sum}}}{\text{sum}},\frac{e^{x_2-\text{sum}}}{\text{sum}},\frac{e^{x_3-\text{sum}}}{\text{sum}},\frac{e^{x_4-\text{sum}}}{\text{sum}},\frac{e^{x_5-\text{sum}}}{\text{sum}},\frac{e^{x_6-\text{sum}}}{\text{sum}},\}$$


```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
```

* Aoid __numerical instability__ when dealing with very large values or very small values.
* If you have very large values or very small values, it generally leads to numerical
instability and leads to errors, which are called __overflow errors__.

***

40:00

***

45:00

```
your journey starts with one step
```

$$\{x_1, x_2, x_3, x_4, x_5, x_6\}$$

* $$z^{2} = \alpha_{21} \times x^{1} + \alpha_{22} \times x^{2} + \alpha_{23} \times x^{3} + \alpha_{24} \times x^{4} + \alpha_{25} \times x^{5} + \alpha_{26} \times x^{6}$$

***

50:00

1. First going to compute the attention scores then 
2. we are going to compute the attention weights
3.  and then we are going to compute the context Vector

***

55:00

```python
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)
```

```python
attn_scores = inputs @ inputs.T
print(attn_scores)
```

***

1:00

```python
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
```

```python
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)

print("All row sums:", attn_weights.sum(dim=-1))
```

```python
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
```

***

* 1:05

***

* 1:10

***

* 1:15

#### Need for trainable weights in the attention mechanism

* __Sentence__: `The cat sat on the mat because it is warm`
* __Query__: Suppose our query is `warm`

* __Without Traninable Weights__: If we only use the dot product between the query "warm" 
and each word's embedding, we might find that "warm" is more similar to itself, 
and maybe somewhat similar to "mat" (if our embeddings capture that mats can be warm). 
Words like "The", "cat", and "sat" might have low-similarity score because they are not sementacially related to "warm"

* __With Traninable Weights__: With trainable weights, the model can learn that "warm" should more attention to "mat" even if "mat" is not semantically similar to "warm" in traditional embedding space. The model learns that "warm" often follows "mat" in contexts like this capturing the long-range dependencies.


***

