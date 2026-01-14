#### Implementing a Self Attention Mechanism with Trainable Weights 

* (Key, Query, Value)

* scaled dot product

* today we are going to look at a more real life situation which is mactually implemented and we are going to consider trainable weights 


***

5:00

```
your journey starts with one step
```

* we'll see how these train weight matrices are constructed and once these weight matrices are trained the model can learn to produce good context Vector for every token so at the heart of this trainable
Key, Query and Value Weight Matrices

* how to convert in input embeddings, which are the input vectors into (key, query, value) vectors
* goal here is the same as the last lecture we want to get from the input embeddings to context embeddings for every token

***

10:00

#### Three Trainable Weight Matrices
1. query weight Matrix W_q
2. key weight Matrix W_k
3. value weight Matrix Q_v

* transformation is not fixed the key the key to these Transformations are these three trainable weight matrices W_q, W_k, W_v
* parameters of these weight matrices are to be optimized

***

```python
x_2 = inputs[1] # second input element
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
query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
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

```python
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
```

***

25:00

#### Computing the Attention Score

***

30:00

```python
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
```

****

#### Attention Weights

35:00

* Problem with attion scores is that:
1. They are not interpretable. So, we do normalization!

* Normalization serves two purposes
1. It helps make things interpretable
2. it helps when we do back-propagation

* Difference between attention scores and weights ...meaning is same but attention weights sum-up to one

#### Scale by $$\sqrt{d_\text{keys}}$$
* all of these values are taken and they are scaled by something which is called as __square root of the keys Dimension__
* Scale by $$\sqrt{d_\text{keys}}$$


#### Attention weight Matrix
1. we scaled by the square root of the dimension ($$\sqrt{d_\text{keys}}$$)
2. we implement __softmax__

***

40:00

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
* Do make the variance of dot product stable
* it turns out that the dot product of Q and K increases the variance because multiplying two random numbers increase the variance. So remember that when we get to the attention scores, we are multiplying q and K right, the query and the key,
* it turns out that if you don't divide by anything the higher the dimensions of these vectors whose dot product you are taking the variance goes on increasing that much and dividing by the square root of Dimension keeps the variance close to one.

* if the variance increases a lot it again makes the learning very unstable and we don't want that we want to keep the standard deviation of the variance closed so that the learning does not fly off in random directions and the values the variance generally should stay to one that helps in the back propagation and that's also generally better for uh avoiding any computational issues.
*  so that's the reason why uh we want the variance to be close to one so this is the second reason why we especially use square root so uh there are two reasons the first reason is of course we want the values to be as small as possible this helps  if the values are not small the __softmax becomes peaky__ and then it starts giving preferential values to Keys, which we don't want it can make the learning unstable but why square root the reason

***

* 50:00

***

* 55:00

***

* 1:00

#### 3.4.2 Implementing a compact SelfAttention class

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


product of the attention weights and the values this is the last step which we saw the context vector
1:05:57
uh yeah so this was the last step which we saw the context Vector is just the product of the attention weights and the
1:06:02
values so this is how we compute the context Vector so some key things to mention
1:06:08
here in this pytorch code the self attention version one is a class derived from nn. module so nn. module uh which
1:06:16
is a fundamental building block of P torch models and that provides necessary functionalities for model layer creation
1:06:23
and management as I mentioned mentioned to you before the init method initializes trainable weight matrices
1:06:29
query key and value for queries keys and values each transforming the input
1:06:34
Dimension D in into an output Dimension D out and during the forward pass which is the forward method what we do is that
1:06:41
we compute the attention scores by multiplying queries and keys normalize the scores using soft Max and finally we
1:06:48
create a context Vector that's the last step so this is just an explanation of the code I'll share this entire code
1:06:54
file with you so you'll have this explanation don't worry so let's try to create an instance of this
1:07:00
class uh so I'm creating an instance of this class with uh two with three as the
1:07:07
input embedding Dimension and D out is equal to two so here you see I have created this
1:07:14
uh so print essay version one inputs so these are the six embedding vectors so here is the Matrix of the six context
1:07:21
vectors so directly returned so what this print statement does is that it Returns the context Vector so actually
1:07:28
when you do this uh self attention version one and you pass the input many
1:07:33
things are happening when you pass the inputs these key query value Matrix matrices are created attention scores
1:07:39
are calculated attention weights are calculated and the context Vector is calculated which is returned over here
1:07:46
so it has six embedding vectors so each row corresponds to the context Vector so the first row corresponds to the context
1:07:52
Vector for first token your second row corresponds to to the context Vector for second tokken Journey Etc similarly the
1:07:59
last row corresponds to the context token for context Vector for step so that's why the dimensions here are six
1:08:05
rows and two columns so since the inputs contain six embedding vectors we get a matrix
1:08:11
storing the six context Vector remember we have we want six uh context we want a
1:08:17
context Vector for each input embedding Vector that's the main goal which we started out in today's class and we have
1:08:23
achieved that goal over here in a very compact manner in just maybe 10 to 15 lines of code so if you have followed
1:08:30
till here it's been a pretty long lecture you should be really proud of yourself because if you have understood
1:08:35
until here I believe you have understood the core of the attention mechanism just write these Dimensions down once take
1:08:42
the dot product yourself and see how the calculations play out on a book or on a piece of paper that's the best way to
1:08:48
learn this concept I it all boils down to matrices and dimensions so as a quick check let's not
1:08:55
notice the second row 3061 8210 and let's see whether it's the same as the
1:09:01
context Vector for Journey which we have calculated earlier so that's the same so it's a good sanity check which means we
1:09:07
are in the right direction now what we can do is that we can we can actually improve this self attention version one
Self Attention Python class - Advanced version
1:09:14
further by uh changing how these are defined So currently we are using nn. parameter right the main hypothesis is
1:09:21
that why don't we use directly a NN do linear function because it automatically
1:09:27
creates the uh initializes the weight Matrix in a manner which is good for
1:09:32
computations so instead of just sampling from random values here why don't we use
1:09:38
the linear function so that the initialization is done in a proper manner using p
1:09:44
torch that's exactly what we do next so we can improve the self attention version one version one implementation
1:09:51
further by utilizing the NN do linear layers of pytorch
1:09:57
which effectively perform matrix multiplication when bias units are disabled so basically we can use nn.
1:10:03
linear to also initialize random values of query key and the value value Matrix but the main advantage is that nn.
1:10:10
linear has an optimized weight initialization scheme and that leads to more stable and effective model model
1:10:16
learning you can of course use NN do parameter also but the main advantage of
1:10:22
nn. linear is that it has a stable uh initialization scheme or rather I should
1:10:27
say more optimized initialization scheme since we always use this for all types of neural network tasks so why not
1:10:34
essentially uh use the linear layer we can just put the bias terms to false
1:10:39
because we don't need this we just need to initialize a weight Matrix with d in
1:10:44
and D out weight weight Matrix for query key and value with d in as the rows and
1:10:50
D out as the columns but we don't need the bias Matrix so you can just use the linear lay and put the bias to


***

1:10:57
false so it will initialize these weight Matrix weight matrices and that's
1:11:02
usually more common practice for implementing the self attention class when we deal with llms so similar to uh
1:11:10
here we created an instance of the self attention version one right similarly we can create an instance of the self
1:11:15
attention version two and pass in the arguments as the input Dimension and the output so here again we get a six rows
1:11:22
and two column tensor uh which is the context vectors for all the six uh input
1:11:27
embedding vectors so you'll notice that these values are different than these
1:11:33
values because the initialization schemes are different so they use different initial weights for the weight
1:11:40
Matrix since nn. linear uses a more sophisticated weight initialization scheme so the linear uses usage of nn.
1:11:48
linear leads to a more sophisticated weight initialization scheme than NN do parameter I won't go into the details of
1:11:55
how the weights are initialized in nn. linear but you can explore this further that also is an interesting topic but
1:12:01
the length of the lecture will increase further okay so that actually brings us
1:12:07
to the end of today's lecture where we implemented query key value Matrix found
1:12:12
the attention scores attention weights and the context vectors for all the input embeddings I just want to end
1:12:18
today's lecture by uh showing you a schematic which illustrates what all we have implemented in today's class
1:12:28
okay so at the end we implemented this self attention python class and uh this
One figure to visualise self attention
1:12:34
schematic actually explains everything so this is our input this is our input Matrix let me actually show it with a
1:12:41
different color so it has six uh it has six rows and three columns so let's
1:12:47
focus on the second row for now which is the input embedding for Journey so then what we do is that we first uh
1:12:54
initialize a weight Matrix for query weight Matrix for key weight Matrix for value and uh we have to specify two
1:13:01
things the input Dimension and the output Dimension the input Dimension here has to be the same as the vector
1:13:07
embedding Dimension here because we are taking a we'll take uh product between the Matrix but the output Dimension can
1:13:14
be anything generally in GPT like llms the output Dimension is the same as the input Vector Dimension but here we have
1:13:21
chosen a different output Dimension so then what we do is we multiply all the input embedding vectors with the query
1:13:28
weight Matrix the key weight Matrix and the value weight Matrix to get the queries Matrix the keys Matrix and the
1:13:34
values Matrix so remember that these three the WQ w k and WV these three are
1:13:42
the trainable weight matrics the parameters are initially initialized randomly but they are trained as the llm
1:13:50
Lars uh okay so these are the queries keys and Valu Matrix then what we do is
1:13:56
we take the queries uh we take a DOT product with the keys transposed and that gives us the attention scores which
1:14:04
are normalized to give us the attention weight Matrix so if you look at Journey For example the first value here tells
1:14:11
us the attention weight between journey and your the second value tells the attention weight between journey and
1:14:17
journey similarly the last value here tells the attention weight between journey and step so this attention
1:14:23
weight tells us how much you should attend to each word when the query is Journey similarly for all the other
1:14:30
rows then what we do is that we take the attention weight Matrix and take a product with the values take a product
1:14:36
with the values uh Matrix and then we finally get the context Vector there are
1:14:42
uh there is one context Vector for each input embedding Vector so since there are six vectors your journey begins with
1:14:49
one step the number of rows here is six the number of columns of the context Vector will will will always be equal to
1:14:56
the D out Dimension which you have chosen here for the query key and the value
1:15:01
Matrix so I believe this diagram illustrates what all we have learned so
1:15:07
far uh okay so so self attention involves the trainable weight metrix
1:15:13
metries WK WK WK WQ WK and WV these

***


1:15:19
matrices essentially transform the input data into queries keys and values which are crucial components of the attention
1:15:27
mechanism awesome now before we end this lecture I just want to tell you why uh
Key, Query, Value intuition
1:15:34
like what is the meaning behind key query and value and why are we giving these fancy terms like key query and
1:15:40
value to these so uh the simplest way to think of query is
1:15:46
that it's analogous to search query in a database so it represents the current token the model is focusing on so if you
1:15:54
ever find your s worried about what is the query just look at just think of it as the current token the model is
1:15:59
focusing on so if I say the query is key if the query is Journey I simply mean
1:16:04
that currently we are focusing on the word Journey that's it uh key in
1:16:09
attention mechanism each item in the input sequence has a key so keys are used to match with the query so even
1:16:16
Keys you can think of as items in the input sequence that's it that's the simplest way to think about
1:16:23
key uh so so the key and the query are important to get the attention uh to get
1:16:29
the attention weight or the attention score and then finally value so value represents the actual content or
1:16:35
representation of the input items themselves so once the model determines which keys are most relevant to the
1:16:41
query it retrieves the corresponding values so that's where the name comes
1:16:47
from so once we find the query we have to find which key or which word relates
1:16:53
more to the query or attends to the query that's why these are called keys like in a dictionary setting and value the
1:17:00
reason these are called values is because when we find the ultimate context Vector we use the attention
1:17:06
scores and then we use the original input embedding value so what is the representation of the input items that's
1:17:13
why this value term comes into the picture so that's the underlying reasoning behind the query key and the
1:17:21
value okay and uh in the next lecture what we'll be looking at is that we'll
1:17:26
be looking at causal attention so until now we have looked at self attention
1:17:33
right in the next lecture we'll modify the self attention mechanism so that we prevent the model from accessing future
1:17:39
information in the sequence and then after that we'll be looking at multi-head attention which is
1:17:46
essentially splitting the attention mechanism into multiple heads so the next lectures are going to be
1:17:51
interesting I know these lectures are becoming a bit long but attention is the engine of Transformers so to truly
1:17:58
understand Transformers and to truly understand large language models we have to have these lectures uh and you need
1:18:05
to write these things down which I'm teaching you you you need to write the codes which I will share with you definitely so that you develop an
1:18:12
understanding for it the lectures serve as a good starting point to cover all the concepts in a clear manner I take a
1:18:18
whiteboard approach intuition Theory and coding in a lot of detail I don't think any other videos or content explain
1:18:25
these Concepts in the level of detail which we are covering here but I believe that once you understand the detail and
1:18:31
the nuts and bolts that's when you will be confident to work on Research problems that's when you'll be confident
1:18:36
to make new discoveries in the field and I think ultimately it all boils down to matrices Dimensions dot product that's
1:18:44
it and Vector Calculus if you understand these U you'll really Master everything
1:18:49
that's that's what I believe so thank you so much everyone I hope you are liking these lectures please put your
1:18:55
comments in the YouTube uh comment section and I'll reply to them thanks
1:19:00
everyone I'll see you in the next lecture

***






