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


* $$\{x^{1}, x^{2}, x^{3}\}$$

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


1:05:56
attention weight Matrix with the input right and we have to do some summations can you think of the matrix
1:06:02
multiplication operation which will directly give us this answer um it's fine if you don't know
1:06:09
the answer but the simplified matrix multiplication is just essentially
1:06:14
multiplying the attention weights with the inputs that's it uh this last step of finding the
1:06:21
context Vector is just taking this attention weight Matrix and multiplying it with the input Matrix and the claim
1:06:27
is that it will give us the context vectors it will give us the six context vectors which we are looking for
1:06:33
remember we need a context Vector for every token right and there are six tokens so here are the six context
1:06:39
vectors now I'm going to try to explain why this matrix multiplication operation
1:06:44
really works so let's go to the Whiteboard once more all right so this is the first
1:06:51
Matrix which we have and that's the attention weights right here
1:06:57
and this is the second Matrix which we have which is the inputs uh so keep in mind here that the
1:07:04
attention weights is a 6x6 Matrix so we have six rows and six columns and the inputs is a 6x3 matrix now we have
1:07:13
already looked at how to find the uh context Vector for the
1:07:18
second uh for the second row right which which essentially corresponds to the word journey and uh so let's see what we
1:07:26
exactly did here so the final attention Matrix will be a 6x3 matrix because so
1:07:33
sorry the final context Vector Matrix will be a 6x3 matrix because every row of this will be a context Vector so the
1:07:41
first row will be the context Vector for the first word the second row will be the context Vector for the second word
1:07:48
so let's look at the second word which is essentially the context Vector for Journey now uh if we take a product of
1:07:57
these two Matrix so let's say if we take the product of the attention weight Matrix and the input Matrix first let's
1:08:02
check the dimensions so this is a 6x6 Matrix and the inputs is a 6x3 so 6X 6
1:08:09
can be multiplied with 6x3 so taking the product is completely possible and it will result in a 6x3
1:08:15
matrix uh so let's look at the second row if you look at the second row it will be uh something like we we'll take
1:08:22
the second row uh so the second row First Column would be the dot product
1:08:29
between the second row of this and the First Column of this the second row second column will be the dot product of
1:08:34
this with the second column of this and the second row third column will be the dot product of this second row and the
1:08:40
third column here so that's what I've written here in the output Matrix so the first element of the
1:08:47
second row will be the dot product between the second row and the First Column the second element of the second
1:08:54
row will be the dot product between the uh second row and the second column and
1:08:59
the third element of the second row will be the dot product between the third row
1:09:04
of the first Matrix and the third column of the second Matrix right now when you compute these dot
1:09:11
products uh very surprisingly you will see that the answer is actually equal to
1:09:17
this the answer is 138 the answer is actually 138 which is
1:09:24
138 multiplied by the first row over here plus 237 multiplied by the second
1:09:32
row over here plus 233 which is multiplied by the third row over here
1:09:37
Etc uh it's just a trick of matricis but what it it turns out that this second
1:09:45
row second row can also be represented by this formulation where you take
1:09:52
the uh first element of this second row multiply it with the first
1:09:59
row of the input Matrix plus the second element of the second row multiply with
1:10:04
the second row of the input Matrix plus the third element multiply it with the
1:10:09
third row of the input Matrix can you see what we are essentially doing here we essentially scaling every input
1:10:15
Vector right we take the first input Vector we take the first input Vector we scale it by 138 we take the second input
1:10:23
Vector we scale it by 237 we take the third input Vector we scale it by 233 isn't this the exact same
1:10:30
scaling operation which we saw uh when we looked at the visual representation of the uh context Vector calculations
1:10:39
remember we have seen seen the scaling operation to calculate the context Vector here where we had taken each of
1:10:45
the input vectors and we had scaled it by the attention weight values and then
1:10:50
we summ them to find the final context Vector this is the exact same thing which I which we are doing over here so
1:10:56
when we take the product of the uh when we take the product of the attention weights and the inputs another way to
1:11:03
look at it is that if you look at the second row it's actually scaling the first input by 138 scaling the second
1:11:11
input by 237 dot dot dot and scaling the sixth input by 0158 so it's the exact same operation as
1:11:18
we performed before for finding the context vector and that's why finding the context vectors is as simple as
1:11:25
multiplying the attention weights with the inputs the first row of this answer will give you the context Vector for the
1:11:32
first uh input embedding Vector the second row will give you the context
1:11:38
Vector for the second token the third row of this product will give you the context Vector for the second token for
1:11:44
the third token and right up till the very end the sixth row will give you the context Vector for the final token and
1:11:51
this is how the product between the attention weights and the inputs will give give you the final context Vector
1:11:57
Matrix which contains the context Vector for all of the tokens which you are looking for and uh with this final calculation
1:12:05
we calculate the context Vector for all um of the input
1:12:11
tokens and this is exactly what I've have tried to do here so finally uh we
1:12:16
generate a tensor which is called as the all context vectors and we multiply the attention weight Matrix with the input
1:12:22
Matrix and then we get this all context Vector tensor and if you look at the
1:12:28
second row here 4419 6515 56 you'll see
1:12:33
that this is exactly the same value of the context Vector which we had obtained over here uh when we looked at Journey
1:12:41
so this again implies that whatever we are doing here uh with the matrix multiplication is leading to the correct
1:12:48
answer so based on this result we can see that the previously calculated context Vector 2 for journey matches the
1:12:55
second row in the in this tensor exactly so remember this operation to get the
1:13:00
final context Vector we just multiply the attention weights with the inputs if you did not understand the matrix
1:13:06
multiplication which I showed on the Whiteboard I encourage you to do it on a piece of paper because this last matrix
1:13:14
multiplication is very important to get the context Vector we just have to multiply the attention weight Matrix
1:13:20
with the input Matrix and what all you are learning right now will Direct L extend to the
1:13:26
key query and value concept which we'll cover when we when we come to causal attention and multi-head attention and
1:13:32
even in the next lecture in the next lecture we are going to look at this exact mechanism but with trainable
1:13:38
weights then we'll come to the concept of key query and value but these operations which we are looking at here
1:13:45
so for example uh here we are taking the matrix product between attention weights and inputs right in the key query value
1:13:52
this will be replaced this inputs will be replaced by value we'll also have a key and a query but
1:13:59
the underlying intuition and the underlying mechanism is exactly the same so if you understand what's going on
1:14:04
here you'll really understand key query value very easily okay one last thing which I want
Need for trainable weights in the attention mechanism
1:14:11
to cover um at the end of today's lecture is that okay so you might think that we already then find the context
1:14:19
vectors like this right then what's the need for trainable weights
1:14:25
we just take the dot product and we then find these context vectors the main problem with the
1:14:31
current approach is that think about how we found the attention weights to find
1:14:37
the attention weights all we did was to just take the dot product right so currently uh in our world the reason why
1:14:45
we are giving more attention to starts is that the alignment between starts and journey is maximum so the only reason
1:14:52
why we are giving more attention to starts is because because it semantically matches with
1:14:57
journey because we are only getting the attention scores and attention weight from the dot product however that is not
1:15:04
correct right because two vectors might not be semantically aligned but maybe
1:15:09
they are more important in the context of the current sentence so for example journey and one are not semantically
1:15:17
related to each other but what if in the current context one is the is the vector
1:15:23
which is more important so apart from the meaning you also need to capture the information of the
1:15:30
context right what is happening in the current sentence and without trainable weights
1:15:35
it's not going to happen we are not going to capture the context effectively right now we did manage to capture the
1:15:41
context somewhat but we only give attention to Words which are similar in meaning to the query but even if a word
1:15:49
is not similar in meaning it still might deserve attention in the context of the current sentence
1:15:55
so let's take a simple example here okay so the example is the cat sat
1:16:01
on the mat because it is warm and let's say our query is warm so in the first
1:16:07
case let's say we do not use trainable weights like what we have done in today's lecture if we don't use
1:16:13
trainable weight we only take the dot product between the query warm and each words embedding and we'll find that warm
1:16:20
is most similar to itself and maybe somewhat related to mat words like the cat and sat might have
1:16:27
low similarity scores because they are not semantically related to or so with with this so if we don't consider
1:16:34
trainable weights we'll only look at Words which are more similar to this query which is warm now with trainable
1:16:40
weights the model can learn that warm should pay more attention to mat even if mat is not semantically related to warm
1:16:47
so what will happen without trainable weights is that mat and warm might be vectors which are like this which have a
1:16:53
90° angle and they might not be related because their meaning is not related but that does not mean we should not pay
1:17:00
attention to mat because in this context probably Matt is the most important if the query is warm because the mat is
1:17:06
warm but the meaning of mat and warm are not not related right that's why we need
1:17:12
trainable weights with trainable weights the model can learn that warm should pay
1:17:17
more attention to Matt even if mat isn't semantically similar to warm in traditional embedding space so this is
1:17:24
where important the trainable weight allows the model to learn that warm often follows mat in context like this
1:17:31
one and that's how it captures long range dependencies that is the reason why we need trainable weights without
1:17:38
trainable weights this meaning would be lost and we would only be looking at Words which are similar to The query by
1:17:45
trainable weights we get more more of this information that okay Matt might not be related in meaning but in the
1:17:52
current context mat is the word which is more important because the mat is warm
1:17:57
this is how trainable weights allow us to capture context and in the next lecture we'll specifically devote the
1:18:04
next lecture to uh the simplified self attention mechanism but with trainable weights so here is the lecture notes for
1:18:12
the next lecture self attention mechanism with trainable weights we'll introduce the concept of key query value
1:18:18
and then slowly we'll move to the concept of uh causal attention and then
1:18:26
we'll move to the concept of multi-head attention so up till now we have covered simplified self attention I know this
1:18:32
lecture became a bit long but it was very important because uh I have seen no
1:18:37
other lecture or no other material which covers this much detail visually theoretically and in code about the
1:18:44
attention mechanism I could have directly jumped to key query and value which will come later but then you would
1:18:50
not have understood the meaning but this lecture allowed me to build your intuition I hope you're liking these set
1:18:57
of lectures if you have any doubts or any questions please ask in the YouTube comment section and I'll be happy to
1:19:03
reply thank you so much everyone and I really encourage you to take notes while I'm making these lectures I'll also
1:19:09
share this code file with you um thanks everyone and I look forward to seeing you in the next lecture

***




























