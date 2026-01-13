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


55:09
and these are the exact same steps which we will follow for other queries as well so let me take you through code right
55:16
now and let us start uh implementing the attention or let us
55:22
start calculating the these three steps for the other queries as well so as we
Coding attention score matrix for all queries
55:29
discussed we have to follow three steps the first step is to find the attention scores and remember how do we find the
55:37
attention score if we have a particular query we'll just take the do product of that with all the other input vectors so
55:44
let's say if the query is in inputs we'll take the dot product of the query
55:50
with all the other vectors in the input so one way to find the attention scores
55:55
is to just Loop through the input two times and essentially find the dot product uh I'll show you what that
56:03
actually means uh right so let me rub these these
56:09
things over here so that I can show you this one method of finding the attention scores so let's say you Loop over the
56:16
input Vector right so the first you'll encounter your then you will find the dot product between your and all these
56:24
other uh all the other inputs so that will be uh the result of
56:31
the first inner loop so what you do is that first you fix an i in the outer
56:37
loop and in the Inner Loop you go through the input entirely so when we fix an i in the outer loop it means we
56:43
fix this sarey then we go through the inner loop entirely and find these six
56:49
dot products now change the outer loop so then the outer loop changes to journey and then similarly find the dot
56:55
product between Journey and all the other vectors now change the outer loop once more so similarly we'll change the
57:01
outer loop and in each outer loop we'll go through the inner loop so that is essentially finding the dot products
57:07
which are the attention scores uh the problem with this approach is that this will take a lot of
57:15
computational time so if you look at the output tensor so this is a 6x6 and each
57:21
element in this tensor represents an attention score between two pairs of inputs so for example this 6x6 Matrix
57:29
which you just saw in the code is very similar to this here I'm showing the normalized attention scores but even the
57:35
attention scores look like the 6x6 so if you look at the first row all of those
57:40
are the dot products between the first query and all the other queries if you look at the second row all of these are
57:46
the dot products between the second query and uh all the other inputs so
57:53
this second this second row actually will be exactly same to the attention uh
57:59
scores which we had calculated earlier see because the second row is
58:05
9444 1.49 ETC so if you look at the second row here that is also 9544
58:12
1.49 because the second row represents the dot product between the second query
58:17
and all the other input vectors similarly the last row represents the dot product between the last query and
58:24
all the other input vectors okay now here we have used two for Loops
58:30
right and that's not very computationally efficient for Loops are generally quite slow and that's the
58:35
reason why matrix multiplication needs to be understood the reason I say that linear algebra is actually the core
58:42
Foundation of every machine learning concept which you want to master is this for someone who does not know about
58:49
linear algebra and matrix multiplication they'll just do these two rounds of four Loops but if you actually know linear
58:55
algebra you'll see that instead of doing this you can just take the uh multiplication of inputs and the
59:02
transpose of the inputs and you will actually get the exact same answer so what this does is that you
59:08
take the input uh input Matrix and what the input Matrix looks like is this you
59:15
take the input Matrix and then you multiply with the transpose of the input Matrix and you'll get the exact same
59:21
answer as uh you'll take you'll get the exact same answer as doing this dot product in the
59:28
for Loop format and you can verify this so if you just take the product between
59:34
inputs Matrix and the inputs transpose what we'll see is the exact same thing
59:40
as the previous answer why because when we multiply two matrices what it essentially does is it just computes a
59:47
bunch of dot products between the rows of the first Matrix and The Columns of the second Matrix which is exactly what
59:53
we are doing here in these two for Loops it just that this matrix multiplication operation is much more efficient than
59:59
using these two for Loops so Step One is completed right now we have found the


***

1:00:05
attention scores I hope you have understood why this is a 6x6 Matrix here and why each what each row represents
1:00:12
each row represents the dot product between that particular query and all
1:00:18
the other input um input embedding vectors now we'll implement the
Coding attention weight matrix for all queries
1:00:24
normalization so remember how we did normalization here we did torch. soft Max right so similarly what
1:00:32
we are going to do here is we are going to do torge do softmax of this attention scores Matrix and what this will do is
1:00:39
that it will implement the soft Max operation to each row so the first row
1:00:44
we'll do the soft Max like we learned before then the second row we do the softmax like we learned before similarly
1:00:51
the last row will do the soft Max like we learned before so if you look at each individual row you'll see that entries
1:00:57
of each row sum up to one so you can look at the second row here1 385 2379
1:01:04
it's the same U attention weights which we have got for the
1:01:10
journey query uh one key thing to mention here
1:01:15
is that what is the dim parameter over here so the dim here I'm saying minus
1:01:21
one and the reason is explained below the dim parameter in functions like like tor. softmax specifies the dimension of
1:01:28
the input function input tensor along which the function will be computed so
1:01:34
by setting dim equal to minus1 here we are instructing the softmax function to
1:01:40
apply the normalization along the last dimension of the attention score tensor and what is the last dimension of
1:01:47
the attention score tensor it's essentially The Columns so if the attention scores is a 2d tensor it's a
1:01:53
2d 6x6 tensor right and it has the shape of rows and columns uh the last
1:01:59
Dimension is the column so Dimension equal to minus one will normalize across the
1:02:05
columns so what what will happen is that for the first row look at the columns so
1:02:10
this is the first column this is the second column actually I have to show here this is the First Column this is
1:02:15
the second column this is the third column so we are normalizing essentially along the columns right because we are
1:02:21
going to take the exponent of what all is there in the First Column second column third column Etc we are going to
1:02:26
sum these exponents that's why it's very important to uh write this dim equal to minus1
1:02:34
because we are normalizing across a column um and that's why the values in
1:02:39
one row sum up to one since we are normalizing in the each column that's
1:02:45
why the values in each row sum up to one it's very important to note that so Dimension equal to minus1 means that we
1:02:51
have to apply the normalization along the last Dimension and for a two dimensional t sensor like this the last
1:02:56
Dimension is the columns so the soft Max will be applied across the columns and
1:03:02
that's why for each row you will see that all the entries sum up to one so these are the attention weights
1:03:09
which we have calculated and the last step which is very important is calculating the context vector and uh I
1:03:16
want to uh show some things to you but before that let's verify that all the rows indeed sum up to one in this
1:03:23
attention weights so what I'm doing here is that I'm looking at the second row here and I'm just going to sum up to one
1:03:30
and I'm just going to sum up the entries of the second row and you will see that uh the second row sums up to one and I
1:03:37
have also included a print statement below which prints out the summation of all the rows and you'll see that the
1:03:44
first row the second row similarly the sixth row all of the rows essentially sum up to one this means that the
1:03:50
softmax operation has been employed in a correct manner for you to explore this
1:03:55
dim further you can try with dimm equal to Z dim equal to 1 also from the errors you will learn a
1:04:02
lot these small details are very important other students who just apply llm Lang chain and just focus on
1:04:10
deployment will never focus on these Minor Details like what is this dim operator over here Etc but I believe the
1:04:17
devil always lies in the details so the students who understand these Basics will really Master large
1:04:24
language models much more than other students now we come to the final step which is essentially Computing the
Coding context vector matrix for all queries
1:04:30
context vectors right and I will take you to code but the final step is
1:04:35
actually implemented in elegant one line of code let me take you to the final
1:04:41
step before what we had done in the final step remember what we simply did was uh we just
1:04:48
uh where was that yeah in the final step what we simply did was we just
1:04:53
multiplied the attention weights for each Vector for each input Vector with
1:05:01
that corresponding Vector right I can show this to you in the Whiteboard
1:05:06
also okay so what we did for the final step of the context Vector was something
1:05:12
like this yeah so what we did was we we got the attention weights for
1:05:18
each input embedding vector and we multiplied those attention weights with each with the corresponding input vector
1:05:25
and we those up now this is exactly what we have to do for the
1:05:30
other uh for the other tokens also but we have to do this in a matrix manner
1:05:35
because we cannot just keep on looping over and use for loops and there is a very elegant Matrix operation which
1:05:42
actually helps us calculate the context vectors it's just one line of matrix
1:05:47
product essentially we have to multiply the uh attention scores Matrix or the


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


























