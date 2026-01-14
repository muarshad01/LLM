#### implementing a self attention mechanism with trainable weights 


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

#### Scale by $$\sqrt{d-\text{keys}}$$
* all of these values are taken and they are scaled by something which is called as __square root of the keys Dimension__
* Scale by $$\sqrt{d-\text{keys}}$$

*  So currently the dimension of the keys is a it's two two Dimension right because remember the keys queries and the values Matrix we uh

* step remember these weights are not optimized but when they are optimized we can make interpretable statements like these all of the rows will sum up to one you can check these and this is called as the attention weight Matrix this is one of the most important step uh in getting to the context Vector please remember this step and uh we did two things here we scaled by the square root of the dimension and then we implement soft Max now let's go go to code and let's implement this in the process we'll also understand why we do the Coding attention weights

***

40:00

scaling by square root of D
40:10
okay yeah so we compute the attention weights by scaling the attention scores and using the soft Max function the
40:17
difference to earlier when I say earlier it's the previous lectures is now we scale the attention scores by dividing
40:23
them by the square root of the embedding dimension of the keys and that embedding Dimensions is two so we are dividing by
40:30
square root of two so let's do the same thing here so we have this attention scores 2 so it's
40:37
a 6x6 Matrix which we have printed out here actually currently I'm just
40:42
employing this on the attention scores for the second query okay no problem so attention scores 2 is the attention
40:49
scores for the query 2 which is Journey so it's actually one row and six columns so what I'm going to do is that I'm
40:55
going to take this attention scores for Journey I'm going to first divide it by the square root of the keys Dimension
41:04
and then uh what I'm going to do is that I'm going to apply soft Max the reason we do dim equal to minus one is that
41:11
because we have to sum over all the columns and all the if so that's why when you look at one row you'll see that
41:17
it sums up to one so two things are important this D of K is the dimension it's the keys do shape minus one because
41:24
we are looking at the column remember keys do shape is 3x 2 so when you do
41:30
keys do shape and index it by minus one the result will be two so we are going to uh divide by square root of two uh
41:37
and in Python remember here we are exponen by 05 so into into .5 means rais
41:44
to.5 that is the same as dividing by the square root of two so every element will be first
41:51
divided by the square root of two in the attention score and then we implement the soft Max so if you look at the
41:58
attention weights for Journey you'll see that these are the attention weights let's actually check whether these are
42:04
correct to what we saw yeah so let's look at the second row here the second row is 0.15 2264 etc for Journey and
42:13
here you will see that the second the output is exactly the same that's a good sanity check and you'll see that all of
42:20
this sum up to one so this is how we calculate the attention weights for one
42:26
uh query and similarly we calculate the attention weights for all the queries so if I just replace this with attention
42:32
scores which is the 6x6 we'll get the attention weight Matrix which is a 6x6
42:38
Matrix okay now let's come to the question which all of you might be thinking and I don't think this is
Scaling by square root of key dimension
42:43
covered enough in other lectures and other videos but it's a very fascinating thing I took some time to understand
42:50
this and I've come up with two reasons why we actually divide by the square root of Dimension the first reason is
42:56
stabil in learning and let me illustrate this with an example so let's say if you
43:02
have this tensor of values which is 0.1 min-2 3 -2 and .5 okay if you take the
43:11
soft Max of this uh versus now let's say if you multiply this with eight and then you
43:17
take a soft Max so you'll see that the soft Max of the first is kind of it's
43:23
good right it's diffused these values are diffused between 0o and one but if if you look at the soft Max of the
43:28
second tensor you'll see that the values are disproportionately high which means
43:34
that if there are some values in the original tensor if some values are very high and when you take the soft Max
43:41
you'll get such kind of peaks in the softmax output I've actually explained this better here so the softmax function
43:48
is sensitive to the magnitude of its inputs when the inputs are large the difference between the exponential value
43:55
of each input becomes much more pronounced this causes the softmax out output to become pey where the highest
44:02
value receives almost all the probability mass and we can check it here so when we multiply with uh8 you'll
44:09
see that this has the highest value which is four and when we take the soft Max you'll see that the value is 08 here
44:18
which is much higher than all the other values in fact it's around 10 to 15 times higher that's what is meant by
44:23
softmax becomes pey if the values inside the soft Max are very large so we don't
44:29
want the values inside the soft Max to be very large and that's one reason why we scale or divide by the square root to
44:37
reduce the values itself before taking soft Max we make sure that the values are not large and that's why we divide
44:43
by the square root Factor so in attention mechanisms particularly in Transformers if the dot
44:50
product between the query and the key Vector remember that we are ultimately applying soft Max on the dot product
44:55
between query and key right because attention scores are just dot product between query and key and if the dot
45:01
product becomes too large like multiplying by eight in the current example which we saw the
45:07
attention scores can become very large and we don't want that this results in a very sharp softmax distribution and uh
45:15
such sharp softmax distribution can become so the model can become overly confident in one particular key so in
45:22
this case the model has become very confident in this fourth key or rather this uh fifth key we don't want that



***


45:30
because that can make learning very unstable that's the first reason why we
45:35
divide by square root to make sure that the values are not very large and to have stability in learning but still so
45:42
I I knew this reason but then I was thinking but why square root why are we dividing by square root why why not just
45:49
only the dimension what is the reason behind dividing by square root and then I came across a wonderful justification
45:56
for this so so the reason for square root is that it's actually related to the
46:01
variance uh so it turns out that the dot product of Q and K increases the
46:06
variance because multiplying two random numbers increase the variance so remember that when we get to the
46:12
attention scores we are multiplying q and K right the query and the key it
46:17
turns out that if you don't divide by anything the higher the dimensions of
46:22
these vectors whose dot product you are taking the variance goes on increasing that much and dividing by the square
46:29
root of Dimension keeps the variance close to one let me explain this also with an example so let's say we have a
46:37
query Vector which is generated randomly and a key Vector which is generated
46:43
randomly uh okay and currently let's say I'm doing a five dimensional Vector so
46:48
let's say I have a key Vector five dimensional key Vector which is sampled from a normal distribution and a five
46:54
dimensional query Vector sampled from a normal distribution and then I'm taking a DOT product between the query and the key
47:01
and then I'm also in the second case taking dividing by the square root of the dimension okay and I'm doing this
47:08
thousand times so that I can get a distribution over the dot product so after I do this a thousand times what I
47:14
do is I plot the variance before scaling and I plot the variance of the dot product after scaling so the results are
47:21
surprising if the dimension is equal to five the variance of the dot product before scaling is actually very close to
47:27
five if the dimension before scaling is 20 the variance before scaling is very
47:32
close to 20 this indicate that if the dimensions of the query and key vectors
47:38
go on increasing and if you don't scale then the variance of the resulting dot
47:43
product scales proportionately so if you have 100 dimensional key and query Vector the variance before scaling will
47:50
be close to 100 and we can actually test this out so here let me do this 100 and
47:56
compute variance 100 so now I'm printing this for 100 and
48:02
let me print this
48:16
out okay I think I should replace this also with 100 uh and let me print this
48:24
out okay so this is exactly what we are predicted right so the variance before scaling in this case is
48:30
107 uh see so as the dimensions increase the variance increases now look at the
48:37
power of scaling when you scale by the square root so see here we are scaling
48:42
by the square root when you scale by the square root of Dimensions no matter how much you increase the dimension if you
48:48
see the variance after scaling the variance is always close to one and that's the reason why square root is
48:54
used if you don't use a square root the variance will not be close to one so let me actually not use the square root here
49:02
and let me do it directly uh if you do it directly then you will see that the variance after
49:07
scaling are some random values they are not close to one having the square root actually
49:13
really uh having the square root make sure that even if the dimensions
49:18
increase the variance after scaling remains close to one of the dot product between the query and the key and this
49:24
is very important uh the reason why the variance should be close to one is that if the variance
49:30
increases a lot it again makes the learning very unstable and we don't want that we want to keep the standard
49:37
deviation of the variance closed so that the learning does not fly off in random directions and the values the variance
49:44
generally should stay to one that that helps in the back propagation and that's
49:49
also generally better for uh avoiding any computational issues so that's the
49:55
reason why uh we want the variance to be close to one so this is the second reason why we especially use square
50:02
root so uh there are two reasons the first reason is of course we want the values to be as small as possible this
50:09
helps uh if the values are not small the soft Max becomes pey and then it starts
50:15
giving preferential values to Keys which we don't want it can make the learning unstable but why square root the reason
50:21
why square root is because specifically when you take the dot product between query and key to find the attention
50:27
and if you don't scale as the dimensions of the query and key increase the dot product variance can become huge we
50:33
don't want that because again that will make learning unstable so scaling by the square root makes the variance close to
50:41
one so if you see after scaling it keeps the variance close to one and that's why we divide by the square root it's very
50:47
important for you to have this understanding and not many people have this understanding but I hope I've
50:52
clarified this um concept to you and you have appreciated why we are dividing by square root that's why this is also
51:00
called as scaled dot product attention because we scale by the square root okay so now until now we have
Calculating context vectors
51:08
reached a stage where we have essentially computed the attention weights and now we essentially come to
51:13
the last step which is now we are ready to compute the context
51:18
Vector so let's go ahead and actually compute the context Vector but first
51:24
what I want to do is I want to show you uh pictorially what all we have done until now so let's see what all we have
51:31
done until now is that let's say if you focus on a particular query we have found the attention score between the
51:38
query and all the input keys by taking a DOT product between the query and the
51:43
keys so the attention scores are shown in the blue over here and then what we do is we divide by the square root of
51:50
the key Dimension and then we normalize using soft Max and then we have found the attention weights
51:57
uh and the attention weights sum up to one awesome so we have reached this stage and the final step essentially is
52:03
to compute the context vectors so let's come to that right now so until now you
52:09
might be thinking that I've used the key and the query but what about the value
52:14
why did we even get the value Matrix the value Matrix will be useful in the final
52:19
step so remember for every input embedding Vector we have also calculated the value Vector so the way the context
52:26
text Vector is now found out is that we have the attention weights right so we just so for the first input embedding we
52:32
multiply the value Vector with the first attention weight we multiply the second value Vector with the second attention
52:38
weight similarly we multiply the last value Vector with the last attention weight and we are going to add all of
52:44
these together and this is going to give us the final context Vector it's very similar to what we did
52:50
earlier remember earlier we did not have this value Vector the value Vector was just the input embedding vector
52:57
but the whole Essence here is that now we have calculated the attention weight so now it's time to assign the weightage
53:03
assign the corresponding weightage to each input embedding vector or the value vector and sum them up to give the
53:09
context Vector I'll show you intuitively what this means in a
53:15
minute uh but let me take you to this whiteboard right now so that I can
53:20
show you the next step okay so we have calculated the attention weights now and now we'll be calculating the cont
53:26
context Vector so let me show you mathematically how we compute the context Vector so we have these
53:32
attention weights which is a 6x6 Matrix and we have this values which you computed at the start of this lecture we
53:38
have this value matrix it's a 6x2 matrix right so the first row are the values
53:44
for your the second row are the values for Journey similarly the sixth row is the value for step now let's say we look
Context vectors visually explained
53:52
at Journey uh and I want to find the context Vector for Journey and let's
53:59
look at the attention weights for Journey it's the second row let me show you intuitively how you find the context
54:06
Vector for Journey and you'll never forget this after I show the illustration okay so let's say these are
54:13
the uh value vectors for the different tokens so this is the value Vector for
54:18
Journey and let's say this is uh 3951 and 1 the way we do
54:25
the uh context Vector calculation is that let's look at the attention weights
54:31
so the attention weights for the journey are this second row which means 0.15
54:36
2264 Etc so this means that I'm paying 15% attention to
54:42
your I am paying 22% attention to journey I am paying 22% attention to
54:50
begins I'm paying only 13% to width I'm paying only 9% to 1 and I'm only paying
55:00
18% to step okay how do I encode all of this information to find the context
55:06
Vector it's pretty simple you take the your vector and you multiply it by5
55:13
because it only contributes 15% you take the journey Vector you multiply it by 22
55:19
because it only contributes 22% you take the begins Vector you multiply it by
55:24
2199 because that also contributes only 22% similarly you take the one vector
55:30
you multiply it with 0.09 because it contributes very less you take the step
55:35
Vector you multiply it by8 because it only contributes 18% and then you add
55:41
all of the small contributions together to give you the context Vector for Journey let me show you how this looks
55:47
like so your the attention for your is how much 15 right so you will scale you
55:55
will scale the your Vector let me show this with a different color you'll scale the your vector
56:01
by5 the attention score for Journey was 22 so you'll scale it by 22 for starts
56:08
was also 22 and for width let's see how much it was for width for width it
56:14
was3 so for width and one it was very low so for width and one they they make
56:20
very less contributions for step it was around5 so now you have the six vectors
56:25
and you will add add all of these six vectors together to give you the context Vector for
56:31
Journey so when you add all of the six vectors together it will give you the context Vector for Journey if you have
56:38
this kind of a visual representation in mind you will never forget what context Vector means now do you understand why
56:44
the context Vector is richer than just the input embedding Vector if you just look at the input embedding Vector for
56:50
Journey it has no information about how much attention should be paid to your step one
56:56
withd and starts but now if you since you have this attention weight Matrix
57:02
since you have this attention weight Matrix over here you exactly know how much relative importance should be paid
57:07
to each of the other words so you scale the other vectors by that much amount and then you add all the vectors
57:13
together to get the context Vector so the context Vector is an enriched Vector it contains the semantic meaning of
57:20
Journey plus it also contains how all the other words attend to Journey remember none of these rates are
57:26
optimized right now we are we have just initialized them randomly but when the llm is trained all of these context
57:32
vectors will be perfectly optimized so you would know that in that particular sentence in that particular paragraph
57:38
which word uh should Journey pay most attention to now this exact thing which I've shown
Context vector mathematical formula
57:45
you in uh in the graphical format can be computed in Matrix if you just multiply
57:51
the attention weights with the values so if you multiply the attention weights with the values your multiplying a 6x6
57:57
Matrix with a 6x2 so of course the matrix multiplication is possible and the resultant will be a 6x2 matrix like
58:04
this so this is a 6x2 matrix which is a context Vector Matrix and each row
58:10
corresponds to a context Vector for that token so if you look at the second row over here the second row corresponds to
58:16
the context Vector for Journey which we have shown over here the first row corresponds to the context Vector for
58:22
your similarly the last row corresponds to the context Vector for step
58:27
one exercise I want to give you is that uh use this this visual representation
58:33
of scaling so take the second row take journey and uh use the scaling approach
58:38
which I showed you in the graphical representation so take the vector for your multiply it by 15 take the vector
58:45
for Journey multiply it by 22 similarly take the vector for step multiply it by8
58:50
add them all together and see whether the result matches with the second row over here that will give you an
58:57
intuition of why this matrix multiplication actually gives us the exact same result as this graphical
59:03
intuition based calculation which we did over here but if you forget this Matrix formula just remember the scaling based
59:10
approach which we discussed in this graphical intuition and you will get the exact same value so remember that the
59:16
context Vector Matrix is just a matrix product of attention weights and values
59:21
attention weights multiplied by the values Matrix gives us the context Vector Matrix and this is exactly what
59:27
we are going to implement in code right now uh so let us go to
59:33
code yeah so we saw this we saw the square root and now we are going to uh
59:40
implement the context Vector so remember that context Vector first we are going to only see the context Vector for
59:47
Journey and it's the product between the attention Matrix attention weight for
59:52
Journey multiplied by values let me explain this a bit so uh on the
59:57
Whiteboard what we saw is we just multiplied the entire attention weights with the value right but if you want
1:00:03
just the uh context Vector for Journey what you can do is just take the second
1:00:09
row it will be uh 1X 6 and you multiply it with this values which is 6x2 and
1:00:15
then you'll get a 1x two Vector which is the second row here and that will be the context Vector for Journey so this is
1:00:22
what I have showed over here the context Vector 2 which is the context Vector for journey is just the product of the
1:00:27
attention weights for Journey multiplied by the values Matrix and the result is 3061 and
1:00:34
8210 and let's actually see the result here and that exactly matches the second row which we have 3061 and 8210 awesome
1:00:44
so our calculation seems to be correct so in the code right now we have only computed the single context Vector right
Self Attention Python class - Basic version
1:00:50
now we are going to generalize the code a bit to compute all the context vectors it's going to be very simple because now
1:00:56
we just multiply the attention weights with the values but we'll do this in a structured manner we'll Implement a self
1:01:03
attention python class and what this class will do is that it will essentially have a forward method this
1:01:09
forward method will compute the keys queries values it will compute the attention scores attention weights and
1:01:15
the context vectors all in a very short piece of code so let's do that right now before that let us summarize what all we
1:01:22
have seen so far so that you'll understand the python class much better so let me zoom out here a
1:01:28
bit so remember how we started the lecture we started the lecture with uh
1:01:33
we started the lecture with taking the inputs and then multiplying them with query key and the value to get the
1:01:39
queries Matrix the key Matrix and the value Matrix okay then remember what we did next then we move to the attention
1:01:46
scores we multiplied the queries with the transpose of the keys to get the attention scores so we had the attention
1:01:52
scores Matrix then what we did is we scaled this by square root of the keys Dimension and then we took the soft Max
1:01:59
this gave us the attention weights then we took the attention weights and we multiplied it by the values Matrix and
1:02:04
that ultimately gave us the context Vector Matrix remember this flow so the flow is in four steps step number one is
1:02:12
at the left side of the page which is converting the input embeddings into key
1:02:17
query value Vector step number two is getting the attention scores step number three is getting the attention weights
1:02:23
step number four is getting the context vector that's it and we are done that's exactly what we are going to implement
1:02:29
in this python class so uh with the llm implementation which we are going to cover next in one
1:02:36
of the subsequent lect lectures it's very useful to organize the code in a python class so we cannot keep on
1:02:43
writing separate lines of codes like what we did over here right it's just better to have a class so that then we
1:02:48
can create an instance of this class and then always return the context Vector okay so we are going to Define in
1:02:55
this class called self attention version one and it will take two attributes the input Dimension and the output Dimension
1:03:02
the input Dimension is the input Vector embedding Dimension the output Dimension is what we want the keys query and value
1:03:10
dimension in GPT and other llms these these two are generally similar okay first thing what we do is
1:03:17
when an instance of this class is created this init Constructor is automatically called and the query key
1:03:22
and the value matrixes matrices are initialized randomly which means that
1:03:27
they have a dimension of D in and D out so D in rows in our case three rows and
1:03:33
D out columns two columns and then each element will be initialized in a random
1:03:38
manner then what we do is we do the forward pass what happens in the forward pass is that it takes an input it takes
1:03:46
X as the input which is the input uh input embedding Vector that needs to be
1:03:51
given as an input to execute the forward method and then in the forward method what we do is we first compute the keys
1:03:57
Matrix which is X multiplied by the uh weight trainable weight Matrix for key
1:04:03
then we compute the query Matrix which is X multiplied by the trainable Matrix for query then we compute the value
1:04:09
Matrix which is X multiplied by the trainable Matrix for value and this is uh exactly what we saw on the left side
1:04:16
of the board over here here so until now we are at this stage where we are taking the inputs we
1:04:23
are multiplying it with the m weight Matrix to get the queries keys and the values and now we'll go to the right
1:04:29
side of the board to compute the attention scores so to get the attention scores we'll multiply queries with keys
1:04:36
transpose so that's exactly what's done here to get the attention scores we multiply queries Matrix with keys
1:04:41
transpose then we get the attention weights to get the attention weights we'll of course apply soft Max but
1:04:48
before applying soft Max we'll divide the attention scores every element of the attention scores with the square
1:04:54
root of the Keys embedding Dimension so keys do shape minus one Returns the
1:05:00
columns which is the embedding dimensions in this case it's two columns of the keys Matrix so it will be square
1:05:06
root of two the reason we do this division as we saw is first of all to make sure the values in The Matrix in
1:05:12
the attention score Matrix are small second it also helps to make sure that
1:05:18
the dot product between the quiz keys and the queries uh does its variance
1:05:24
does not scale too much so we want its variance to be very close to one that's why we specifically divide by the square
1:05:30
root of the dimension and here the DM equal to minus one just tells the soft Max that you
1:05:36
have to sum across the columns and that's how we make sure that each row if you take each row it sums up to one so
1:05:43
if you look at each row of the attention weight Matrix it will sum up to one and then the context Vector is just the
1:05:49
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



