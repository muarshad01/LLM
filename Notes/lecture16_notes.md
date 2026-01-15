#### Causal Self Attention

***

* 05:00

#### What is Causal Attention?

* Causal attention also known as mask attention is a special form of attention.

* It restricts the model to only consider the previous and the current inputs in a sequence, when processing any given token.

***

* 10:00

* This is in contrast to the self attention mechanism, which allows access to the entire input sequence.

* When Computing attention scores the causal attention mechanism ensures that the model only factors in tokens that occur at or before the current token in the sequence.

* To achieve this in GPT like LLM, for each token processed, we mask out the future tokens which come after the current token in the input text.

* we mask out the attention weights above the diagonal and set those attention weights to be equal to zero and then we normalize the nonmass attention weights such that the attention weights sum up to one in each row.

***

* 15:00

#### 3.5 Hiding future words with causal attention

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

```python
# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

***

20:00

```python
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```

```python
masked_simple = attn_weights*mask_simple
print(masked_simple)
```

```python
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```

***

* 20:00

***

* 25:00



sum so this leads to a data leakage problem why data leakage because the
25:58
although we zero out the elements above the diagonal since we are taking soft
26:03
Max before the elements which come in the future do affect the previous elements also so we need a way to avoid
26:10
this so there is a smarter way to do this renormalization and let me tell you
26:16
what that smarter way is so if you look at what all we have
26:21
done until now so what we did is essentially this we took the attention scores we applied soft Max so this
26:28
already brought in the influence of future tokens then we mask with zero
26:33
then we again normalize the rows and then we got the attention weight Matrix right this is what we did right now what
Negative infinity masking and softmax
26:39
if there is a more efficient way so the efficient way is that what if we have the attention scores then we apply
26:46
something called as an upper triangular Infinity mask and then we just apply softmax once this will ensure that there
26:53
is no leakage problem let me explain what I mean by the upper triangular Infinity mask so let's
27:00
say we have the let me first show you the attention scores so let's say we
27:05
have the attention score Matrix right uh instead of applying soft Max earlier and
27:11
getting the attention weight Matrix what if we replace so let's say for the first row what if we replace these values with
27:18
negative Infinity for the second row we'll replace these values with negative Infinity basically what if we replace
27:24
all the entries above the diagonal with negative Infinity like this and then we take the soft Max what that will ensure
27:31
is that anyway when we take the soft Max when you do the exponent of negative Infinity it's going to be zero so when
27:38
you take the soft Max of let's say this row all of these entries will anyway be zero and then they will automatically
27:44
sum up to one because we are taking the soft Max so this kind of a trick will ensure that we are not having the data
27:51
leakage problem because the attention scores are calculated so now when you look at each row there is no influence
27:58
of future tokens yet because we have not done the soft Max then we just replace the elements above the diagonal with
28:04
negative Infinity there is no influence of future tokens now we have cancelled the influence of future
28:09
tokens by replacing them with negative infinity and we have not even done soft Max now and then we will do soft Max to
28:17
this Matrix what the soft Max will do is that it will kill two birds with the same Stone it will replace all of these
28:23
entries with zero because exponent of negative Infinity is anyway zero and since we are applying soft Max it will
28:29
anyway ensure that the sum of every row is equal to one so it will ensure that the attention weight Matrix the rows all
28:35
sum up to one and that is exactly what we are going to do next so now if if I
28:42
give you this Matrix and if I tell you that you want to replace all the elements above the diagonal with zero uh
28:50
or negative Infinity which whether you will use the upper triangular Matrix or
28:56
whether you will use the lower triangular
29:07
Matrix okay so let me tell you how this is actually done the way this works is that we first make a upper triangular
29:13
Matrix so let me print this out to show you what we are doing here so we print this out right
29:22
now incomplete input maybe I need one more bracket over here yeah so what we are going to do is
29:29
that we are going to take uh again 6x6 Matrix of ones and we are going to take
29:35
an upper triangular Matrix this time remember earlier we took a lower triangular Matrix let me tell you why we
29:41
take an upper triangular Matrix so we take an upper triangular Matrix where all of these are ones so what we are
29:47
going to code later is that we are going to say that look at all of the places where there are ones and replace those
29:52
ones with negative Infinity that is the mask which we are going to construct so
29:58
so we have this mask tensor over here which is a vector of essentially zeros
30:05
but all the elements above the TR above the diagonal are one then what we do is
30:11
we use this attention scores do mask fill function so what this mask fill function does in tensor flow or torch.
30:19
tensor I'll share this link with you what this uh function does is that uh it
30:24
looks at the argument first so what's there inside is that we take this mask we take this mask Matrix and we find out
30:31
all of the places where the uh Matrix returns a positive or True Value and


***



30:36
those are all the places which are above the diagonal right and we'll replace this so what this Mas field does is that
30:44
it looks for all the places where uh this mask Matrix has positive values and then in the attention score
30:51
Matrix will replace all of those with negative Infinity so effectively what
30:56
this uh ATT attention scores. mask fill function does is that it takes the attention scores Matrix and it replaces
31:03
all of the elements above the diagonal with a negative Infinity this is exactly what we wanted and now what we do is we
31:10
take this Matrix and we apply torge do soft Max so again as we did previously First We Take The Mask Matrix and divide
31:17
it by the square root of the keys Dimension and then we apply the soft Max so then that will ensure that all the
31:23
infinity values will anyway become zero and each row will sum up to one so now
31:28
my final attention weight Matrix looks something like this where you will see that the data leakage problem is not
31:34
there because I apply soft Max after all of the future elements are set to negative infinity and second all of the
31:41
attention weight Matrix rows sum up to one so the causal attention mechanism is
31:47
satisfied and also the soft Max is satisfied the data leakage problem is not there and each row sums up to one so
31:54
I have essentially obtained everything which I wanted in calcul of these attention
31:59
weights just to I've written some of these explanations over here so that you can understand it better so masking in
32:06
Transformers set scores for future tokens to a very large large negative value such as these uh making their
32:13
influence in the softmax calculation effectively zero the softmax function then recalculates attention weights
32:19
among the unmask tokens this process ensures no information leakage from the mass tokens focusing the model solely on
32:27
intended data now uh since we have got the attention weight Matrix we can just simply
Dropout in causal attention
32:34
multiply them with the values Matrix to get the context Vector that's it this is
32:39
the implementation of the causal attention mechanism in Python but there is one more additional step which is
32:46
typically implemented along with the causal attention mechanism and that is implementing the causal attention
32:53
mechanism with Dropout so if you're not familiar with Dropout it's actually a deep learning
32:58
technique where you take a neural network and you randomly switch on neurons in different layers to zero what
33:04
this does is that usually when you are training some neurons become lazy and
33:09
they do not do any work because they realize that other neurons are anyway doing most of the work and the result is
33:15
pretty well so I'll just switch off so that's a lazy neuron problem or codependency problem what Dropout
33:21
ensures is that when a lazy neuron sees that the other neuron is Switched Off it it's forced to do the work uh that's the
33:29
simplest way of thinking about it so Dropout randomly turns neurons off so it ensures that all the neurons essentially
33:35
participates and this leads to better generalization it prevents overfitting
33:41
and it does better on the test data we will so the main advantage of Dropout is
33:47
that it prevents overfitting and improves generalization performance in Transformer architecture
33:53
including models such as GPT Dropout in the attention mechanism is implemented and it's applied usually in two specific
33:59
areas first it's applied after the calculation of the attention scores and
34:04
second it's after applying attention weights to the value vectors so there
34:10
are two specific uh ways in which um Dropout can generally be
34:17
implemented first is after you get the context Vector itself after applying attention weights to the value vectors
34:22
you can Implement Dropout but the more common way is to Implement Dropout after
34:29
calculation of the attention weights or the attention scores and hence we are going to consider that so essentially
34:35
what is done in the dropouts is that let's say if you have an attention weight Matrix which with causal attention implemented so all future
34:42
tokens have been masked what we will do is that we will first create a Dropout mask what this Dropout mask specifies is
34:49
what all neurons need to be randomly turned off so let's say if we Implement a Dropout with a probability of 0.5 this
34:56
means that on average 50% of the attention weights in each row will be turned off so let's say if you look at
35:02
the second row let's say this will be turned off if you look at the third row 50% right so three entries so randomly
35:09
this will be turned off this will be turned off if you look at the uh fifth row 50% so you'll you'll randomly zero
35:17
out certain elements so this this is how Dropout is implemented so this is the Dropout mask
35:23
which you can see over here wherever the mask appears those particular element M will need to be zeroed out so if you
35:29
look at the fourth row over here uh let me rub some of the things over here yeah
35:35
so if you look at the fourth row in this Dropout mask we have a mask position
35:40
here here and here so we have a position at 1 four and five so the first entry
35:46
will be masked it will be removed the fourth entry will be masked so only two entries are going to survive here 24 and
35:53
24 so here you can see over here 24 and point 24 are the only two entries
35:59
surviving in this row so essentially what the Dropout uh does in very simple terms is that it looks at rows and then


***




36:06
it randomly switches switches off attention weights with a particular given
36:12
probability uh so now let me Implement first the Dropout in um in Python so in
Coding causal attention dropout in Python
36:19
the following code example what we are going to do is we are going to use a dropout rate of 50% which means that we
36:25
are going to mask out half of the attention weights later when we train the GPT model we are
36:31
going to use a lower dropout rate of around 0.1 or02 so uh in the following code we
36:37
apply pytorch Dropout implementation to a 6x6 tensor consisting of just ones for
36:42
illustration purposes and then we'll actually apply it on the attention weight Matrix which we have so let's say
36:48
we have a 6x6 uh we have an example which is a 6x6 Matrix of on let me print
36:54
it out over here uh uh
36:59
yeah so let me print print example so
37:05
let's say we have a matrix 6x6 so these are all ones then we'll Implement tor. nn. Dropout point5 what this is going to
37:13
do is that it will look at each row and then on average it will switch off 50%
37:18
of the weights and what this will do is that since the 50% of Weights are
37:23
Switched Off which means 0.5 all the other weights are rescaled by that that much amount so all the other weights
37:30
which are not Switched Off will be rescaled by two it will be divided by 0.5 or they'll be multiplied by two so
37:36
if you look at the first row over here you'll see that two weights are switched off if you look at the second row you'll
37:42
switch you'll see that four weights have been switched off you look at the third row you'll see that one weight has been switched off so remember this is
37:49
probabilistic so if you take 10,000 rows you'll see that on an average
37:55
50% of every row will be switched off so that does not guarantee that three exact
38:01
neurons or three exact weights will be switched off in every row two three or four neurons might be switched off but
38:07
on an average three neurons will be switched off in every row so when applying Dropout to an attention weight
38:13
Matrix with the rate of 50% half of the elements with the of the Matrix are randomly set to zero remember this is
38:19
probabilistic to compensate for the reduction in active elements the values of the remaining elements in The Matrix
38:25
are scaled up by a factor of two this is how tor. nn. Dropout is implemented and
38:30
you can even check this so if I click on tor. NN drop. Dropout you can see the
38:36
documentation for the dropout dropout class in tensor flow or
38:41
py torch rather so this is a pytorch Dropout class so you'll see that during training randomly zeros out some of the
38:48
elements with probability P outputs are scaled by a factor of 1 upon 1 minus P
38:53
that is exactly the kind of scaling which we are seeing over here so the the scaling is crucial to maintain the
38:59
overall overall balance of the attention weights uh ensuring that the average
39:04
influence of the attention mechanism remains consistent during training and inference phases now let us actually
39:11
take the attention weights which we had over here these were the final attention weights and we are going to apply Dropout layer so I take the attention
39:19
weight Matrix and I apply Dropout to it and here Dropout is being defined as a
39:24
class and the class takes here an instance of the Dropout class is created that the input argument is 05 which
39:31
means the dropout rate is 0.5 so here you can see that compared to this um
39:37
versus let's say if you see the attention weight Matrix you drop out you'll see that some attention weights will be randomly set to zero and the
39:44
weights which are not set to zero will be scaled by two so in the first row you'll see that this first weight is not
39:49
set to zero so it will be multiplied by two let's look at the second row so we have 3986 and 60 and4 after implementing
39:57
Dropout both of them are set to zero uh then let's look at the third row
40:03
2526 3791 3683 after implementing Dropout none of
40:09
them are set to zero so since it's probabilistic in nature some weights will be set to zero some will not but
40:16
overall 50% of the attention weights will be set to zero so as you can see the resulting ATT
40:22
attention weight Matrix now has additional elements zeroed out and the remaining ones are rescaped SC this is
40:28
exactly what we wanted now we have gained an actual
40:33
understanding of causal attention and Dropout masking we will Implement a causal attention class in Python so this
40:40
is also what we are going to see next on the Whiteboard so the next section which we
Coding the Causal Attention Class in Python
40:46
are going to see is uh implementing a causal attention class which incorporates causal attention and
40:52
Dropout into the self attention class which we have implemented earlier so to do this first I want you to have a


***





40:59
visual understanding of what we are going to implement that will make understanding the code so much easier so
41:05
if you have understood the self attention class when we implement this causal attention it's exactly going to be the same except for a few small
41:12
changes so we will have the inputs we will multiply it with the weight query
41:17
weight key and the weight value trainable matrices then we'll obtain the queries keys and the value Matrix then
41:25
we'll get the attention scores by multiplying queries with keys transfer then what we'll do is that we'll
41:30
Implement uh we'll mask out so all of these diagonals will be replaced with
41:36
minus all the elements above the diagonal will be replaced with minus infinity then we will do scaling by
41:42
square root of the dimension and we'll do Dropout and we'll do soft Max so that will give us the attention
41:49
weights so remember now the attention weights all the elements above the diagonal will be equal to zero and some
41:55
of the elements will be randomly switch switched off because we are implementing Dropout and then we'll get the we'll
42:01
multiply the attention weights with the values and we'll get the context Vector Matrix this is all which we are going to do in the uh causal attention class one
42:11
more additional step which we are going to do is that we are going to look at batches so this is the first batch right
42:17
so this is the first sentence your journey begins with one step what we ideally want to do is we want to develop
42:24
the attention class which can handle multiple Cent sentences at once so what if there is a second sentence also that
42:30
second sentence can be my name is something something let's say so that second sentence will also be handled in
42:36
a very similar Manner and then I'll get another context weight Matrix in a similar manner so my the class which I
42:44
Define should be able to handle both of these batches together so let's see how we can Implement
42:51
that uh one more thing yeah so as I mentioned one more thing is to ensure that the code can handle batches
42:57
consisting of more than one input as I showed you earlier so what we are going to do is
43:03
that we are going to create a simple batch which has two inputs so as we already saw the first input is a six row
43:09
and a three column Matrix now we are just going to add one more input so then the batch will be two a tensor which has
43:17
two so we have two batches and each has 6x3 so this is the incoming tensor which
43:24
our class should be equipped to handle so so this results in a 3D tensor consisting of two input text with six
43:30
tokens the first text can be your journey begins with one step the second sentence can let's say be my name is uh
43:40
my name is so and so let's say that's the second sentence now uh why is this
43:45
6x3 because each sentence has six tokens and each token has three dimensional
43:51
Vector embedding so the following causal attention class is very similar to the
43:57
self attention class except that we are going to add two things we are going to add the Dropout and we are going to add
44:02
the causal mask okay so let's go through this class now uh first what we are
44:08
going to do is that the shape of the input is now different because now the input shape has the First Dimension as
44:14
the batch size uh whereas in the self attention class if you scroll up earlier
44:20
the shape of the input which was there here the shape of the input was just the number of rows were six and the number
44:26
of columns were three because it did not have batches but now the shape of the input is different the shape of the
44:31
input let's say is 2x 6x3 so first is the batch number or the batch size the
44:37
second is the number of tokens and the third is the vector embedding Dimension so B comma number of tokens comma the
44:45
embedding Dimension is X do shape then what we do is that we multiply the input with the key weight
44:53
Matrix we multiply the input with the query weight Matrix we multiply the input with the value weight Matrix to
44:58
get the keys queries and the values then what we are going to do next is that we are going to multiply the queries with
45:05
the keys transpose and why we are doing 1 comma two over here because we are only interested in the number of tokens
45:11
Dimension and the inputs Dimensions remember that we are looking at it in batches right so when you look at the
45:18
first batch uh you when you look at the first batch you only care about the
45:23
number of tokens and the input Dimensions when you look at the second batch you only care about the number of tokens and the input Dimensions so when
45:30
you get the attention scores you multiply you take the queries and you multiply keys. transpose 1 comma
45:37
2 uh let me explain this to you right now yeah so let me explain how this uh
45:43
queries and key transpose actually works so now the queries which I have will be


***


45:48
in batches right so if you look at the first batch uh let's say these are the
45:54
so this is the first token and this this is the second token I'm just showing two tokens now and if this is the second
46:00
batch then this is the first token and this is the second token both of these batches are now coming together in the
46:06
queries whereas if you look at the keys let's say these are the keys the reason
46:11
we are taking the keys transpose is that for the queries to be multiplied with the keys the keys need to look like this
46:17
otherwise the matrix multiplication cannot happen so if you look at the queries now the queries are this and
46:23
when you do keys. transpose 1 comma 2 they will look look something like this which means that we'll still have two
46:29
rows but inside the so it was 2A 6 comma 2 right so we'll still have two rows but
46:35
inside each row the that particular Matrix will be transposed so without transpose the keys look like this and
46:42
when you do uh when you do this so when you do keys. transpose 1 comma 2 what will be preserved is that inside the uh
46:50
so the rows will be converted into columns so the keys will Keys transpose will start looking like this and when
46:57
you multiply the queries with the keys transpose what will happen is that the batches will be processed sequentially
47:03
so in the first batch this queries will be multiplied with this Keys transpose
47:08
and you'll get a result then this then this queries will be multiplied with
47:13
this Keys transpose and you'll get a result and both those results will be stacked together so here just the
47:19
multiplication has been shown and finally you'll get the attention scores where both the results have been stacked
47:26
together this is how uh it actually works in
47:32
batches and then what we do is that once we get the attention scores uh as I told
47:37
you earlier first we are going to uh we are essentially going
47:43
to create yeah so here we are creating an upper triangular mask which is of all
47:49
ones uh except so which is once so let's see the upper triangular so upper
47:56
triangular Matrix is ones above the diagonal right so there are ones above the diagonals and everything below the
48:02
diagonal is zero and then wherever there is one those will be replaced with minus infinity in the attention scores Matrix
48:10
then what we are going to do is we are going to divide by square root of the key Dimension and we are going to take
48:16
the soft Max this is exactly what we saw this is the attention weight Matrix where all the rows sum up to one and the
48:23
causal attention mask is applied and then finally what we do is that we uh apply the Dropout to these attention
48:30
weights and the Dropout has been defined over here where the dropout rate is taken as an attribute when we create an
48:36
instance of the causal attention class and then the context Vector is a product of the attention weights
48:42
multiplied by the values and then we ultimately return the context Vector so let's see now how this
48:49
actually works out in practice so uh first the context length is batch. shape
48:55
of one because why one because we have a batch size of six so batch do shape of one will be six and then what we'll be
49:02
doing is that we'll be defining a caal attention class with the input dimension of D in now let's see what D in actually
49:09
is so D in will be uh let's print this out actually let's
49:16
see what DN is so I think DN is three because the vector size is three and D
49:22
out so if I print D in uh that will be equal to three correct and if I print D
49:29
out I think that will be equal to two because those are the dimensions which we have used yeah D out will be equal to
49:36
two the context length will be equal to 6 and then 0 comma 0 is 0. 0 is
49:41
essentially the Dropout so here we are saying that don't do so put the dropout rate to be zero so then what we do is
49:47
that we create an instance of this causal attention class and then we pass in the batch which we have defined
49:53
earlier so now you can see that here we defined a batch where we stack two inputs on top of each other when we
49:59
process the first input when we process the first input we should get a context Vector Matrix of
50:05
size U so here there are six rows and two columns right so when we process the first input of this batch we'll get a
50:10
context Vector of the size 6x2 um 6x2 and when we process the second
50:19
input in the batch we'll get a context Vector Matrix of size again we'll get a context Vector Matrix of size 6x2 so
50:27
there will be two context vectors of size 6x2 so the resultant answer should be 2x 6x2 it will be a 3D
50:33
threedimensional tensor let's see if that's indeed the case um so now here


***



50:38
you can see that I've have passed in my batch of inputs and here are the context
50:44
vectors here's my resultant answer and if you print the shape of the context Vector which is the resultant answer
50:50
it's 2x 6x2 why because we have two matrices of 6x2 which we are stack which are stacked on top of each other so so
50:56
you can even print out the so you can even print out the
51:01
context vectors now so if I do print context
51:07
Vex this will print out the context vectors and you'll see that here we get so the first is uh this is the context
51:14
Vector Matrix of the first input this is the context Vector Matrix of the second input and they are stacked on top of
51:19
each other awesome so which means the causal attention class which we have which we have written is capable of
51:26
handling B I did not explain this thing here which is register buffer so why do we need a
register_buffer in PyTorch
51:32
buffer when we create this mask so the main thing is that it's not really necessary for all use cases but it
51:37
offers some Advantage here so when we use the causal attention class in a large language model buffers are
51:43
automatically moved to the appropriate device CPU or GPU along with our model which will be relevant when training the
51:50
llm in future chapters usually matrices are this matrices like this which are fixed which which need not need to be
51:57
trained so this is an upper triangular Matrix right all the elements above the diagonal will be one will be one it's a
52:03
fixed Matrix we will not train this usually it's better to Define all of these as the using the register buffer
52:11
because then these are automatically move to the appropriate device along with our model and since we are anywh
52:17
not training them um it's much more convenient so we don't need to manually
52:22
ensure that these tensors are on the same device as the model parameters avoiding device M mismatch errors later
52:29
when we move to GPU calculations this will be very important so just remember
52:34
that U these masks or these matrices which are not trained it's it's good to
52:39
Define them using register buffer so that they can be automatically moved to the appropriate devices we don't need to
52:46
ensure that they are on the same device as our model parameters so that is an important thing
52:51
to be aware of the second thing is that here we have used the uh colon number of
52:57
tokens so this is to ensure for cases where the number of tokens in the batch is smaller than the supported context
53:03
size if this were not written that's also fine then the mask will be of the context size but if the number of tokens
53:10
are smaller than the context size the mask is created only up till the number of tokens this might happen if one of
53:16
the batch has smaller number of tokens than the contact size especially one of the ending batches Etc but these are
53:22
edge cases which you now don't need to worry about all you need to understand is that what this class has effectively
53:28
done is that we have uh implemented the causal attention mask which means that
53:33
all the elements above the diagonal are set to zero and we have ensured that all the rows of the attention weight sum up
53:39
to one using the soft Max and we have also implemented the Dropout layer to ensure generalizability and to prevent
53:47
overfitting um so I think this actually brings us to the end of this lecture on the causal attention class uh in the
Next steps
53:54
next section what we'll be doing is that we will expand on this concept and Implement a multi-head attention module
54:00
that implements several of these causal attention mechanisms in parallel so what the attention mechanism
54:07
in GPT and in other llms what they are doing is that they take these causal
54:12
attention mechanisms and they stack them together so let me show you this graph
54:18
this plot of what all we have learned so far so until now what we have learned is that we have learned about causal
54:25
attention in this section when multiple causal attention heads are stacked together it leads to multi-head
54:30
attention and that's what's actually implemented in GPT but now try to think about it without covering these lectures
54:37
how can you understand multi-head attention directly without understanding causal attention you cannot understand
54:43
multi-ad attention to understand causal attention you need to understand key query value that's why I have developed
54:49
these lecture Series so that we cover each module in a sequential Manner and in a lot of detail I know the lectures
54:55
are becoming tough the lectures are becoming long but if you follow what I'm doing and if you implement this code
55:02
you'll start to understand things in a much better manner I believe that all the other tutorials all the other videos
55:07
on YouTube out there currently they are very short they do not explain all of the details I think the devil lies in
55:13
the details we need to understand the details we need to deal with Dimensions you need to understand how batches work
55:19
why we have a three-dimensional tensor don't be scared of dimensions and matrices the student who Masters
55:25
dimension matrices linear algebra fundamentals they will really understand what is
55:31
going on here I'm deliberately trying to have a mix of the Whiteboard notes and the coding in Jupiter notebook so that
55:38
you understand the basics the theory as well as you implement the code thank you so much everyone I'll see you in the
55:44
next lecture where we'll cover multi-head attention in a lot of detail thanks everyone

***

