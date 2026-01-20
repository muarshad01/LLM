```
Every effort moves you
```

***

* 15:00

* Embedding Size = 768

1. Token Embedding 
2. Positional Emdedding
3. Input Embedding = Token Embedding + Positioal Embedding
4. Droupout (We randomly turn off some elements of every uh every input embedding to zero. Benefits: a) Prevent overfitting; b) Improve generalization)
5. Transformer
6. Layer Normaliztion
7. Masked Multi-head Attention
8. Dropout layer
9. Layer Norm
10. Feed forward neural network
11. Dropout layer
12. Shortcut connections

*** 

* 30:00

***


screen U let me minimize this
30:47
color yeah I hope uh you can see the entire Transformer workf flow on the screen right now we have we start with
30:55
the layer normalization so I'll show show this with a different color probably yeah we start with the layer
31:01
normalization then we move to the uh mask multihead attention then we then we
31:07
have a Dropout layer uh then we have a shortcut connection over
31:13
here let me Mark this
31:21
line yeah we have a shortcut connection over here then we have another layer
31:26
normalization layer we we have a feed forward neural network we have a Dropout layer then we again have a final
31:32
shortcut layer which actually takes us to the Transformer block output um now let us actually go to
31:40
the uh visual diagram again to see whether we indeed have implemented all the steps which were mentioned in the
31:47
Transformer block so I'm zooming in over here right now and if you see there are actually 1 2 3 4 5 6 7even eight
31:54
different steps right we start out with the again let me switch to a different color
32:00
here
32:07
yeah yeah we start out with the layer Norm we go to the multi-ad attention we go to the Dropout we have the shortcut
32:13
connection that's the first block then we have again the layer normalization the feed forward neural network the
32:19
Dropout and then the last shortcut so there are eight different steps here which were the same eight different steps we implemented in the uh visual
32:27
flow map and now there are the final two steps which are remaining so let's look at the last two steps of the GPT model
32:34
in the visual flow map so the fifth step was the Transformer step which we implemented
Post transformer layer normalisation
32:40
within the Transformer also there were eight different steps now we go to the next block which is
32:48
uh uh the next block is another layer of normalization so the Transformer block
32:53
itself has two normalization layers but we have another normalization layer after the Transformer block output so
33:00
here we have the Transformer block output and here you can focus on the dimensions again every vector or every
33:06
token has an embedding size of 768 right again the dimensionality is preserved
33:12
and then we apply the layer normalization so same thing as what the near layer normalization does is done
33:18
here also in every token we normalize the embedding values so that the mean of
33:25
the resultant embedding values is zero and the variance is equal to one and this is done for all the four
33:32
tokens so again the layer normalization does not change the dimensions and then the last layer which we have is the
Output layer
33:38
output head so the output head is a neural network at the final stage of the GPT model so the input to the output
33:46
head is this input which has four tokens and each token has a dimension of 768
33:51
then what we do is that we pass this through a neural network whose size is the embedding dimension multiplied by
33:58
the vocabulary Dimension and I'll tell you why just in a moment but if you take this input input tensor and if you pass
34:06
in through the neural network which has uh which takes in inputs of 768 dimensions and the output is
34:13
50257 the resultant uh output which you get it's called as the logits Matrix and
34:20
this logits Matrix actually has four rows because we have four tokens every
34:26
so every f effort moves you and the number of columns here is not equal to 768 the number of columns is equal to
34:33
the vocabulary size because we are passing in passing the input through a neural network whose final output size
34:41
is 50257 so for every token we have this logits which is a
34:46
50257 dimensional Vector so for every there is a 50257 dimensional Vector for
34:53
effort there is a 50257 dimensional Vector similarly for move and U there
34:58
are 5 257 dimensional vectors and the reason here is because we want to predict the next word right
35:06
based on the input so when every is the input we want to predict what's the next word and that should be effort so we
35:12
look at the vocabulary and we look at that that token or that column which has
35:18
the highest value and that column should ideally correspond to effort when every is the input similarly when every effort
35:26
is the input we we will look at the column which has the maximum value and that column corresponds to certain word
35:32
in the vocabulary and that word should be moves because when every and effort are the input moves is the output
35:39
similarly when we look at every effort moves as the input we will predict the next word and that should be U so we
35:45
look at that column which has the highest value and that should hopefully be U and finally we look at every effort
35:51
moves you that's the final prediction task and then uh when every effort moves
35:56
you is the input we look at the final output and that should hopefully be the
36:02
token which corresponds to the next word which is forward every effort moves you forward we'll see how to make the next
36:09
word prediction in the next class but for now I just want you to appreciate that this last step is the only step
36:15
where the dimensions of the input change the dimensions of the input all along where four which were the number of
36:21
tokens multiplied by 768 which was the embedding Dimension but now the final
36:26
output size four which were the number of tokens multiplied by 50257 and based on this final output
36:34
we'll get the next word prediction now when you look at every effort moves you the context size is
36:40
four whenever we have a context size we have that many prediction tasks so
36:45
remember here there are four tokens right so we don't only have one prediction task you might think that the
36:51
only prediction task is every effort moves you is the input and we have to predict the next word no there are four
36:56
prediction tasks the first prediction task is every is the input then effort should be the output every effort is the
37:02
input movees should be the output every effort moves should be the input U should be the output and only the fourth
37:08
prediction task is every effort moves you what's the output so that's why we have this four rows here because there
37:15
are four prediction tasks we'll see in the next uh next class how to predict
37:20
the next word from this output tensor which we have obtained now one thing to mention is that uh so here if you you
37:27
see the output tensor value will actually be four so let me write this
37:32
down over here the output tensor value will be four multiplied
37:40
by uh the vocabulary size which is uh
37:48
five so it's 5 0
37:54
2 5 and 7 so that is the fin final tensor value uh remember one thing which
38:00
I have considered here is only one batch if you have two batches this will be the final tensor generated for both those
38:07
batches so similar similar to every effort moves you if you have another sentence of four words such as I like uh
38:15
movies and and you have to predict the next word that's the second batch so then the output tensor will also include
38:20
the batch size so then it will be 2 multiplied by uh 4 by 5 2 57 why this
38:29
initial two because the batch size is equal to two so the First Dimension is always the batch size the second
38:35
dimension is the number of tokens which we have or the context size uh and the third dimension is the
38:43
50257 which is the vocabulary size so this is the output tensor Dimension
38:49
format which we have now let me zoom out here and show you the entire flow map which we have
38:54
seen it was a pretty long flow map but I I hope you all have understood what we
38:59
are trying to do over here the reason I went through this entire flow map starting from token embedding positional
39:05
embedding to input embedding to drop out to the Transformer block uh then we went
39:10
to layer normalization then finally we went to Output head the reason I showed you all these things is just so that you
39:16
understand dimensions and what's going on with Dimensions I did not just want to show you the code because when you
39:21
see the code you cannot visualize the dimensions but now once you have seen this lecture and when whenever you let's
39:28
say you're looking at the layer normalization part of the Transformer right you can just visualize what's
39:34
happening uh what are the dimensions of the input to a particular block what are the dimensions of the output and so so
39:42
that will make your learning process much easier so now actually uh on the
39:47
Whiteboard we have seen how the building block stack up for the entire GPT model
39:52
we started with tokenization input embedding positional we went to Transformers and then we saw the final
39:58
layer nor norm and then we also saw the final linear output layer okay so now once your intuition is
40:06
clear once your visual understanding is clear we are pretty much ready to move into code so now let us dive into code
40:13
and uh see how these different blocks can be arranged together to code the entire GPT model
Coding the entire GPT-2 architecture in Python
40:20
all right so let's jump into code right now here's the GPT configuration it's
40:26
the same configuration is gpt2 which we are going to uh use when we are going to
40:32
look at this entire coding module so here you will see that the vocabulary size is equal to
40:38
50257 this was actually the vocabulary size which was implemented when gpt2 was
40:44
trained the context length is equal to 1024 the vector embedding Dimension
40:50
which we also saw on the Whiteboard that is equal to 768 the number of attention heads is equal to 12 the number of
40:58
Transformers U which are there which we are going to implement are equal to 12 the drop rate or the dropout rate is
41:05
equal to 0.1 and the query key value bias is false this is for setting or
41:11
initializing the weight metrices for the queries the keys and the values we don't need the bias term in that for now you
41:19
can focus on a couple of things the first is the embedding Dimension uh which is going to stay 7608 throughout
41:25
the second is the 502 spice and vocabulary size these are the same dimensions which we just saw on the
41:31
Whiteboard so you might be able to relate to them uh 12 is the number of Dropout uh 12 is the number of
41:38
Transformer blocks so in gpt2 when it was trained they had 12 Transformer blocks we are also going to use a
41:45
similar number of Transformer blocks then the number of attention heads so every Transformer block has a multi-ad
41:51
attention module right and that module has a certain number of attention heads we are going to use that equal to 12
41:57
dropout rate is 0.1 I think this much should be enough to understand the entire code now as we
42:04
are going through this code I want you to just keep this diagram in mind and the dimensions which we just saw in the
42:11
visual flow map everything will follow from this particular diagram since we are exactly going to uh code it in a
42:18
similar manner okay so we started out this
42:23
lecture series four to five lectures back with this GPT model class where we
42:29
had the forward method but the Transformer block was not implemented so this was a dummy Transformer block the
42:36
layer normalization was not implemented this was a dummy layer normalization
42:42
what we did over the last three to four lectures is that we coded the layer normalization class feed forward neural
42:47
network class and also the Transformer class so here's the layer normalization class which we had coded given a
42:55
particular uh layer now you can think of when whenever you see layer normalization remember we have seen the
43:02
visual visual flow map now right so you can uh try to visualize what happens in
43:08
the layer normalization layer we have seen that already so see this is exactly what happens in a layer normalization
43:14
layer we look at every token that has 768 Dimensions right then what we are
43:19
doing is that we are going to subtract the mean and divide by the square root of the variance so that the resulting
43:26
values have a mean of zero and standard deviation or variance of one we also have a scale and a shift here which are
43:32
trainable parameters then we have the feed forward block and remember where the feed forward block comes in the
43:39
Transformer architecture it's this uh it's this feed forward block um let
43:46
me actually change my pen color to it's this feed forward neural network which I have shown you over here the expansion
43:52
contraction feed forward neural network this is that feed forward block so you'll see that the input is the
43:58
embedding Dimension the layer is four times the embedding Dimension size and the output is the embedding Dimension
44:04
once you are able to visualize this you'll be able to easily understand this code the input is the embedding
44:09
Dimension the hidden layer is four times the embedding Dimension that's the expansion layer then there is a
44:15
contraction layer from the four times embedding Dimension to the embedding Dimension and there is a Jou activation
44:22
function we spent a whole lecture on the J activation because it's a bit different than the
44:27
it's smoother on the negative side I think I also have a plot for the J activation over here so let me quickly
44:33
show you that um if I can find it yeah so on the right hand side you can see
44:40
railu it's zero for negative inputs for the J it's not zero and also it's
44:45
differentiable at x equal to Z that's why in llm training generally the J activation uh is used so we coded the
44:53
layer normalization class the j class and the feed forward class and then in
44:58
the last lecture we coded the entire Transformer block class itself so here we had the so the eight steps in the
45:06
Transformer if you remember take a look at these eight steps which we saw um let
45:12
me zoom out here further so these are the eight steps which we saw in the Transformer we start with input
45:18
embeddings add a Dropout layer normalization then uh we go to the mass
45:24
multi attention then drop out and then the shortcut connections that's the first of these eight steps in the that's
45:30
the first four of these eight steps in the Transformer block then what we do is that then we again have a layer
45:37
normalization feed forward neural network with J then Dropout and then
45:42
another set of shortcut connections these are the last four steps so overall there are these eight steps which are
45:47
happening in the Transformer block architecture and it's the same eight steps which we wrote down when we wrote
45:53
the code for the Transformer block class these are the first four steps and it
45:59
ends with the shortcut connection and these are the last four steps and here again it ends with the shortcut
46:05
connection the main point which we also recognized when we implemented the Transformer block class is that the
46:11
dimensions are preserved the dimensions of the input to the Transformer are the same as the dimensions of the output
46:18
from the Transformer block awesome now we have all the knowledge and now we are
46:23
ready to code the entire GPT architecture so first first let's look at the forward method um which takes in
46:30
the input and remember the input looks something like this let me show you how the input actually looks like the input
46:37
might look something like this where here we are showing two batches and each batch has let's say four token IDs uh so
46:44
the number of rows here in this tensor are equal to the number of batches and the number of columns are the sequence
46:50
length or the number of tokens which we are considering that's why if you see the number of rows here in the input
46:56
shape are size number of columns are essentially the sequence length or the number of
47:02
tokens whenever now I'm explaining this next whole part you can keep that flow
47:07
map in mind which I showed you on the Whiteboard and think about the flow which we had for every effort moves you
47:13
so first we convert the tokens into token IDs and then we convert the token
47:19
IDs into embedding vectors that's what's done in this token embedding uh so we initialize an
47:26
embedding layer from the pytorch module and then from this embedding layer we
47:33
get the embedding token embedding vectors for the input IDs and then similarly we get the positional
47:39
embedding vectors for the four positions so see sequence length is equal to four so we get the positional embedding
47:45
vectors for the four positions and we add the token embedding to the positional embeddings so whenever you
47:51
try to make sense of Dimensions here always think of one batch at a time when you think of one batch things will be
47:57
simplified because one batch only has four tokens let's say so when you look at this step the output of this step uh
48:04
you can visualize the output exactly from the flow map which we had seen so let me take you to that yeah look at the
48:12
first step over here which was titled as positional embedding when you look at this code the token embeddings these are
48:20
essentially uh these are essentially this kind of a
48:26
tensor where for every token you have the 768 dimensional Vector similarly you
48:31
can think of the same visualization for positional embeddings and when you look at this x equal to token embeddings plus
48:38
positional embeddings you can uh you can try to visualize this this part which we
48:43
saw we added the token embeddings to the positional embeddings right over
48:48
here so for every token we added the token embeddings and the positional embeddings to get the input embeddings
48:55
that's exactly what I'm I've written in the code over here and then after this part we go to the next steps which were
49:02
which were the same in the visual flow so the next step Next Step
49:08
was addition of the Dropout which was step number four over here I'm showing that with a star right now that was the
49:15
step number four so Dropout layer followed by the Transformer followed by
49:22
the Transformer block and then the last two layers were another layer normalization layer which which was step
49:28
number six here this was the second last layer which was another layer of
49:33
normalization and then the final layer was output head so after you get the input
49:39
embeddings there are four things you apply Dropout layer you apply the Transformer blocks you have the another
49:46
layer of normalization and then you apply the output head that's it now if you see the Transformer blocks
49:54
we are chaining different Transformer blocks together together based on the number of layers so in the configuration
50:00
we have seen that the number of layers is equal to 12 right in this configuration we have seen that the
50:05
number of layers is equal to 12 so uh we are actually chaining 12 Transformer
50:11
blocks together using NN do sequential this NN do sequential is a tensor flow or pytorch module which allows us to
50:18
chain um different neural network blocks together so we have chained 12
50:23
Transformer blocks so when you see the cell trf blocks it just looks like one line
50:29
of code right but there are actually 12 Transformer blocks chain together here and in each Transformer block there are
50:36
eight different steps which are being performed so it's a huge number of operations being performed in this one
50:41
simple line of code that's it actually if you think of the building blocks if you have
50:47
understood the Transformer block if you have understood the layer normalization class uh if you have understood the
50:53
dimensions with respect to the Token embeddings positional embeddings that's all what
50:58
the um Transformer or what the GPT model
51:03
class is actually doing it takes in the input and then the outputs are the logits which we saw on the Whiteboard so
51:11
the dimensions of the logits if you remember let me take you to the Whiteboard once more so if you go to the
51:16
Whiteboard and if I zoom into this part of the if I zoom into this part of the Whiteboard right now you'll see that the
51:23
output of the GPT model is this output Logics and the shape of these logits is for each batch we have the number of
51:30
tokens uh in this case those were four tokens and the number of columns are equal to the vocabulary size and then
51:37
the First Dimension is the number of batches which we have okay so now what we can actually do
Testing the GPT model class on a simple example
51:43
is that we can take a simple input batch and then we can pass in through this entire GPT
51:49
model uh so what we are doing here is that we are taking an input batch which has every which has two batches and each
51:56
batch has two tokens what we are doing is that first we create an instance of the GPT model and pass in the model
52:02
configuration and then we create this object so we create an instance of model
52:08
uh which Returns the output so if you print out the input batch you'll see this and if you print out the output
52:14
batch let's look at one batch currently if you look at one batch you will see that there are four tokens and each
52:19
token has number of columns equal to the vocabulary size which is 50257 this is exactly what we had seen
52:25
in the output logic right so if you see in one batch there are four tokens four rows and there are 50257 columns this is
52:33
exactly the same thing which is happening for the second batch also there are four tokens here and there are
52:39
50257 columns because the vocabulary size is 50257 so as we can see the output tensor
52:46
has the shape uh two batches four tokens in each batch and 50257 columns since we
52:53
passed in two input text and four tokens each the last Dimension 50257 corresponds to the vocabulary size
53:00
of the tokenizer in the next class we are going to see how to convert these 50257 dimensional output token vectors
53:07
back to tokens and how to predict the next word but for now you can see that
53:12
the trans the GPT model class which we have constructed is working fine and there are a huge number of parameters
53:19
which we are actually dealing with here this small piece of code over here has more than 100 million parameters can you
53:25
imagine that there are just a huge number of parameters because there are 12 Transformer blocks chained together
53:31
there are eight steps in each Transformer block then uh and remember there is that expansion contraction
53:38
layer in each Transformer block that that layer itself has a huge number of parameters we have parameters for token
53:45
embedding positional embedding uh Etc so totally all of the parameters add up we
Parameter and memory calculations
53:51
can actually print out the number of parameters so what we can do is that we can have have total params and we can do
53:58
p. numl what p. numl will do is that it will actually print out uh so using the
54:04
numl method short for number of elements we can collect the total number of parameters in the model's parameter
54:09
tensors so if you print out the total number of parameters you will see that the number of parameters is equal to 163
54:16
million huge number of parameters right and all of this is running on our local
54:22
machine so you might be thinking earlier we spoke of initializing a 12 4 million parameter GPT model right so then why is
54:29
the actual number of parameters 163 million why is it so higher so this actually relates back to
54:37
a concept which is called weight time that is used in the original gpt2 architecture so which means that the
54:43
original gpt2 architecture is reusing the weights from the token embedding layer in its output layer so if you go
54:51
back to the Whiteboard right now yeah in the schematic which we saw over here there is an output layer right there is
54:57
an output layer towards the end and there is a token embedding layer over here when GPT model when gpt2 model was
55:04
constructed the parameters which were used for the token embedding layer were the same parameters which were used in
55:10
the linear output layer and that's why uh the total number
55:16
of parameters was less in our case right now we did not reuse the parameters so when we print out the parameters they
55:22
say 163 million uh but in gpt2 architecture they are reusing the weight
55:27
so we can actually print this out so in our case what we can do is that we can print out the embedding layer shape of
55:33
the token embedding layer and we can see that um the embedding layer shape is 50257 comma 768 and we can also print
55:42
out the output layer shape and these both have exactly the same shape so that's why in gpt2 model they just
55:48
reused this these parameters um so what we can actually do
55:54
is that we can um let we can remove we can remove the output layer parameter
55:59
count from the total parameter count and check the number of parameters so in the output layer parameter parameters are
56:06
removed the total number of parameters comes up to be exactly 124 million parameters which is the same size of the
56:12
gpt2 model so weight Ty actually reduces the overall memory footprint from 163
56:19
million we go to 124 million and it also reduces the computational complexity of the model however using separate token
56:26
embedding and output layers results in better training and model performance that's why when in our code which we are
56:32
writing we are using separate layers in our GPT model implementation and this is true for modern llms as well weight time
56:40
is good for reducing the memory footprint and reducing the computational
56:45
time and complexity but to get a better model training and performance it's good
56:50
not to reuse the parameters um we can also print out the space which is taken by our model
56:58
so we can compute the memory requirements of the 163 million parameters so the total size of the
57:04
model it turns out it's around 600 megabytes of space so in conclusion by
57:09
calculating the memory requirements for the 163 million million parameters in our gpt2 model um and assuming that each
57:17
parameter is a 32-bit float taking up to four bytes that's why we have multiplied by four over here we have assumed that
57:23
each parameter takes around four bytes so we multiplied the total number of parameters by four to get the total
57:29
size that 620 MB which illustrates the relatively large storage capacity
57:34
required to accommodate even relatively small llms so gpt2 is relatively small
57:39
120 million compared to there are now models which have more than b billion parameters imagine how much space such a
57:47
model will take on your device it's impossible to run extremely large models on local devices and that's why we need
57:53
GPU access so in this lecture today we implemented the GPT model architecture and saw that
Conclusion and summary
58:01
it outputs numerical tensors right so you might be curious how do we go from that numerical tensor output into the
58:08
next word prediction task uh so that we'll be doing next in the next lecture we are going to
58:14
generate text from the output tensor which we have obtained today today what I want to what I want you to appreciate
58:20
is just within 10 lines of or even 15 20 lines of code we have actually run a
58:25
very power powerful large language model completely on our laptop and we buil this model from scratch this model had
58:32
more than 160 million parameters it took 600 megabytes of space on our machine
58:38
indicating the size required by these models and I want you to appreciate that once you understood the visual flow map
58:44
I showed you on the Whiteboard it's just eight lines of code but to understand these eight lines of codes we need to
58:50
put in a lot of effort because there is a lot of theory and lot of intuition behind every single line of code over
58:57
here but once you get that it just these uh stacking of the token embeddings
59:02
positional embeddings Dropout layer Transformer blocks another layer normalization and the output output head
59:08
that's it and then we directly get the Logics um I hope you are following these
59:14
lectures I know this lecture is also become a bit long but it is essential to show you the theory as well as the code
59:21
in today's lecture I could have directly showed you this code which was only this much but then at every step you would
59:27
not have visualized the dimensions what is exactly happening can we take a specific example and see what is
59:32
happening for that example and now I've given you a visual understanding intuitive understanding as well as a
59:38
coding understanding in the code I written uh the description of what what
59:43
is happening in every layer for example here you will see that the forward method takes a batch of input tokens
59:50
computes their embeddings applies the positional embeddings passes the sequence through the Transformer block
59:56
normalizes the final output and then computes the logits representing the next tokens unnormalized
1:00:04
probabilities we will convert these Logics into tokens and text outputs in
1:00:10
the next class but whenever this when you access this code file Ive also written a number of such comments and
1:00:17
paragraphs here which you can read and try to understand the code for yourself
1:00:22
one thing which I will very very highly suggest is that whatever I written on the Whiteboard
1:00:28
right when I was I was myself writing the flow map of the Transformer on the Whiteboard this this flow map I
1:00:35
understood that my understanding also became very clear if I just ran the code
1:00:40
my understanding was not that strong but when I wrote every single step of the code on a whiteboard um and when I
1:00:48
created a visual flow map uh which I demonstrated to you in today's lecture
1:00:54
such kind of visual flow map which I'm showing right now on the screen it really strengthened my understanding and
1:00:59
it made me confident about this subject if you can also write this similar flow map on your notebook or on the
1:01:06
Whiteboard which you are using it will tremendously improve your understanding thanks a lot everyone I hope you are
1:01:12
enjoying from this lectures next lecture will probably be the last lecture in the GPT architecture series uh where we'll
1:01:19
generate text from the final output tensor which we have obtained today thanks a lot everyone and I look forward
1:01:24
uh to seeing you in the next lecture

***







