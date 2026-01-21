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
  13. Layer Normalization

***

* 35:00

* Fixal Step = Number of Tokens X 50,257

***

* 40:00
  
***

* 45:00

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





***

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



***


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









