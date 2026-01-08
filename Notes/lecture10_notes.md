## Token Embeddings
* Also called Vector Embeddings / Word Embedding
* Preferred token embeddings 

* What are token embeddings?

* small Hands-On demo where we'll play

* Conceptual understanding of why token embeddings are needed

***

* semantic meaning between these individual words so cat and kitten

* convolutional neural networks work so well because convolutional neural networks don't just use the pixel values and stretch it out as one input vector they actually encode the spatial relation between the pixels so the these two eyes are close to each other right.

*  one hot encoding

***

* 10:00 
* we have to train a neural network to create Vector embedding

***

* 15:00 
* I want to prove to you that well trained Vector embeddings actually the semantic meaning right so king plus

***

* 20:00 

***

* 25:00
* embedding weight Matrix

1. vocabulary-size
2. vector Dimension
  
* GPT-2 was 768
* vector dimension 768
* matrix size = 50257 x 768

***

* 30:00
* embedding layer weight Matrix

* initially when this weight Matrix is initialized
*

 * embedding layer weight Matrix which has 50257 rows and which has 768 columns but
you you don't know what each weight value will be so then

* initialize the embedding weights with random values that's the First Step
* this initialization serves as the starting point for the llm learning
process then what do you do these weights are then optimized as part of
the llm training process

* embedding layer weight Matrix so if

***

* 35:00

***

* 40:00

* embedding layer consists of small random values initially and these are the values which are optimized during llm training as part of the llm optimization itsel

***

* 45:00

* (e) - lookup operation that retrieves rows from the embedding layer weight Matrix using a token ID 

* (f) what an embedding does is actually?
*
*
*
* is that you have inputs right so let's say we have
53:27
three tokens three token IDs those are converted into one hot representations
53:32
they are fed into a neural network with five neurons why five because the vector Dimension is five and then we have a
53:39
linear layer whose output is X into W transpose this gives the same output as
53:44
the embedding layer so earlier I showed you tor. nn. embedding right here I'm showing you tor. nn.
53:52
linear then you might be thinking why is nn. linear not used used to define the
53:57
embedding Matrix the reason is it's not used is because so both embedding layer
54:03
and NN layer lead to the same output both embedding layer and the NN linear layer lead to the same output but
54:09
embedding layer is much more computationally efficient because in the NN layer we have many unnecessarily
54:15
multiplications with zero so you could just use NN do linear operation over here and do the X into W transpose but
54:22
here you see you have to do one hot encoding so there are many zeros and a lot of unnecessary computations are
54:28
there these unnecessary computations really scale up when we are dealing with vocabulary sizes of chat gp2
54:35
gpt2 and that's why we use the embedding layer here so there is also the
54:41
torch. nn. linear this layer is also there there are two layers and even you can use nn.
54:49
linear to create the embedding Matrix but the reason it's not used is because it's not efficient so the embedding
54:55
layer is much more preferred or the nn. linear layer when you create the embedding Matrix this is just a small
55:02
takeaway for anyone who is familiar with neural networks but if you are not don't worry if you did not understand this
55:08
part mostly I wanted to cover some other major points in today's lecture and some
Lecture recap
55:14
of them are first I wanted you to understand the conceptual understanding of why token embeddings are needed in
55:21
convolutional neural networks we exploited the spatial features of an image before giving it as as input for
55:26
training this is exactly what we do in uh token embeddings words can be
55:32
represented as vectors and those vectors can carry meaning that is called as Vector embedding or token
55:38
embedding and if the token embeddings are trained properly we show I showed you an example where the words the
55:46
vectors can actually carry meaning so if you have Vector for King which encodes some masculinity if you have Vector for
55:53
woman which encodes some femininity and we subtract the Vector for man which also encodes masculinity the answer is a
56:00
vector which encodes femininity which is Queen what also we can do is that we can
56:05
show that if you take two vectors of Words which are similar to each other
56:10
and if you take the magnitude of the difference between those vectors that magnitude is much lesser than words
56:16
which do not mean anything which means that if you if you have vectors for
56:21
Words which are similar to each other they might be closer together in space so if embeddings are created nicely they
56:28
can actually encode the meaning between words I found this concept very hard to
56:34
understand so I wanted you to First understand that it is possible to have vectors in such a way that they encode
56:40
meaning many people don't even understand what does it mean that vectors have meanings so the first two
56:46
points in today's lecture were devoted for you to get an conceptual understanding of why token embeddings are
56:52
needed then we looked at a practical aspect of how token embeddings are are created to create a token embedding
56:58
Matrix you need two parameters you need your vocabulary size and you need the vector dimension for gpt2 the vocabulary
57:06
size was 50257 and the vector Dimension was 768 so you essentially have an embedding
57:12
weight Matrix which has 50257 rows and 768 columns for each token ID in the
57:19
vocabulary you have to construct a vector now how are these weights of the
57:26
embedding Matrix determined they are initialized randomly these weights are initialized randomly and then the
57:33
embedded weights are optimized as part of the llm training process that's very important uh so this this is how the
57:40
embedding weight Matrix is created but what's also quite interesting is that at the heart of it the embedding weight
57:45
Matrix is just a lookup operation which means that if you have a embedding weight
57:51
Matrix if you have an embedding weight Matrix you can just pass in the input ID
57:56
or the token ID for which you want the vector embedding and then you get it corresponding to the particular row you
58:02
can even pass in a bunch of input IDs and then the embedding layer will just look for that the row corresponding to
58:08
the input ID and retrieve the vector embedding for you so simple way to look
58:14
at the embedding layer is that it's essentially just a lookup operation that's it now towards the end we also saw one
58:21
more thing that whatever the embedding layer does can actually be done using the neur oral Network linear layer but
58:28
the reason it's not preferred is because the embedding layer is much more computationally
58:33
efficient awesome in this lecture I have not covered how to train the embeddings but I just wanted to give you an overall
58:40
understanding of what token embeddings are what is the embedding layer weight Matrix but in in subsequent lectures we
58:46
are also going to see how to train the embedding layer so in this example which we saw I directly use the pre-trained
58:53
word to Google news right later we'll also see how this pre-training is done and how gpt2 gpt3 and gp4 did the
59:01
pre-training for the for creating the vector embeddings in the next lecture we are going to look at another important
59:07
concept which is called as positional embedding so until now we looked at how to connot words into vectors right but
59:14
when sentences are given the positioning of the sentence also matters a lot uh the cat sits on the mat so cat and mat
59:21
are close by but if mat is somewhere far away they are not related so position of the words also matter a lot apart from
59:28
their semantic meaning up till now the vector embeddings which I have showed you do not encode the position for where
59:34
the word comes in the particular sentence but that is another uh feature of words and sentences which we are
59:41
going to exploit in images we exploited transational invariance and we also
59:46
exploited spatial similarities between features in vctor embeddings we have already exploited the semantic
59:53
relationship and the semantic meaning between words but we'll also see how to exploit the position and where the words
1:00:00
are positioned in sentences and that will be the subject of the next lecture
1:00:05
where we are going to learn about positional embeddings thank you so much everyone I know these lectures are
1:00:11
becoming a bit long but I deliberately want to construct everything so that I show you a whiteboard approach um and I
1:00:18
also show you presentations and I also show you the code files I'll be sharing this code file and also this code file
1:00:24
with you so that you can play around with it have access to it please comment in the chat if you're liking these
1:00:30
lectures because then I will modify adapt it accordingly and as I say many times the most important thing is
1:00:36
showing up for these lectures uh don't lose interest don't lose motivation and keep on learning along with me thanks so
1:00:43
much everyone and I look forward to seeing you in the next lecture














