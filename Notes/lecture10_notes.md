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

has six rows
40:07
and three columns here also you see we have six rows and three columns every row here corresponds to the vector
40:14
associated with that token ID so this is the three-dimensional Vector with the zero token ID this is the
40:20
threedimensional vector with the first token ID this is the threedimensional vector of the second token ID Etc
40:28
uh so now you can see a tensor has been returned and these are the initial weights which need to be
40:33
optimized so as we can see here the weight Matrix of the embedding layer consists of small random values
40:39
initially and these are the values which are optimized during llm training as part of the llm optimization itself
40:46
which we will see in further chapters so when the llm is optimized there are actually two broad level things which
40:52
are optimized first is the embedding layer weights and second is actually the we
40:58
uh which are needed later to also predict the next word I'll come to that
41:04
in one of the upcoming chapters moreover we can see that the weight Matrix has six rows and three columns as we saw
41:11
over here and there is one row for each of the six possible tokens in the vocabulary which we already discussed
41:18
each row is essentially the vector embedding of each token or each token ID great
Embedding matrix as a lookup table
41:27
now what we can do is that uh once this embedding layer has been created right uh what I want to show you is that how
41:34
can we get the vectors for each ID and that's actually pretty simple because
41:39
this this is the first row is the vector for the zeroth ID the second row is the vector for the first ID Etc so let's say
41:47
uh if you want to get the vector for ID number three uh let me show this to you in code
41:54
ID number three is is right and if you you want to get the vector for is how do
41:59
you do it you first look at ID number three so ID number 0 1 2 3 so this this is that ID number right so then all you
42:06
need to do is look at this corresponding Row in the embedding weight Matrix that's why it's called a lookup table to
42:14
find the vector associated with a particular ID you just need to take this
42:19
Matrix you look at the vector corresponding with that particular ID row number that's it this is exactly
42:26
what we are going going to do over here uh we want to obtain the vector representation for ID number three right
42:32
so that's what we are going to do we are going to access the embedding layer it's a lookup table and we are going to
42:38
access the uh embedding Matrix for the token ID
42:44
3 and what will this be this will be the fourth row because the zero ID is the first row the first ID is the second row
42:51
second ID is the third row and third ID is the fourth row that's it and then when you print this you will get this
42:56
vector so this is the vector which is the vector embedding for that particular ID which is ID number three so I also
43:03
written this over here if we compare the embedding Vector for token ID3 we see
43:09
that it is identical to the fourth row so look at this this Vector it's exactly
43:14
the same as the fourth row right uh in other words the embedding
43:20
layer is essentially a lookup operation that retrieves rows from the embedding layers weight Matrix via a token ID
43:27
let me explain this in simpler words the embedding layer is essentially just a lookup operation and what this lookup
43:34
operation does is that if you give it an ID number it looks for that particular row and retrieves a vector for you so
43:41
for example if you want the vector for ID number five all you will need to do is look at this particular row which is
43:48
row number six and then it will retrieve or it will give you that particular
43:53
Vector if you look at if you want the vector for ID number Z just look at row number one if you look at the vector for
44:01
ID number or if you want the vector for ID number one just look at row number two so that's why you can so the
44:08
embedding weight Matrix is of course a matrix of the weights for Vector embeddings but it's also a lookup table
44:15
if you specify the ID number you can use the embedding weight Matrix to find the exact Vector representation for that
44:22
particular ID so that's why if you look at the P documentation this embedding is
44:27
also called as a simple lookup table I hope you have understood this right now okay so uh one major portion of this
44:36
lecture was for you to understand the embedding weight Matrix and why it is actually considered to be a lookup
44:43
table awesome now let's come to the next part so previously we have seen how to
44:48
convert a single token ID into three dimensional embedding right we just gave a single token ID and converted it into
44:55
an embedding vector but remember what we started from I wanted the vector representations for these four IDs so I

***

* 45:03
wanted the vector representation for ID number uh one ID number five ID number
45:09
two and ID number three so how can we give these four to the lookup table we
45:14
just specify the particular array so here we have the input IDs right uh we
45:19
have the input IDs for which we want the vector representation all we do is that we just use the embedding layer and pass
45:26
in the input IDs so similar to what happened here what this operation does is that it
45:33
first looks at the input IDs and it sees that there are actually Four values in the input ID then what it does it goes
45:40
through each individual ID and looks up the embedding Vector for that ID that's it so when you
45:47
pass in the input IDs it will first look at uh this thing and it will look at
45:52
first it will look at row number three then row number four row number six and row number two two so it will look at
45:58
row number four row number six row number three and row number two and then it will print out the
46:05
answer so essentially each row in this output Matrix is the corresponding Vector embedding for that particular ID
46:13
so the only thing which you have to remember right now is the embedding layer is a lookup Matrix and you can
46:18
pass a single ID to this lookup Matrix you can even pass multiple IDs or a group of IDs and in just one line of
46:25
command you can get all the vector embeddings for that particular token ID
46:32
this is how embedding layer is actually implemented in practice right now uh small random values have been initiated
46:39
but um we are going to train these values so that they actually capture the
46:45
meaning and that is what we'll come to in one of the subsequent
46:50
lectures okay so let's see how much of the lecture we have covered so far we covered this part with where we saw that
46:58
uh we saw that the embedding layer is essentially a lookup operation that retrieves rows from the embedding layer
47:05
weight Matrix using a token ID here is an image which also explain this so
47:10
let's say this is the weight Matrix uh embedding Matrix and let's say uh these
47:16
are the token IDs which we want to embed or we want to find the vector
47:21
representations so if you actually pass in these token IDs to the embedding layer what it will do is that it will
47:28
first look at each particular ID so it will look at ID number two which means it will go to row number three which is
47:35
highlighted in blue that will be the first answer of this lookup table then
47:40
it will look at ID number three and that will mean row number four that is the second row of the final answer then ID
47:48
number five which is essentially row number six and then uh it will give the
47:54
vector corresponding to row number six and finally uh ID one which means row number two so
48:01
then it will give the vector corresponding to row number two that's it so this is the embedding weight
48:07
Matrix and then it just looks at the particular row based on these IDs and
48:12
then it gives the vector embeddings for all the IDS which we asked for so here we asked for fox jumps over dog and then
48:19
it gives the vector embeddings for all those IDs so if someone asks you what's an
48:25
embedding layer you can say that it's a simple lookup operation that retrieves the vector for the particular token ID
Embedding layer vs neural network linear layer
48:32
that's it now I want to show you one last thing actually uh it is a bit of a finer
48:40
detail but I think it's important for you to also think about an embedding layer in some another dimension right so
48:47
let's say we have the following three training examples let's say we are we want to have the embedding for ID number
48:53
two ID number three and ID number one so let's say there are four words in our
48:59
vocabulary and uh which means that 0 1 2 3 these are the IDS with these four
49:05
words and we want to encode each of these IDs into a vector so uh let's say
49:12
the embedding Dimension is five so each of these each of these IDs will have a vector with basically 1
49:20
2 3 4 five with five Dimensions so the ID number zero will have a vector embedding of five dimensions ID number
49:27
one will have a vector embedding of five Dimensions Etc ID number two will have a
49:32
vector embedding of five dimensions and ID number three will have Vector embedding of five
49:37
Dimensions so the embedding Matrix which we create we pass in the first argument
49:42
the first argument is the vocabulary size which is the four number of rows and then the second argument is the
49:48
embedding Dimension so that is five right and so then if you print out the embedding weights you will see that and
49:55
you can even try this in python the embedding weights will be these which are initialized to random values and
50:07
U you'll see that there are four rows and five columns because we have four IDs in the vocabulary and each ID has
50:14
five um has a vector with five Dimensions this is great then what we do is that we just
50:22
retrieve those weight Vector weights which we need so then we passing this
50:27
idx so we only need vectors for ID is 2 3 1 we don't need the vector for ID Z so
50:33
then these are the IDS which are passed and then we do the lookup operation and then this is the embedding table which
50:41
is extracted or the embedding Matrix which is extracted for IDs 2 3 and 1 now
50:46
one thing which I want to explain to you that this embedding layer is actually the same as a neural network linear
50:53
layer so uh I'll try to explain this quickly because I don't want to divert
50:58
you from the main purpose of the lecture but let's say if you have a neural network with four
51:03
inputs uh four as the input Dimension and then you have three batches of
51:10
inputs coming in the first batch is basically the first ID number which is two and encoded as a one hot
51:17
representation the second batch is the ID number uh so let's see the IDS which we needed ID is 231 right so the second
51:25
batch is the ID number three which is encoded as the one hot Vector 001 and the third ID is the ID number
51:31
one which is encoded as 0 1 0 if these three are the input batches
51:38
which are fed to this neural network and let's say there are five neurons here the output of this linear layer is X
51:44
into W transpose so what is w every neuron here will have four weights associated with
51:51
it because every input has four dimensions if you look at the first input it's zero 0 1 0 it has four uh
51:59
four dimensions so every neuron here will have four weights associated with it so if you look at this weight
52:05
transpose Matrix the First Column will be the weights of the first neuron the second column will be the four weights
52:11
of the second neuron the third column will be the four weights of the third neuron fourth column will be the four
52:17
weights of the fourth neuron and the last colum will be the four weights of the fifth neuron so let's say if x is
52:24
the input Matrix W transpose is the weight Matrix when you do X into W
52:29
transpose you will get essentially a matrix which has three rows and uh it has five columns why
52:38
three rows because for every input we have three inputs input number input
52:43
number one input number two and input number three and for each input we want a vector embedding with five Dimensions
52:50
so there is three rows and there is five columns so each row corresponds to the vector embedding for that particular
52:57
input now if you look at this if you look at this output look at the first
53:02
row 0. 6957 and if you look at this output you'll see that it's exactly
53:07
similar in fact both these outputs are completely similar so what is exactly
53:13
happening here what the embedding Matrix is actually doing underneath is that it's the same operation as a neural
53:19
network linear layer so what an embedding does is actually is that you have inputs right so let's say we have
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













