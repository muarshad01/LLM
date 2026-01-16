ecture
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from
0:10
scratch series first let me take you through what all we have learned so far in this
0:17
lecture series through this diagram so in this lecture series we are going to
0:22
build a large language model completely from scratch and we are going to do that in three stages in the stage one we will
0:31
uh lay the foundations for building an llm in the stage two we will pre-train
0:37
the llm and in stage three we are going to fine tune the llm we are still at stage one and until
0:44
now we have covered two aspects of stage one we have looked at the data preparation and sampling which included
0:52
tokenization vector embeddings and positional embeddings and very recently we have looked at the attention
0:58
mechanism in a lot of detail DET in particular those of you who have followed the attention mechanism we had
1:05
a very detailed uh four to five lectures which started from simplified self attention self attention causal
1:12
attention and multi-head attention if you have not been through these lectures I highly encourage you to
1:18
go through them because attention really serves as the fundamental building block
1:23
to understand everything which follows if you have watched all the previous
1:28
lectures and if you have run the code which I have been providing it's amazing and I would like
1:34
to congratulate you that you have reached this part understanding attention is one of the most difficult
1:41
aspects of understanding large language models and if you have reached up till here the rest of this will be easier for
1:48
you so let's get started in these subsequent lecture videos which are to
1:53
follow we are going to learn about this part number three which is the large language model architecture as I always
2:00
do I'm going to break this into multiple videos I will not cover everything in one video um we will cover every single
2:08
video in a lot of detail and completely from scratch today right now it's the first
2:14
video in the large language model architecture module so let's get started
2:19
I think this will be a very interesting module for all of you especially to those of you who have followed until now
2:25
we have learned about the attention mechanism we learned about input embeddings we learned about position
2:30
embeddings but all of you must be thinking how does all of this really come together to give me something like
2:36
a GPT where does the training happen where does back propagation happen where
2:41
are the neural networks here if you remember at the start I told you that large language models are just deep
2:48
neural networks where are neural networks and what exactly is the Transformer we learned about the
2:54
attention mechanism and uh you must have heard about this that um attention
3:00
mechanism is at the heart of Transformers but what really is Transformers when do we do the training
3:06
and where do we generate the next word as the output all of that will become
3:11
pretty clear to you as we slowly start unraveling this box of the llm
3:16
architecture I really had a lot of fun learning about this and uh let's get
3:22
started as I told you llm architecture I'm planning to cover in four to five videos and today is the first video
3:30
so after learning about the attention mechanism in the previous lectures let us learn about the llm architecture now
Overview of the LLM architecture
3:38
I want to give you initially a view of what the llm architecture really looks
3:44
like this is the birds ey View and we are going to cover every single aspect of this in detail but right now I want
3:50
to show you what all you have learned and how does that fit in the context of
3:55
what's to come next this always helps in the learning process imagine if you are getting
4:02
walking through a forest right and if you want to get to the other side it's always good to know to track your path
4:09
to have some kind of feedback like okay this is what you have covered right right now and this is what's next to
4:16
come so that you can relate what you're learning next with the learnings from the past and that helps you reach the
4:22
end of the forest in our case learning about how the previous
4:27
knowledge fits into what we are are going to learn about next will really help you learn about llms in a much
4:33
better manner so initially we started with tokenizing then we looked at Vector
4:40
embedding and positional embedding the final embedding lay vectors which we had
4:46
for every token were then converted into context vectors through M MK multi-head
4:53
attention so the main aim of attention or rather the multi-ad attention was to
4:59
take the input embedding vectors and convert them into context vectors context vectors are a much richer form


***


5:05
of representation than embedding vectors because they not only contain semantic meaning of the token but they also
5:12
contain information about how the token relates to all the other tokens in the sentence
5:18
now uh mask multi-head attention forms a very important part of something which
5:24
is called as the Transformer block Transformer block is the most important part of the large language model
5:32
architecture and it's a block which actually consists of many different aspects which are linked together so let
5:38
us zoom into this Transformer block a bit unravel it open this block and see
5:43
what it contains if you zoom into the Transformer block you'll see that it contains a number of things and mask
5:49
multi-head attention forms a part of this so whatever you have learned in the multihead attention comes over here so
5:55
imagine you have a sentence such as every effort moves you and you want to
6:00
predict the next word right the first step is to convert each of these into input embeddings or vector embedding so
6:07
these are these and let's say we also add a positional embedding right these embedding vectors are then passed onto
6:14
the Transformer block the first part of the Transformer block is a layer normal normalization the second is M multihead
6:22
attention which converts the input embedding tokens into context vectors these are then passed into a Dropout
6:29
layer you can notice this plus signs so these arrows which run from here to this
6:35
plus sign they are called as shortcut connections the output of the shortcut connection goes to another layer
6:41
normalization then we have a feed forward neural network here then another Dropout layer is connected and there's
6:47
one more shortcut uh connection here and if you zoom into the feed forward neural
6:53
network further you will see that it has something which is called as the JLo activation if you look at all these
7:00
terminologies and you you think what does it mean what is layer normalization what is Dropout what is the JLo
7:06
activation why do we have a feed forward neural network here and why are all these things stacked together like this
7:13
that's all what we are going to cover in this video and the four to five videos which are going to follow forward but
7:19
remember this entire architecture has a large number of trainable parameters and trainable weights when the llm is
7:26
pre-trained these weights and these parameters are optimized and ultimately we get the output the outputs are such
7:33
that they have the same form and dimensions as the inputs and the outputs are then processed further which gives
7:40
the final text so once we get the output from the Transformer block it goes to these output layers and then the which
7:47
decodes the output from the Transformer block and we get the next word so every effort moves you was the input if you
7:54
remember and the next word is forward I just wanted to give you this bird eye view of what exactly is going on and
8:02
what we are building what you have learned so far and how it fits into what we are planning to learn next in these
8:08
set of lectures which we which are going to follow we are going to zoom into this Transformer block and we are going to
8:14
understand every single thing which has been mentioned here we will learn about first of all we'll learn about how to
8:20
stack these different layers together which will be in today's lecture then we will dive into each individual layer and
8:26
learn about them we'll have a separate lecture on layer normaliz ation a separate lecture on the shortcut
8:31
connections a separate lecture on feed forward neural network with J activation we'll stack all of these together and
8:38
then finally we'll have a separate lecture on how this output from the Transformer is decoded to produce the
8:44
next World okay so I hope you have understood why we learned about the mask
8:49
multihead attention because if we had not learned about this see this forms such a critical part of this Transformer
8:55
block right to learn about this one small block is it took us five lectures spanning over 7 hours but that's the
9:02
importance of the attention mechanism if this block is removed uh if this Mass multi-head block
9:08
is removed it's like the large language models would lose all their power and
9:13
then we are back to the age of recurrent neural networks and long short-term memory networks okay so let's see what we have
9:21
learned so far we have learned about input tokenization we have learned about embedding token plus positional and we
9:27
have learned about mask multi head attention Okay so uh let me first give
9:33
you a brief overview of the Mask multihead attention in which you have if in case you have forgotten so we have
9:39
the input embedding vectors which are stacked together like this we have a bunch of keys queries and the value
9:46
matrices which are multiplied with the inputs to give the queries the keys and
9:51
the values the queries are multiplied with the keys transposed to give us attention
9:57
scores which are then converted into attention weights attention weights are then


***



10:02
multiplied with the values Matrix to give us the context vector and since we have multiple attention heads the
10:08
context vectors are stacked together to give us a combined context Vector this is what is happening in the multi-ad
10:14
attention block now uh this whole process of what all we have learned so far can be visualized like this also if
10:21
you have the input text which is every effort moves you it's first tokenized and GPT uses a bite pair tokenizer which
10:28
we learned about before every single token is converted into a token ID every
10:33
single token ID is converted into a vector embedding which is a vectorized representation these Vector embeddings
10:39
are passed into the GPT model which consist of the Transformer block which I showed you before then there is an
10:45
output that output is further decoded and that gives us the output text for gpt2 the token embeddings which
10:53
were used had a embedding Vector size of 768 Dimensions which means each token ID
10:58
was converted into a vector of 768 Dimension and the output is generated such that the dimensions are matched so
11:05
the output is a 768 dimensional Vector for each 768 dimensional input token
11:11
embedding and then we do some postprocessing with the output so that we generate the next word which is
11:16
forward so every effort moves you forward great so what we are yet to
11:23
learn is the Transformer block and we'll start learning about this in today in today's lecture we'll dive slowly deeper
11:30
and deeper into every single layer of this block in subsequent lectures so for this set of four to five lectures we
GPT-2 model architecture overview
11:36
will not use a toy problem we will not use a toy model we are directly going to use
11:42
gpt2 so we will use the same architecture which was used to build the gpt2 model so if you look at this
11:49
paper this was the paper which introduced gpt2 and if you look at the models which
11:55
they had they had uh they had a small model model and they had a large model which has 1542 million
12:02
parameters if you look at the small model it had 117 million parameters this was revised later to be 124 million
12:10
parameters which is what we are going to use for these set of lectures and for the rest of these video series as well
12:16
so we are going to construct an llm with 124 million parameters which has 12 layers what are these layers which means
12:23
we'll have 12 Transformer blocks and uh D model which is the vector embedding
12:28
size is 76 these are the parameters which we are going to use in today's lecture and also in the rest of the
12:35
lectures um so why are we using gpt2 and not gpt3
12:40
or GPT 4 one reason is that gpt2 is smaller so it's better to run it locally
12:46
on our local machine uh and second reason is that open AI has made only gpt2 weights public opena has really not
12:54
made the weights of gpt3 and gp4 public yet uh so that's the thing with open
13:00
source right open a is closed Source right now whereas meta's Lama models are open source so all weights have been
13:06
released so that's why we are sticking with gpt2 because its weights have been made public we'll we'll load these
13:12
weights later in one of the subsequent videos so here is the configuration which we are going to use and uh to all
13:20
those who are watching the video you can pause here and try to understand whether you understand every single terminology
13:25
here we have covered all of these in the previous lecture so I'm I'm going to pause here and ask you to also pause on
13:32
your end and try to think about these terminologies I'll anyway explain each of these terminologies but I want you to
13:39
just give it a shot and try to understand okay so so let's go step by
13:44
step the first is the vocabulary size this means that uh every we start with a
13:49
vocabulary so um the gpt2 uses a bite pair encoder right so it's a subword
13:56
tokenizer so the vocabulary is how many subwords are basically there uh this
14:01
will be used for tokenization so if the vocabulary is a word level tokenization so if the sentence is every step moves
14:09
you forward then the vocabulary will have every step moves you forward so that way the tokenization will happen
14:16
but if you use a bite pair encoder with gpt2 uses it's a subword tokenizer so
14:21
the vocabulary size is 50257 and it may contain of characters
14:26
it may contain subwords it may contain full words also but this is the vocabulary size which we deal with when
14:33
we consider uh gpt2 this will be very useful for tokenization so when we do
14:41
tokenization what happens is we have a vocabulary and there are tokens in the vocabulary and there's a token ID with
14:48
respect to every single token and whenever whenever a new text is given to us using that vocabulary
14:54
that text is converted into tokens and then those tokens are converted into to token IDs if some text does not belong
15:02
to the vocabulary that's called as the out of vocabulary problem the bite pair encoder does not face this issue because
15:08
it's a subw tokenizer we have covered about vocabulary size in our lecture on
15:14
embedding so if you are unclear about this please refer to that the second is the context length the context length
15:20
basically refers to how many maximum words are used to predict the next word so if there is context length is 1024
15:28
which was actually used in gpt2 we are going to look at one24 words and we are going to predict the next word maximum
15:35
there will be no case when we are looking at 2,000 words let's say and predicting the next word when I say word
15:40
I'm actually meaning token here which is not exactly correct because gpt2 uses



***



15:45
the bite pair encoder toker to tokenizer which is subword tokenization scheme but
15:51
for the sake of this lecture if I use word and token interchangeably it's because it's good for intuition the
15:58
second thing is the embedding Dimension now every token in this vocabulary which we have will be projected into a vector
16:04
space such as this so for example the tokens are your journey starts with one step here is a three-dimensional Vector
16:12
representation of every single token right um and the embedding should be
16:18
such that the meaning is captured so for example if journey and starts are more similar in meaning they would be closer
16:25
together in this embedding space so this is a three-dimensional embedding embedding space in gpt2 we are using a
16:31
768 dimensional embedding space it's very difficult to show this over here but you can imagine a 768 dimensional
16:38
embedding space in which the words are projected now if you are thinking how do we learn about these projections how do
16:44
we know which Vector Journey corresponds to now that's also trained in gpt2 when
16:50
we look at the Transformer block you'll see that the embedding itself is not fixed we are going to train the
16:56
embedding layer so that uh every word is embedded correctly so that semantic
17:02
meaning is captured the next thing is the number of heads and these are the number of attention heads which are
17:08
equal to 12 so if you look at this diagram over here I told you that multiple queries keys and values Matrix
17:15
matrices are created right so the more the number of attention heads the more the number of these matrices are created
17:22
so if we have 12 attention heads it means there will be 12 such queries keys and value matrices so here the number of heads is
17:30
12 number of layers is the number of Transformer blocks remember this is different than the number of attention
17:35
heads number of layers is how many such layers are we going to have so this is one one Transformer block layer and it
17:43
includes multi-ad attention so within this one layer there will be 12 attention heads but in terms of these
17:50
Transformer blocks itself there can be 12 blocks so it's not necessary that the
17:55
number of layers and number of heads are similar here we are using 12 Transformer blocks
18:02
U which will which will see later how they are stacked up together okay so number of layers is 12
18:08
then drop rate is basically the dropout rate and uh query key value bias is or Q
18:15
KV bias is the bias term when we initialize the query key and the value matrix by default this is always set to
18:23
false okay so the number of Transformer blocks one more thing which I want to mention here is is that we are looking
18:29
at the gpt2 small model which use 12 transform which uses 12 Transformer blocks right but as we saw over here
18:36
they had four models of gpt2 so if you go from left to right here you'll see small the medium has 24 transformer
18:43
blocks the large has 36 Transformer blocks and the largest which is extra
18:49
large that has 48 Transformer blocks and you'll see that the dimensionality also increases from left to right we are
18:56
using 768 Dimension gp22 small but if you go from left to right you'll see that 10241 1280 and finally the gpt2
19:03
extra large has a dimensionality of 1600 okay so I hope you have understood
19:10
this this configuration and what we are now going to do is that now I'm going to take you
19:15
to code and I'm going to build a GPT placeholder architecture what does this mean this
19:22
basically means that whatever I showed you over here right this thing this
19:27
thing whatever I showed you I know that you have not yet understood the layer normalization the shortcut connection
19:34
even the Transformer block what it exactly has has what the speed forward neural network is what the JLo
19:39
activation is right now what I want to do is I just want to create a skeleton for our code where these different
19:45
blocks will come in together we'll code them later in subsequent parts and we'll have a separate lecture for each of them
19:51
but right now we'll build a GPT placeholder architecture which will also called as the dummy GPT model
19:59
this will actually give a bird's eyee view of how everything fits together so here I have shown a bird's eye and this
20:05
is a bird's eye view so the reason this Birds Eye is again very important is that you'll see what we are planning to
20:11
do in the subsequent lectures and that's why the skeleton is very important especially for a complicated topic like
20:17
the llm architecture where multiple things have to fit in together first let's zoom out and see what all has to
20:23
fit in together and then in subsequent lectures we'll start coding it out so I'm going to take you to code right now
Begin coding the GPT-2 architecture
20:29
this is the GPT configuration 124 million parameters which we are going to use so let's jump right into



***



20:36
it okay so now what we are going to do is we are going to implement a GPT model
20:42
from scratch to generate text and I'll show you exactly how the code is executed but at every single step of the
20:48
code I'll again take you to the Whiteboard so that you can visualize what every parameter means it's very
20:54
important for you to read a sentence of the code and to visualize how it how how it looks like only then you'll really
21:01
understand the code so this is the GPT configuration which we covered on the Whiteboard I hope you have understood
21:06
the meaning of every single terminology here if not just look up the meaning once more or go through our previous
21:12
lectures but it's very important that you don't just skim through it without understanding the
21:17
meaning okay now as I told you we are going to build the GPT architecture so this is a dummy GPT model class we'll
Dummy GPT model class
21:25
use a placeholder for the Transformer block we'll use a placeholder for the layer normalization okay so first let me
21:32
give you a broad overview of what all do we have here so we have a Dy GPT model over here and it has the forward pass
21:40
what this forward or rather I should call it the forward method what this forward method does is that it takes an
21:46
input and at the end of this forward method we are going to uh print out the output this is what we are aiming to do
21:54
so if you look at the figure which um Let me show this figure to you
21:59
here this is the main thing right so what that forward method does is that it takes an input which basically can just
22:06
be these words and then the aim of this is to the aim of the forward pass is to
22:11
give you the next word in this case the next word is forward so that's the output so all of what we want to
22:17
implement somewhere lies in the middle right um so there are two main blocks
22:24
which will be very important to us so there is first the Transformer block we we are going to create a class for the
22:29
Transformer block not in this lecture but in later lectures and we are going to create a class for the layer
22:35
normalization let me show you where these come into the picture so if you look at the Transformer block over here
22:42
the Transformer block consists of all of these things right and layer normalization is a very important part
22:47
of it layer normalization will also be implemented before the Transformer and after the Transformer but it is also
22:54
present within the Transformer itself so we'll have a Transformer block we'll in which we'll put all of these things what
23:00
I'm showing here and we'll have a separate layer normalization block the reason we are having a separate layer
23:06
normalization block is that it comes in the Transformer block that's fine but it also comes at other places so it's
23:13
better to define a separate class of it so this is the class which will Define later not now this is the class will
23:19
which will Define later not now now let us see what what this forward method is
Passing the input
23:24
actually doing okay so the forward method first takes in an input
23:30
and uh let me show you what that input actually looks like
23:35
um okay so I have just made some visualizations
23:42
over here so that you understand what's going on yeah okay so the forward method
23:47
is going to take an input right and the input let's say is this same thing let me write it over here what's that input
23:55
the input is every effort moves you let's say this is the input which is
24:00
which is passed to the forward method let me write over here right and I'm going to write this with a different
24:06
color so let's say the input is
24:14
every every effort
24:21
moves you okay great so this is my input right
24:27
now the way this will be fed to the forward method is that uh let me actually show you
24:34
that so we are going to feed this input to the forward method doing something
24:39
like this so let's say every effort moves you is the input right we are first going to use the tick token
24:46
tokenizer which is the bite pair encoder and we are going to convert these tokens into token IDs so remember the workflow
Vector embeddings
24:53
which we saw over here every token here see every token is essentially converted
24:59
into token IDs and then everything later after this point happens within the GPT
25:05
model class but till this stage we have to do it outside and then pass the token
25:10
IDs to the GPT model class so now we have this every effort moves you right
25:16
this will be converted into a token ID this will be converted into a token ID this will be converted into a token ID
25:22
and this will be converted into a token ID right the first step is that every token ID will be converted into token
25:30
embedding uh and so what that means is every token ID so let's say this is ID
25:37
1 let's say this is ID 1 this is id2 this is ID3 and this is ID 4 right each
25:45
of these token IDs will need to be converted into a 768 a 768
25:53
Vector 768 Vector embedding essentially that is going to be the input embedding
25:58
and the way we are going to do that is that we are going to first create a token embedding
26:04
layer and for that we will use the nn. embedding in pytorch what this layer
26:10
actually does is that uh it creates this Matrix which is called as the token embedding Matrix it has rows which is
26:18
equal to the model vocabulary size and every row basically corresponds to one token ID and every Row the length of
26:25
every row is essentially 768 so now if you want to uh find the vector embedding
26:31
for ID number one let's say ID number one is 44 you just look at the 44 throw
26:36
over here and you get the 768 dimensional Vector if you want to look at the let's say for effort the ID is 64
26:44
you look at ID number 64 for effort you get the 768 dimensional Vector similarly
26:50
U let's say the ID for U is 85 or or rather 40,000 you go downward and you
26:56
get you go to the 40,000 row and you get the 768 Vector 768 dimensional input
27:02
embedding Vector now that's why this token embedding Matrix is also called as the
27:08
lookup Matrix you just pass in the token IDs and it gives you the vector embeddings remember that all of the
27:14
parameters here here everywhere in this token embedding Matrix they will be initialized randomly for now and we will
27:20
train these parameters so when we initialize this token embedding layer it initializes the parameters from a goian
27:27
distribution and then they are initialized randomly later when we do back propagation we'll train these for
27:33
now when when you look at all these embedding matrices just just know that their values are random for now okay so
27:40
the first step is to convert all of these tokens into um token embeddings which are 768
27:46
embedding uh vectors and you'll see that that has been done over here so when you
27:52
go to the forward method first what you do is you look at the input shape right the input shape is basic basically batch
27:58
size which are the number of rows and the sequence length which is essentially
28:05
uh the length of the number of tokens which we are considering so for example let us look at
28:12
this this is for example one such batch right so I have in this batch two the
28:18
batch length the batch size is two so there are two rows in this tensor and the number of columns are the number of
28:23
tokens which I'm going to use for now let's just look at one batch so I'm I'm feeding reading in four tokens and which
28:30
are my input inputs and then I want to get the next word so every effort moves you forward which will be the next to it
28:36
right so that's why the shape is batch size and sequence length so batch size is in the example which I showed you
28:42
there are two batches so two rows and sequence length is the number of tokens basically great so the first thing what
28:48
we are going to do is that we are going to create the token embeddings out of
28:54
the input um input index which is the inputs which we have given now take a
29:00
look at these inputs and just look at the first batch the first batch is a list of token IDs which you which uh
29:06
have been mentioned over here what we'll do with these token IDs is we will then query or look up the token embedding
29:13
Matrix and then retrieve those input embedding vectors so for this token ID there will be a 768 dimensional Vector
29:20
for this token ID there will be a 768 dimensional Vector Etc so you might be thinking where is that embedding Matrix
29:27
so that has been created in the init Constructor which is invoked by default so see first we have a token embedding
29:33
Matrix which has been created uh the number of rows of this Matrix are equal to the vocabulary size exactly what we
29:40
have written over here the number of rows of this token embedding Matrix is equal to the
29:45
vocabulary size and the number of columns of this token embedding Matrix is the embedding Dimension why because
29:52
every token or every token ID has a 768 dimensional Vector associated with it so
29:58
the number of uh columns which are there is equal to the 768 so for every token
30:04
ID essentially there will be 768 columns uh so this is the embedding uh token
30:11
embedding weight Matrix which has been created using this pytorch embedding class and what we are doing here
30:18
essentially in the forward method is that we are looking at the input token IDs which have been mentioned in our
30:24
batch and we are going to look up that token embedding Matrix and we'll get the
30:29
token embeddings for the inputs so we'll essentially have four four 768
30:34
dimensional vectors for the first batch and we'll have four 768 dimensional vectors for the second
30:40
batch great The Next Step which we are going to do after getting the token IDs is we have to get the positional
30:47
embedding right so remember up till now we have uh we have a 768 dimensional
30:52
Vector for id1 a 768 dimensional Vector for id2 a 768 dimensional VOR VOR for ID
30:58
3 and a 768 dimensional Vector for ID number four now what we are essentially going
Positional embeddings
31:05
to do is that we are going to add a positional embedding to each of these
31:11
four vectors okay so for that first we need a positional embedding weight Matrix very similar to the Token
31:17
embedding weight Matrix so remember the positional embedding really depends on the context size because uh at for uh
31:25
we'll take let's say Contex size is 1024 at Max we'll use 1024 tokens to predict
31:32
the next word right so we just need to know the uh let's say the position is
31:37
one so let's say the position is one we need a positional embedding Vector for
31:43
this position if the position is three or two we need a positional embedding Vector for this position similarly if
31:49
the position is 1024 we need a positional embedding Vector for this we don't need a positional embedding Vector
31:54
for one to5 why because we are not we are not looking at uh the context window
32:00
of 1025 we are only going to look at maximum one24 tokens at once and predict
32:06
the next word so that's why the number of rows in the positional embedding Matrix is the
32:11
equal to the context size but if you see the number of columns they are still equal to the embedding size and that is
32:18
important because we are going to add the positional embedding vectors to the Token
32:23
embedding uh to the Token embedding vectors so here there are four token embedding vectors of 768 Dimensions each
32:30
right to each of these we will have four positional embedding vectors so now uh
32:37
every effort moves you let's say so it's position number one position number two position number three and position
32:43
number four right so we will get the positional M Vector corresponding to First Position second position third
32:50
position and fourth position and we'll add them uh to each of these token
32:56
embedding vectors the context size is 1024 but for now I'm just showing you a
33:01
simple version of four context size but the main point I'm trying to illustrate
33:07
here is that once we get the token embedding vectors for these four tokens We'll add them with the positional
33:12
embedding vectors okay so here you can see that first a positional embedding Matrix is
33:18
also initialized when this init Constructor is called and the number of rows here are equal to the context
33:24
length which is exactly what I showed you uh over here the number of rows are equal to the context length or the
33:31
context size and the number of columns are equal to the embedding Dimension so this is again a nn. embedding which is
33:38
very similar to the embedding class or exactly the same embedding class which we used for token embedding again these
33:44
will be initialized randomly for now we'll train them later now what we'll be doing is that
33:50
based on the positions um in the in the tokens we are
33:55
going to uh qu or we are going to look up the positional embedding Matrix so torch.
34:02
arrange sequence length what this will this is going to do is that it will look at the token length so in this case the
34:08
token length uh let's say right now the token length is equal to 4 right the
34:14
token length is equal to four in this batch which I given so it will arrange it as 0 1 2 and three and it will get
34:20
the positional embedding uh it will get the positional embedding vectors for row number zero row number one row number
34:26
two and row number three three and then it will then what we are going to do is we are going to add the token embeddings
34:33
for the four tokens and we are going to add the positional embeddings for the four tokens so the X the um we have the
34:42
input initially right the way it's been transformed is like this so the four tokens which we had in every batch uh
34:48
we'll convert them into 768 dimensional input vectors we'll then add the 768
34:54
dimensional positional vectors to each of them and then finally we have a 768 dimensional uh embedding Vector for each
35:02
of these tokens awesome so now next what we are going to do is the next step is
35:08
something which is called as drop EMB which is the Dropout embedding uh so this is just the dropout rate so what it
35:15
will do is that it will take these uh it will take the embedding vectors for um
35:20
all the tokens and it will randomly turn off some weight values this generally helps the generalization performance and
35:27
prev overfitting we'll look at this in detail in one of the next classes after
35:32
we get these uh embedding vectors let me show you the figure what happens next so
35:39
once we get these embedding vectors as I showed you over here
35:45
um uh yeah actually let me show it over
35:51
here yeah so once we get these uh token IDs and the token embeddings right uh
35:57
we'll then pass it to the GPT model which essentially consists of the Transformer block and then the output is
36:06
generated so after we get the yeah so after we get the input embedding and
Transformer block
36:12
after we add the positional embedding next what we have to do is we have to pass it through the entire Transformer
36:17
block and also we later have a layer normalization layer so this is exactly
36:22
what is done over here after we get these embeddings and we apply the Dropout layer we then pass it through
36:28
the Transformer block so in this one step actually several things are happening in this one step what we are
36:34
doing is that we are implementing multi-head attention we are implementing a Dropout layer we are implementing
36:40
shortcut connections we are implementing layer Norm we are implementing feed forward neural network with JLo
36:46
activation then another Dropout layer uh and then remember we have 12 of these Transformer blocks in
36:53
gpt2 and then finally we have another final Norm which is the um layer normalization layer towards the
Output logits
37:00
end and the important step which I want to highlight in today's lecture is this last step which is the output which we
37:07
have which are called as logits and there is a reason why they are called as logits so let me explain that to
37:14
you uh okay okay so when we reach the
37:20
when we reach the output what we'll be having is that we'll have four tokens and each of those four tokens we have a
37:27
76 68 dimensional representation that's the output vectors right but now we want to predict the next word based on the
37:35
input token based on the input sentence the input sentence as I mentioned was every effort moves you we have to
37:41
somehow predict the next word which is forward so after all of the Transformer
37:49
blocks have been implemented the output is such that for every token uh for every token we'll have a
37:58
768 dimensional Vector that's the output now what the main thing is that how will
38:05
we predict the next word and so the way this is done is that the final output
38:11
Matrix which we have which we have will have this format where there will be four tokens which are the number of rows
38:17
but there will be columns which is equal to vocabulary size which is 50257 and let me tell you why so if you
38:23
look at token number one uh actually before that when we look at an input
38:28
batch every effort moves you there are actually Four prediction tasks which are happening here uh you have you first
38:36
look at one word every and you predict the next word which is effort then you look at the next which is every
38:43
effort this becomes an input in the next and then you predict the next word then
38:48
the next input is every effort moves
38:57
and then you predict youu only then there is the fourth task which is every effort moves you and then you predict
39:03
the next word which is forward so when you look at this input batch which has four tokens or a context length of four
39:09
in this case we are actually doing four prediction tasks so when you look at token number one which is every we need
39:16
to predict what what's the next token right out of the vocabulary what has the
39:21
highest probability of coming next so if you look at the rows there will be the
39:27
column length will be 50257 and every element here will represent probabilities so you will then
39:33
take that element which has the highest highest probability so let's say that is the 40,000 that is the 40,000 column over
39:41
here so then we look at the vocabulary we'll look at the 40,000 token in the
39:46
vocabulary which and that seems to have the highest prior probability we'll choose only that token which has the
39:53
highest probability so that's the 40,000 column and then that 40,000 column will be effort now similarly when you look at
40:00
token two so the input will be every effort right and then you'll again look at the row and you'll see uh that column
40:09
number which has the highest probability and let's say this column number is 20,000 so you look at the token
40:14
corresponding to 20,000 in the vocabulary of 50257 and that should be
40:21
moves similarly every effort moves you will be the input and we'll again have a
40:26
token corresponding to you and then every effort moves you will be the input and then we'll have a token
40:32
corresponding to forward so that's why the output which we expect will have this format which has these tokens which
40:38
is the input sequence length and in this case it's equal to four and the vocabulary size will be
40:44
50257 and then this will essentially give us an idea of what the next word will be at every single prediction stage
40:51
since there are four input output prediction tasks in this uh input sentence so that's why if you look at
40:58
the output head later when we print out the output Dimensions it Dimensions will be the number of tokens and the number
41:05
of columns will be equal to vocabulary size so even if you look at the output head Dimension uh remember when we reach
41:12
up till this point the number of rows are equal to the number of tokens which is four number of columns is the
41:17
embedding Dimension which is 768 that will be multiplied by this neural network which has 768 rows and 50257
41:25
columns so ultimately the result which will come in logits will have four rows and 50257 columns very similar to uh
41:35
very similar to what we have seen over here four rows and 50257 columns don't worry in the subsequent lectures we'll
41:41
have a separate lecture for each of each of these but right now I just wanted to show you this overall thing of what we
41:47
are going to implement when we reach the end of the next four to five lectures we'll get these logits Matrix so that
41:53
we'll know what the next word in the prediction is now now see here what we are doing is that we have taken two
42:00
texts every effort moves you is text number one which is also batch one and the second text is everyday holds up
42:07
right so we are creating a batch which has two texts and the first step as I mentioned is to get the token IDs so
42:13
these are the token IDs these four token IDs for the first batch these are the four token IDs for the second batch what
42:20
we do then is that we create an instance of the dummy GPT model with the configuration as I mentioned above with
42:27
the configuration as this right and although we have not defined anything over here and
42:33
everything is placeholder right now we can still run this code um nothing is initialized here right so this block
42:39
currently does nothing and even this layer currently does nothing but we can still execute this code and get the
42:44
output so what will happen is that these two blocks will not essentially do anything but we can still it's a
42:50
functional code so we'll still get the Logics so let's see what the result is uh so we'll pass in this batch to the
42:57
model right now and let's see the output shape so once we pass in this batch try to visualize the steps which are
LLM Architecture workflow summary
43:03
happening right uh and your visualization should follow this workflow for now think of the workflow
43:11
so first I have this first look at only one batch so I have four token IDs these are the four token IDs these four token
43:18
IDs will be converted into 768 dimensional input embedding vectors then
43:24
I will add positional embedding vectors to each of them the resultant will be passed through um let's see what the
43:31
result yeah the resultant will be passed through a Dropout layer then I'll have then will go through the Transformer
43:36
block then the result will go through the another uh normalization layer which
43:44
is also called as layer normalization layer and then until this point when I reach this stage the output will have
43:51
four rows corresponding to the four tokens and 768 columns because the embedding Dimension is 768 that's for
43:57
one batch now this will go through the output embedding uh output head which is the final neural network and then the
44:04
result will be number of tokens which is four and 50257 columns because I want to
44:10
now get logits and see which one which word should come next so here you can see the result for the first
44:17
batch there are four rows and then 5 0257 columns this is exactly similar to
44:25
the output which I was showing you over here the output Logics should have four tokens as the rows and 50257 columns and
44:32
similarly for the second batch there are four rows and 50257 columns each
44:37
parameter here should ideally represent the probability of the next token remember when you look at these
44:44
four tokens there are four input output tasks Happening Here not just one so every is the first input and effort
44:51
should be the first output every effort is the first is the second input and moves should be the second output every
44:58
effort moves should be the third input and U should be the third output every effort moves U should be the final input
45:05
and the output should be forward so right now these these outputs are random because we have not trained anything but
45:11
ultimately we'll just add the back propagation algorithm and then all of these probabilities will start to make
45:17
sense later we'll also apply soft Max Etc to these logic so that they'll be between 0 to
45:23
one okay so here you can see that the output tensor has two rows corresponding to the two text samples this first row
45:30
corresponds to the first text sample the second row corresponds to the second text sample each Tex sample consist of
45:37
four tokens so first token first row correspond to First token second to the
45:42
second token Etc and each token is a 50257 dimensional Vector which matches
45:47
the size of the tokenizers vocabulary so yeah here you can see each token is a 50257 dimensional vector and it encodes
45:54
the probability of what should come next the embedding has 50257 Dimensions because each of these Dimensions refers
46:01
to a unique token in the vocabulary at the end of the series of lectures when we implement the postprocessing code
46:08
we'll convert these 50257 dimensional vectors back into token IDs which we can decode into what word comes
Next steps and recap
46:16
next okay uh so in this lecture we have looked at a top down View at the GPT
46:22
architecture what are the inputs what are the outputs Etc and uh I hope you have gotten a sense of
46:29
what all we are going to implement in the subsequent lectures so in the next lectures we'll go through every single
46:35
block sequentially so my next lecture is planned for layer normalization then after that we have a lecture on feed
46:42
forward neural network with J activation then we have a lecture on shortcut connections then we'll have a lecture on
46:48
uh how all of these come together in the Transformer block and then finally we'll have the entire GPT model implementation
46:55
and the last lecture will be gener text from output tokens so this logits Matrix which is there right which you obtained
47:01
over here uh where was that yeah here was the
47:06
Logics Matrix which was uh which was returned how to convert this into the next word we'll see that in the last
47:12
lecture in this series of lectures okay so that that actually brings me to the end of this lecture I
47:18
want to leave you with this one image this one image so what we have
47:25
learned right now is this GPT backbone so we have started this series with understanding the GPT backbone but
47:31
remember this GPT backbone consists of layer normalization jalu activation feed
47:37
forward Network and shortcut connection and all of these actually come together
47:42
all of this feed in together um to make what is called as
47:47
the Transformer block um that's why it's called
47:53
Transformer so you might be thinking what exactly is the Transformer and why do we say that that attention is the heart of it the reason people say
47:59
attention mechanism is the heart of Transformer because if you if you unlock or Unravel the Transformer block you'll
48:06
see that the mass multi-ad attention is a crucial component of it but there are several other components which you also
48:12
should be aware of and we'll cover that I hope you have got a bird's eye view which I planned for in today's lecture
48:19
um thank you everyone I hope you're having a lot of fun with these whiteboard notes as well as through this
48:25
coding assignments as well as through this coding part the Transformers lectures which came before were a bit
48:32
complicated but now it's getting a bit easier so you have been through the hard part of the course so congrats for that
48:39
and now comes the very interesting part later thanks everyone I'll look forward to seeing you in the next lecture


