
0:00
[Music]
0:04
hello everyone welcome to this video in
0:06
the build large language models from
0:08
scratch Series this is a conclusion
0:11
video in which I'll try to quickly
0:14
summarize what all we have completed in
0:16
this lecture Series so
0:18
far you may be into two categories of
0:21
students the first is who has followed
0:23
this entire series and who is watching
0:26
this video congratulations you have made
0:28
it this far you now one you are now one
0:30
of the few students in the whole world I
0:32
would say who knows how to build an
0:34
entire large language model completely
0:36
from scratch if you have stumbled upon
0:39
this video and if you don't know about
0:41
the rest of the playlist I highly
0:43
encourage you to go through all of the
0:45
videos to develop a very firm
0:47
understanding of large language models I
0:50
started this lecture series with the
0:51
goal of teaching you how to be a
0:54
fundamental researcher or a fundamental
0:57
engineer in machine learning the main
1:00
thing which you should really focus on
1:02
is to understand the nuts and bols of
1:04
how things work instead of running fancy
1:06
large language model applications we
1:09
should focus our attention to how large
1:11
language models are really built we
1:13
should understand the nuts and bols of
1:15
large language models and that has been
1:17
the whole focus of the series The videos
1:19
have been long but they have been very
1:21
satisfying um as I have seen from
1:24
interactions with many students who have
1:26
watched the
1:28
videos uh when when I started this
1:30
lecture series I took a lot of
1:32
inspiration from this book build a large
1:34
language model by Sabastian rashka so
1:36
thank you
1:38
Sabastian and uh this really helped me
1:41
construct the courses or construct the
1:43
entire lecture series here is the code
1:46
file which we have run throughout these
1:48
entire or throughout this entire lecture
1:51
series as you can see the code has
1:54
become pretty long but every single
1:56
thing has been explained in a lot of
1:57
detail in the videos which we have seen
2:00
throughout this course I've explained
2:01
every single code block to you and we do
2:04
not assume anything in this lecture
2:06
series everything is covered in depth
2:08
from scratch I shared this code file
2:10
with all of the students who have been
2:12
watching and uh I hope that once you
2:15
have access to this Cod code file you
2:17
are able to run and construct the entire
2:19
large language model on your
2:21
own here is the workflow of everything
Recap
2:24
which we covered in this series first we
2:26
started with stage number one then we
2:29
sequentially we move to stage number two
2:31
and then we move to stage number three
2:33
just like to build a house we first need
2:35
to lay the foundation in stage number
2:37
one we laid the foundation for building
2:40
the large language model we looked at
2:42
data preparation and sampling then we
2:44
looked at the attention mechanism which
2:46
is the engine behind which behind which
2:49
or uh using which large language models
2:52
derive their
2:53
power um and finally we looked at the
2:56
large language model architecture in
2:58
stage one how different blocks are are
3:00
arranged together to construct the large
3:01
language model once stage one is
3:04
finished then we move to stage number
3:05
two which is essentially pre-training in
3:08
pre-training we looked at how to
3:10
implement the loss function for the
3:12
large language model how to do a
3:14
backward pass and how to train more than
3:17
100 million or the billion parameters
3:19
which are typically involved in the llm
3:21
we also saw how to load pre-trained
3:23
weights from uh large language model
3:26
such as open air gpt2 and that EX Ates
3:30
the pre-training process and then in
3:32
stage number three we saw that
3:34
pre-training is not enough to make sure
3:36
that the llm performs well on specific
3:38
tasks such as classification or making
3:41
our own personalized chatbot we learned
3:43
about fine tuning we completed two
3:45
handson projects fully from scratch we
3:47
made an llm classifier which can
3:49
distinguish between spam versus no spam
3:51
emails and we also built our own
3:54
personal assistant which can follow
3:56
instructions these were incredibly
3:58
rewarding experiences and we displayed
4:01
and understood everything from the very
4:03
Basics now what I'll do is that I'll
4:06
take you through individual steps and
4:08
key components of what all we have
4:10
covered in this journey so far if you
4:13
have watched the lecture Series this
4:14
will serve as a good recap for you and
4:16
you can also use it as a quick uh
4:19
revision before you go for interviews
4:21
and uh if you are if you have not
4:24
watched the previous lecture videos this
4:26
will serve as a good video for you to
4:28
see what all we will be covering in this
4:30
lecture Series so the first step of a
Data Preprocessing Pipeline
4:33
large building a large language model is
4:35
to implement the data pre-processing
4:37
pipeline the data pre-processing
4:39
pipeline for an llm is much different
4:42
than a regression model or a
4:44
classification model when you have input
4:46
texts in a large language model the goal
4:49
is to predict the next token right so
4:51
the first thing what you do is that you
4:52
have to convert this input text into
4:55
tokens and then you have to convert the
4:57
tokens into token IDs
4:59
after this the next step is to convert
5:01
these token IDs into the higher
5:03
dimensional Vector space so that the
5:05
semantic meaning of different tokens is
5:07
capture when you project the tokens into
5:10


***



higher dimensional Vector space then you
5:12
have to add uh the positional embeddings
5:15
to these token embeddings so the
5:17
projections of token IDs into higher
5:19
dimensional Vector spaces as has been
5:21
shown here are called as token
5:22
embeddings you have to add positional
5:24
embedding vectors to the Token embedding
5:26
vectors the reason we add positional
5:28
embeddings is because along with
5:30
converting tokens into Vector
5:32
representations the positions at which
5:35
individual tokens show up is also very
5:37
important when predicting the next token
5:40
when you add token embeddings to
5:41
positional embeddings it results into
5:43
something which is called as input
5:44
embeddings now these input embeddings
5:47
are the output which we expect from the
5:51
data pre-processing pipeline the whole
5:53
goal of the data pre-processing pipeline
5:55
is to take the input text from the huge
5:57
number of documents the training data
5:59
data which we are feeding to the large
6:01
language model and to convert all of
6:03
those documents into sentences then into
6:06
tokens then into token IDs then into
6:08
token embeddings then add positional
6:10
embeddings and convert it into input
6:12
embeddings to give you a sense of the
6:14
token embedding dimension in gpt2 the
6:16
embedding Dimension is of
6:18
768 right so that's the first thing the
Attention Mechanism
6:21
data pre-processing pipeline after this
6:23
we move to understand the attention
6:25
mechanism attention mechanism is the
6:27
driving engine which gives llm power so
6:31
the main idea of attention is that when
6:33
you look at Vector embeddings you just
6:35
look at semantic meaning of one token
6:37
right you do not look at the
6:38
relationship of one token with all the
6:40
other tokens however when you're
6:42
predicting the next token context is
6:44
very important you need to know how one
6:46
token relates to all the other tokens
6:49
when you look at one token how much
6:51
importance to should you give to other
6:52
tokens and that importance is also
6:54
called as attention so the whole goal of
6:57
the attention mechanism is to convert
6:59
the input vectors to convert the input
7:01
embedding vectors which look like this
7:03
into something which is called as the
7:04
context vectors so at the end of the
7:07
attention when the attention mechanism
7:08
is implemented the input vectors are
7:10
converted into context vectors so you
7:12
see the input embedding Vector for
7:14
Journey and here is the context Vector
7:16
for Journey context vectors are much
7:19
richer than input embedding vectors
7:20
because they also contain information
7:22
about how Journey let say relates to all
7:24
the other tokens in the
7:26
sentence and to go from the input
7:29
embedding vectors to the context vectors
7:31
there is a huge sequential flow uh which
7:34
we need to understand we first multiply
7:36
the inputs with the trainable query key
7:38
and the value matrices which give us the
7:41
queries keys and the value Matrix we
7:43
multiply queries with the keys transpose
7:45
which gives us the attention score then
7:47
we scale the attention score with square
7:49
root of the keys Dimension we add
7:51
Dropout we Implement caal attention
7:54
which means that we add a mask to all of
7:57
those tokens um which are not involved
8:00
in the next token prediction task and
8:01
then we add a soft Max layer so after
8:04
implementing the scaling plus Dropout
8:06
plus softmax we convert the attention
8:08
scores into attention weights the
8:10
attention weights are then multiplied
8:12
with the values and then we get the
8:13
context Vector Matrix now this is only
8:16
for one attention head when we consider
8:19
a large language model there are
8:20
multiple attention heads which are
8:22
acting the reason we have multiple
8:24
attention head is to capture multiple
8:25
dependencies and long range dependencies
8:28
within a large paragraph So when you
8:31
combine the context Vector matrices from
8:34
multiple attention heads you get this
8:35
ultimate final context Vector Matrix
8:38
that's the output when the input
8:40
embedding Matrix passes through the
8:42
attention head so the whole Revolution
8:45
which happened with respect to large
8:46
language models is because of this
8:48
workflow which I'm sharing on the screen
8:50
right now taking the input embedding
8:52
Matrix and converting it into this
8:54
context Vector Matrix that's the key
LLM Architecture
8:57
after you understood the attention
8:58
mechanism then we move to the llm
9:00
architecture now remember that I'm going
9:03
a bit fast here because this is a recap
9:05
uh this is a summary if you want to
9:07
understand each and every element I
9:09
highly encourage you to go to that
9:10
specific lecture and watch that entire
9:12
video again now the next step after
9:15
understanding attention is to look at
9:16
the llm architecture so this is the
9:18
bird's eye view of how the llm
9:20
architecture looks like and actually let
9:22
me take another figure which I think is
9:24
a better
9:25
representation of the llm architecture
9:28
so I'm just taking a screenshot sh of
9:30
this figure here and I'm going to move
9:32
it above so if you look
9:36
at right so if you look at this this
9:40
architecture this gives us
9:43
a a bird's eye view of what actually
9:46
goes on when you look at a large
9:47
language model so first as I mentioned
9:49
you have inputs U yeah first as I
9:52
mentioned you have inputs those are
9:53
converted into bunch of tokens then uh
9:56
the tokens are converted into Vector
9:58
embeddings we add embeddings and that
10:00
gives us input embeddings the input
10:02
embeddings are then passed into this
10:04
block into this uh yeah so token
10:07
embedding layer positional embedding
10:08
layer and then we have a Dropout layer
10:11
and then we have these input embeddings
10:13
which are then passed into this Blue
10:15
Block which is the Transformer block the
10:17
Transformer block is where all the magic
10:19
happens so you might be thinking okay
10:21
now where does attention fit within the
10:23
Transformer block within the Transformer
10:25
block there is another module which is
10:27



***



called as the attention module so the
10:29
attention mechanism which we learned
10:31
about earlier and the key query value
10:33
all of these things are actually
10:34
happening inside this mask multi-ad
10:36
attention modu so within the Transformer
10:39
block there is a normalization layer
10:40
multi-head attention Dropout layer these
10:42
are shortcut connection then other
10:44
normalization layer a feed forward
10:46
neural network another Dropout layer one
10:49
more shortcut connection and then we
10:50
come out of the Transformer block after
10:53
coming out of the Transformer block
10:54
there's a layer normalization layer and
10:56
a final neural network which converts
10:58
the trans former outputs into something
11:00
which is called as the loged sensor the
11:03
loged sensor is then used to predict the
11:05
next token given a given uh given an
11:08
input
11:09
sequence now when you look at a gpt2
11:12
let's say there are 12 such Transformer
11:14
blocks which are arranged together for
11:15
larger llms even multiple more
11:18
Transformer blocks are arranged together
11:20
and Within These Transformer blocks
11:22
there is multi-ad attention so within
11:24
every Transformer block there can be 12
11:26
or there can be 24 attention heads so
11:28
there are multip mle Transformer blocks
11:30
and within each Transformer blocks there
11:32
are multiple attention heads uh the
11:34
terminology is a bit complex but once
11:37
you get a visual feel of this
11:39
architecture uh it's actually quite easy
11:42
to code it sequentially so once you
11:44
understood this architecture what we did
11:45
in the lecture series is that we coded
11:47
every single thing with respect to this
11:49
architecture completely from scratch so
11:51
you can
11:52
search feed forward here and here is the
11:55
part three of the architecture so in
11:57
fact we went sequentially with respect
11:59
to to all the parts we first covered
12:01
layer normalization then we covered uh
12:04
the feed forward neural network then we
12:06
covered the shortcut connections then we
12:09
covered the coding attention layers so
12:11
everything is covered in coding as well
12:13
along with this whiteboard approach so
12:15
once you figure out this llm
12:17
architecture you will have a bird's ey
12:19
view of what exactly happens with the
12:20
input sequence how it goes through the
12:22
Transformer how it comes out of the
12:24
transform Transformer we have the logic
12:26
sensor and that is then used to predict
12:28
the next two token so this is my llm
12:31
prediction now this next token which is
12:33
predicted by my llm is used to calculate
12:35
the loss function between the llm output
12:39
and the True Result after we get the
12:41
loss function the next step is to run
12:43
the llm pre-training loop which means
12:45
that once we have understood how to do
12:48
the forward pass which means that how to
12:50
have an input sentence or how to have an
12:52
input sequence get the output from the
12:54
llm and get the loss based on the next
12:57
token and this loss by the way is the
12:59
loss entropy loss between the next token
13:01
prediction which we have and the actual
13:03
next token once we know how to get the
13:05
loss then we have to do a back
13:06
propagation which means that then we
13:08
have to take the partial derivative of
13:10
the loss with respect to all of the
13:11
parameters in the llm architecture so
13:14
here I have just mentioned the training
13:16
Loop first you calculate the loss on the
13:18
entire batch then you do the backward
13:19
pass which means that you calculate the
13:21
partial derivative of the loss with
13:23
respect to all of the trainable weights
13:25
so here you might be thinking what all
13:27
are the different trainable weights so
13:29
so there are trainable weights in the
13:30
token embedding positional embedding
13:32
because we don't know the so when I say
13:34
transform it into a vector space we
13:36
don't know what the ideal transformation
13:38
so we need to figure out those
13:40
parameters we need to figure out the
13:41
positional embedding parameters we need
13:44
to have the we need to figure out the
13:46
scale and shift parameters in the layer
13:49
normalization we need to train the query
13:51
key and the value trainable weight
13:53
matrices these parameters in the mass
13:56
multihead attention module then the
13:58
second layer nor alization layer also
14:00
has trainable parameters the feed
14:02
forward Network also has trainable
14:04
parameters and the final output layer
14:06
and the final layer Norm also have
14:08
trainable parameters and remember that
14:10
we have 12 such Transformer blocks so
14:12
that's why all of these parameters when
14:14
you add up it leads to more than a
14:16
million or more than even a billion
14:17
parameters which we need to train so we
14:19
need to find the gradients for all of
14:21
these parameters and then we need to do
14:23
a gradient update by something like wi +
14:26
1 is equal to Wi IUS Alpha * the partial
14:29
gradient of loss with respect to the
14:31
weights this is just a vanilla gradient
14:33
descent which I'm showing you usually we
14:35
Implement some more sophisticated
14:36
schemes like ADM or Adam with weight
14:39
Decay once you do this pre-training Loop
14:42
you will actually get loss function as a
14:45
uh loss as a function of the epox and
14:48
this we have done on our own laptop on
14:50
our own system but on a very small data
14:52
set keep in mind that the pre-training
14:55
which is needed for actual llms like
14:57
gpt2 gpt3 GPT 4 Etc they are done on
15:00
huge amounts of data set with millions
15:02
of uh news articles with millions of
15:06
blogs books Etc and that pre-training
15:09
costs more than $1 million also so it's
15:12
impossible for us to pre-train an actual
15:14
full-blown large language models on our
15:16
laptop but once you go through the
15:18
lecture videos which I'm showing you you
15:21
can run the pre-training loop for a
15:22
small data set and that gives you an
15:24
entire feel of how GPT is constructed
15:26
what we did after this is that we took
15:28
our architecture we took our llm
15:30
architecture and we loaded pre-trained
15:32
weights from
15:33
gpt2 and then we actually predicted the
15:36
next token based on the input input
15:39


***



sentence this was the first fundamental
15:41
result which we achieved in this lecture
15:44
series after the pre-training is
LLM Fine Tuning
15:46
completed then we move to llm fine
15:48
tuning we learned about two types of
15:50
fine tuning classification fine tuning
15:52
in which we built an email
15:53
classification llm so when you have
15:56
given an email the llm can classify
15:58
whether it's SP or not a Spam and then
16:00
we also built an instruction finetuned
16:03
llm fully from scratch so you have to
16:05
give a bunch of instructions inputs and
16:07
outputs and train the llm to do a good
16:09
job on instructions so let's say if the
16:11
instruction is convert the active
16:12
sentence to passive the active sentence
16:14
is the chef Cooks the meal every day the
16:16
passive sentence is the meal is cooked
16:18
every day by the chef and we train this
16:20
fully from scratch whatever I'm showing
16:22
you right now has been
16:24
implemented uh based on the architecture
16:26
which we have developed in code we have
16:28
not used used it from anywhere else and
16:31
then finally we learned about llm
16:33
evaluation we learned about three types
16:35
of evaluation the first is M
16:37
mlu which is based on this paper
16:40
measuring massive multitask language
16:42
understanding what they show in this
16:43
paper here is that basically we have uh
16:46
57 tests which we can eval use to
16:49
evaluate the llm performance that's one
16:52
type of evaluation the second type is
16:53
using humans to compare and rate llms
16:56
and the third type is using a powerful
16:59
large language model to to evaluate
17:01
another llm so this is the approach
17:03
which we followed so we used a tool
17:04
called o Lama to access Lama 3 llm
17:08
especially we use this
17:10
llama 8 billion parameter Lama 38b
17:14
instruct model which is already
17:15
finetuned and it has 8 billion
17:17
parameters so it is super powerful and
17:20
uh what we did with this larger llm is
17:22
that if the true output is this and if
17:25
the model response is this we ask the
17:26
llm to compare the output with theel
17:28
model response and to give an evaluation
17:30
score and this is the evaluation score
17:33
given by the
17:34
llm So based on the model response and
17:37
the actual response it assigns a score
17:39
out of 100 so that's the third
17:41
evaluation tactic which we learned about
17:44
this is the entire details of what all
17:46
we implemented in this course we
17:48
implemented an next word or next token
17:51
prediction llm from scratch we
17:53
implemented an email classification fine
17:55
tuned llm and we implemented an
17:57
instruction fine tuned llm once you
17:59
complete this lecture series I'm sure
18:01
that you will understand the nuts and
18:03
bolts of building a large language model
18:05
and you will be a much stronger machine
18:07
learning and llm engineer if you have
18:09
completed this series and if you're
18:11
watching this lecture now I highly
18:13
encourage you as next steps to dive into
18:15
fundamental research you have this code
18:17
file right start making changes start
18:20
exploring small language
18:22
models why what's the need for large
18:24
language models can we just have three
18:26
Transformer blocks you can now start
18:28
making all these edits to the code
18:30
because the code is in building blocks
18:31
format right you can change hyper
18:33
parameters explore the effect of
18:35
different optimizers explore the effect
18:37
of different learning rates different
18:39
number of Transformer blocks explore the
18:41
effect of different evaluation
18:43
strategies it's an area of active
18:45
research try to dive into research and
18:48
that's the best way to stay at The
18:49
Cutting Edge and contribute to
18:51
Innovative and impactful llm research I
18:54
hope all of you enjoyed this lecture
18:56
series my whole goal for everyone who's
18:58
learning through these lectures is to
18:59
train you to become fundamental llm and
19:02
machine learning Engineers who can
19:03
contribute to Innovative research and
19:06
not just deploy llm apis thank you so
19:08
much everyone and I look forward to
19:10
seeing you in the next lecture

***

