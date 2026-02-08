
* $$(W_Q, W_K, W_V)$$
* $$Q, K, V$$
* $$Q \times K^{T}$$
* Attention Score
* Scaling by $$\sqrt{d-keys}$$ + dropout + softmax
* Attention Weights
* Context Vector = Attention Weights X $$W_{V}$$
* Head-1: Context Vector Matrix
* Head-2: Context Vector Matrix
* Concatenate along columns

***


***

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









