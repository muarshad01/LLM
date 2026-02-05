netuning recap
0:00
[Music]
0:05
hello everyone and welcome to this
0:08
lecture in the build large language
0:09
models from scratch Series today we are
0:13
going to look at step number four in the
0:16
instruction fine-tuning Hands-On project
0:18
which we have started and that step is
0:20
loading a pre-trained large language
0:23
model until now we have looked at three
0:25
steps the first one involved data set
0:28
download and formatting the second
0:30
involved batching the data set and the
0:32
third involved creating data loaders so
0:35
here's the data set which we have been
0:37
looking at which consists of
0:41
1100 entries of instructions inputs and
0:44
outputs they are in various fields from
0:47
English to geography to general
0:49
knowledge Etc and essentially through
0:51
this data set our aim is to teach the
0:53
llm to be able to correctly follow
0:58
instructions um and that's what
1:00
finetuning refers to in pre-training if
1:03
we just used a pre-trained llm it does
1:06
not follow instructions very well so
1:08
that's why we need to train it with a
1:10
specific data set now before we come to
1:13
loading a pre-trend llm we first needed
1:16
to spend a lot of time on downloading
1:19
this data set then converting it into a
1:21
format called The alpaka Prompt style
1:24
then batching the data set which is
1:26
creating different batches and
1:27
ultimately creating data loaders now
1:29
that we we are ready with the data
1:30
loaders we can go ahead and load a
1:33
pre-trained
Why do we need a pre-trained LLM?
1:34
llm what does it mean loading a
1:36
pre-trained llm well what we are going
1:38
to do here is that this is the llm
1:41
architecture which we have constructed
1:42
in this lecture series and there are
1:44
several places where there are trainable
1:46
weights so for example in multi-head
1:48
attention there are the query key value
1:51
weight metries then in feed forward
1:53
neural network there are the neurons and
1:56
their weights in layer normalization one
1:58
and two there are the scale and the
2:00
shift parameters which have parameters
2:02
or weights associated with it then there
2:04
is the token embedding layer and the
2:06
positional embedding layer there are
2:09
trainable weights associated with these
2:10
two layers as well if you add up all of
2:13
these trainable weights then we have
2:14
more than 1 million or more than 100
2:16
million trainable such
2:18
parameters uh in the pre-training
2:20
process these parameters are trained on
2:22
a huge amount of data so for example
2:25
gpt2 gpt2 if you search about gpt2
2:29
weights released um gpt2 weights
2:32
released open AI actually pre-trained
2:35
gpt2 on a huge amount of data and not
2:38
just that when they pre-trained the llm
2:40
and optimized
2:41
the trainable weights optimized all of
2:44
these weights which I just described
2:45
over here they released those weights
2:47
publicly which means that as users we
2:50
can have access to those weights and we
2:52
can utilize those weights now when open
2:54
a released their weights they released
2:56
it for four different models so if you
2:58
see the open a g pt2 weights they
3:00
released it for actually multiple models
3:03
124 million 150 1558 million 345 million
3:08
774 million Etc so you have different
3:12
folders corresponding to the parameter
3:14
size and then based on each folder you
3:16
can download these weights on your local
3:18
PC why are we downloading these weights
3:21
in the first place before fine tuning
3:23
the reason we are downloading these
3:25
weights and reusing them is that when
3:28
the when we reuse these weights it means
3:31
that a lot of information is already
3:34
captured correctly so for example when
3:37
we do token embedding we want to project
3:39
the words or tokens into a higher
3:42
dimensional Vector space right and we
3:44
want to project them so that the
3:45
semantic meaning is captured to train a
3:48
huge neural network again from scratch
3:50
here would take very a very long time
3:52
for us and computational resources
3:54
instead we will directly use the
3:56
pre-trained gpt2 weights so that will
3:58
help us in fact in all of these
4:01
parameters reusing the gpt2 weights
4:04
actually helps us uh because the model
4:07
now starts from a much more
4:08
knowledgeable State instead of a random
4:11
initialization so the main goal of
4:13
reusing the pre-trained weights is that
4:16
the
4:17
model
4:19
starts
4:21
from a knowledgeable state
4:29
instead
4:33
of instead of a random
4:43
initialization so we are going to load
4:45
these pre-train weights and then we are
4:47
going to train on the specific data set
4:49
which we have curated so we are going to
4:51
load the pre-train weights and then use
4:53
these training data loaders testing data
4:56
loaders and validation data loaders so
4:58
the weights and parameters which we load
5:00
from gpt2 of course they will change
5:02
during the process because we will do
5:03
one more training procedure on the
5:06
specific data set but at least we won't
5:08
start from a random state of all these
5:10
parameters we'll start from a
5:11
knowledgeable State and so the
5:13
computational time it will take to fine
5:16
tune or to train on our specific uh data
5:18
set will be much lower so that will be
5:20
much more efficient that's why the fine
5:22
tuning process always happens after the
5:24
pre-training process in the pre-training
5:27
we always take the model to a
5:28
knowledgeable state and then we train it
5:30
again on a specific data set so that it


***



5:32
can further improve its weights and its
Coding - load a pre-trained LLM
5:35
parameters so now let us go into code
5:37
and let us load the pre-train llm which
5:40
means that let us use the weights um
5:43
which we get from gpt2 one thing which I
5:46
would like to mention here is that when
5:48
we developed the spam versus no spam
5:50
email classifier and when we use the GPT
5:53
to weights we used the weights from a
5:55
model which is 117 million parameter but
5:58
now we we have to use the model which is
6:01
much higher because we because I saw
6:03
that a smaller parameter which is 124
6:06
million parameter model does not perform
6:09
very well in the instruction fine tuning
6:11
task that's why we have to use a larger
6:13
model so as I've indicated on the
6:15
Whiteboard we will load gpt2 with 355
6:18
million model 355 million uh which are
6:22
the number of model
6:24
parameters okay awesome so instead of
6:27
using the smallest 124 million parameter
6:30
model as before we load the medium siiz
6:32
model with 355 million parameters right
6:35
and uh that takes approximately
6:38
1.42 gabt of space so please make sure
6:42
that as I'm showing you the code below
6:44
you have this much amount of space on
6:46
your local machine I'm not using any
6:49
fancy GPU or any uh High processing
6:52
speed computer I'm simply using a
6:54
MacBook Air
6:56
2020 so these These are the model
6:59
configur ation for which GPT 2's weights
7:01
are publicly released and we are going
7:03
to use the GPT medium which is the 355
7:05
million parameter model it has the
7:08
context size of 1024 it has
7:11
24 uh Transformer blocks and within each
7:15
Transformer block There are 16 attention
7:17
heads so in choose model we have chosen
7:20
gpt2 medium 355 million parameters and
7:23
in the base configuration we have
7:25
vocabulary size 50257 context length of
7:28
1024 right and uh dropout rate we are
7:32
using zero and the query key value bias
7:35
we are setting this to be equal to true
7:37
and that is because when we initialize
7:39
the query key and the value weight
7:41
matrices we are also going to initialize
7:43
the bias terms 50257 is essentially the
7:47
vocabulary size on which gpt2 was
7:49
trained
7:50
on then what we are going to do is that
7:52
we are going to run this download and
7:54
load gpt2 function and I'm just going to
7:56
show you this function briefly what this
7:59
down download and load gpt2 function
8:01
actually does is that it uh it accesses
8:04
this URL and it downloads the seven
8:07
files onto the local machine so you can
8:10
even access these files on kaggle if you
8:12
click on this 355 if you click on the
8:15
355 million and click on the seven files
8:17
you can get these seven files and you
8:19
can download it to your local PC however
8:21
just downloading these files is not
8:23
enough you need to do some
8:25
pre-processing steps on this file so
8:27
that um so check this function which is
8:30
load gpt2 parameters from the tensorflow
8:33
checkpoint what this does is that after
8:35
downloading the parameters uh from the
8:38
seven files it actually converts it into
8:40
a specific dictionary and through this
8:42
dictionary you can handle or access
8:44
these parameters in a much more easier
8:46
manner we have covered the details of
8:48
this in a previous lecture which is
8:50
titled pre-training using gpt2 weights
8:53
and that's a 1 hour lecture in which
8:54
this code has been explained in a lot of
8:56
detail for now all you need to know is
8:58
that when you you run this code it first
9:00
of all downloads the seven files it
9:02
creates a folder called 355 million and
9:05
it downloads the seven files and then it
9:08
also has this the thing which it returns
9:10
mostly is that this params dictionary
9:12
the params dictionary essentially
9:14
consists of the Transformer blocks the
9:16
token embeddings the positional
9:17
embeddings the final normalization layer
9:20
parameters which are very accessible if
9:22
you just download the seven files it
9:24
becomes difficult for us to integrate
9:27
these parameters with the code which we
9:28
have written earlier here that's why
9:30
it's very essential for you to run this
9:33
load gpt2 params from TF checkpoint uh
9:36
so that the params dictionary is created
9:40
successfully uh if you don't understand
9:42
this code right now it's fine all you'll
9:44
have to do is just execute this line of
9:46
code which is settings comma params
9:48
equal to download and load gpt2 what
9:51
this will essentially do is that first
9:53
of all from GPT download 3 we have to
9:55
import this function so this file which
9:58
I showed you over here is called GPT
9:59
download 3py from this file we are
10:02
importing this function and uh the
10:04
results of that function we are storing
10:06
in something called settings comma
10:08
params so when you run up till here the
10:11
1.42 gbt model which has been shown over
10:14
here so see this is the 1.42 GB that
10:17
will be downloaded so for me this took a
10:20
long time to download Because the
10:21
Internet connection was not very strong
10:23
and uh the file kept pausing a lot in
10:26
the middle so please make sure that you
10:28
are sitting in an area which has strong
10:30
internet connection and you have memory
10:32
on your device if you have 1.4 or 1.5 GB
10:35
in a strong uh internet area I think
10:39
this download should happen in a much
10:40
quicker Manner and then what we are
10:43
going to do is that after we download
10:45
these weights we are going to load



***



10:46
weights into GPT what this function does
10:49
is that we have constructed this GPT
10:52
architecture right at all of the places
10:54
where there are trainable parameters
10:56
this function appropriately Maps the
10:58
download parameters in the params
11:01
dictionary into our model so remember I
11:04
mentioned that this function essentially
11:06
returns this dictionary called as params
11:08
in which the weights and parameters have
11:11
been arranged in a specific format when
11:13
you run this function load weights into
11:15
GPT the parameters from that params
11:18
dictionary are downloaded into our
11:20
model and you can actually control F
11:24
contrl F
11:26
load
11:28
load with
11:30
into GPT and you'll see that we have
11:33
defined this function before load
11:35
weights into GPT um what this function
11:39
asence essentially does is that the
11:40
params dictionary it Maps the values
11:43
extracted from the params dictionary
11:45
into the GPT architecture which we have
11:47
constructed before so you can think of
11:49
this whole code code block as one
11:51
assignment step we are assigning the
11:53
downloaded parameters to our model we
11:55
have had a separate lecture to explain
11:57
this fully so I'm not covering this in
11:59
detail right now let me take you to the
12:02
current code which we are on so uh then
12:05
you have to run after you run the
12:07
settings comma params then you have to
12:08
run load weights into GPT which loads
12:11
the downloaded weights from the 355
12:13
million model into uh your GPT
12:16
architecture and then you set the model
12:18
into evaluation mode so if your laptop
12:21
does not have a strong configuration
12:23
with respect to memory or processing
12:25
speed you can even choose the model here
12:27
to be gpt2 small which which has 124
12:30
million parameters and that should take
12:31
you onethird of the time or half of the
12:34
time it takes you to load the weights of
12:36
this GPT to medium model if on the other
12:39
hand your laptop has a very high
12:41
processing speed and if you have GPU
12:43
access I recommend you can use gpt2
12:46
large or even gpt2 X Excel if you have
12:48
GPU access because then the results
12:51
which you will get will be much better
12:52
than uh what we will obtain with smaller
12:56
models okay when you run when you run
12:58
this file you'll see that you'll get
13:01
outputs such as these I I have already
13:03
downloaded the gpt2 parameters so I'm
13:06
getting file already
13:08
exists now what we can do is that before
Pre-trained LLM performance
13:10
diving into fine tuning the model which
13:12
we'll come to in the subsequent section
13:15
what we can do is that we can actually
13:17
check the pre- trend llms performance on
13:19
one of the validation tasks by comparing
13:22
its output to the expected response we
13:24
have not even trained or fine-tuned our
13:26
llm on the data set yet on this data set
13:29
but I just want to check the performance
13:31
of this llm by taking some very specific
13:33
example so what I'm going to do now is
13:36
that I'm going to take this example from
13:38
our data set and the example is that
13:40
below is an instruction that describes a
13:42
task Write a response that appropriately
13:44
completes the request and the
13:47
instruction is that convert the active
13:49
sentence to passive and the sentence is
13:51
the chef Cooks the meal every day um and
13:54
you can see that I've taken this
13:56
sentence from the validation data it is
13:58
the first sentence from the validation
14:00
data so actually let me check it out
14:02
over here so if I put Chef yeah this is
14:05
the instruction input and output so the
14:08
instruction is convert the active
14:09
sentence to passive the chef Cooks the
14:11
meal every
14:13
day and the correct
14:16
output uh I think
14:22
convert yeah I think this is the one let
14:25
me check it again yeah the instruction
14:28
is convert the active sentence to
14:29
passive the chef Cooks the meal every
14:31
day it's this sentence exactly and the
14:33
correct output is the meal is cooked by
14:35
the chef every day so if the G if the
14:37
llm is fine tuned correctly the output
14:39
which it should give should somewhat be
14:41
of a passive tense so if the active
14:43
tense is the chef Cooks the meal every
14:45
day the passive tense should be the meal
14:46
is cooked by the chef every day we have
14:49
not trained our llm on this data at all
14:51
so we are not expecting to get a correct
14:53
answer but I just want to see how wrong
14:55
we are or how far off we are from the
14:57
correct answer so you can use the
14:59
generate function which you have defined
15:01
previously what this function does is
15:03
that you have to pass in the maximum
15:04
number of new tokens to generate and
15:07
then based on the model and based on the
15:09
trained weights remember we are just
15:11
using the weights downloaded from GPT to
15:13
medium based on this based on these
15:16
weights the next tokens or the new


***


15:18
tokens will be generate new tokens will
15:20
be generated and here we have to specify
15:22
the maximum number of new tokens which
15:24
I'm mentioning to be
15:26
35 and the context length is of course
15:29
1024 which is the context length of the
15:32
medium
15:34
configuration uh that's it actually and
15:36
then you have to pass in the input text
15:38
so I'm passing in the input text over
15:40
here and then the model which is the
15:42
model with the pre-trained weights and
15:45
then the generated text we have to
15:46
convert it back from token IDs into text
15:49
so let us print this out right now and
15:51
let us see the
15:54
response so the response here so one
15:57
thing to mention is that when you use
15:58
the erated text uh it Returns the
16:01
combined input as well as output this
16:04
Behavior was convenient in the previous
16:05
chapters because pre-trained llms are
16:08
primarily designed to complete the text
16:10
right so we can predict the input and we
16:12
can predict the output so it will just
16:14
look like an input output pair where the
16:16
input is completed but now we actually
16:18
just need to focus on the model
16:20
generated response right every time we
16:22
print the generated text we don't need
16:24
the input text we don't need the
16:26
instruction we just need the response
16:28
right
16:29
uh so now what I'm doing is that when
16:31
you print out the response we need to
16:33
subtract the length of the input
16:34
instruction from the start of the
16:36
generated text so we have to just print
16:39
out the response here so we mention that
16:41
you subtract the input you subtract the
16:43
instruction and just print the response
16:45
text so here is the response which our
16:47
model gives the response which is given
16:49
by the model is the chef Cooks the meal
16:51
every day so it has just recycled the
16:53
first sentence in the response itself it
16:56
has included an instruction and in the
16:58
instruction it is retaining the same
17:00
instruction convert the active sentence
17:01
to passive the shf cooks the which means
17:04
that the model has not at all followed
17:05
my
17:07
instruction uh in fact the pre-train
17:10
model is not yet capable of currently
17:12
correctly following the given
17:13
instruction it creates a response
17:15
section that is good but it simply
17:18
repeats the original input sentence it
17:20
simply repeats the original input
17:22
sentence and it also repeats a part of
17:24
the instruction just as it is but it
17:26
fails to convert the active sentence to
17:29
passive voice that's the main uh thing
17:31
which I want to convey to you that
17:33
without fine-tuning the model itself
17:36
just using the pre-trained weights it's
17:39
not doing a good job at all in fact it
17:41
fails to convert the active sentence to
17:43
the passive it's recycling the same text
17:45
which we provided as an instruction and
17:47
it's creating this hashtag hashtag
17:49
hashtag instruction in the response
17:51
itself that is also not good so now what
17:53
we'll do in the next section is that we
17:55
are going to implement the finetuning
17:57
process to imp to improve the model's
18:00
ability to comprehend and appropriately
18:02
respond to such requests so the reason
18:04
fine tuning exists in the first place is
18:07
because without fine tuning even if we
18:09
load the weights from a parameter which
18:12
is 355 million or even if you load from
18:15
774 million you'll see that even for a
18:17
774 million gpt2 param gpt2 model the
18:22
response is not coherent without fine
18:24
tuning and that's why in the next
Recap and next steps
18:27
section we are going to look at fine
18:28
tuning the llm on instruction data which
18:31
means that we'll actually model or we'll
18:33
actually modify the weights and
18:34
parameters of this GPT model so that it
18:37
can try to understand these instruction
18:40
input output pairs from this specific
18:42
data
18:44
set uh okay everyone this brings us to
18:46
the end of the lecture where we looked
18:48
at loading a pre-trained llm I initially
18:51
plan to cover the fine tuning in this
18:53
lecture itself but I thought it will be
18:55
good to cover it in the next lecture
18:57
otherwise the duration of the lecture
18:58
would have been pretty long I hope you
19:00
are liking these lectures we are now
19:02
very close to actually finishing the
19:04
entire finetuning and then extracting
19:07
the responses evaluating the responses
19:09
and the code file which I'm going to
19:10
share with you can be used to perform a
19:13
wide range of instruction fine tuning
19:14
tasks so thanks a lot everyone in the
19:17
next lecture I'll explain the
19:19
fine-tuning the model process and I look
19:21
forward to seeing you in the next
19:23
lecture
