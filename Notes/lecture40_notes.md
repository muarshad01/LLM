* [OpenAI GPT-2 Weights](https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights)

* Model starts from a knowledgeable state, instead of a random initialization.

***

* 5:00

***

* 10:00


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





