* [OpenAI GPT-2 Weights](https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights)

* Model starts from a knowledgeable state, instead of a random initialization.

***

* 5:00

***

* 10:00

***

* 15:00


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






