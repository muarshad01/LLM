
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

* 15:00

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










