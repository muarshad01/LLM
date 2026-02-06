
***

* 5:00

***

* 10:00

***

* 15:00

***

* 20:00

active research right now if you have
20:05
quantitative mathematical answers it's
20:07
very easy to say you scored this much
20:09
but what if the answer is qualitative
20:11
like the response which we just got
20:13
right now how do we evaluate llms in
20:15
that case we'll look at that in one of
20:17
the next lectures but for now let's just
20:19
go ahead and plot the losses so here you
20:22
can see I have plotted the training loss
20:24
and have plotted the validation loss for
20:26
one Epoch and you can see that the
20:28
model's performance on both the training
20:30
loss and the validation training and
20:32
validation set improve substantially
20:34
over the course of the training there is
20:36
a rapid decrease in losses during the
20:38
initial phase which indicates that the
20:40
model is quickly learning meaningful
20:42
patterns and representations from the
20:44
data then as training proceeds to to the
20:47
second Epoch the losses continue to
20:49
decrease but at a much slower rate
20:52
suggesting that the model is fine-tuning
20:54
its learned representations and then
20:55
it's converging to a stable Solution
20:58
that's how you should evaluate this plot
21:00
already we can see that the validation
21:02
loss is still a bit higher and the
21:03
training loss has the potential to go
21:05
down further but due to memory and
21:07
compute require compute limitations I
21:09
was not able to increase the number of
21:11
epo but I highly encourage you to do so
21:14
if you do have the compute power if you
21:16
do not and even if you have reached this
21:18
stage it's awesome because now you have
21:20
implemented finetuning on your own local
21:25
machine uh so while the loss plot
21:28
indicates that the model is training
21:29
effectively the most crucial aspect is
21:32
its performance in terms of respect in
21:34
in in terms of response quality and
21:37
correctness so although the loss
21:38
function looks good as I mentioned
21:40
earlier we need a way to evaluate the
21:42
responses of this model we need a way to
21:45
say whether the responses make sense how
21:47
well the responses look qualitatively do
21:49
they really answer the question which
21:51
has been posed and that's why there is a
21:53
separate lecture which we will devote to
21:55
evaluating the large language models in
21:57
this case
21:59
so today was a very important lecture
22:01
because we successfully fine tuned the
22:05
instruction large language model and we
22:08
demonstrated that without using this
22:09
fine tuning data set the model cannot
22:12
follow instructions but when you use the
22:14
fine tuning data set it really learns to
22:16
follow instructions very well and I will
22:18
demonstrate this further in the next
22:19
lectures as well but one of the outputs
22:22
so if the in if the input is something
22:24
like this rewrite the sentence using a
22:26
simile simile means using similar kind
22:29
of meaning so the car is very fast the
22:32
correct response is the car is as fast
22:34
as Lightning and the model our model
22:36
predicts the car is as fast as a bullet
22:38
awesome right it really learns how to uh
22:43
how to answer based on the instruction
22:45
which has been given sometimes it does
22:47
make mistakes because we are training
22:48
for only one Epoch but we can clearly
22:51
see that if you train it for more epox
22:52
it will get better and better and better
22:54
and it's learning the
22:56
instructions so thanks a lot everyone
Recap and next steps
22:58
this brings us to the end of today's
23:00
lecture where we learned about
23:03
llm um fine tuning which is probably the
23:07
most important step in instruction fine
23:10
tuning so now if you look at this
23:12
flowchart let me rub all of this and
23:14
show you till where we have reached in
23:16
this flowchart in this flowchart we have
23:18
reached this stage where we even
23:20
inspected the modeling loss so we have
23:22
finished stage one which is preparing
23:24
the data set we have finished stage two
23:26
fine tuning the llm in the the next
23:28
lecture we'll move to stage three which
23:30
is evaluating the llm so thanks a lot
23:33
everyone I hope you are liking these
23:34
lectures which are a mix of whiteboard
23:36
plus coding approach I look forward to
23:39
seeing you in the next lecture





