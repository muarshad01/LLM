
***

* 5:00

***

* 10:00

***

* 15:00


took for me to run this code and the
15:49
configuration of the laptop which I'm
15:51
using so first thing to not is that I'm
15:54
using number of aox equal to 1 because
15:56
if I use number of aox equal to 2 It
15:58
crack is my laptop and it takes a huge
16:00
amount of time for the training process
16:03
that's number one number two I'm not
16:05
using a GPU over here which I highly
16:07
recommend you to do I'm using a simple
16:09
CPU so my configuration here is MacBook
16:12
Air
16:14
2020 so if I type MacBook Air 2020
16:19
configuration yeah this is the this is
16:22
the configuration which I'm using right
16:23
now I'm I just have 8 GB of RAM um I
16:27
think I have 8 yeah I have 8 GB of RAM
16:30
over here and I not using a GPU I'm
16:33
running it on a CPU so this is a pretty
16:35
basic configuration but I want to run it
16:38
on a basic configuration to show you the
16:40
results even without using fancy gpus so
16:43
if you have access to GPU I would highly
16:46
recommend you to connect to a Google
16:48
collab GPU instance or rent out an AWS
16:52
ec2 or a GPU instance from Amazon I'll
16:55
make a tutorial on it pretty soon but
16:58
here you see I'm using number of epochs
16:59
equal to one on my CPU MacBook Air and
17:02
when I run this you'll see that just for
17:04
one Epoch it took 2 hours for me to run
17:06
this code now if I would have changed
17:08
this to two it would have taken four to
17:10
five hours plus it was crashing my
17:12
system so I'm not on a very optimal
17:14
system here but if you have a laptop
17:16
with minimal configurations I've written
17:18
this code so that it will run on your
17:20
end as
17:21
well so now you can monitor the training
Monitoring the finetuned LLM results
17:24
and validation losses you'll see that
17:25
the training loss goes on decreasing I'm
17:28
doing only one Epoch and there are 115
17:30
batches here so it runs for all of those
17:34
batches for one Epoch and then you can
17:36
see that the training loss has decreased
17:38
and the validation loss has also
17:40
decreased sufficiently now let us see
17:42
the response so I'm also printing out
17:46
the uh response based on the start
17:49
context so now the instruction was that
17:52
instruction was convert the active to
17:55
passive the chef Cooks the meal every
17:56
day and here's the response the meal is
17:59
prepared every day by the chef end of
18:01
text isn't that awesome this is almost
18:04
close to the correct passive tense the
18:06
correct passive answer is the meal is
18:09
prepared or the meal is cooked every day
18:11
by the chef and here instead it's the
18:12
meal is prepared every day by the chef
18:15
so it's almost exactly correct whereas
18:17
earlier if you if you saw the earlier
18:19
output without fine tuning the llm so
18:22
without fine tuning the llm it could not
18:24
convert the active into passive it just
18:26
recycled the same text which we had
18:27
given in the instruction
18:29
but right now when we finetuned the
18:31
large language model on this custom data
18:33
set so this was the data set with 1100
18:36
instruction input output pairs when we
18:39
fine tune the large language model on
18:41
this we can see that the response is
18:43
passive in the passive tense which is
18:44
awesome which is exactly what we wanted
18:47
and then the llm continues with the rest
18:48
of the generation but essentially here I
18:52
have demonstrated that using
18:54
finetuning and just by training on one
18:57
epoch just by training for one Epoch on
19:00
a machine which has no GPU you can still
19:03
obtain good results in in 2 hours you
19:06
are able to generate a active active
19:08
tense the chef Cooks the meal every day
19:10
to passive tense the meal is prepared
19:12
every day by the chef so I I've checked
19:15
this for two EPO on a system with a
19:18
better configuration and just with two
19:19
EPO you can get it to the correct
19:21
response that the meal is cooked by the
19:23
chef every day instead of prepared so
19:25
just by increasing the number of epo
19:27
from 1 to two you actually get to the
19:29
correct
19:30
answer so as we can see based on the
19:33
output about the model trains well as we
19:35
can see from as we can tell based on the
19:37
decreasing training loss and the
19:39
validation loss values and based on the
19:42
response text we can see that the model
19:43
almost correctly follows the instruction
19:45
to convert the input input sentence from
19:47
active to passive awesome we have an
19:50
evaluation section later in which we
19:52
will learn how to evaluate the responses
19:54
of our llm we cannot just qualitatively
19:56
say that this is good this looks good
19:58
Etc there is a whole separate field of
20:00
llm evaluation and that's the subject of
20:03
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




