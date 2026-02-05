ocess
0:00
[Music]
0:05
hello everyone and welcome to this
0:07
lecture in the build large language
0:09
models from scratch Series today we are
0:12
actually going to f tune the model which
0:16
we have been developing for the past
0:17
four to five lectures we have been
0:20
looking at an example where we do have a
0:22
pre-trained llm but the llm is not
0:25
really good enough to understand or
0:26
follow
0:27
instructions so we want to employ
0:30
instruction
0:31
finetuning and to do this we have been
0:34
following these three stages the stage
0:36
one is preparing the data set we have
0:38
implemented this stage where we
0:40
downloaded the data and formatted it
0:42
using the alpaka prompt format then we
0:45
divided the data set into multiple
0:47
batches and then we created the
0:50
training testing and the validation data
0:55
loaders that was the stage one on
0:58
preparing the data set in stage two
1:00
which is fine-tuning the llm what we
1:03
have done in the previous lecture is
1:04
that we have the llm architecture which
1:07
looks something like this and we have
1:09
loaded this llm architecture with
1:11
pre-trained weights from GPT model so we
1:14
have used the pre pre-trained weights
1:16
from gpt2 355 million parameter model
1:20
and we have kept our architecture ready
1:22
now with these pre-trained weights and
1:24
now the stage is set for us to finetune
1:27
the llm what this means is that we
1:29
already already have pre-trained weights
1:32
right but we have not yet trained this
1:34
llm on data set on the instruction data
1:37
set this is the instruction data set
1:40
which we have been using in this
1:42
Hands-On example and which we will
1:44
finetune the lmon so this is a data set
1:48
which consists of 1100 instruction input
1:50
output Pairs and our hope is that when
1:54
we train the weights of the llm again um
1:58
using this instruction data it will get
2:00
much much better at responding to
2:03
instructions so let me show you in code
2:05
where we ended the previous lecture in
2:08
the previous lecture we just had the
2:09
pre-trained llm without any fine tuning
2:12
and we asked the llm the question that
2:16
we use this
2:17
particular um input and instruction so
2:20
the instruction was convert the active
2:22
sentence to passive and the sentence was
2:25
the chef Cooks the meal every day the
2:27
output which the llm gave was the Chef
2:30
Cooks the meal every day and instruction
2:32
convert the active sentence to passive
2:34
basically it just regenerated the same
2:37
thing what we had given as input it did
2:39
not it was not able to convert the text
2:41
into out into from active to passive at
2:45
all so as you can see the response
2:48
section was created but the llm simply
2:50
repeats the original input sentence and
2:52
part of the instruction and it fails to
2:55
really convert the active sentence to
2:57
the passive sentence or to passive voice
2:58
as requested
3:00
this is the stage which we are in right
3:02
now and we want to really improve the
3:04
llm performance when it receives an
3:07
instruction so what we'll be doing is
3:09
that if you look at this architecture
3:11
there are number of areas where there
3:13
are trainable parameters right token
3:15
embedding has trainable weights
3:17
positional embedding has trainable
3:18
weights layer normalization has the
3:20
scale and shift parameters which are
3:22
trainable the multi-ad attention has the
3:25
query key and value weight metries which
3:27
are trainable again layer normalization
3:29
has trainable scale and shift the feed
3:31
forward neural network has trainable
3:33
weights the final layer normalization
3:35
and the final um output layer also has
3:38
trainable weights we will train this
3:42
architecture again on this particular
3:44
data set which I just showed you this
3:45
instruction data so that the weights are
3:48
optimized and the llm can answer the
3:51
instructions pretty well so then your
3:53
question would be why did we pre-train
3:55
the llm if we are training the llm again
3:57
on this specific data set well the
4:00
reason we pre-train the llm is because
4:02
now the model essentially starts from a
4:05
knowledgeable state so due to
4:07
pre-training the model starts from a
4:09
knowledgeable State instead of a random
4:11
state if we just started from a random
4:13
State then it would have been extremely
4:16
difficult for the model to learn
4:17
everything based on this limited data at
4:19
least through pre-training the model is
4:21
able to understand so many things such
4:23
as the semantic meaning of words when
4:25
it's given a text it knows what to
4:27
respond it cannot follow instructions
4:29
that's fine but at least it can know the
4:31
semantic meaning between different words
4:33
it knows uh it knows a lot about the
4:36
language itself due to pre-training and
4:39
that helps us to set a foundational
4:41
knowledge base before we find tune on
4:43
the specific data set so fine tuning as
4:46
a process always comes after the
4:48
pre-training stage so let's go ahead and
4:51
begin the fine tuning process now before
4:53
we do the fine tuning we need two things
4:55
we we first of all need to understand
4:57
the training Loop which is this Loop and
5:00
the second we need to understand the
5:01
loss function one thing I would like to
5:04
mention here is that the training Loop
5:05
and the loss function is exactly the
5:07
same as the uh training Loop and the
5:09
loss function which we had looked at
5:11
when we pre-trained the model so just to
Understanding the finetuning training loop
5:14
revise here's the training Loop what we
5:16
do is that we go through each batch in
5:17
the data set we calculate the loss
5:20
gradients we reset the loss gradients in
5:22
a new Appo in a new EPO then we
5:25
calculate the loss on a current batch
5:26
and then do the backward pass in the
5:29
backward pass we essentially calculate
5:30
the gradient of the loss the partial
5:32
derivative of the gradient of the loss
5:34


***



with respect to all of the parameters
5:36
and remember that here we have 355
5:38
million parameters we will calculate the
5:41
partial gradient of the loss with
5:43
respect to all of these and in the
5:45
update step we'll use an we'll use a
5:47
gradient descent based Optimizer which
5:50
looks something like this so this is the
5:52
simplest form of gradient descent W old
5:54
is W new is equal to W old minus Alpha
5:57
times the partial derivative of loss
5:59
with with respect to W and we'll do this
6:01
many times and we'll hope that the
6:03
parameters eventually get to a stage
6:05
where the loss function is minimized so
6:08
if the loss function landscape looks
6:09
something like this initially we start
6:11
somewhere here and then we later go down
6:13
to the bottom of this Valley and hope
6:15
that the loss function is minimized
6:17
right now uh if this is the training
6:20
Loop then we'll need to know what the
6:21
loss function is and the loss function
Understanding the finetuning loss function
6:23
is essentially obtained from the input
6:25
and Target pairs so remember when we
6:27
created this data loader every data
6:30
loader essentially has input and Target
6:33
pairs so let me write this down over
6:37
here so let's say we looked at we are
6:39
looking at batch number one batch number
6:42
one has eight samples sample one sample
6:44
two so each batch has eight samples
6:46
because we have set the bat size to be
6:48
eight and in each sample we have the
6:50
input and we have the
6:54
Target and since it's the next token
6:57
prediction task the target is just the
6:58
input which is shifted to the right by
7:00
one and we add the end of text token you
7:03
can think of the target as the True
7:05
Value which we
7:06
have True Value and then to get the loss
7:10
we'll need a predicted value
7:12
right so I'm going to write here
7:15
predicted
7:17
value and the predicted value is
7:19
obtained from the generate function so
7:22
remember in the last lecture we
7:23
discussed about the generate function
7:25
where given an input we generate the
7:27
next token IDs so essentially what
7:30
happens is that when an input to text or
7:34
yeah a sequence of input tokens passes
7:36
through this GPT architecture which I
7:39
have shown to you right now so let's say
7:40
the input instruction is convert active
7:43
to passive it will pass through this GPT
7:46
architecture and then when it comes out
7:48
of the GPT architecture we have
7:50
something called as the logic sensor
7:52
that's then converted into a tensor of
7:55
probabilities and through this tensor of
7:57
probabilities we get the predicted
7:59
output of the llm that's not the True
8:02
Value but that's the predicted value and
8:04
then based on the true value and the
8:06
predicted value we then compute the loss
8:08
function So based on the true value and
8:11
the predicted value both of these values
8:13
will be needed to then calculate the
8:15
loss function and this loss function is
8:18
the categorical cross entropy loss or
8:21
rather it is the cross entropy
8:25
loss the simplest way to think about
8:28
cross entropy loss is is that it's
8:30
negative of the logarithm so essentially
8:33
since probabilities are involved we want
8:35
the probability to be as close to one as
8:37
possible so that the loss function
8:39
decreases and becomes as close to
8:42
zero I'm not going into the details of
8:45
this because we have already covered
8:46
this in a lot of detail when we learned
8:48
about llm pre-training but I'm just
8:51
giving you a qualitative flavor for you
8:52
to understand what is the loss which we
8:54
are trying to minimize over here so this
8:57
is the loss function which is the cross
8:58
entropy loss once the loss function is
9:00
defined then all we need to do is learn
9:02
the training loop on the instruction
9:04
data set so the data set is already
9:06
divided into input and Target Pairs and
9:10
uh we have spent the last three to four
9:12
lectures dividing the data set into
9:14
input Target Pairs and constructing
9:16
these data loaders all of that hard work
9:19
will come into use right now so now I'm
Coding the finetuning training loop in Python
9:21
going to take you to the code and we are
9:23
going to finetune this model
9:26
together great so we have already done
9:29
the hard work when we implemented the
9:31
data set processing including the
9:32
batching and creating the data loaders
9:35
now we can reuse the loss calculation
9:37
and training functions which we have
9:38
implemented in pre-training so as I told
9:41
you here are some functions which I have
9:43
collected here and we'll be reusing
9:45
these functions the first function is
9:47
the calculation of the loss on a batch
9:49
of data so here you can see is the cross
9:51
entropy loss which I have mentioned over
9:55
here the cross entropy loss and then
9:59
when we want to calculate the loss over
10:00
the entire data loader we just take a
10:02
summation of all of the U summation of
10:05
losses calculate calculated in
10:07
individual batches and then to normalize
10:10
the total loss we just divide by the
10:11
number of batches that's what's
10:13
happening in the calculation of loss in
10:15
a loader so the same function can be
10:17
used and if you want to calculate it for
10:19
the training data loader we pass the
10:21
train loader if you want to calculate
10:23








***


the loss for the testing or validation
10:25
we pass the test loader or validation
10:27
loader respectively then then the
10:29
training Loop which I've described over
10:32
here this training
10:35
Loop or I should Mark this entire thing
10:39
this entire training ROP training Loop
10:41
we have we have written a code for this
10:43
before when we looked at pre-training
10:45
and this essentially the most important
10:47
part of this code is the backward pass
10:49
where the gradients of the loss are
10:50
calculated and then we do this Optimizer
10:53
step so here I implemented a simple
10:55
vanilla gradient descent right the
10:57
actual Optimizer which is used is Adam
10:59
with weight DEC so you can either use
11:01
Adam or you can use Adam with weight
11:03
Decay which is Adam W that's the
11:06
optimizer will will be using and then
11:09
the remaining of the code is just about
11:11
printing the loss which we obtain at
11:13
different uh iterations and then
11:16
visualizing the testing loss visualizing
11:18
the validation loss and visualizing the
11:20
training loss together so as you can see
11:22
here we are going to print the training
11:24
loss and the validation
11:26
loss um yeah so initially before we even
11:30
begin the training let us first
11:32
calculate the training and the
11:33
validation loss um initially the
11:35
training has not yet happened on this
11:37
data set on this data set I have not yet
11:41
run the training loop I just want to see
11:42
the loss at the initial moment so here
11:44
if you see this is equivalent to being
11:48
at this early point where the training
11:50
has not yet started so the loss will
11:52
naturally be very high I just want to
11:54
see the training and the validation loss
11:56
so the training loss is 3.82 and the
11:58
valid ation loss is 3.76 so this is
12:01
pretty high and we want to bring both of
12:03
these losses together now the model is
12:06
prepared and the data loaders are
12:07
prepared we can now proceed to train the
12:10
model so now the code which we are
12:14
implementing below it sets up the
12:15
training process including initializing
12:18
the optimizer setting the number of EPO
12:20
and defining the evaluation frequency so
12:23
here you can see we Define the optimizer
12:25
which is storage. op. adamw so let me
12:28
just
12:30
torch. optim do
12:33
adamw so this is the Adam with weight DK
12:35
Optimizer which we are going to use if
12:37
you don't know how Adam works it's fine
12:39
but it's the most widely used Optimizer
12:42
I would say in machine learning
12:44
algorithms these days it keeps a track
12:46
of the gradients of the loss function it
12:49
keeps a track of the gradient Square
12:51
also um and uh it's just very useful to
12:55
prevent local
12:57
Minima and uh to accelerate the
13:00
convergence that's why the Adam or the
13:02
adamw optimizer is predominantly used we
13:05
are using a learning rate of
13:08
0.005 and the weight decay of 0.1 now
13:11
please note that these are hyper
13:13
parameters which means that you can
13:14
change them and you might obtain better
13:16
results these are just some parameters
13:18
which we have used right now and they
13:20
lead to reasonably good results I
13:22
encourage all of you who are watching
13:24
this video and who will later receive
13:25
access to this code file to change these
13:27
hyperparameters as much as possible so
13:30
that you see the results for yourself
13:32
and you see how the results are
13:34
improving or not improving you'll learn
13:35
a lot through this
13:37
process so this is the train model
13:39
simple we defined the train model simple
13:41
code over here we need to pass in the
13:44
model uh we need to pass in the model we
13:47
need to pass in the train loader the
13:48
validation loader the optimizer which is
13:50
the admw the device which is CPU in my
13:53
case I'll come to this device in a lot
13:56
of detail very soon then we have to
13:59
specify the number of epoch so I have
14:00
defined the number of epoch equal to one
14:03
we have to specify evaluation frequency
14:05
so here you see when we start printing
14:06
out after every five batches I'm
14:08
printing out the results you can change
14:11
this based on after how many iterations
14:13
you want to print out the results all of
14:15
this is mentioned in this part of the
14:17
code whereare evaluation frequency
14:19
another parameter is evaluation
14:20
iterations which also shows essentially
14:23
evaluation iterations is after how many
14:25
iterations you're calculating the loss
14:27
on the evaluation data set and
14:29
evaluation frequency mentions after how
14:31
much frequency you want to display that
14:33
loss so here we are setting both to be
14:36
equal to five which means that after
14:37
five batches are processed the
14:39
evaluation loss will be calculated and
14:41
it will be printed also then we have to
14:44
give a start uh start context so as you
14:47
have as I've mentioned over here uh we
14:50
have to give a starting context to
14:52
evaluate the generated responses during
14:54
training based on the first validation
14:56
set so the first validation which we are
14:59
giving is format input and that is
15:02
validation data so that's the first
15:03
validation data which we are going to
15:05
give as the start context and that is
15:07
because if you see the example which we
15:10
have taken over here the example which
15:12
we have checked is validation data zero
15:15
and that is convert the active sentence
15:18
to passive the chef Cooks the meal every
15:19
day now for this same example we want to
15:22
check what the fine tuned llm response
15:24
so that's why we have used this as the
15:26
start
15:27
context so the number of tokens which
15:30
will be printed or the output will be
15:32
starting from this particular
15:35
instruction all right
15:40
so now one thing which I would like to
15:43
mention over here is that the time it
15:46
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

