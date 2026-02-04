tion finetuning recap
0:00
[Music]
0:05
hello everyone and welcome to this
0:07
lecture in the build large language
0:09
models from scratch series we are
0:12
looking at instruction F tuning and we
0:15
have finished step number one which is
0:17
data set download and formatting and in
0:19
the last lecture we have finished step
0:21
number two which is batching the data
0:23
set in this lecture we are going to look
0:25
at creating data loaders so you can
0:28
think of data loaders as an efficient
0:30
way to just collect the different
0:33
batches whenever we need to do
0:35
so using data loaders the process of
0:38
training a large language model becomes
0:41
very easy because then you can access
0:43
batches in a sequential Manner and in a
0:45
much more efficient manner so here's the
0:48
pytorch data sets and data loaders
0:51
documentation I'll be sharing the link
0:53
to this and as you can read here data
0:55
loader wraps an iterable around the data
0:58
set which means that the B which we have
1:00
created can be accessed in an iterative
1:03
manner using the data loader and by
1:05
iterative I just mean in a sequential
1:07
Manner and essentially it makes it very
1:09
easy to access the different data
1:11
samples so the reason we use data sets
1:14
and data loaders is more for efficiency
1:17
of access to the data itself and that
1:19
makes a very big difference when
1:21
training a huge model um such as llms
1:25
with more than 100 million
1:28
parameters so let me quickly recap
1:30
whatever we have done so far so the
1:32
first step was data set download and
1:34
formatting here's the instruction fine
1:37
tuning data set so we have 1100 pairs of
1:40
instructions inputs and outputs so an
1:43
instruction could be something like edit
1:45
the following sentence for grammar the
1:47
input can be he go to the park every day
1:49
and the correct output is he goes to the
1:51
park every day the instruction can be
1:54
convert 45 kilomet to meters and here it
1:57
does not have any input because there is
1:58
one fixed answer and the output is 45
2:01
kilom is 45,000
2:03
M so what we are essentially doing is
2:06
that we are telling the large language
2:08
model that hey we look we know that you
2:10
are pre-trained but you are not very
2:12
good at following instructions here is a
2:14
training data set which I have that will
2:17
teach you how to follow instructions and
2:19
how to respond effectively so we are
2:21
providing this training data set as an
2:23
additional specific data set and then we
2:26
will train the large language model
2:28
again so that it it gets better at
2:30
responding to
2:32
instructions so that's the first step
2:34
which we implemented and that was data
2:36
set download and formatting what's meant
2:38
by formatting well there is this uh
2:40
template which is called as alpaka
2:42
format and what this template
2:45
essentially does is that when we are
2:46
going to do instruction fine tuning um
2:49
so let me go
2:51
to Stanford
2:54
alpaca
2:56
GitHub right so this is the GitHub
2:58
repository and and basically they have a
3:02
data set of 52,000 instruction response
3:05
pairs such as what I showed you here and
3:08
they have a specific way to convert
3:10
these into prompts so when we are going
3:12
to train the large language model we
3:14
need inputs and outputs right so there
3:16
is a specific prompt which we are going
3:19
to use and that prompt itself will serve
3:21
as input output input Target pairs so if
3:24
the instruction is something like let's
3:27
say convert 45 km to meters and the
3:30
output is 45 km is 45,000 M The Prompt
3:34
which we are going to give the llm is
3:35
that below is an instruction that
3:37
describes a task paired with an input
3:40
that provides further context Write a
3:42
response that appropriately completes
3:44
the request and then in the instruction
3:47
we provide the instruction which was
3:48
convert 45 km to meters then we have the
3:52
input the input in this case does not
3:54
exist because there's only one answer
3:56
and then we have the
3:57
response now this is the prompt which
4:00
will be provided to the llm and then we
4:02
will train the llm with the next word
4:05
prediction tasks which means that this
4:07
is an auto regressive model within this
4:09
prompt itself there are input and Target
4:11
pairs so when below is the input below
4:14
is is the output when below is is the
4:16
input below is and is the output when
4:19
below is an instruction is the input
4:21
below is an instruction that will be the
4:23
output so similarly when we train the
4:26
llm on this entire prompt we will show
4:29
early reach a stage when it reads when
4:31
it reads the instruction and the input
4:33
it learns how to uh provide the
4:37
response so that's what's meant by
4:39
formatting every input every input
4:43
instruction and response is formatted
4:45
into this uh alpaka style
4:49
format and that's done in the data set
4:52
downloading and formatting stage then
4:54
what we do is that we batch the data set
4:56
this was not as easy as simply creating
4:59
batches but there were multiple steps
5:01
involved in data batching itself and
5:04
these steps are divided into five
5:06
substeps uh or these steps can be
5:09
bucketed into
5:10
five uh five main components the first
5:13
as I mentioned is format data using the
5:15
prom template and we use the alpaka
5:18

***


template there is also the 53 template
5:20
which is used by Microsoft uh and you
5:22
can try that when I'm going to share
5:24
this code file with you so that's the
5:26
first step the Second Step which we are
5:28
going to do is we are going to to toize
5:30
the formatted data which means that
5:33
here's the formatted data right which is
5:36
in the alpaka prompt style we are going
5:37
to convert this entire data into token
5:40
IDs so let's say one batch has four
5:43
instruction input output pairs or two
5:45
instruction output input output pairs
5:48
the First Data will look something like
5:49
this when tokenized the second data will
5:51
look something like this when tokenized
5:53
now naturally the length of the tokens
5:56
here and the length of the tokens here
5:58
won't be the same and that's a problem
6:00
since we are dealing with batches so the
6:02
next step is that so after tokenizing
6:04
the next step is that we are going to in
6:06
one batch we need to adjust so that all
6:09
of the data samples have the same token
6:11
ID so consider this batch for example
6:14
where there are three inputs when the
6:16
first input is tokenized it has five
6:18
token IDs when the second input is
6:20
tokenized it has two and when the third
6:22
input is tokenized it has three and
6:24
that's not good we need to make sure
6:26
that all of these have the same number
6:28
of token IDs so the way we do it is that
6:30
we take that input which has the maximum
6:32
number of token IDs in this case it's
6:34
the first input which has five token IDs
6:37
and then we we take the remaining two
6:39
inputs and we pad them with the end of
6:40
text token which is
6:42
50256 and we pad them so that their
6:45
length becomes equal to the maximum
6:47
token ID length which is equal to five
6:49
since the first one has five token IDs
6:52
and once you do this padding you make
6:54
sure that in every batch all the inputs
6:56
of every batch have the same number of
6:59
token so all of these three have the
7:01
same token ID if you look at the second
7:03
batch all of these have the same token
7:05
ID that's the third step in the fourth
7:08
step we have to create the target pairs
7:10
so we have inputs and targets right and
7:12
targets are just the input pairs which
7:14
are shifted to the right by one so if
7:16
you have an input which looks like this
7:18
the way to create the target pairs is
7:20
just you get rid of the first element
7:22
and have the remaining so input shifted
7:24
to the right by one and then you add an
7:27
end of text token to convey that this is
7:29
the end end of the instruction output
7:31
PIR so that's what I mentioned earlier
7:33
by the next toen prediction task right
7:35
when below is an input below is is the
7:37
output when below is is the input below
7:40
is and is the output this input Target
7:43
pair creation is probably the most
7:45
important step in fine tuning and it's
7:47
also not very intuitive because you are
7:49
not saying that the response is the
7:51
output you're saying that the whole
7:53
thing itself is your data and within
7:55
that you have input and Target pairs it
7:57
turns that it turns out that with this
7:59
next token prediction task the llm
8:01
learns to uh segregate or recognize that
8:05
up till here I have the instruction with
8:06
the input and then this is the response
8:08
which I have to predict so the next word
8:10
or the next token prediction task is
8:13
What's Done in fine tuning as well
8:15
similar to pre-training in fine tuning
8:17
we are also doing the next token
8:19
prediction and this is how we create the
8:21
input and Target Pairs and the last step
8:24
here is that we replace the padding
8:26
tokens with minus 100 so all the 5025 6
8:29
here is replaced with minus 100 and the
8:32
reason it's replaced with Let me show
8:35
this to you um right over here yeah so
8:40
here you see all the 50256 tokens are
8:43
replaced with minus 100 the reason they
8:45
are replaced with minus 100 is that in P
8:48
torch when you use the categorical cross
8:50
entropy loss this minus 100 is the
8:53
ignore index which means that when we
8:56
use these inputs and Target pairs to
8:58
calculate the loss
9:00
the token tokens associated with these
9:03
IDs will not be counted for getting the
9:06
loss function and that makes sense
9:07
because all of these are useless end of
9:10
text tokens which we have just padded
9:11
over here they should not be used for
9:13
the loss calculation we retain one 50256
9:17
token in every input because that
9:19
conveys the end of text but all the
9:21
remaining 50256 we replace them with
9:24
minus 100 so that they don't influence
9:26
the loss and you can check this so you
9:28
can search
9:31
pytorch cross entropy and if you see
9:34
this function you'll see that the ignore
9:36
index is equal to minus 100 and that's
9:39
why all the target pairs which have
9:41
minus 100 in the token ID they will not
9:44
be utilized for the loss function
9:46
calculation the reason I went through
9:48
all this summary again is that it's very
9:50
important for you to understand how
9:51
batches are created and once you
9:54
understand how batches are created it's
9:55
pretty straightforward to create data
9:57
loaders which is what we'll understand
9:59
in today's
10:00
lecture great so now let's get started
10:03
with today's lecture where what we'll do
10:04
is that uh we have the training testing
10:07
and validation data sets right and we
10:09


***



have also batched the data set which
10:11
means that we have created multiple
10:12
batches we are going to pass these data
10:15
sets as inputs to data loaders and we
10:17
are going to create the training the
10:18
testing and the validation data loaders
10:22
so I'm going to take you to code right
10:23
now um so here let me first show you the
10:27
code to create the batches itself so
10:29
this this custom colate function this is
10:31
the final function what this does is
10:33
exactly what I had outlined on the
10:35
Whiteboard right it first converts the
10:39
tokens into token IDs then it does the
10:42
padding so that in every batch the
10:44
length of the number of tokens is the
10:46
same and then it replaces the 50256 with
10:50
a token ID of minus 100 and it creates
10:53
the input sensor and the target sensor
10:56
so these are the inputs and the target
10:58
batches which have been created
11:01
uh so we have to also specify the batch
11:04
length because in one batch there might
11:06
be eight different uh instruction
11:09
response pairs or there might be 10 or
11:11
there might be 50 we have to specify
11:13
what's the batch size and based on the
11:15
batch size inputs and targets will be
11:17
created for every instruction response
11:21
pair so let's say if we have this
11:23
instruction input output input sensor
11:25
and Target sensor will be created if we
11:27
have this instruction input output the
11:30
input and the targets will be created if
11:32
the batch size is equal to three all of
11:34
these will be in one
11:36
batch uh awesome so we have the inputs
11:40
and the target tensor which has been
11:42
created for the different batches but we
11:43
have not yet created the data loaders
11:45
right so that's what we have that's what
11:48
we have to do right now so now the main
Coding the data loaders in Python
11:50
goal of uh this lecture is to create
11:53
data loaders for an instruction data set
11:56
one small thing which I would like to
11:57
mention before we start this lecture is
12:00
about the device so if you see this this
12:04
piece of code over here we have
12:07
the uh we have the input sensor and we
12:09
have the target sensor for a particular
12:12
let's say for a particular instruction
12:14
input and output and uh we have this
12:16
code which lets which lets us transfer
12:18
these tensors to the Target device which
12:21
we have we already have integrated this
12:22
in our code now we'll see why we have
12:25
done that and what is the use of that so
12:27
the custom colate function includes code
12:30
to move the input and Target tensors to
12:32
a specific device that's the piece of
12:35
code which I just showed you right now
12:36
that device can be CPU or Cuda for gpus
12:40
or it can be MPS also for Max with apple
12:43
silicon chips so uh the advantage of
12:46
this is that in the previous chapters or
12:48
in the previous lectures we moved the
12:50
data onto the target device for
12:52
pre-training itself for example we used
12:54
the GPU memory when device equal to QA
12:57
and we did this in the training Loop
13:00
now we have this as a part of the custom
13:02
colate function itself so we have this
13:05
device transfer in the custom colate
13:07
function which is not part of the
13:08
training Loop and that gives a
13:10
significant Advantage having this code
13:13
as part of the custom colate function
13:15
offers the advantage of Performing the
13:17
device transfer process as a background
13:20
process outside the training Loop and
13:22
that prevents it from blocking the GPU
13:25
during model training so let's say if we
13:27
have if we are doing model training
13:29
we'll need to use the GPU right um and
13:32
we don't need to block the GPU during
13:34
that time because that's the most
13:35
important part so this this process we
13:39
are doing kind of in the background we
13:41
have not in included this transfer
13:43
process during the pre-training itself
13:46
so this process takes place in the
13:48
background and that makes sure that when
13:50
we do the training and when we are using
13:52
the GPU the GPU is not
13:54
blocked um so I hope you have understood
13:56
this and then we have to specify the
13:59
device itself so here device equal to
14:01
torch. device if you have GPU available
14:04
you can specify Cuda else it will
14:06
utilize your CPU I'm using a Macbook
14:08
2020 and I'm utilizing the CPU which is
14:12
the lowest configuration and uh whatever
14:15
I'm showing to you in this uh in these
14:17
set of lectures I'm deliberately showing
14:19
it on a low configuration device without
14:21
any GPU so that you can also replicate
14:23
it on your device but if you do have uh
14:28
if you do have a laptop with an apple
14:29
silicon chip you can uncommit this piece
14:31
of code because if the Silicon chip is
14:33
available you can just use storage.
14:35
device MPS that will run the code in a
14:37
much faster manner for now I'm just
14:40
going to use device equal to
14:41
CPU and then what we do is that if you
14:44
look at the custom colate function we
14:46
have to pass in the different arguments
14:48
right and one of the argument is uh
14:50
device and one such is allowed maximum
14:53
length now I want to Define another
14:55
function which defines these arguments
14:57
by default uh and I don't have to pass
15:00
in them separately and I'll do that
15:02
using the partial
15:04



***




command so we will be using the partial
15:07
function from pytorch Funk tools
15:09
standard library from Python's
15:11
functional tool standard library to
15:13
create a new version of the function
15:15
with the device argument pre-filled so
15:17
we'll already fill the device argument
15:19
with CPU in this case the device will be
15:21
CPU if you have a GPU the device will be
15:24
GPU and then I'll say the allowed
15:26
maximum length which is the context
15:27
length is going to be 1 02 4 what's the
15:31
context length here well we are doing
15:33
the input Target pairs right so the
15:35
entire prompt which we are going to look
15:37
at um let's see so if the prompt looks
15:40
something like this this entire thing um
15:44
will have the context length which means
15:46
that at one time what's the maximum
15:48
number of tokens which I'm going to look
15:51
at Great and then what we are doing is
15:54
that now we are ready to set up the data
15:56
loader so uh you see we have the
16:00
instruction data set class which we
16:02
defined in the previous lecture let me
16:04
take you to that class right now so that
16:06
you can understand what this instruction
16:08
data set class is doing so see this is
16:11
the instruction data set class what this
16:14
class does is that it takes the um it
16:17
takes the data set it converts it into
16:20
uh it converts it into token IDs so then
16:24
what we are doing is that we are
16:25
creating an instance of this instruction
16:27
data set class
16:31
yeah here you see we are creating an
16:32
instance of this data set class which is
16:34
the training data set so the instruction
16:36
data set also uh takes the training data
16:40
and converts it into token IDs then we
16:43
have to pass the training data set into
16:45
the data loader itself um and uh we have
16:49
to use the colate function this is where
16:51
most of the magic actually happens uh we
16:54
have used a customized colate function
16:55
here which means that uh whatever I told
16:59
in this part the process which we create
17:02
the process through which we create
17:03
batches that has been implemented in the
17:05
customized colate function you have a
17:08
batch you have uh multiple prompts in
17:11
that batch you first format them using
17:13
the alpaka style format you make sure
17:16
that in one batch the length of the
17:17
token IDs is the same then you pad with
17:20
50256 tokens and then except for 5256
17:23
you replace everything with minus 100
17:26
that's what's happening in this colate
17:28
function
17:29
and then the reason we are calling it
17:31
training loader is that the data set is
17:33
the training data set remember that we
17:35
are using uh 10% we are using 80% of the
17:39
data as training let me actually check
17:42
that how much percentage of the data we
17:44
are using for
17:48
training yeah we are using 85% of the
17:50
data for training 10% for testing and 5%
17:53
for validation so of this entire data of
17:56
this entire data set 85% is used for for
17:59
training uh so the train data set
18:01
consists of this training data but it's
18:03
converted into token IDs and then using
18:06
the colate function we collect it into
18:09
batches so here the batch size is equal
18:11
to eight so one batch will have eight
18:13
prompts and uh each alpaka prompt has
18:16
the instruction input output Pairs and
18:18
remember that in each batch the number
18:20
of token IDs um or the length of each
18:23
input the number of token IDs in each
18:25
input is the same that's the train
18:27
loader and and then similarly we create
18:30
the validation loader and we create the
18:31
test loader to create the train loader
18:34
the validation loader and the test
18:36
loader we use this data loader function
18:38
and I've already uh mentioned to you
18:40
about the data loader it helps us access
18:42
the data in a very easy manner so the
18:45
simplest way to think about the train
18:46
loader the validation loader and data
18:48
loader is that it creates batches so now
Visualizing training data loader output
18:51
if you actually run this piece of code
18:52
and if you print out the train loader
18:55
and if you print out the shape let's see
18:57
so the train loader looks like this if
18:59
you look at uh the first entry here
19:02
that's the first batch and the first
19:04
batch has eight samples because the
19:06
batch size is equal to 8 and why are
19:08
there two there are two such things here
19:10
the first is the input sensor and the
19:12
second is the target stenor so what's
19:15
happening in this 8X 61 and 8x 61 is
19:19
that um let's say you have prompt number
19:23
one which looks something like this
19:24
prompt number two similarly there are
19:27
eight prompts
19:29
and all of them have the same length
19:31
which is equal to 61 so this is my
19:34
inputs input sensor and similarly there
19:37
is a target sensor the target sensor
19:40
will just be the input shifted to the
19:42
right by one right U and then there will
19:45
be eight such Target tensors here so
19:47
this will be 8X 61 and this will be 8X
19:52
61 this is exactly the first row so the
19:55
first row the input sensor is 8x 61 the
19:57
target sensor is 8x 61 that's the first
20:00
batch now let's look at the second batch
20:02
why is the second entor 76 and in the
20:05



***




first it was only 61 the reason it's 76
20:08
is that when we look at the second batch
20:11
when we look at the second batch the
20:13
eight the eight samples might have
20:15
varying length right so we have to look
20:18
at that sample which has the maximum
20:20
length right and in the second batch it
20:22
will be 76 so then all the other samples
20:25
will be appended so that their length is
20:28
also equal to
20:29
76 so that's why every batch has
20:32
different number of token IDs in the
20:33
first batch it was 61 because the input
20:36
with the maximum token ID length would
20:38
have been 61 in the second it's 76 so it
20:41
might change because we are calculating
20:42
the maximum length for each batch
20:45
separately so this is the training
20:47
loader so now I hope you understand how
20:49
to access the data in the training
20:51
loader if you want to access the first
20:53
input Target pair in the training loader
20:56
you just look at the first row if you
20:57
want to access the second you just look
20:59
at the second row so that's why data
21:01
loaders are used because it's much
21:02
easier to access the inputs and Target
21:05
uh pairs if you want to access the
21:07
second batch and if you want to access
21:09
the second prompt of the second batch
21:10
you just look at the second row of the
21:13
second batch and you'll get the input
21:14
and the target pairs for the second
21:17
batch uh similarly you can print out the
21:20
inputs and targets for the validation
21:22
loaders as well the batch size will be
21:24
equal to eight but the number number of
21:28
batches will be small because the
21:30
validation data is 5% and the test data
21:33
is 10% the training data is 85% that's
21:36
why we have so many batches over here
21:39
but each batch has a batch size equal to
21:41
eight which means that the number of
21:42
samples in each batch will be equal to
21:45
eight uh okay so this is how we have
21:48
created the training loader we have
21:49
created the validation loader and we
21:51
have created the test loader as well and
21:53
we have printed the TR training loader
21:55
and I hope you have understood the shape
21:57
of the training loader because once you
21:59
understand the dimensions over here you
22:01
will really understand what we have done
22:02
in the train train loader and the
22:05
validation
22:05
loader so in the preceding output you
22:08
can see that the first input and Target
22:10
batch have dimensions of 8X 61 as you
22:13
can see over here 8 by 61 input and
22:16
Target batch of dimensions 8 by 61 where
22:19
8 represents the batch size and 61 is
22:21
the number of tokens in each training
22:23
example or number of token
22:26
IDs the second input and Target taret
22:28
batch have a different number of tokens
22:30
for instance 76 and that is because in
22:33
our custom colate function the data
22:34
loader is looking at each batch
22:36
separately and then it creates the token
22:39
length separately for each batch based
22:41
on the input with the maximum number of
22:43
tokens in that
22:44
batch this brings us to the end of
Recap and next steps
22:47
today's lecture where we looked at uh
22:50
data loaders and especially we looked at
22:53
this third step which
22:56
is creating data loaders
22:59
so one thing which I would like to
23:00
emphasize over here is that when you
23:02
think of fine
23:04
tuning uh a lot of time should be spent
23:07
on batching the data set and creating
23:09
the data loaders itself because once you
23:11
do that correctly the rest of the parts
23:13
essentially step number four step number
23:15
five and step number six can actually
23:17
proceed in a much more simplified and
23:19
easy manner if you understand how the
23:21
data loaders have been created so we
23:23
have spent a lot of time on SP uh step
23:26
number one step number two and step
23:28
number three and that is because as with
23:30
all machine learning projects the real
23:32
skill of an llm engineer or a machine
23:34
learning engineer is how you pre-process
23:36
the data how you clean the data and then
23:38
you feed it to the model uh that's why
23:40
it's very very important to spend a lot
23:42
of time on the data preparation step
23:44
itself so at the end of this lecture we
23:47
have actually finished stage one of the
23:48
instruction fine tuning which is
23:50
preparing the data set itself and now we
23:52
are ready to move to stage two which
23:55
we'll be covering in the next video so
23:57
thanks a lot every everyone I hope you
23:59
enjoyed this lecture where you
24:00
understood how to create data loaders
24:02
for an instruction data set in the next
24:04
lecture we'll be diving further into
24:07
loading the pre-trained llm and actually
24:10
fine-tuning um the llm and let's see the
24:13
performance thanks a lot everyone and I
24:15
look forward to seeing you in the next
24:17
lecture

***
