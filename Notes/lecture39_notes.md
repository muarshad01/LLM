* [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)

***

* 5:00
  
***

* 10:00
  
***

* 15:00



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




