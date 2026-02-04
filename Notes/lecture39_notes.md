* [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)

***

* 5:00
  
***

* 10:00
  
***

* 15:00

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





