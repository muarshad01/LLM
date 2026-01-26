## LLM finetuning approaches

* __Finetuning__: Adapting a pretrained model to a specific task by training the model on additional data.

#### What is LLM Fine-tuning?

* Fine-tuning LLM involves the additional training of a pre-existing model,
which has previously acquired patterns and features from an extensive data set, using a smaller, domain-specific dataset. In the context of "LLM Fine-tuning," LLM denotes a "Large Language Model," such as the GPT series by OpenAI.
 
***

* 5:00

#### Finetuning Types
1. Instruction finetuning
* Training a language on a set of tokens using specific instructions
2. classification finetuning
* Model is trained to recognise a specific set of class labels, such as spam or no spam

***

* 10:00


to me because I did not know that llms
10:37
could be used for classification tasks
10:39
in fact classification fing is also used
10:42
for sentiment classification such as
10:44
angry sad happy Etc if you have given
10:46
piece of text and if you want to
10:48
classify it into either of these five
10:49
buckets you can use class you can use
10:52
classification fine
10:53
tuning but instruction fine tuning is a
10:56
much more uh common B common fine tuning
11:01
it can handle broader set of
11:03
tasks and usually it can even handle or
11:06
it rather needs larger data sets because
11:09
you want it to handle a broader set of
11:11
tasks right so greater amount of
11:14
computational power is also usually
11:16
needed for instruction based Point
11:17
tuning why is greater amount of
11:19
computational power needed because you
11:21
have given instructions to the model and
11:23
it has to search for the entire Corpus
11:25
based on the instructions which you have
11:27
given right and the actions which it
11:29
perform is not specific so for example
11:31
here the action is pretty specific spam
11:33
or no spam the action here which it has
11:36
to perform can be something complex as
11:38
translation using the same instruction
11:40
based model you can do other other
11:42
things also so one instruction based
11:44
model can actually handle broader set of
11:46
tasks such as translation summarization
11:49
Etc but in classification fine tuning
11:52
the only action which is performed is
11:55
classifying in which category the text
11:57
belongs so so that's why the
12:00
classification fine tuning can handle an
12:01
arrow set of
12:03
prompts so when you think of fine tuning
12:05
people usually only talk about
12:07
instruction based fine tuning but I find
12:09
that classification fine tuning is also
12:11
equally important so the example which
12:13
we are going to start in today's lecture
12:15
is actually based on the concept of
12:16
classification fine
12:19
tuning okay so now in instruction based
12:22
fine tuning what usually people also
12:25
talk about is methods to make the fine
12:27
tuning more efficient So within
12:29
instruction based fine tuning there are
12:31
actually two other methods which are
12:34
called as Laura and QA so basically
12:37
these come under the category of
12:39
parameter efficient fine tuning so it's
12:41
a form of instruction fine tuning which
12:43
is more efficient than full fine tuning
12:46
essentially what is done in parameter
12:48
efficient fine tuning is that we only
12:51
update a subset of parameters and freeze
12:53
the rest of the parameters at a time so
12:56
this reduces the number of trainable
12:58
parameters making memory requirements
13:00
for instruction fine tuning more
13:02
manageable so Laura is
13:05
uh basically an improved fine tuning
13:08
method where instead of fine tuning all
13:10
the weights that constitute the weight
13:12
Matrix two smaller matrices that
13:15
approximate this larger Matrix are fine
13:17
tuned right now we are not going into
13:19
details of Laura we'll cover that in a
13:21
subsequent lecture as it's a pretty
13:22
broad topic but I just want to introduce
13:24
the concept to you today and the second
13:27
is Q Laura which is basically quantized
13:31
Laura and this is a more memory
13:33
efficient iteration of Laura and Q Laura
13:36
takes Laura a step further by quantizing
13:39
the weights of the Laura adapters to
13:41
lower Precision so right now just keep
13:43
in mind that Laura and qora are forms of
13:46
more efficient fine tuning which reduce
13:49
the memory requirement which is
13:51
traditionally needed in instruction
13:53
based V
13:54
tuning awesome so I hope you have
13:57
understood the differences between
14:00
uh instruction fine tuning and
14:03
classification fine tuning right and now
14:06
what we are going to do in today's
14:07
lecture is we are going to start working
14:09
on a Hands-On problem which is a fine
14:12
tuning classification problem so we are
14:15
going to look at the second category now
14:17
and we are going to take a real data set
Hands on project: email classification finetuning
14:20
we are going to look at emails a youth
14:22
set of emails which are spam as well as
14:24
no spam and we are going to train a
14:26
large language model to classify whether
14:28
it's spam or or nopan we won't be doing
14:31
all of this in today's lecture because
14:33
this this involves a lot of steps first
14:35
we have to download the data set
14:37
pre-process the data set create data
14:39
loaders this is in the stage one of data
14:41
set
14:42
preparation this data set will be the
14:44
all the emails which will be classifying
14:46
into spam and no spam then in stage
14:49
number two we'll have to initialize the
14:50
llm model load pre-train weights then
14:53
modify the model for fine tuning
14:55
Implement evaluation utilities and then
14:57
in the final stage we are going to
14:59
finetune the model evaluate the finetune
15:02
model and use model on new data I could
15:05
have covered all of this in one lecture
15:07
but then I would have to rush through it
15:09
instead I'm going to split this into



***


15:11
five to six lectures so that you
15:13
understand the entire fine tuning
15:14
process sequentially this is the
15:16
philosophy which we follow in all of the
15:18
lectures in this series I take you
15:20
through every single step in a lot of
15:22
detail first on a whiteboard and then
15:24
through code today what we are going to
15:27
do is we are going to do two things we
15:28
are going
15:29
to first download the data set and the
15:33
second thing what we are going to do is
15:34
we are going to pre-process the data set
15:37
and in the next lecture we'll create
15:39
data loaders initialize the model and
15:41
then later we'll also load the pre-train
15:43
weights for now let's just focus on step
15:45
number one and step number yeah actually
15:48
just step number one in this lecture
15:49
which is download and pre-process the
15:51
data set in the next lecture we'll look
15:53
at step two which is creating data
Coding: downloading the email classification dataset
15:55
loaders so let me take you through code
15:57
right now uh um right so in this section
16:02
of the this section of the code I have
16:03
titled as fine tuning for classification
16:06
the first step as I mentioned is
16:07
downloading the data set this is just
16:09
the code for downloading and unzipping
16:11
the data set let me take you through the
16:14
place where the data set exists so this
16:16
is the UC arwine machine learning
16:18
repository it's quite famous because it
16:20
has a large number of data sets within
16:23
this repository there is also an SMS
16:25
spam collection data this is the data
16:27
set which we'll be using it contains a
16:29
huge list of emails both spam as well as
16:32
non-spam so if you scroll down here you
16:35
will see that uh you'll see information
16:38
about what exactly is present in this
16:39
data set first you will see that 425 SMS
16:43
spam messages were manually
16:46
extracted uh this is a UK Forum in which
16:48
cell phone users make public claims
16:50
about SMS spam messages so 425 spam
16:54
messages and they also have 322 more
16:56
spam messages which are publicly
16:58
available some other website so overall
17:00
there are 747 spam messages now let's
17:04
look at the no spam so in no spam they
17:07
have a large number of messages as you
17:08
can imagine it's it's easier to Source
17:12
legitimate no spam messages the way they
17:15
collected no spam messages is at the
17:18
department of computer science at the
17:19
National University of Singapore so
17:22
naturally if it's emails originating in
17:25
the University between students uh they
17:28
are likely to be not spam so we have
17:30
much larger no spam messages compared to
17:33
spam messages so the data set is a bit
17:36
not balanced and the no spam messages
17:38
are also called ham messages I'm not
17:41
sure why this terminology exists but no
17:44
spam messages are called ham and the
17:46
spam messages are just called spam
17:47
messages so this is the data set you can
17:50
even download the data set from here or
17:53
you can just run the code which I will
17:54
be providing to you which downloads and
17:56
unzips the data once you run this piece
17:59
of code the data set will be downloaded
18:01
onto your local machine and uh it will
18:05
be downloaded in this collection SMS
18:07
spam collection. tsv so here you can see
18:11
on my vs code there is this folder
18:12
called SMS spam collection and this is
18:15
the tsv file which I'm showing on the
18:17
screen to you right now uh so here you
18:19
can see that the messages can either be
18:21
ham which is no spam or spam so ham
18:25
means not a Spam and spam is of course
18:27
spam so as expected spam uh is free
18:30
entry and uh winner basically as things
18:33
like this so here we can see that these
18:35
emails are indeed making sense and this
18:37
classification is making sense so this
18:40
is the entire data set which you can
18:42
download either through code or you can
18:44
download it from the UC repository which
18:47
I just showed you once the data set is
18:49
downloaded what you can do is that you
18:51
can use pandas and you can convert it
18:53
into a data frame so that reading the
18:55
data becomes much more easier so you can
18:57
print out the data frame and it looks
18:59
something like this so the label is
19:00
either ham or spam so here we can see
19:04
that there are total 5572 rows so 5572
19:08
emails but now we can print the value
Coding: Balancing the dataset
19:11
counts for the label in this data frame
19:14
so the label is a column name and we can
19:17
print out the number of ham entries
19:19
which are 4825 which are no spam and the
19:22
number of spam entries is much lesser
19:24
which is
19:25
747 now uh what what we can do here is
19:29
that we need to make the data set
19:31
balanced right there are number of ways
19:33
to make the data set balanced but we
19:35
will uh take a simple approach here
19:38
since this is not a classification
19:39
machine learning class this is a class
19:42
on large language models so we are going
19:44
to take a simple approach and we are
19:45
just going to randomly take 747 entries
19:49
from no spam so that the number of no
19:51
spam and the number of spam emails match
19:53
each other both should be
19:55
747 so as has been written here for simp
19:58
licity and because we prefer a small
20:00
data set for educational purposes we
20:03
subsample the data set so that it
20:04
contains 747 instances from each class
20:08
so this is a function which creates a
20:10
Balan data set what this function is
20:11
going to do is that it's it's going to
20:14
randomly sample ham instances so that we
20:17
can match the number of spam instances
20:19
that's equal to 747 it's done by this
20:22
line of code uh and then we combine the
20:25
ham subset with the spam so the total
20:28
now the new data frame is balance DF and
20:30
if you print out the balance DF value
20:33
counts you'll see that the number of ham
20:36
which is no spam emails are 747 and the
20:38
number of spam emails are
20:41
747 so after executing the previous code
20:44
to balance the data set we can see that
20:46
we now have an equal amount of spam and




***

* 20:00


no spam messages great this is exactly
20:50
what we wanted now we can go a step
20:52
further and we can to take a look at the
20:54
labels instead of having ham and spam uh
20:58
we can assign ham to be equal to zero
21:00
and spam to be equal to
21:02
one so these are the label encodings of
21:06
each of our emails so one note which
21:09
I've written here is that this process
21:11
is similar to converting text into token
21:13
IDs remember in uh when we pre-trained
21:17
the large language model we had a big
21:18
vocabulary the GPT vocabulary which had
21:22
uh more than 50,000 words in fact it had
21:24
50257 tokens and every token had a token
21:27
idid
21:29
this is a much simpler mapping we have
21:30
only two tokens kind of and they're
21:33
mapped to zero and
21:35
one now as we usually do in machine
Training, validation and testing dataset splits
21:37
learning tasks we'll take the data set
21:39
and split it into three parts we will
21:42
take this 747 data and we'll split it
21:44
70% will be used for training 10% will
21:47
will use for validation and 20% will use
21:50
for
21:52
testing um as I've mentioned here these
21:55
ratios are generally common in machine
21:56
learning to train adjust and eval at
21:58
models so here you can see I've written
22:01
a random split function what this
22:03
function is doing is that it just takes
22:05
the train end which is the fraction of
22:07
the training data which is train Frack
22:09
it's going to be 7 validation Frack is
22:11
going to be 0.2 so we first construct
22:14
the training data frame which is 70% of
22:17
the main data frame the validation data
22:20
frame is the remaining is 20% and the
22:22
test data frame is the remaining 10% is
22:25
the remaining 20% sorry the validation
22:27
data frame is the 10% the test data
22:30
frame is the remaining 20% of the full
22:32
data
22:33
frame and then when this function is
22:35
called out it will actually return the
22:37
training data frame the validation data
22:40
frame and the testing data frame so it
22:42
will return three data frames to us so
22:45
we now we can actually test this
22:47
function so we have this balance data
22:48
frame and we pass it into this function
22:50
called random split and once it is
22:52
passed into this function we also
22:54
specify the train fraction which is 7
22:57
and we specify the validation fraction
22:59
which is 0.1 and then we construct the
23:01
train data frame the validation data
23:03
frame and the test data frame so let us
23:06
check
23:08
uh whether the length makes sense so I'm
23:12
just going to type in new code here
23:14
which is length of train DF let's see
23:18
what's the
23:19
length it's
23:22
1045 yeah so length of train DF is
23:25
1045 uh because the total number of spam
23:28
and not spam is 747 + 747 which is
23:32
1494 then let me also print out length
23:37
of validation
23:40
DF and let me also print out length of
23:44
test DF and let me print out all of
23:48
these actually so that uh we can see
23:51
whether all of them indeed add up to
23:54
1494 Okay so
23:59
right so now I'm printing
24:01
this
24:18
uh okay so here you see that the length
24:21
of train DF is 1045 validation DF is 149
24:25
and test DF is 300 so let's add them
24:27
here 1045 1045 + 149 +
24:33
300 and let's see so it's 1494 and this
24:36
is 747 + 747 so that makes sense this is
24:40
kind of a check that the training
24:42
validation and testing data frames have
24:43
been created correctly I like to do
24:46
these checks once in a while to just
24:48
make sure that we are on the right track
24:49
in the code now what you can do is that
24:52
we'll also convert these data frames
24:54
into CSV files because we'll need to
24:56
reuse them later so we are just going to
24:58
use the 2 CSV function so you can search
25:02
this 2
25:04
CSV pandas what this does is that it can
25:07
take your data frame and it can convert
25:09
it into a CSV file I'll also add the
25:12
link to this in the information
25:14
description uh so you can apply this
25:17
function to the training data frame
25:18
validation data frame and also to the
25:20
testing data frame and then you can get
25:22
the train. CSV validation. CSV and the
25:25
test. CSV files so until now we have we
25:28
have reached a stage where we have
25:30
finished the first step which I had
25:32
mentioned over here and that first step
25:34
was to download and pre-process the data
25:36
set we downloaded the data set we
25:38
balanced the data set which was a part
25:40
of pre-processing so that the number of
25:42
spam and not spam are the same which is
25:45
747 and then we cons divided the data
25:48
set into training 70% validation 20% and
25:52
testing
Summary and next steps
25:54
10% and now in the next lecture what we
25:56
are going to see is that we are going to
25:58
first create data loaders so that we can
26:00
also do batch processing and it's
26:02
generally much better when you work with
26:04
large language models to use data
26:06
loaders and then we are going to see how
26:08
can we load the pre-trend Ws how can we
26:10
modify the model Etc you might be
26:13
thinking right how can a classification


***


26:15
task be done using large language models
26:18
so what happens towards the end is that
26:19
we usually fit another neural network at
26:22
the end of this and that will have a
26:24
softmax output so that the uh class so
26:28
that will be the
26:30
classification output so there is some
26:32
augmentation which we'll need to do here
26:34
so that the output is either spam or no
26:36
spam which is zero or one so we'll take
26:39
the architecture the same architecture
26:41
we worked on earlier but we'll augment
26:42
it we'll augment the end part of it the
26:45
architecture so that it's suitable for
26:48
classification so this brings us to the
26:50
end of the lecture thanks a lot everyone
26:52
I hope you are liking this approach of
26:54
whiteboard notes plus coding we have
26:57
covered a huge number of lectures in
26:58
this series before but if you are
27:00
landing onto this video series for the
27:02
first time it's fine I usually try to
27:04
make every lecture self-content but if
27:06
you want to revise any of your Concepts
27:08
or want to learn these Concepts from
27:10
scratch please go through the previous
27:12
lectures in this series also thanks a
27:15
lot everyone and I look forward to
27:16
seeing you in the next lecture

***





