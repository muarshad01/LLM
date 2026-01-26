
0:00
[Music]
0:05
hello everyone and welcome to this
0:07
lecture in the build large language
0:09
models from scratch Series today what we
0:12
are going to do is that we are going to
0:14
have an introductory lecture on large
0:17
language model fine tuning if you have
0:20
followed this series so far we have
0:21
conducted around 30 to 32 lectures and
0:25
Within These lectures we have finished
0:27
the stage one which is understanding the
0:29
L Market Ure the attention mechanism
0:31
data preparation and sampling and we
0:34
have also finished stage number two
0:36
which is pre-training the large language
0:38
model model evaluation and loading
0:40
pre-trained
0:42
weights now that stage one and stage two
0:44
are finished we are now ready to move to
0:47
the last stage of building a large
0:49
language model from scratch and that is
0:51
the stage of
0:53
fine-tuning with pre-training we saw
0:55
that we are getting good results and
0:57
when we give an input text we are
1:00
getting output which makes a lot of
1:02
sense and it's pretty awesome so it's
1:04
like we have built our own GPT from
1:07
scratch and here's the architecture
1:09
which we use to construct our large
1:11
language model uh we spent a huge amount
1:14
of time and a number of lectures to
1:15
understand this
1:17
architecture awesome so now we are ready
1:19
to begin with this next stage which is
1:22
the stage of fine tuning so first you
1:24
might be thinking that okay we already
1:26
pre-trained the large language model and
1:28
it seems to already work pretty well
1:30
right so then does that mean we have
1:33
finished building our llm
1:35
no let me explain to you why do we need
1:38
fine tuning so let's say if you have
What is finetuning?
1:41
pre-trained the model that's fine but
1:43
what if you have a specific task so that
1:46
specific task can be constructing a
1:48
chatbot based on your own data as a
1:50
company or let's say you're an
1:52
educational company who wants to make an
1:55
educational app using your data let's
1:58
say if you want to make a chatbot as an
2:00
airline using your
2:03
data essentially if you want to make a
2:06
specific
2:08
application the pre-trained model is not
2:10
enough because it's pre-trained on
2:12
General data available from all over the
2:14
Internet you need to train the model
2:18
again on additional data this is called
2:21
as
2:22
finetuning so the formal definition of
2:25
fine tuning is adapting a pre-trained
2:27
model to a specific task
2:30
by training the model again on fine tune
2:33
on additional data there are some things
2:35
which are very important here some
2:37
terminologies the first terminology is
2:39
this specific
2:41
task so uh fine tuning is needed when
2:44
you have certain specific tasks which
2:46
need to be performed such as if you are
2:48
a company and if you want to develop a
2:50
model put it into production you cannot
2:52
just use a pre-trained model as I
2:55
mentioned it's trained on a generic data
2:57
right it will not give answers which you
2:59
expect based on your individual private
3:01
data so there is a specific task if you
3:05
need to do a specific task you need to
3:07
fine tune the pre-trend model and the
3:09
second thing is training the model again
3:12
that means that until now the certain
3:14
weights and biases of the model have
3:16
been optimized during the pre
3:18
pre-training but now you are going to
3:20
feed the model with additional data so
3:22
naturally you will need to train the
3:24
model again the parameters the weights
3:25
the biases are going to change and
3:28
that's usually done in the process of
3:29
fine
3:30
tuning so there are some in fact a lot
3:34
of Articles written about fine tuning so
3:36
here is the schematic you have a large
3:38
language model which is pre-trained then
3:40
you train it further on custom data set
3:43
and that leads to a fine tuned large
3:46
language model okay so the formal
3:48
definition of llm fine tuning is that
3:51
fine tuning llm
3:54
involves the additional training of a
3:58
pre-existing model
4:00
which has previously acquired patterns
4:02
and features from an extensive data set
4:04
using a smaller domain specific data set
4:08
so see we are training a pre-existing
4:10
model again so that's why it's called
4:12
additional training why are we training
4:14
this model again because we have a newer
4:16
smaller domain specific data set so we
4:19
need to train the model again so that it
4:21
adapt its parameters it adapt it adapts
4:24
its weights and its
4:26
biases okay so fine tuning is necessary
4:31
to to be done after pre-training is done
4:34
okay so open AI even provides this
4:38
uh description about fine tuning where
4:42
it gives you instructions about how you
4:44
can do fine tuning using open models so
Finetuning practical example
4:47
to give you a practical example of fine
4:49
tuning this is the website which I had
4:51
made in the final year of my PhD it
4:54
talks about my Publications my talks my
4:56
media Etc so let's say if you also have
4:59
a website or if you have a blog post
5:01
site like this and if you want to make a
5:03
chatbot which does not answer like how
5:06
chat GPT answers but you want the
5:09
chatbot to answer like how you speak
5:11
right you want the chatbot to answer
5:15
based on your data how you generally
5:17
write articles how your how you word the
5:20
different paragraphs on your website you



***


5:23
have a specific tone and you want that
5:25
tone to come in your chat bot how will
5:27
you do this with pre training it's not
5:30
possible because in the pre-train model
5:33
might be your data is not even there in
5:35
the pre-training so then you will need
5:37
to find tune you will need to give
5:38
additional data such as whatever is
5:40
present on my website my blogs my
5:42
Publications my talks so that the model
5:45
can understand and learn what is your
5:47
tone of speaking how do you generally
5:49
construct sentences and then it will
5:51
adapt so the resulting chatbot which
5:55
will be developed will now be speaking
5:58
in your tone or my tone rather in this
6:01
example that is called as fine tuning so
6:03
the specific application which I'm
6:05
looking for is constructing a chatbot
6:07
which speaks in my tone for this
6:09
specific application I have this
6:11
additional data which is my website my
6:13
Publications my talks and my media which
6:15
I'll feed to the model and I'll ask the
6:17
model to be trained again or rather I'll
6:19
train the model again based on this
6:22
additional data this process of training
6:24
the model again is called as finetuning
6:27
so now the finetune model will behave EX
6:29
exactly like how I want it will speak in
6:32
my tone it will use grammatical
6:34
sentences like I do that's a practical
6:36
example of fine tuning another example
6:39
is let's say you have uh your research
6:43
Publications like how I'm describing
6:45
right here and you want to make a
6:46
chatbot which basically answers people's
6:50
queries about your Publications so then
6:52
you can feed specific data based on your
6:54
Publications that's another example of
6:57
fine tuning in fact uh the reason I
7:00
thought of this example was I saw a Blog
7:03
uh in fact I saw a question asked on the
7:06
open AI question Forum where the person
7:09
was saying that they are a beginner at
7:11
learning fine tuning and their purpose
7:13
is to create the model which could use
7:15
the tone of their voice from their blog
7:17
exactly like what we had discussed and
7:20
they are asking a number of questions
7:22
about how exactly to go about this how
7:25
to make sure that the accuracy obtained
7:27
is very good how to prevent
7:28
hallucinations which are wrong answers
7:30
Etc so and then there are number of
7:32
answers which is given by the opena
7:34
community to help this person F tune
7:36
their
7:38
model great so this is the general
7:40
introduction of fine tuning now within
Instruction and classification finetuning
7:42
fine tuning itself there are two broad
7:45
categories of fine tuning and let me
7:48
Mark them here for you the first
7:49
category is called as instruction fine
7:51
tuning and this is much more common and
7:53
much more broader the second category is
7:56
called as classification F unun what's
7:59
the difference between the two so in
8:01
instruction find tuning what we do is
8:03
that we train the language model on a
8:06
set of tasks using Specific Instructions
8:10
so here I have given two examples for
8:13
instruction uh instruction based F
8:15
tuning so let's say the instructions are
8:19
U is the following text spam answer with
8:22
a yes or no this is an example of
8:24
instruction based fine tuning because we
8:26
are asking the llm that you will be
8:28
given text like this and your task is to
8:31
look at the text and answer whether it's
8:33
spam so classify or say whether it's yes
8:35
or no that's why this is an example of
8:37
instruction fine tuning we are adding
8:39
instructions here which are in blue
8:41
color that's very important or the
8:43
instruction could be your given sentence
8:45
and translate it into German
8:47
appropriately that is another example of
8:49
instruction find tuning so the llm takes
8:52
the input sentence in the first case it
8:54
gives the output as yes or no in the
8:56
second case it translates the English
8:58
sentence into German
9:00
the reason these are examples of
9:01
instruction fine tuning is that we are
9:03
giving a specific instruction and the
9:05
llm is behaving based on that
9:09
instruction the second example is
9:11
classification fine tuning this is not
9:13
as common but readers or listeners
9:16
rather who have learned about machine
9:18
learning and classification examples
9:21
such as brain tumor classification image
9:23
classification between cats and dogs
9:25
this is pretty similar but now you just
9:27
use a large language model inste of
9:29
let's say a convolutional neural network
9:31
for example or any other neural network
9:34
to make the classification so here no
9:37
instruction is given to the llm just the
9:39
input is given without
9:41
instructions and the llm has to classify
9:44
whether the email is Spam or not spam
9:46
here also we saw a Spam not spam example
9:49
right but here instructions were given
9:51
that the llm has to answer with a yes or
9:53
no but here instruction is not given
9:56
just two categories will be there spam
9:57
and not spam and we train the llm to
10:00
make the output based on these two
10:02
categories so the llm receives a text
10:05
and then it predicts whether it's a Spam
10:06
or no spam although the result is same
10:08
in this case and this case the way the
10:12
prompt itself is constructed or way the
10:13
llm is constructed is different that's




***



10:16
why this what I'm highlighting right now
10:18
this is an example of instruction F
10:20
tuning and this example where the model
10:23
input is given without instructions
10:24
that's an example of classification
10:26
based fine
10:28
tuning uh um where we just classify into
10:31
categories using large language models
10:33
when I saw this it was pretty surprising
10:35
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
20:48
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

