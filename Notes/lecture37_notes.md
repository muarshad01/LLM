

* Pretrained LLMs are good at text completion
* They struggle with following instructions:
  * Fix the grammer in this text
  * convert this text into passive voice

***

* 5:00

if a
5:03
user asks about the refund policy
5:06
provide these these instructions you can
5:08
train or fine-tune the llm further so
5:11
that it can it can respond to queries
5:12
such as these so this ensures that the
5:15
chatbot not only understands General
5:17
language but also responds with company
5:20
specific information improving accuracy
5:22
and relevance that's why it's called
5:24
instruction fine tuning we have to train
5:27
the llm by saying that if you are asked
5:29
these instruction here is the answer
5:31
which you
5:32
provide the second example is that of
5:34
personalized Healthcare so a virtual
5:37
assistant in a healthcare setting is
5:39
designed to help patients schedule
5:41
appointments remind them to take Med
5:43
medications Etc now if you just take the
5:45
pre-trained model it may have General
5:48
Healthcare knowledge but it does not
5:50
have knowledge about the specific health
5:52
care provider practices treatment plans
5:55
Etc so the base language model may have
5:58
General Healthcare knowledge but find in
5:59
tuning it with The Specific Instructions
6:02
related to the healthc care providers
6:04
practices treatment plans and patient
6:06
needs is
6:08
crucial so fine tuning ensures that the
6:10
assistant understands medical
6:12
terminology follows specific Healthcare
6:15
guidelines and personalizes the
6:17
responses based on the patient history
6:19
for example when a patient asks about
6:21
diabetes management so uh how should I
6:25
or weekly what should be my plan with
6:27
respect to managing diabetes the
6:29
assistant can be fine tuned to provide
6:31
personally personalized advice based on
6:33
specific medical
6:35
guidelines so these are some practical
6:37
examples we show why fine tuning is
6:39
necessary fine tuning is necessary for
6:41
two main things first of all to train
6:44
the llm to follow instructions and
6:46
second to train on domain specific data
6:49
with respect to the company or with
6:51
respect to anyone who wants to deploy a
6:53
large language
6:54
model now let us look at some of the
6:56
instructions on which we can find tune
6:59
the large language model so one such
7:01
instruction can be convert 45 km to
7:04
meters and then the answer is 45 km is
7:07
45,000 M the second instruction can be
7:11
provide a synonym for bright and then
7:13
the desired response can be a synonym
7:15
for bright is radiant the third
7:18
instruction can be edit the following
7:20
sentence to remove all passive voice the
7:22
song was composed by the artist and the
7:25
desired response is the artist compos
7:27
the song
7:29
so now we have removed all the passive
7:31
voice and convert converted it into
7:33
active voice right so these are the type
7:36
of data sets on which we have to train
7:38
the large language model so that it can
7:40
understand how to follow
7:42
instructions and this is what is called
7:44
finetuning so training on such a data
7:46
set which I just showed where the input
7:49
output pairs are explicitly provided
7:52
such as in this case where we provide
7:53
the instruction the input and the output
7:55
and tell the llm that hey I know you are
7:57
pre-trained but you can do better with
8:00
respect to instructions here is a
8:01
training data set which will help you
8:03
get better at following
8:05
instructions now this approach is has
8:08
also another terminology which is called
8:10
as supervised instruction fine tuning
8:12
the reason it's called supervised is
8:14
that we are providing it a pair or a big
8:16
set of instructions and the responses
8:18
which we
8:20
expect all right now let's get started
8:22
with a Hands-On project which we are
8:24
going to implement so as I mentioned our
8:27
main goal is that we want to construct a
8:29
personal assistant so we won't be doing
8:31
all of that in this lecture but we'll be
8:33
covering a part of it here is the
Steps for instruction fine-tuning
8:35
sequential workflow which we are going
8:37
to follow to build our personal
8:39
assistant first we'll load the training
8:41
data set which consists of such
8:43
instructions and responses it won't be
8:46
just three but there will be more than
8:47
thousand instruction response pairs then
8:50
we will batch the data set create data
8:53
loaders then we'll load the pre-train
8:55
llm and then we'll finetune the llm
8:58
inspect the loss
9:00
extract responses do the evaluation and
9:02
then score the output or score how well
9:05
we are doing in today's lecture we are
Preparing and loading the dataset
9:07
going to see the first step which is
9:09
data set download and formatting this is
9:12
a bit different than what all we have
9:14
seen so far so this will be a very
9:15
interesting lecture all right so let's
9:18
get
9:19
started so we are looking at the first
9:21
step which is preparing a data set for
9:23
instruction fine tuning right so first
9:26
let us go to code and let me show you
9:28
how to download the data set
9:30
itself uh okay so here we are in the
9:33
code where we have started to look at
9:35
instruction find tuning in this section
9:37
we download and format the instruction
9:39
data set for instruction fine tuning a
9:42
pre-trend llm so we already have a
9:44
pre-trend llm in this section we are
9:46
going to download and format the
9:48
instruction data set the data set
9:50
consists of 1100 in 1100 instruction
9:54
response pairs let me just show you the
9:56
data set which we are going to use so
9:58
here's the data set which we are going
10:00
to use as you can see if you scroll down
10:03
below this is a huge data set which
10:06
consists of 1100 pairs of instruction
10:09
and responses and let's see how some
10:12
examples are defined over here so let's
10:14
take any example so let's take this so
10:18
you see there are if you think of it as
10:21
dictionary there are these keys so there
10:22
is an instruction for
10:25
each uh for each instruction response
10:28
pair we have an instruction an input and
10:30
an output so instruction is edit the
10:32
following sentence for grammar the input
10:35
is he goes to the park every day and the
10:38
output is he goes so the input is he go
10:40
to the park every day and the output is
10:43
he goes to the park every day so
10:44
remember three things we have to give
10:46
the instruction we have to give the
10:47
input and we have to give the output
10:49
that's done for all the questions so
10:51
here see instruction what are the first
10:53
10 square numbers in this case there is
10:55
no need to give an input because there
10:57
is a fixed answer and then the output is
11:00
14916 up to
11:03
100 uh then instruction is translate the
11:06
following sentence into French the input
11:08
is where is the nearest restaurant and
11:10
the output is it's translated into
11:12
French so as you scroll down you'll see
11:14
that there are instruction input output
11:16
pairs there are 1100 of them and some of
11:20
them have
11:21
inputs as you can see some of them have
11:24
inputs but some of them don't have an
11:25
input at all so for example if the
11:28
instruction is converted fet to meters
11:30
there is one answer to this right we
11:31
don't need to specify an input
11:33
separately so there is just the
11:35
instruction and there is just the
11:37
output so this is the data set which we
11:39
are going to load in the first step and
11:41
we are going to use the function called
11:42
download and load file so what this
11:45
function does is that it goes to this
11:46
URL which I just shared with you and it
11:49
downloads the data set from this URL so
11:51
you can run this function and it even
11:52
prints out the number of entries in this
11:54
function so we can see that the number
11:56
of entries is
11:58
1100 that indicates that we have indeed
12:01
downloaded the data set correctly now
12:04
let's go to the next step uh the data
12:06
list which we loaded from the Json file
12:09
contains 1100 entries of the instruction
12:11
data set let us print one of these so
12:14
let us print the 50th entry so data is
12:17
my final uh data set and I'm going to
12:19
access the 50th entry so this says that
12:22
identify the correct spelling of the
12:24
following word occasion and the correct
12:26
spelling is occasion let's check if this
12:28
is present in this data set so I'm going
12:30
to control F occasion and see it's
12:33
present this is the 50th entry great let
12:37
us print another entry which is 999 so
12:40
here you can see that instruction what
12:41
is the antonym of complicated there is
12:43
no input over here the output is the
12:46
antonym of complicated is
12:48
simple antonym is opposite awesome so we
12:52
have just tested that the data is loaded
12:55
correctly and we are able to access
12:57
specific instances or specific um index
13:02
of the data to see what is the
13:03
instruction input and the output
13:05
remember three things instruction input
13:09
and the
13:10
output next thing which we have to do is
Converting instructions into Alpaca prompt format
13:13
that we cannot leave the instruction
13:14
input and output in this format because
13:17
researchers have discovered that or
13:19
found out through experimentation that
13:21
there is a specific way in which the
13:23
prompt needs to be given to the llm
13:25
during the training process and that
13:28
specific way is
13:30
documented in terms of formats so for
13:32
example there is a specific format
13:34
through which you have to give these
13:36
instructions as mentioned by Stanford
13:38
alpaka there is one more different
13:40
format which was used by 53 so 53
13:45
Microsoft uh so for 53 open models there
13:48
was a different format which was used
13:50
for fine tuning but the most common one
13:53
which I've seen is uh the Stanford
13:56
alpaka based format in which what these
13:59
people do is that they also have similar
14:02
Json file where they have a list of
14:03
52,000 input output Pairs and they have
14:07
constructed a very specific prompt out
14:09
of these input output pairs so you can
14:11
see that as we showed you they have an
14:13
instruction they have an input and they
14:14
have an output for every uh input
14:17
response pair but then they convert it
14:20
into a prompt like
14:21
this so the prompt which they give to
14:24
finetune the alpaka model is that below
14:27
is an instruction that describes a task
14:30
paired with an input that provides
14:33
further context Write a response that
14:36
appropriately completes the request this
14:38
is the prompt which they train the llm
14:40
with below is an instruction that
14:42
describes a task paired with an input
14:45
that provides further context Write a
14:49
response that appropriately completes
14:51
the request and then they provide the
14:53
instruction the input and the response
14:56
so instruction is provided through this
14:59
this instruction which is over
15:02
here uh input is the input which is
15:06
accessed through the input field and
15:08
then the response is accessed through
15:10
this output field so this instruction
15:12
input and output is converted into a
15:15
prompt like this which is then provided
15:17
to the large language
15:19
model uh and this is exactly what we are
15:21
going to do in the code right now before
15:24
that I just want to show you these two
15:26
types of
15:27
formatting uh so if so currently we have
15:31
this instruction input and output right
15:33
there are two ways to actually format
15:35
this data set and convert it into a
15:37
prompt the first is the alpaka prompt
15:39
style and the second is the 53 prompt
15:42
style so as I showed you the alpaka
15:44
prompt style is that uh you convert you
15:48
have the prompt which is as follows
15:49
below is an instruction that describes a
15:52
task Write a response that appropriately
15:55
completes the request then in the
15:57
instruction you
15:59
uh add this you add this
16:03
instruction uh then in the input you add
16:07
the input and in the response you add
16:08
the output so that's the alpaka prompt
16:11
so this instruction input and output
16:13
which was there uh that is converted
16:16
into this type of a prompt that below is
16:18
an instruction here is the instruction
16:20
here is the input occasion and the
16:22
response is the correct spelling which
16:23
is occasion with One S now 53 which was
16:27
developed by Microsoft it's another fine
16:29
tuning style where the prompt is user
16:32
and assistant in the user you directly
16:35
give the instruction which is identify
16:36
the correct spelling of the following
16:38
word occasion and then in the assistant
16:40
you directly give the output so here you
16:42
see the difference between F and the
16:44
alpaka is that in the F prompting the
16:47
user what the user has is instruction
16:50
plus the input
16:53
so so instruction
16:59
so instruction
17:01
plus the input is actually fused over
17:04
here whereas in the alpaka prompt the
17:07
instruction and the input is separated
17:10
we can use either of these in fact when
17:12
I share the code file with you I will
17:14
encourage you to try the 53 prompt style
17:16
as well uh but since the alpaka prompt
17:19
style is more common we are going to use
17:21
this and we are going to convert the
17:24
instruction input and output which we
17:25
have into prompts such as what is
17:29
mentioned in the alpaka prompt
17:31
style okay so let us convert our
17:33
instructions into alpaka format we are
17:36
going to define a function which is
17:37
called format input it's going to take
17:39
an entry uh so you can think of as one
17:42
entry as this thing which has key value
17:45
pairs the key has instruction input
17:47
output and corresponding values right so
17:51
when this
17:52
function uh returns an entry you first
17:55
construct the instruction text which is
17:57
below is an instruction that describes a
17:59
task Write a response that appropriately
18:01
completes the request and then in the
18:03
instruction you take the dictionary
18:07
which is entry and then you find the
18:09
value corresponding to the instruction
18:11
key so in this case the value will be
18:13
identify the correct spelling of the
18:15
following word so then the prompt will
18:17
be below is an
18:18
instruction uh and then this is identify
18:21
the correct uh identify the correct
18:24
spelling of the following word that's
18:26
the instruction text then you have to
18:28
specify the input text which is input
18:30
and then you specify that particular
18:32
input in this case the input is occasion
18:35
now see what we are doing here if the
18:37
input is not present then you just
18:39
return blank so in cases like these
18:41
where the input is not present the input
18:44
will be left blank and this is mentioned
18:46
in the alpaka repository also if the in
18:49
input is not present then we just have
18:51
below is an instruction instruction and
18:53
the
18:55
response right so this is my format in
18:57
input function which takes the entry
19:00
dictionary and then it gives me the
19:02
instruction text and it gives me the
19:04
input text and it combines them together
19:06
so when you run the format input it will
19:09
give you this output if the input is
19:12
present and it will give you this output
19:14
if the input is absent currently we have
19:17
not added the response but I'll show you
19:19
where we can add it okay so this is the
19:22
format input function now let us test it
19:25
uh on a data set so we'll take the data
19:27
index by 50 and we have already seen
19:29
what that is before identify the correct
19:31
spelling of the following word and we
19:33
will pass this input to the format input
19:36
so now the format input takes in this
19:38
data and gives the model input the model
19:41
input is basically until this point
19:43
below is an instruction and then input
19:45
is the occasion and then we have to add
19:47
the response to this right so then here
19:49
we say that the response will be the
19:51
output um so the dictionary
19:54
indexed uh dictionary and then we look
19:56
at the value corresponding to the output
19:59
key so then we have the instruction and
20:01
the input and then we append this
20:03
desired response to the model input and
20:06
so the desired response is the correct
20:07
spelling is occasion so when you print
20:10
the model input plus the desired
20:11
response you'll get the model input as
20:13
the prompt and then you'll get the
20:15
response itself now this full thing is
20:17
later fed as an input to the large
20:19
language model so that it trains on this
20:21
entire
20:23
prompt So currently we saw an example of
20:25
a data which has the input right what if
20:27
we have an example of a data which does
20:29
not have the input so data index by
20:32
999 you see here the instruction is what
20:35
is the opposite of complicated there is
20:36
no input over here so let's see how our
20:39
code deals with that so when you input
20:41
the data index by 999 into the format
20:43
input function it gives the model input
20:46
and this will be below is an instruction
20:48
and then we just have the instruction
20:49
there is no input and then you have to
20:52
give the desired response which is
20:53
response and then the output here so the
20:56
output in this case was an antonym of
20:59
complicated is simple right so then that
21:02
will be the desired response and then
21:05
the model input will be combined with
21:07
the desired response and then we'll get
21:09
this entire answer so this entire output
21:12
is a mix of the prompt and the response
21:15
and then this whole thing is fed to the
21:16
large language model when we do the fine
21:18
tuning
21:19
later for now I just want to show you
21:22
that the data set uh was first
21:26
formatted uh through the Alpac style
21:28
format and converted into a specific
21:30
prompt and response output like this you
21:33
can of course change this when I share
21:35
this code file with you there is no need
21:36
to stick with this particular prompt but
21:39
for the sake of Simplicity and to follow
21:41
the convention we are doing this in this
21:43
video because uh if you see the Stanford
21:46
alpaka repository there are about
21:49
29,000 um stars
21:51
and around 4,000 Forks which means that
21:55
it's a pretty popular repository and
21:57
many people use this this kind of a
21:59
configuration this this kind of a
22:01
configuration when they do find
Splitting dataset into train-test-validation
22:04
tuning great so up till now what we have
22:06
done is that we have converted our
22:08
instructions into alpaka format now the
22:10
next thing is we will split our data set
22:12
into training testing and validation
22:14
right so we have the data now uh which I
22:17
have showed over here this has 1100
22:19
pairs we'll split them into training
22:22
testing and validation so we are going
22:24
to use 85% for training 10% for testing
22:27
and the remaining five 5% for validation
22:30
so we'll just uh index or we'll just get
22:33
the train data the test data and the
22:35
validation data from our main data based
22:38
on this these
22:39
fractions so the initial 85% is the
22:42
train data then the 10% is the test data
22:45
and the remaining 5% is the validation
22:47
data you can even print out the training
22:50
data set length the validation data set
22:52
length and the testing data set length
22:54
so you'll see that the training data is
22:56
9 935 pairs the validation data is 55
23:00
Pairs and the testing data is 110
23:04
pairs even on the Whiteboard I have seen
23:06
that or I've written rather that the
23:08
next step is partitioning the data set
23:10
into training testing and validation
23:13
training is 85% testing is 10%
23:16
validation is 5% of course you can feel
23:19
free to play around with these
23:20
parameters when I share the code with
23:22
you there are many things in this code
23:24
which are not set in stone which means
23:26
they are not fixed and we can continue
23:28
changing so many things we can change
23:30
these these fractions we can change this
23:34
format we can use the
23:36
Microsoft 53 format which I showed you
23:39
and all of this is open for exploration
23:41
right now this field is so new that uh
23:44
right now is the time to really start
23:46
exploring get into research that way you
23:48
will also develop lot of confidence as a
23:50
machine learning and an llm
23:53
engineer so today we are going to end
23:55
this lecture until this part where we
23:57
saw the dat data set download and
23:59
formatting in The Next Step what we are
24:02
going to see is that we are going to
24:03
batch the data set now this is a um
24:07
topic which will need some amount of
24:09
detailing because it's not very
24:11
straightforward we have to make sure
24:12
that the input length is the same for uh
24:16
all of the instructions then we have to
24:19
convert the instructions into token IDs
24:21
we have to pad them with tokens uh there
24:23
are some specific things which we need
24:25
to do which we'll look at in The Next
24:27
Step which is organizing data into
24:29
training batches and I've also started
24:31
writing the code for the next
24:34
lecture uh in four to five lectures
24:36
we'll build our own personal assistant
24:38
chatbot so then you will have built your
24:40
own chat GPT completely from scratch um
24:43
and that will set you apart from all the
24:45
students who are just consumers of chat
24:47
GPT you'll now build your own
24:49
personalized assistant and then you will
24:51
have the confidence that whenever you
24:53
approach any company you can build a
24:55
custom chatbot for that company as well
24:57
by following the same procedure thank
25:00
you so much everyone I hope you are
25:01
enjoying these lectures where it's a mix
25:03
of whiteboard approach as well as coding
25:06
approach uh please try to follow along
25:09
write notes if you are coming to this
25:11
lecture for the first time I encourage
25:12
you to watch all the previous lectures
25:14
which have happened so far so that you
25:16
can strengthen your understanding
25:18
anyways I make the lecture so that it's
25:20
as selfcontained as possible thanks
25:23
everyone and I look forward to seeing
25:24
you in the next lecture

