stages of building
Pretraining
1:44
llms there are basically two stages
1:47
which we are going to look at creating
1:49
an llm involves the 
first stage which is called as pre-training and second stage which is called as fine-tuning.

What exactly is pre-training
What exactly is finetuning

pre-training just means training on a large and diverse data set
2:24




they are trained
3:11
on a huge and diverse set of
3:14
data.



you can think of one token as equal to
3:51
one word there is a bit of detailing
3:54
here but that's not the scope of today's
3:57
lecture 

so for the purposes of this
3:59
lecture just think of one token to be
4:00
equal to one word







task
5:59
which is called called as word
6:00
completion which means you are given a
6:04
set of words let's say you are given uh
6:07
this
6:09
sentence the
6:11
[Music]
6:14
lion is in
6:18
the dash so this next word is not known
6:23
and chat GPT then predicts or llms then
6:27
predict this word as let's say forest

if you train the llm for this
6:41
simple task it turns out that it can do
6:44
a wide range of other tasks as well so
 the underlying language model to
7:13
begin to perform tasks without even
7:16
training on
7:18

them for example Performance on tasks
7:21
like picking the right answer to a
7:23
multiple choice question steadily
7:25
increases as the underlying language
7:27
model improves this means that even
7:30
though you just train the llm for
7:32
predicting the next word 

like how how I
7:35
mentioned to you before it turned out
7:37
that the llm can also do variety of
7:39
other things such as translation such as
7:42
answering multiple choice question such
7:44
as summarizing a text then sentiment
7:47
detection
7:48
Etc 

***

pre-trained so why why is the second
10:22
stage which is fine tuning why is that
10:24
really needed the reason fine tuning is
10:27
important is because let's say you are
10:31
uh you are a manager of an airline
10:32
company or you are the CEO of an airline
10:36
company and you want to develop a
10:39
chatbot so that users can interact with
10:42
the chatbot and the chatbot responds
10:45
let's a user can ask some question like
10:47
hey what's the price for the Lanza
10:49
Airline which leaves at 600 p.m. now the
10:54
response which you want is very specific
10:56
to your company the resp response which
10:59
you want is not generic response which
11:03
is collected from all the places on the
11:06
internet so if you just use the
11:08
pre-trained model and if you use that
11:11
model to build the chatbot it it has
11:13
been trained on a huge amount of data
11:15
not just your company data Maybe the
11:17
pre-trend model does not even have
11:18
access to your company data so the
11:21
answer which will come is pretty generic
11:23
it will not be specific to your company
11:26
it will not be specific to your
11:27
application
11:29
secondly as I showed you before we can
11:33
use chat GPT to generate multiple choice
11:36
question but let's say if you are a very
11:37
big educational company and if you want
11:40
to develop really very high quality
11:43
questions maybe you should not just rely
11:45
on the pre-train model and you should
11:47
fine tune the model so that it's better
11:50
for your specific
11:52
application so the main purposes of of
11:55
fine tuning are it is basically a
11:58
refinement on the pre-training on a much
12:01
narrower data set so let's say you are a
12:04
bank JP Morgan and if you have collected
12:06
huge amount of data which is not
12:08
publicly available what you can do you
12:10
can take the pre-train model and then
12:12
you can give your own data set and then
12:15
train the model again on your data set
12:17
so that when it answers it
12:20
answers uh in such a way which is
12:22
specific to your
12:24
company so finetuning is needed if you
12:27
want to build an application which is
12:28
specific to a particular task or a
12:30
particular domain if you are a general
12:33
user if you are a student who just wants
12:35
to let's say use chat GPT to get
12:37
questions to uh get information about
12:40
certain things then you can just use gp4
12:43
you will not need um fine tuning too
12:46
much but if you are a big company let's
12:47
say and if you are wanting to deploy llm
12:52
applications in the real world on your
12:54
data set you will need fine tuning let
12:57
me give you some examples of this which
12:58
are mentioned on open A's website also
13:01
so open a actually mentions so many
13:04
things on their blog posts and their
13:05
website which not many people know about
13:08
so let's say let's look at this company
13:10
called SK Telecom and it wanted to build
13:13
a chatbot to improve customer service
13:16
interactions for Telecom related
13:18
conversations in Korean now if it just
13:21
used the gp4 it's not suited for this
13:25
particular requirement right gp4 maybe
13:27
is not trained on Telecom conversation
13:31
in Korean so the training data did not
13:33
involve this probably so what this SK
13:36
Telecom will do is that it will finetune
13:38
gp4 by using its own training data so
13:41
that it gets a fine tune model which is
13:44
specific for its purpose as you can see
13:46
for SK Telecom this resulted in
13:48
significant improve in per Improvement
13:50
in performance 35% increase in
13:53
conversation summarization quality and
13:56
33% increase in intent recognition
13:59
accuracy that just one
14:01
example uh the second example which you
14:04
can see is the an example called Harvey
14:08
so Harvey is basically an AI legal tool
14:10
for attorneys so now imagine that if you
14:14
have open a if you just use gp4 without
14:18
fine tuning what if gp4 is not trained
14:21
on legal cases what if the data is not
14:25
does not cover the legal cases which
14:27
happened in countries
14:29
so then that's not a good tool for
14:31
attorneys right attorneys ideally or
14:34
lawyers want an AI tool which is trained
14:37
on legal case
14:39
history so as you have seen here while
14:42
foundational models so pre-train models
14:44
are also called foundational models
14:47
while foundational models were strong at
14:49
reasoning they lacked the extensive
14:52
knowledge of legal case history and
14:55
other knowledge required for legal work
14:59
so the training data set lack the
15:01
knowledge of legal case history and
15:03
that's why if you were to build such an
15:05
AI tool which can assist lawyers and
15:08
attorneys you have to include the
15:11
specific legal case history data and
15:13
that's why you will have to fine tune
15:15
the llm further remember one key
15:18
terminology which I used here the
15:20
pre-trained data or the pre-trained
15:22
model is also called as the foundational
15:24
model and the fine tuning happens after
15:27
that
15:29
so here's Harvey basically harvey. you
15:32
can go to this link right now so this is
15:33
a trusted legal AI platform and if you
15:37
are thinking how different is it from
15:39
gp4 now you have your answer this is a
15:41
fine-tuned model which is specifically
15:45
fine tuned on data sets which include
15:47
legal case
15:49
history and as you can see Harvey works
15:52
with the world's best legal teams it it
15:54
works really well here is another
15:56
article which says JP Morgan Chase UNS
16:00
AI powered llm Suite may replace
16:03
research analysis now you might be
16:05
thinking if gp4 is already there why did
16:07
JP Morgan unve its own AI power llm and
16:11
the reason is because it's fine-tuned
16:13
with their own data it's fine tuned for
16:17
their employees specifically maybe the
16:20
JP Morgans data is not available
16:22
publicly to anyone so only they have the
16:24
data and they have trained the llm which
16:26
is fine tuned so that the answers are
16:28
specific for their
16:32
company okay so this is so I showed you
16:35
examples in the tele communication
16:38
sector which is this SK Telecom I showed
16:41
you examples in the legal sector which
16:43
is the example of Harvey and then I also
16:47
showed you examples in the economics or
16:49
banking sector essentially you will see
16:51
that when you go to a production level
16:53
or when you think of startups or
16:55
Industries you will definitely need fine
16:57
tuning uh directly using gp4 is good for
17:01
students because it satisfies their
17:03
purposes but fine tuning is needed as
17:05
you build more advanced
17:07
applications so these are the two stages
17:10
of building an llm the first stage is
17:13
pre-training as I mentioned and the
17:15
second stage is fine tuning I hope you
17:17
have understood up till this point if
17:19
something is unclear please put it in
17:21
the comment section of this particular
17:23
video now just so that this explain this
17:27
concept is explained to you in a better
17:29
manner I have created this pre-training
17:31
plus fine tuning schematic so that you
17:34
can go through this schematic step by
17:36
step to get a visual
17:38
representation so let's start with the
17:40
first block the first block is the data
17:43
on which the models are trained on
17:46
whether you do pre-training and later
17:48
whether you do F tuning you cannot get
17:50
anywhere without data so the data is
17:53
either internet text books media
17:55
research
17:56
articles um we saw this data over here
17:59
right you need huge amount of data and
18:02
you need to train the large language
18:06
model on this data set this data set can
18:09
include billions or even trillions of
18:11
words now one more point which I want to
18:14
raise here is the computational cost for
18:17
training the llm right so you're
18:19
training this llm on a huge amount of
18:21
data which is also the second step of
18:23
this schematic now to train this you
18:25
need computational power you need
18:27
computational units and it's not
18:30
possible for normal students or even for
18:32
normal people who don't have access to
18:34
powerful gpus to do this just to give
18:37
you a sense of the cost the total
18:39
pre-training cost for gpt3 is $4.6
18:44
million this is a huge amount $4.6
18:47
million think about it for pre-training
18:51
of gpt3 it cost this
18:53
much so the first two steps in this
18:56
schematic which is collecting this huge
18:58
amount of data and then training the llm
19:01
requires a lot of computational power
19:03
requires a lot of computational energy
19:05
and of course a lot of
19:07
money so when you train a pre-trained
19:10
llm like this it's also called as
19:11
foundational model and it is this also
19:14
is awesome it has huge amount of
19:16
capabilities like I'm interacting with
19:18
gp4 right now it's a foundational model
19:20
it's a pre-trend model it still has huge
19:23
amount of
19:25
capabilities and then the third step in
19:27
this is find tuning and fine tuning is
19:31
also so the third step as I mentioned is
19:33
fine tuned llm and after fine tuning you
19:36
can get specific applications like you
19:38
can build your own personal assistant
19:40
you can build a lang translation bot you
19:42
can build a summarization assistant you
19:44
can build your own classification bot so
19:47
if you are a company or an industry or a
19:49
startup who is looking for these
19:51
specific applications using your own
19:53
data you will fine tune the pre-trend
19:56
llm on the label data set
19:59
that's the three steps so the first step
20:02
is data second step is training the
20:05
foundational model and the third step is
20:08
fine tuning on the specific task now
20:12
there is also one thing between
20:14
unlabeled data and label data for this
20:17
pre-training we usually don't label the
20:19
data set uh it's an unsupervised
20:22
learning task and it's also called Auto
20:24
regression because if you are
20:26
considering a sentence and predicting
20:28
the next word the part of the sentence
20:30
is used for training and the next word
20:32
is used for forecasting or testing and
20:35
then for the next set of training the
20:37
earlier forecasted word is used as
20:39
training it's fine if you don't
20:41
understand this terminology right now
20:43
but just remember that pre-training is
20:45
done on unlabeled text Data whereas
20:47
finetuning is mostly done on labelled
20:49
text
20:50
Data let's say if you are doing a
20:53
classification task right or if you want
20:55
a specific llm task which classifies
20:58
email say spam or no spam you will need
21:01
to provide labels for which emails were
21:03
spam which were not spam Etc so
21:06
generally for fine tuning we will have a
21:09
labeled data
21:11
set let's say Harvey you considering the
21:14
case of Harvey which we s for legal data
21:16
we had these case studies uh legal case
21:19
studies right you will need to provide
21:21
those case studies and mention that for
21:23
these questions these were the answers
21:25
in this particular case study Etc so in
21:28
a sense that that becomes label
21:30
data so please keep this pre-training
21:33
plus fine tuning schematic in mind what
21:35
many students do is that they do not
21:37
understand the difference between
21:38
pre-training and fine tuning and they
21:41
also don't understand why is it called
21:42
pre-training in the first place
21:44
shouldn't it just be training ideally it
21:46
should just be called as training but
21:48
it's called pre-training because there
21:50
is also this fine-tuning component so
21:52
it's just a terminology thing so if you
21:55
understand this then you will get a
21:57
better Clarity of when you interact with
21:59
GPT you are essentially interacting with
22:02
a pre-trained or a foundational
22:05
model okay so I hope you have understood
22:09
the schematic and I hope you have
22:10
understood the difference between
22:12
pre-training and fine tuning now I want
Steps for building an LLM
22:14
to just show you the steps which are
22:16
needed for building an L which basically
22:18
summarizes what all we have covered in
22:20
today's lecture so the first step is to
22:23
train on a large Corpus of Text data
22:26
which is also called raw text remember
22:29
some of these terminologies keep on
22:30
appearing everywhere they actually mean
22:32
something very simple but they just
22:34
sound a bit fancy so raw text is just
22:37
basically training on a large Corpus of
22:39
text Data a more formal definition of
22:41
raw text is regular text without any
22:44
labeling information so raw text means
22:46
text which does not come with predefined
22:48
labels so first you train on such such
22:52
kind of a raw text and you need huge
22:54
amount of raw
22:55
text the second task is the first
22:58
training stage of nlm and this is called
23:01
as pre-training as we already saw this
23:03
task involves creation of an initially
23:06
pre-trained llm which is also called as
23:07
the base or a foundational model gp4 for
23:11
example what we saw in the paper this
23:13
paper over here I'll also be sharing the
23:16
link to this paper in the information
23:19
section uh this paper is a pre-train
23:22
model and uh it is actually capable of
23:25
text completion but it also turns out
23:27
it's capable of many other things like
23:30
uh as we saw over here sentiment
23:32
analysis question answering Etc so
23:34
pre-training also gives the llm a lot of
23:37
other
23:39
powers so that is Step number two step
23:42
number three after obtaining the
23:44
pre-trained llm the llm is further
23:46
trained on label data and this is fine
23:50
tuning so let me just mention over here
23:53
this is called as fine tuning and we saw
23:55
a number of applications of this in
23:57
different areas we saw applications in
24:00
the legal area then we saw applications
24:03
in the Comm telecommunication sector
24:06
this es Telecom Harvey we also saw an
24:09
application related to JP Morgan Chase
24:12
so fine tuning is essential as and when
24:14
you go into production and if you want a
24:17
really powerful model train specifically
24:19
on your data set even within fine tuning
24:22
there are usually two categories the
24:25
first category it's called instruction
24:26
fine tuning and second is called fine
24:29
tuning for classification tasks so let's
24:31
say you are an education company and if
24:33
you want
24:35
to uh if you want to or let's say if
24:38
you're a company like u a text
24:40
translation company which converts
24:42
English into French so here you might
24:45
need to give some sort of an instruction
24:47
like this is the English language and
24:49
you convert it into the French language
24:51
so you might need to give instruction
24:53
answer pairs as label data set so the
24:56
label data set means showing that okay
24:58
these are all the English language these
24:59
are all the French language use this to
25:02
further fine tune what you already know
25:04
so that's called as instruction fine
25:06
tuning even for Airline customer support
25:09
let's say someone is asking a question
25:11
and then the customer support responds
25:13
you might have a label data set which
25:15
consist of usually for this instruction
25:17
these are the responses let's say and
25:20
then you give this label data for fine
25:22
tuning the second type of fine tuning is
25:24
for classification even here you need a
25:27
label data let's say you want to build
25:29
an AI agent which classifies emails into
25:31
spam and no spam you need to give a
25:33
label data set which consists of the
25:36
text and the associated labels maybe
25:38
you'll give 10,000 emails and maybe
25:41
5,000 are spam 5,000 are not spam as the
25:44
training data to this llm and then you
25:47
will find tune it further so remember
25:50
pre-training usually does not need label
25:52
data and pre-training is so pre-training
25:55
does not need any without any labeling
25:57
information that's also fine but for
26:00
fine tuning you typically need a labeled
26:02
data
26:04
set okay so this actually brings us to
26:07
the end of today's lecture where we
26:09
covered about the two stages of building
26:11
an llm in particular pre-training and
26:13
fine tuning in pre-training we saw that
26:16
you have to train on a big Corpus of
26:18
diverse data and you need a lot of
26:21
computational power retraining is not
26:23
possible unless you have first of all
26:25
access to GPU and access to this kind of
26:27
money $4.6 million for pre-training
26:31
gpt3 and then the third step so after
26:34
data after pre-training those are the
26:36
first two steps final step is finetuning
26:39
which is usually done on a labeled data
26:41
set and using fine tune fine tuned llm
26:44
you can do specific tasks such as
26:47
classification
26:48
summarization uh translation and
26:50
building your own chatbot so nowadays
26:53
you must have seen companies are
26:54
building their own llm specific chatbots
26:57
all of these companies will do some kind
26:59
of fine tuning they never use just
27:01
foundational models you will see that
27:04
big companies never only use
27:05
foundational model they will have to go
27:07
that next step of fine tuning and that
27:10
is much more expensive rather than using
27:12
the foundational
27:14
model I hope you are understanding these
27:17
lectures and I'm keeping it a bit visual
27:19
so that there is some sort of visually
27:23
visual stimulation as you learn so and
27:26
I'm also writing these notes on a white
27:28
so that you if you see someone doing
27:30
this handson in front of you you will
27:32
also remain motivated for these lectures
27:35
I encourage you to also write down some
27:37
key points in a notebook or in a mirror
27:40
white board like I'm doing right now in
27:42
the next lecture we are going to start
27:44
looking at basic introduction to
27:46
Transformers and we'll also have a brief
27:48
look at the attention is all you need
27:50
paper we'll have maybe two to three more
27:52
lectures on this uh initial modules and
27:56
then we'll dive into uh
27:59
coding so thank you everyone and I look
28:02
forward to seeing you in the next
28:04
lecture





