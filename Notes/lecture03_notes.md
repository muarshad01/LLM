### Stages of building LLMs
1. Pre-training: Means training on a large and diverse dataset
2. Fine-tuning: To get a more specific-answer compared to a generic-answer.

* __Token__: you can think of one-token as equal to one-word. There is a bit of detailing here ...discussed later.

* __Task__: Word-completion, which means you are given a set-of-words, let's say you are given this sentence the "lion is in the ---" so this next word is not known and ChatGPT then predicts or llms then predicts this word as let's say "forest"

* __Observation__: If you train the LLM for this simple-task it turns out that it can do a wide range of other-tasks as well. For example, picking the right answer MCQs steadily increases as the underlying language model improves. Other examples, language translation, answering MCQs, summarizing a text, sentiment detection, etc. 

***

* Why fine-tuning stage is needed?
* Answer: Generic versus Specific Answer
* The main purposes of of fine-tuning are it is basically a refinement on the pre-training on a much narrower data set.
* finetuning is needed if you want to build an application which is specific to a particular task or a particular domain.

* Example Airline ChatBot: User: What's the price for the PIA Airline,  which leaves at 600 p.m.?
* Example Bank ChatBot: JP Morgan Chase...
* Example Legal Attorneys: AI-legal-tool for attorneys ...

* While foundational models ((also called pre-train models)) were strong at reasoning they lacked the extensive knowledge of legal case history and other knowledge required for legal work.
*  __computational cost__ 
* fine-tune the pre-trend llm on the label dataset that's the three steps so the first step

***

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



