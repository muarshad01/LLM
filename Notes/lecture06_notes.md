hello everyone welcome to this lecture in the building large language models from scratch series we have covered five lectures up till now and in the previous lecture we looked at the gpt3 architecture in a lot of detail we also saw the progression from GPT to gpt2 to gpt3 and finally to GPT 4 uh we saw that the total pre-training cost for gpt3 is around 4.6 million which is insanely high and up till now we have also looked at the data set which was used for pre-training gpt3 and we have seen this several times until now in the prev previous lecture we learned about the differences between zero shot versus few shot learning as well so if you have not been through the previous lectures we have already covered five lectures in this series and uh all of them have actually received a very good response from YouTube and I've received a number of comments saying that they have really helped people so I encourage you to go through those videos in today's lecture we are going to be discussing about what we will exactly cover in the playlist in these five lectures we have looked at some of the theory modules some of the intuition modules behind attention behind self attention prediction of next word uh zero short versus few short learning basics of the Transformer architecture data sets used for llm pre pre-training difference between pre-training and fine tuning Etc but now from the next lecture onwards we are going to start with the Hands-On aspects of actually building an llm so I wanted to utilize this particular lecture to give you a road map of what all we will be doing in this series and what all stages which we will be covering during this playlist so that is the title of today's lecture stages of building a large language model towards the end of this lecture we will also do a recap of what all we have learned until now so let's get started with today's lecture okay so we will break this playlist into three stage stages stage one stage two and stage three remember before we get started that this material which I showing is heavily borrowed from U the book building a large language model from scratch which is written by sebasian rashka so I'm very grateful to the author for writing this book which is allowing me to make this playlist okay so we'll be dividing the playlist into three stages stage one stage two and stage number three unfort fortunately all of the playlists currently which are available on YouTube only go through some of these stages and that two they do not cover these stages in detail my plan is to devote a number of lectures to each stage in this uh playlist so that you get a very detailed understanding of how the nuts and bolts really work so in stage one we are going to be looking at uh essentially building a large language model and we are going to look at the building blocks which are necessary so before we go to train the large language model we need to do the data pre-processing and sampling in a very specific manner we need to understand the attention mechanism and we will need to understand the llm architecture so in the stage one we are going to focus on these three things understanding how the data is collected from different data sets how the data is processed how the data is sampled number one then we will go to attention mechanism how to C out the attention mechanism completely from scratch in Python what is meant by key query value what is the attention score what is positional encoding what is Vector embedding all of this will be covered in this stage we'll also be looking at the llm architecture such as how to stack different layers on top of each other where should the attention head go all of these things essentially uh the main understanding or the main part of this stage will be to understand understand the basic mechanism behind the large language model so what exactly we will cover in data preparation and sampling first we'll see tokenization if you are given sentences how to break them down into individual tokens as we have seen earlier a token can be thought of as a unit of a sentence but there is a particular way of doing tokenization we'll cover that then we will cover Vector embedding essentially after we do tokenization every word needs to be transformed into a very high dimensional

***

Vector space so that the semantic
5:04
meaning between words is captured as you
5:07
can see here we want apple banana and
5:10
orange to be closer together which are
5:12
seen in this red circle over here we
5:14
want King man and woman to be closer
5:17
together which is shown in the blue
5:18
circle and we want Sports such as
5:20
football Golf and Tennis to be closer
5:22
together as shown in the green these are
5:25
just representative examples what I want
5:27
to explain is that before we give the
5:30
data set for training we need to encode
5:32
every word so that the semantic meaning
5:36
between the words are captured so Words
5:38
which mean similar things lie closer
5:40
together so we will learn about Vector
5:43
embeddings in a lot of detail here we'll
5:45
also learn about positional encoding the
5:47
order in which the word appears in a
5:49
sentence is also very important and we
5:52
need to give that information to the
5:54
pre-training
5:55
model after learning about tokenization
5:58
Vector embedding we will learn about how
6:01
to construct batches of the data so if
6:04
we have a huge amount of data set how to
6:06
give the data in batches to uh GPT or to
6:09
the large language model which we are
6:11
going to build so we will be looking at
6:14
the next word prediction task so you
6:16
will be given a bunch of words and then
6:18
predicting the next word so we'll also
6:20
see the meaning of context how many
6:22
words should be taken for training to
6:25
predict the next output we'll see about
6:27
that and how to basically Fe the data in
6:31
different sets of batches so that the
6:33
computation becomes much more efficient
6:36
so we'll be implementing a data batching
6:38
sequence before giving all of the data
6:41
set into the large language model for
6:44
pre-training after this the second Point
6:46
as I mentioned here is the attention
6:48
mechanism so here is the attention
6:50
mechanism for the Transformer model
6:52
we'll first understand what is meant by
6:54
every single thing here what is meant by
6:56
multi-ad attention what is meant by Mas
6:59
multi head attention what is meant by
7:01
positional encoding input embedding
7:03
output embedding all of these things and
7:05
then we will build our own llm
7:08
architecture so uh these are the two
7:11
things attention mechanism and llm
7:13
architecture after we cover all of these
7:15
aspects we are essentially ready with
7:17
stage one of this playlist and then we
7:20
can move to the stage two stage two of
7:23
this series is essentially going to be
7:25
pre-training which is after we have
7:27
assembled all the data after we have
7:29
constructed the large language model
7:31
architecture which we are going to use
7:33
we are going to write down a code which
7:35
trains the large language model on the
7:37
underlying data set that is also called
7:40
as pre-training so the outcome of stage
7:43
two is to build a foundational model on
7:45
unlabeled
7:47
data now uh I'll just show a schematic
7:50
from the book which we will be following
7:52
so this is how the training data set
7:53
will look like we'll break it down into
7:56
epox and we will compute the gradient
8:00
uh of the loss in each Epoch and we'll
8:02
update the parameters towards the end
8:04
we'll generate sample text for visual
8:06
inspection this is what will happen
8:08
exactly in the training procedure of the
8:11
large language model and then we'll also
8:13
do model evaluation and loading
8:15
pre-train weaps so let me show you the
8:17
schematic for that so we'll do text
8:19
generation evaluation training and
8:21
validation losses then we'll write the
8:24
llm training function which I showed you
8:26
uh and then we'll do one more thing we
8:28
will Implement function to save and lo
8:30
load the large language model weights to
8:33
use or continue training the llm later
8:35
so there is no point in training the LM
8:38
from scratch every single time right
8:39
weight saving and loading essentially
8:41
saves you a ton of computational cost
8:43
and
8:44
memory and then at the end of this we'll
8:47
also load pre-trained weights from open
8:49
AI into our large language model so open
8:52
AI has already made some of the weights
8:54
available they are pre-trained weights
8:56
so we'll be loading uh pre-trained
8:58
weights from open a into our llm model
9:02
this is all what we'll be covering in
9:04
the stage two which is essentially
9:06
training Loop plus uh training Loop plus
9:09
model evaluation plus loading
9:10
pre-trained weights to build our
9:12
foundational model so the main goal of
9:15
stage two as I as I told you is
9:17
pre-training and llm on unlabelled data
9:20
great but we will not stop here after
9:22
this we move to stage number three and
9:25
the main goal of stage number three is
9:27
fine tuning the large language model so
9:29
if we want to build specific
9:31
applications we will do fine tuning in
9:33
this playlist we are going to build two
9:35
applications which are mentioned in the
9:37
book I showed you at the start one is
9:39
building a classifier and one is
9:41
building your own personal assistant so
9:44
here are some schematics to show so if
9:46
you want to let you have got a lot of
9:48
emails right and if you want to use your
9:50
llm to classify spam or no spam for
9:54
example you are a winner you have been
9:56
uh specially selected to receive th000
9:58
cash now this should be classified as
10:01
spam whereas hey just wanted to check if
10:03
we are still on for dinner tonight let
10:05
me know this will be not spam so we will
10:08
build a large language model this
10:10
application which classifies between
10:12
spam and no spam and we cannot just use
10:14
the pre-trained or foundational model
10:16
for this because we need to train with
10:17
labeled data to the pre-train model we
10:20
need to give some more data and tell it
10:22
that hey this is usually spam and this
10:24
is not spam can you use the foundational
10:26
model plus this additional specific
10:28
label data asset which I have given to
10:30
build a fine-tuned llm application for
10:34
email classification so this is what
10:36
we'll be building as the first
10:38
application the second application which
10:40
we'll be building is a type of a chat
10:42
bot which Bas basically answers queries
10:44
so there is an instruction there is an
10:46
input and there is an output and we'll
10:48
be building this chatbot after fine
10:51
tuning the large language model so if
10:54
you want to be a very serious llm
10:56
engineer all the stages are equally
10:58
important many students what they are
11:00
doing right now is that they just look
11:02
at stage number three and they either
11:04
use Lang chain let's
11:06
say they use Lang chain they use tools
11:09
like
11:10
AMA and they directly deploy
11:13
applications but they do not understand
11:15
what's going on in stage one and stage
11:17
two at all so this leaves you also a bit
11:19
underc confident and insecure about
11:21
whether I really know the nuts and bolts
11:23
whether I really know the details my
11:25
plan is to go over every single thing
11:27
without skipping even a single Concept
11:30
in stage one stage two and stage number
11:33
three so this is the plan which you'll
11:35
be following in this playlist and I hope
11:37
you are excited for this because at the
11:39
end of this really my vision for this
11:42
playlist is to make it the most detailed
11:44
llm playlist uh which many people can
11:46
refer not just students but working
11:48
professionals startup Founders managers
11:51
Etc and then you can once this playlist
11:53
is built over I think two to 3 months
11:56
later you can uh refer to whichever part
11:59
you are more interested in so people who
12:01
are following this in the early stages
12:03
of this journey it's awesome because
12:05
I'll reply to all the comments in the um
12:09
chat section and we'll build this
12:11
journey
12:13
together I want to end this a lecture by
12:16
providing a recap of what all we have
12:18
learned so far this is very uh this is
12:21
going to be very important because from
12:22
the next lecture we are going to start a
12:24
bit of the Hands-On
12:26
approach okay so number one large
12:29
language models have really transformed
12:31
uh the field of natural language
12:34
processing they have led to advancements
12:36
in generating understanding and
12:38
translating human language this is very
12:40
important uh so the field of NLP before
12:43
you needed to train a separate algorithm
12:45
for each specific task but large
12:47
language models are pretty generic if
12:49
you train an llm for predicting the next
12:51
word it turns out that it develops
12:53
emergent properties which means it's not
12:55
only good at predicting the next word
12:57
but also at things like uh multiple
13:00
choice questions text summarization then
13:03
emotion classification language
13:05
translation Etc it's useful for a wide
13:07
range of tasks and it's that has led to
13:10
its predominance as an amazing tool in a
13:13
variety of
13:15
fields secondly all modern large
13:18
language models are trained in two main
13:20
steps first we pre-train on an unlabeled
13:23
data this is called as a foundational
13:25
model and for this very large data sets
13:28
are needed typically billions of words
13:31
and it costs a lot as we saw training
13:33
pre-training gpt3 costs $4.6 million so
13:37
you need access to huge amount of data
13:39
compute power and money to pre-train
13:42
such a foundational model now if you are
13:45
actually going to implement an llm
13:47
application on production level so let's
13:49
say if you're an educational company
13:51
building multiple choice questions and
13:53
you think that the answers provided by
13:55
the pre-training or foundational model
13:57
are not very good and they are a bit
13:58
generic
13:59
you can provide your own specific data
14:02
set and you can label the data set
14:04
saying that these are the right answers
14:06
and I want you to further train on this
14:07
refined data set uh to build a better
14:10
model this is called fine tuning usually
14:14
airline companies restaurants Banks
14:16
educational companies when they deploy
14:19
llms into production level they fine
14:21
tune the pre-trained llm nobody deploys
14:23
the pre-trend one directly you fine tune
14:26
the element llm on your specific smaller
14:29
label data set this is very important
14:31
see for pre-training the data set which
14:33
we have is unlabeled it's Auto
14:35
regressive so the sentence structure
14:37
itself is used for creating the labels
14:39
as we are just predicting the next world
14:42
but when we F tune we have a label data
14:44
set such as remember the spam versus no
14:47
spam example which I showed you that is
14:49
a label data set we give labels like hey
14:51
this is Spam this is not spam this is a
14:53
good answer this is not a good answer
14:55
and this finetuning step is generally
14:57
needed for Building Product ction ready
14:59
llm
15:01
applications important thing to remember
15:03
is that fine tuned llms can outperform
15:06
only pre-trained llms on specific tasks
15:09
so let's say you take two cases right in
15:11
one case you only have pre-trained llms
15:13
and in second case you have pre-trained
15:15
plus fine tuned llms so it turns out
15:18
that pre-trained plus finetune does a
15:20
much better job at certain specific
15:22
tasks than just using pre-rain for
15:24
students who just want to interact for
15:26
getting their doubts solved or for
15:29
getting assistance uh in summarization
15:32
uh helping in writing a research paper
15:34
Etc gp4 perplexity or such API tools or
15:39
such interfaces which are available work
15:41
perfectly fine but if you want to build
15:43
a specific application on your data set
15:46
and take it to production level you
15:48
definitely need fine
15:50
tuning okay now uh one more key thing is
15:54
that the secret Source behind large
15:55
language models is this Transformer
15:57
architecture
15:59
so uh the key idea behind Transformer
16:02
architecture is the attention mechanism
16:05
uh just to show you how the Transformer
16:07
architecture looks like it looks like
16:08
this and the main thing behind the
16:10
Transformer architecture which really
16:12
makes it so
16:14
powerful are these attention
16:17
blocks we'll see what they mean so no
16:19
need to worry about this right
16:21
now but in the nutshell attention
16:24
mechanism gives the llm selective access
16:26
to the whole input sequence when
16:28
generating output one word at a time
16:31
basically attention mechanism allows the
16:33
llm to understand the importance of
16:36
words and not just the word in the
16:39
current sentence but in the previous
16:41
sentences which have come long before
16:42
also because context is important in
16:45
predicting the next word the current
16:47
sentence is not the only one which
16:48
matters attention mechanism allows the
16:51
llm to give access to the entire context
16:53
and select or give weightage to which
16:55
words are important in predicting the
16:57
next word this is a key idea which and
17:00
we'll spend a lot of time on this
17:02
idea remember that the original
17:04
Transformer had only the had encoder
17:07
plus decoder so it had both of these
17:10
things it had the encoder as well as it
17:11
had the decoder but generative pre-train
17:15
Transformer only has the decoder it did
17:17
not it does not have the encoder so
17:20
Transformer and GPT is not the same
17:22
Transformer paper came in 2017 it had
17:24
encoder plus decoder generative pre-rain
17:27
Transformer came one year later
17:29
2018 and that only had the decoder
17:32
architecture so even gp4 right now it
17:34
only has decoder no encoder so 2018 came
17:38
GPT the first generative pre-trend
17:40
Transformer architecture 2019 came gpt2
17:43
2020 came gpt3 which had 175 billion
17:47
parameters and that really changed the
17:49
game because no one had seen a model
17:51
this large before and then now we are at
17:53
GPT 4
17:55
stage one last point which is very
17:57
important is that llms are only trained
18:00
for predicting the next word right but
18:02
very surprisingly they develop emergent
18:04
properties which means that although
18:07
they are only trained to predict the
18:08
next word they show some amazing
18:11
properties like ability to classify text
18:14
translate text from one language into
18:16
another language and even summarize
18:17
texts so they were not trained for these
18:20
tasks but they developed these
18:22
properties and that was an awesome thing
18:23
to realize the pre-training stage works
18:26
so well that llms develop all of these
18:28
wonderful other properties which makes
18:30
them so impactful for a wide range of
18:33
tasks
18:35
currently okay so this brings us to the
18:37
end of the recap which we have covered
18:39
up till now if you have not seen the
18:41
previous lectures I really encourage you
18:43
to go through them because these
18:45
lectures have really set the stage for
18:46
us to now dive into stage one so from
18:49
the next lecture we'll start going into
18:51
stage one and we'll start seeing the
18:53
first aspect which is data preparation
18:55
and sampling so the next lecture title
18:58
will be be working with Text data and
19:00
we'll be looking at the data sets how to
19:03
load a data set how to count the number
19:05
of characters uh how to break the data
19:07
into tokens and I'll I'll start sharing
19:10
sharing Jupiter notebooks from next time
19:12
onward so that we can parall begin
19:15
coding so thanks everyone I hope you are
19:17
liking these lectures so lecture 1 to
19:20
six we kind of like an introductory
19:23
lecture to give you a feel of the entire
19:24
series and so that you understand
19:26
Concepts at a fundamental level from
19:28
from lecture 7 we'll be diving deep into
19:30
code and we'll be starting into stage
19:33
one so I follow this approach of writing
19:36
on a whiteboard and also
19:38
coding um so that you understand the
19:40
details plus the code at the same time
19:43
because I believe Theory plus practical
19:44
implementation both are important and
19:47
that is one of the philosophies of this
19:49
lecture Series so do let me know in the
19:51
comments how you finding this teaching
19:53
style uh because I will take feedback
19:56
from that and we can build this series
19:58
together 3 to four months later this can
20:00
be an amazing and awesome series and I
20:03
will rely on your feedback to build this
20:05
thanks a lot everyone and I look forward
20:07
to seeing you in the next lecture

***

