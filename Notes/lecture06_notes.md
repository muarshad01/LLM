
* GPT-3 architecture in a lot of detail we also saw the progression from GPT to GPT-22 to GPT-33 and finally to GPT.4 
* Total pre-training cost for GPT-3 is around 4.6 million, which is insanely high.
* We have also looked at the dataset, which was used for pre-training GPT-3. 
* We learned about the differences between zero-shot versus few-shot learning.
* Attention behind self-attention prediction of next-word.
*  uh zero-short versus few-short learning.
*   basics of the Transformer architecture datasets used for llm pre pre-training
*   difference between pre-training and fine tuning.
*    stages of building a large language model.
* three stage stages stage-one stage-two and stage-three
*  so in stage one we are going to be looking at uh essentially building a large language model and we are going to look at the building blocks which are necessary so before we go to train the large language model.
*  we need to do the data pre-processing and sampling in a very specific manner
*   we need to understand the attention-mechanism and we will need to understand the llm architecture so in the stage-one, we are going to focus on these three things understanding how the data is collected from different datasets
*    how the data is processed
*     how the data is sampled
* number one then we will go to attention mechanism
* how to C out the attention mechanism completely from scratch in Python what is meant by key-query-value
* what is the attention-score
* what is positional encoding
* what is Vector embedding
* all of this will be covered in this stage we'll also be looking at the llm architecture such as
* how to stack different layers on top of each other where should the attention-head go
* so what exactly we will cover in data preparation and sampling
*  first we'll see tokenization if you are given sentences how to break them down into individual tokens
*   as we have seen earlier a token can be thought of as a unit of a sentence but there is a particular way of doing tokenization
*   we'll cover that then we will cover
*   Vector embedding essentially after we do tokenization every word needs to be transformed into a very high dimensional Vector space so that the semantic

***

* meaning between words is captured.
*  Encode every word so that the semantic meaning between the words are captured
*  so Words which mean similar things lie closer together
*  so we will learn about Vector embeddings
*  positional encoding the order in which the word appears in a sentence is also very important
*  pre-training model after learning about tokenization
*  Vector embedding we will learn about how to construct batches
*  meaning of context how many words should be taken for training to predict the next output we'll see about that and how to basically Fe the data in different sets of batches
*  so that the computation becomes much more efficient
*  so we'll be implementing a data batching sequence
*  before giving all of the data set into the large language model for pre-training
#### Second Point
* I mentioned here is the attention mechanism
* so here is the attention mechanism for the Transformer model we'll first understand what is meant by every single thing here what is meant by multi-ad attention
* what is meant by Mas multi head attention
* what is meant by positional encoding
* input embedding output embedding
* all of these things and then we will build our own llm architecture so uh these are the two things attention mechanism
*  outcome of stage two is to build a foundational model on unlabeled data
* we'll break it down into epox and we will compute
*  the gradient uh of the loss in each Epoch and we'll update the parameters towards the end
*   we'll generate sample text for visual inspection this is what will happen exactly in the training procedure of the large language model and then
*   we'll also do model evaluation and loading pre-train weaps
*    evaluation training
*    and validation losses
*    then we'll write the llm training function
*     Implement function to save-and-load the LLM weights
*   load pre-trained weights from open AI into our large language model.
*    stage-two which is essentially training Loop plus uh training Loop plus model evaluation plus loading pre-trained weights to build our foundational model.
*     so the main goal of stage two as I as I told you is pre-training and llm on unlabelled data great
* but we will not stop here after this we move to
*

#### stage-three 
* Fine-tuning the LLM so if we want to build specific applications we will do fine tuning in this playlist
* 

***

spam whereas hey just wanted to check if
we are still on for dinner tonight let
me know this will be not spam so we will
build a large language model this
application which classifies between
spam and no spam and we cannot just use
the pre-trained or foundational model
for this because we need to train with
labeled data to the pre-train model we
need to give some more data and tell it
that hey this is usually spam and this
is not spam can you use the foundational
model plus this additional specific
label data asset which I have given to
build a fine-tuned llm application for
email classification so this is what
we'll be building as the first
application the second application which
we'll be building is a type of a chat
bot which Bas basically answers queries
so there is an instruction there is an
input and there is an output and we'll
be building this chatbot after fine
tuning the large language model so if
you want to be a very serious llm
engineer all the stages are equally
important many students what they are
doing right now is that they just look
at stage number three and they either
use Lang chain let's
say they use Lang chain they use tools
like
AMA and they directly deploy
applications but they do not understand
what's going on in stage one and stage
two at all so this leaves you also a bit
underc confident and insecure about
whether I really know the nuts and bolts
whether I really know the details my
plan is to go over every single thing
without skipping even a single Concept
in stage one stage two and stage number
three so this is the plan which you'll
be following in this playlist and I hope
you are excited for this because at the
end of this really my vision for this
playlist is to make it the most detailed
llm playlist uh which many people can
refer not just students but working
professionals startup Founders managers
Etc and then you can once this playlist
is built over I think two to 3 months
later you can uh refer to whichever part
you are more interested in so people who
are following this in the early stages
of this journey it's awesome because
I'll reply to all the comments in the um
chat section and we'll build this
journey
together I want to end this a lecture by
providing a recap of what all we have
learned so far this is very uh this is
going to be very important because from
the next lecture we are going to start a
bit of the Hands-On
approach okay so number one large
language models have really transformed
uh the field of natural language
processing they have led to advancements
in generating understanding and
translating human language this is very
important uh so the field of NLP before
you needed to train a separate algorithm
for each specific task but large
language models are pretty generic if
you train an llm for predicting the next
word it turns out that it develops
emergent properties which means it's not
only good at predicting the next word
but also at things like uh multiple
choice questions text summarization then
emotion classification language
translation Etc it's useful for a wide
range of tasks and it's that has led to
its predominance as an amazing tool in a
variety of
fields secondly all modern large
language models are trained in two main
steps first we pre-train on an unlabeled
data this is called as a foundational
model and for this very large data sets
are needed typically billions of words
and it costs a lot as we saw training
pre-training gpt3 costs $4.6 million so
you need access to huge amount of data
compute power and money to pre-train
such a foundational model now if you are
actually going to implement an llm
application on production level so let's
say if you're an educational company
building multiple choice questions and
you think that the answers provided by
the pre-training or foundational model
are not very good and they are a bit
generic
you can provide your own specific data
set and you can label the data set
saying that these are the right answers
and I want you to further train on this
refined data set uh to build a bette
model this is called fine tuning usually
airline companies restaurants Banks
educational companies when they deploy
llms into production level they fine
tune the pre-trained llm nobody deploys
the pre-trend one directly you fine tune
the element llm on your specific smaller
label data set this is very important
see for pre-training the data set which
we have is unlabeled it's Auto
regressive so the sentence structure
itself is used for creating the labels
as we are just predicting the next world
but when we F tune we have a label data
set such as remember the spam versus no
spam example which I showed you that is
a label data set we give labels like hey
this is Spam this is not spam this is a
good answer this is not a good answer
and this finetuning step is generally
needed for Building Product ction ready
llm
applications important thing to remember
is that fine tuned llms can outperform
only pre-trained llms on specific tasks

***

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






