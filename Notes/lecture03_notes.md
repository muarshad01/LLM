## Stages of building LLMs
1. __Pre-training__: Means training on a large and diverse dataset
2. __Fine-tuning__: To get a more specific-answer compared to a generic-answer.

* __Token__: you can think of one-token as equal to one-word. There is a bit of detailing here ...discussed later.

* __Task__: Word-completion, which means you are given a set-of-words, let's say you are given this sentence the "lion is in the ---" so this next word is not known and ChatGPT then predicts or llms then predicts this word as let's say "forest"

* __Observation__: If you train the LLM for this simple-task it turns out that it can do a wide range of other-tasks as well. For example, picking the right answer MCQs steadily increases as the underlying language model improves. Other examples, language translation, answering MCQs, summarizing a text, sentiment detection, etc. 

***

#### Why fine-tuning stage is needed?
* Generic versus Specific Answer
* It is basically a refinement on the pre-training on a much narrower dataset.
* Finetuning is needed if you want to build an application, which is specific to a particular task or a particular domain.

#### Examples
* Airline ChatBot: ...
* Bank ChatBot: ...
* Legal Al-Tool for Attorneys: ...

#### Foundational models (also called pre-train models)
* Fine-tune the pre-trend LLM on the label dataset.

***

#### unlabeled data and label data for this

* For pre-training,  we usually don't label the dataset. It's an unsupervised learning task and it's also called Auto regression because if you are considering a sentence and predicting the next word the part of the sentence is used for training and the next word is used for forecasting or testing and then for the next set of training the earlier forecasted word is used as training it's fine if you don't understand this terminology right now but just remember that pre-training is done on unlabeled text Data whereas finetuning is mostly done on labelled text Data. 

* let's say if you are doing a classification task right or if you want a specific llm task which classifies email say spam or no spam you will need to provide labels for which emails were spam which were not spam Etc so generally for fine tuning we will have a labeled data set let's say Harvey you considering the case of Harvey which we s for legal data we had these case studies uh legal case studies right you will need to provide those case studies and mention that for these questions these were the answers in this particular case study Etc so in a sense that that becomes label data so please keep this pre-training plus fine tuning schematic in mind what many students do is that they do not understand the difference between pre-training and fine tuning and they also don't understand why is it called pre-training in the first place shouldn't it just be training ideally it should just be called as training but it's called pre-training because there is also this fine-tuning component so it's just a terminology thing so if you understand this then you will get a better Clarity of when you interact with GPT you are essentially interacting with a pre-trained or a foundational model okay so I hope you have understood the schematic and I hope you have understood the difference between pre-training and fine tuning now I want Steps for building an LLM to just show you the steps which are needed for building an L which basically summarizes what all we have covered in today's lecture so the first step is to train on a large Corpus of Text data which is also called raw text remember some of these terminologies keep on appearing everywhere they actually mean something very simple but they just sound a bit fancy so raw text is just basically training on a large Corpus of text Data a more formal definition of raw text is regular text without any labeling information so raw text means text which does not come with predefined labels so first you train on such such kind of a raw text and you need huge amount of raw text the second task is the first training stage of nlm and this is called as pre-training as we already saw this task involves creation of an initially pre-trained llm which is also called as the base or a foundational model gp4 for

example what we saw in the paper this paper over here I'll also be sharing the link to this paper in the information section uh this paper is a pre-train model and uh it is actually capable of text completion but it also turns out it's capable of many other things like uh as we saw over here sentiment analysis question answering Etc so pre-training also gives the llm a lot of other powers so that is Step number two step number three after obtaining the pre-trained llm the llm is further trained on label data and this is fine tuning so let me just mention over here this is called as fine tuning and we saw a number of applications of this in different areas we saw applications in the legal area then we saw applications in the Comm telecommunication sector this es Telecom Harvey we also saw an application related to JP Morgan Chase so fine tuning is essential as and when you go into production and if you want a really powerful model train specifically on your data set even within fine tuning there are usually two categories the first category it's called instruction fine tuning and second is called fine tuning for classification tasks so let's say you are an education company and if you want to uh if you want to or let's say if you're a company like u a text translation company which converts English into French so here you might need to give some sort of an instruction like this is the English language and you convert it into the French language so you might need to give instruction answer pairs as label data set so the label data set means showing that okay these are all the English language these are all the French language use this to further fine tune what you already know so that's called as instruction fine tuning even for Airline customer support


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






