## A closer look at Generative Pre-trained Transformer (GPT) 

* we are going to see how the versions of GPT evolved?
*  how the different papers on GPT evolved?
*   what's the progression from Transformers to
*   GPT to gpt2 to gpt3 and finally to GPT 4?
*    where we are at right now?
*     Transformers we looked at a simplified architecture for Transformer and we also saw the difference between BERT and GPT model.
* we saw what's the meaning of an encoder what's the meaning of a decoder Etc?
### zero-shot versus few-short learning
* Transformers, GPT, GPT-2, GPT-3 and GPT-4.
*  Transformers to GPT to gpt2 and and then finally to gpt3 and then GPT 4.
*  [2017 - Attention is all you need](https://arxiv.org/abs/1706.03762)
    *  Introduced the __self-attention__ mechanism, where you capture the long-range dependencies in a sentence.
    *   A significant advancement compared to RNN and LSTM Networks.
    *   Transformers: (encoder, decoder)
    *   GPT:  Only decoder. 
* [2018 - Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    * __Unsupervised learning__, basically what they basically says that NLP as such up till this point had been mostly supervised-learning.
    * Label data for learning is scarse
    * generative pre-training of a language model on a diverse Corpus of unlabeled text.
    *  __generative pre-training__ and __unlabeled__ text data.
    *  pre-training: what is done here is that the text, which is used is not labeled. Let's say you have given a sentence right and you use that sentence itself as the training data and the next word prediction, which is in that sentence itself as the testing data. So, everything is self-content and you don't need to provide labels.
    *  just predict the next word in an unsupervised learning manner.
    * Why generative because we are generating the next word.
    * [OpenAI Blog]() 
    * Our approach is a combination of two ideas Transformers and __unsupervised pre-training__. it it had not entered the commercial domain then what happened is that in 2019 just the next year came one more paper which is called as
### [language models are unsupervised multitask learner]() 
so what they basically did is they just
6:03
took more amount of data than was used in the earlier paper and they also used a generative pre-train Network and you
6:11
can see that they actually showed four types of generative pre-train networks they showed a smaller model a slightly
6:17
larger model and the largest model which they used had 1542 which is 1,000 or 1
6:23
billion parameters almost uh so here you can see I have just shown a pictorial representation
6:30
here this was the gp2 gpt2 architecture which was introduced in this paper gpt2
6:37
generative pre-train Transformer 2 and here you can see that they released four things gpt2 small gpt2
6:45
medium gpt2 large and gpt2 extra large this was the first time when a paper was
6:51
published in which a large language model was so large in fact 1 billion
6:56
parameters were used in gpt2 extra large and it led to very very good results at
7:01
that time open AI was already working on more complex and more advanced GPT models but when this paper was released
7:09
in fact even this has very good number of citations if you if you just go to Google Scholar and
7:15
search this right now you'll see that this has uh around more than 10,000 plus
7:20
citations so this was the gpt2 paper which had around uh the largest model in
7:26
gpt2 really had around 1,000 million or 1 billion parameters then in 2020 came the real
7:34
boss which was gpt3 uh gpt3 had 175 billion parameters
7:41
let me show you where they actually mention about yeah so they they also
7:47
released a number of versions of gpt3 small medium large extra large a version
7:52
with 2.7 billion parameter a version with 6.7 billion parameter but there was one specific version which was released
7:59
which had 175 billion parameters which was gpt3 and when people started
8:05
exploring this model they could really see that it's amazing it could do so many things although it was just trained
8:11
to predict the next word it can do number of other things like translation sentiment analysis answering questions
8:19
uh answering multiple choice questions emotional recognition it can do so many
8:25
things and this was a huge model 175 billion parameters people had not seen
8:30
language models of this size then two years after this came GPT 3.5 which
8:36
became commercially viral everyone started using it and saw how good it was
8:41
and right now I'm using chat GPT 4 so if you uh see here I'm using chat GPT 40 so
8:49
GPT 4 is where we are right now but you just see this gradual transformation
8:55
which has happened from 2017 to 2024 in a space of 7 years we have gone from
9:00
this original Transformers paper we have gone then to the GPT paper in 2018 2019
9:07
came gpt2 this 2019 came gpt2 then in 2020
9:14
came gpt3 which really changed everything then came GPT 3.5 and then finally we are at GPT 4 this is the
9:21
whole uh transformation from Transformers to GPT gpt2 gpt3 GPT 3.5
9:28
and then G G pt4 many people don't know the difference between Transformers and GPT GPT essentially borrows from the
9:36
Transformer architecture but it's a bit different in that it does not have really the encoder
9:41
block so I just wanted to start off this lecture by giving you this historical perspective of how the generative
9:48
pre-train Transformer has evolved the next thing which I want to cover today is the difference between zero shot and
Zero Shot vs Few Shot learning
9:55
few shot learning zero shot is basically the ability to generalize to completely
10:02
unseen tasks without any prior specific examples and few shot is basically
10:08
learning from a minimum number of examples which the user provides as
10:13
input good so let me actually directly go to the gpt3 paper G this was the gpt3
10:19
paper and they have wonderful illustrations of zero shot and few short learning usually people think research
10:26
papers are hard to read but they have some very nice examp examples which really clarify the concept so in zero
10:32
short learning the model predicts the answer given only a description no other
10:39
Assistance or no other support for example The Prompt can be that hey you have to translate English to French and
10:46
take the word cheese and translate it into French if the model is able to do that that's an example of a zero shot
10:53
learning because we have not provided any supporting examples to the model
10:58
great then there is also one shot learning which means that the model Sees In addition to the task description the
11:05
model also sees a single example of the task so for example look at this I tell
11:11
the model that look C otter translates like this to French use this as a supporting guide or
11:18
like a hint if you may and translate cheese into French so this is one shot learning
11:25
where the model sees a single example of the task and then is few short learning
11:31
where the model basically sees a few examples of this task so for example in
11:37
few short learning uh we say that sea otter translates to this peppermint
11:42
translates to this and giraffe translates to this use these as the supporting examples and then translate
11:49
English to French and then translate GES so this is called as few short learning so I hope
11:58
you understood the difference between zero shot one shot and few shot zero shot is basically you provide no
12:03
supporting examples to the model you just tell it to do that particular task such as language translation and it does
12:10
it for you in one shot the model sees a single example of the task and in few shot the model sees a
12:18
few examples of this task these beautiful examples are provided right in
12:23
the GPT paper itself so you no no need to look anywhere further so let's see what's the claim of
12:30
these authors so what they were saying was that uh we train gpt3 and auto
12:37
regressive language model we'll see in a moment what this means with 175 billion
12:42
parameters 10 times more than any previous language model and test its
12:47
performance in a few short setting so let's see what the results
12:53
are gpt3 provides or achieves a strong performance perance on translation
13:01
question answering as well as several tasks that require on the-fly reasoning
13:06
or domain adaptation such as unscrambling words using a novel word in a sentence or performing three-digit
13:14
arithmetic so this paper basically implied that GPT 3 was a few short
13:20
learner which means that if it's given certain examples it can do that task
13:25
very well although it is trained only for the next word prediction what this paper
13:32
claimed was that gpt3 is a few short learner which means that if you wanted to do a language translation task you
13:39
just need to give it a few examples let's say if you want gpt3 to do a language translation task you just give
13:45
it few other examples of how that example is translated into another language and then gpt3 will do that task
13:51
for you so they claimed that gpt3 is a few short learner you will encounter zero shot
13:58
versus few shot at a number of different in a number of different books articles and blogs so I just wanted to make sure
14:05
it's clear for you so then your question would be okay if gpt3 is a few short learner what about GPT 4 which I'm using
14:12
right now is it a zero short learner or is it a few short learner because it seems that I don't need to give it
14:18
examples right it it just does many things on its own so let me ask gp4
14:24
itself are you a zero short
14:29
learner or are you a few short
14:37
learner let's see the answer so gp4 says that I a few short learner this means I
14:44
can understand and perform tasks better with a few examples while I can handle
14:49
many tasks with without prior examples which is zero short learning providing
14:54
examples helps me generate more accurate responses so this is a very smart answer because gp4 is saying that it does
15:02
amazingly well at few shot learning which means that if you provide it with some examples it does a better job but
15:09
it can even do zero shot learning so this is very important for you all of you to know when you are interacting
15:15
with GPT right if you provide some examples of the output which you are looking at or how you want the output to
15:21
be gp4 will do an amazing job of course it has zero shot capabilities also uh
15:28
but the two short capabilities are much more than zero short capabilities let me ask
15:33
it do you also have zero
15:40
short capabilities so when I ask this question
15:45
to gp4 it says that yes I also have zero shot capabilities this means that I can
15:50
perform tasks and answer questions without needing any prior examples or specific context so this is very
15:58
important I would say GPT 4 is both a zero shot learner as well as a few shot
16:03
learner but to get a more accurate responses as gp4 says itself you need to
16:09
probably provide it few examples to get better responses so in that sense it is
16:14
a better few short learner even when the authors release this paper they say confidently that
16:22
gpt3 is a few short learner but it can also do zero short
16:27
learning it just that its responses may not be as accurate awesome so this is really the
16:35
difference between zero shot learning and fot learning and you need to keep this in mind because when you think
16:42
about large language models this distinction is generally very very important I think when we go to GPT 5 or
16:48
GPT 6 even we might get really better at zero shot learning we are already there but it can be
16:55
better so here I've just shown a few examples of zero short versus few short learning so that this concept becomes
17:02
even more clear so let's say the input is translate English to French for zero
17:08
short learning we will just give the input which you want to translate like
17:13
breakfast and then it's translated here for few short learning as I already showed to you before we give it few
17:20
examples like uh let's say here we want to unscramble the words or make the
17:26
words into correct spelling let's say that's the task to do this task we can we we can give some example so let's say
17:33
this is a wrong spelling and the correct spelling is goat this is a wrong spelling the current correct spelling
17:38
isue so we give GPT or the llm these two examples and then we tell it that based
17:44
on these two examples now translate this into a correct word and then it translates it correctly as F so this is
17:51
an example of few short learning because as you see we do provide two supporting
17:56
examples so that the llm can actually make a better
18:03
translation okay so zero shot learning is basically completing task without any
18:09
example and uh few short learning is completing the task with a few
18:16
examples okay now let's go to the next section which is utilizing large data
Datasets for GPT pre-training
18:22
sets uh we also looked at this in the previous lecture but I just want to reiterate this based on this paper so
18:28
let's look at the data set which gpt3 have used we already saw that they the model was 175 billion parameters right
18:35
like see this but let's see the data on which the model is trained on let's look
18:40
at this data so the data set is the common craw data let's go
18:47
to uh internet and search common craw so you can see that this is the common crawl data set and let me click on this
18:55
right now and it maintains basically a free open repository of web scrw data that can be used by anyone it has over
19:02
250 billion Pages spanning 17 years and it is free and open Corpus since 2007
19:07
great so gpt3 uses 410 billion tokens from the common craw data and this is
19:14
the majority of the data it consists of 60% of the entire data set what is one
19:20
token so one token can be basically you can think of it as a Subs subset of the
19:26
data set so for the sake of of this lecture just assume one token is equal to one word there are more complex
19:33
procedures for converting sentences into tokens that's called tokenization but
19:38
just assume for now just to develop your intuition that one token is equal to one word so that will give you an idea of
19:45
this quantity so there are 410 billion Words which have been used as a data set
19:51
from common crawl then gpt3 also utilize data from web text to let's go and
19:57
search a bit about web text 2 so this is also an enhanced version of the original
20:02
web Text corpus so this covers all the Reddit submissions from 2015 to
20:08
2020 and I think the minimum number of apports here should be three or something but basically open web text
20:15
Data consists of a huge amount of Reddit posts and how huge so basically gpt3
20:21
uses 19 billion words from this web text2 data set which constitutes 22% of
20:26
the data the remaining I would say 18 to 19% of
20:32
the data comes from books and it comes from Wikipedia so this is the whole data
20:37
set on which gpt3 is trained on the total number of tokens on which it is trained on is 310 300 billion tokens
20:45
although if you add up these tokens they are more than 300 billion but maybe gpt3 took a different mix of uh the data set
20:53
but overall they took 300 billion words as the training data uh to generate the
20:59
gpt3 model think of that for a while 300 billion
21:05
tokens that's huge number of tokens and that's huge amount of data and it would
21:10
need a huge amount of compute power and also cost so that's an important point to
21:15
remember training the gpt3 is not easy pre-training rather I should call it
21:21
pre-training is not easy you need a huge amount of computer power and you need huge amount of data
21:28
so as I mentioned to you before a token is uh a unit of a text which the model
21:34
reads this is a good definition to think of token is a unit of a text which the
21:40
model reads for now you can just think of one token is equal to one word we'll cover this in the tokenization part in
21:47
the subsequent lectures we already looked at the gpt3 main paper which is titled language
21:53
models are few short Learners and this paper came out in the year 2020
21:59
one more key thing to keep in mind is that the total pre-training cost for gpt3 is $4.6 million keep this in mind
22:07
this is extremely important you have a huge amount of data set right and you need compute power to run your model on
22:14
the data set you need access to gpus and that's expensive but imagine this cost
22:19
the total pre-training cost for gpt3 is 4.6 million now you must be thinking what
22:26
exactly happens in this pre-training why does it cost this much what exactly are we training here so let me go that into
22:33
a bit more detail for that but before that realize that the pre-trained models
22:40
are also called as the base models or foundational models which can be used later for fine tuning so when you look
22:46
at the generative uh pre-training paper they also mention about fine tuning so
22:52
they say that we do pre-training first and then F tuning this means that let's say if you are a banking company or an
22:58
airline company or an educational company which wants to use gpt3 but you
23:03
also want the output to be specific to your data set then you need to fine-tune
23:08
the model further on label data which belongs to your organization that
23:13
process is called fine tuning that needs less amount of data than the amount of
23:19
data needed in pre-training but fine tuning is very important as you go into production level settings so if you are
23:26
an educational company who is building multiple choice questions let's say you can of course use gpt3 or gp4 but if you
23:33
want more robust more reliable outcomes you need to fine tune it on a label data
23:38
set which maybe your company has collected for the past 5 to 10 years okay now remember that many
23:45
pre-rain llms are available as open-source models uh and can be used as
23:51
general purpose tools to write extract and edit text which was not part of the training data so even gpt3 and G gp4 you
23:59
can uh you can use it yourself as a student gpt3 and GPT 4 can be used and
24:06
it's good you don't need fine tuning for this purpose let's say if you want to get some information or if you want your
24:12
PDF to be analyzed you can use gpt3 and gp4 as well there is one distinction
24:18
which I want to point out and that's between open source and closed Source language models so let me show you that
24:25
so look at the year on the XX so 2022 uh on the this red curve is the
24:32
closed Source model so gp4 is a closed Source model which means that the parameters and the weights really are
24:38
not known too much you can just use the end end output like this interface which
24:43
I have right now but there are many open source models which were releasing during that time it's just that their
24:50
performance was not as good as the closed Source model such as GPT so on the y- axis you have MML you can think
24:57
of it as a performance right right now so you can see the green line which is open source performance was much lesser
25:03
and now as we are actually entering August the performance of the open
25:08
source models in August 24 is actually comparable to closed Source models so
25:13
Lama 3.1 was released recently and uh the Lama 3.1 llm is an open source model
25:21
but it's one of the most powerful open source models which was released by meta
25:27
it has 400 5 billion parameters and its performance is a bit better I think than gp4 so the gap between open source and
25:35
close source llm is closing that being said for students you can still continue
25:40
interacting with gp4 and gpt3 that serves very well for you you don't need to think about fine tuning an llm or
25:47
accessing its weights or parameters great now uh I want to talk a
25:53
bit about uh so the GPT for architecture so up
25:59
till now we have looked at the total pre-training cost for gpt3 then we saw pre-training versus fine tuning and then
26:07
we saw the open source versus closed Source ecosystem now let's come to the
26:12
point number three which is GPT architecture we have already seen this in the previous lecture the GPT
26:18
architecture essentially consists of only a decoder block so let me show you this to uh refresh your understanding so
26:25
this is how the GPT architecture looks like it only consists of the decoder block as I've have shown here um whereas
26:33
the original Transformer consists of encoder as well as decoder so here the gpt3 is a scaled up
26:42
version of the original Transformer model which was implemented on a larger data set okay so gpt3 is also a scaled
26:50
up version of the model which was implemented in the 2018 paper so after the Transformers
26:57
paper there was this paper as I showed which introduced generative pre-training gpt3 is a scaled up version of this
27:05
paper as I already mentioned it has around uh uh 175 billion parameters so I
27:13
think we are aware of this so we can move to the next point now comes the very important task of uh Auto
Next word prediction
27:20
regression why is GPT models called as Auto regressive models and why do they come under the category of unsupervised
27:27
learning so let's look at that a bit further the main part which I want to mention here is that GPT models are
27:34
simply trained on next word prediction tasks which means that let's say you have the lion RS in the and you want to
27:41
predict the next word then you predict that it is going to be jungle this is all what GPT models are trained for so
27:49
let me show you how the training process actually looks like okay so uh I think this plot sums
27:56
it up better so let let's say you give multiple examples right so the first
28:01
input which you give to GPT is second law of Robotics colon this is the input
28:07
based on this input it has to predict the next word that's the output and then it predicts a in the next round the
28:14
input will be second law of Thermodynamics colon U so see the output of the previous label is now the input
28:21
here and then again it has to predict the next word so then it must be robot and then the input is second law of
28:27
robotics colon a robot that's the input and the output it has to predict the next World and then it predicts must
28:34
then the next input will be second law of Robotics colon robot must basically you see what we are doing here at every
28:42
stage of this training process the sentence is broken up into input and output the input consists of let's say
28:49
three words or four words but the output is always one word now you see we don't
28:54
give any labels here what happens is that the sentence itself breaks it break is broken into two parts the first half
29:01
and the second half the second half is where we have to predict the next world that's why it is called as
29:07
unsupervised learning because we do not give labels the sentence itself consists of the labels because the label is the
29:14
next word uh okay so uh GPT is trained on
29:21
next World predictions one awesome thing is that although it is only trained on predicting the next World they can still
29:28
do a wide range of other tasks like translation spelling correction Etc but
29:34
for the training itself GPT architectures are only trained to predict the next word in the sequence
29:40
and this takes a huge amount of compute power the 4.6 million dollars which I showed you is needed for this because
29:47
imagine you have those amounts of data which I showed you before let me show you them again so let's say if you have
29:54
these these many amounts of words 410 billion words so there might be around 40 billion sentences right and each of
30:01
the sentence will need to be broken up let's say the sentence is uh of 10 words
30:07
okay each of the sentences will then needed to be broken up and then into input pair and the
30:13
output uh and then in the output you have to predict the next word it takes a huge amount of time right because you
30:19
will need to do this for the billions of sentences in the data that's why it takes a huge amount of compute power for
30:25
this training procedure and uh how is it trained so basically initially it will predict wrong outputs
30:32
but then we have the correct output U which we know what should be the next word from the sentence itself so then
30:38
the error will be computed based on the predicted output and the difference between the corrected output and then
30:44
similar to The Back propagation done in neural networks the weights of the Transformer or the GPT architecture will
30:51
adapt so that the next word is predicted correctly so please keep in mind that
30:58
that is why it is an example of self-supervised learning uh because let's say you have a
31:03
sentence right what is done is that in the sentence itself we are divided we are dividing it into
31:10
training and we are dividing it into testing so this is the true we know the
31:15
next word this is the next word and we know its true value what we'll do is that using this as the input we'll try
31:22
to predict we'll try to predict the next word so then we'll have have something
31:28
which is called as the predicted word and then we'll train the neural
31:35
network or train the GPT architecture to minimize the difference between these two and update the
31:40
weights so these four these 175 billion parameters which you see over here are
31:46
just the weights of the neural network which we are training to predict the next word so that's why it's called as
31:52
unsupervised because the label for the next word we we do not have to externally label the data set it already
31:58
is labeled in a way because we know the true value of the next word so uh to put it in another words we
32:05
don't collect labels for the training data but use the structure of the data itself to make the labels so next word
32:12
in a sentence is used as the label and that's why it is called as the auto regressive model why is it called
32:19
Auto regressive there is one more reason for this the prev previous output is used as the input for the future
32:25
prediction so let's say let me go over this part again the previous output is
32:31
used as the input so let's say the first sentence is second law of Robotics the output of this is U right this U becomes
32:37
an input to the next sentence so now the input is second law of Robotics colon U
32:43
and then the next word prediction is robot then this robot becomes an input to the next sentence that's why the
32:49
model is also called Auto regressive model so two things are very important for you to remember here the first thing
32:56
is that GPT models are the pre-training part rather I would I
33:01
should say the pre-training part of GPT models is
33:09
unsupervised why is it unsupervised because we use the structure of the data itself to create the labels the next
33:15
word in the sentence is used as the label and the second thing which is very important is that these are
33:22
Auto these are Auto regressive models
33:27
which means that the previous outputs are used as the inputs for future predictions like I showed you over here
33:35
so it is very important to note these key things when you pre-train the GPT so
33:40
in pre-training you predict the next word you break you use the structure of the sentence itself to have training
33:46
data and labels and then you do the training you train the neural network uh which is the GPT
33:53
architecture and then you optimize the parameters the 175 billion parameters
33:59
now can you think why it takes so much compute time for pre-training because 175 billion parameters have to be
34:05
optimized so that the next word in all sentences is predicted
34:12
correctly okay now as I have mentioned to you before uh compared to the original Transformer architecture the
34:19
GPT architecture is actually simpler the GPT architecture only has the decoder
34:25
block it does not have the encoder block block so again let me show you this just for reference the original Transformer
34:32
architecture looks like this if you see it has the encoder block as well as the decoder block
34:38
right but now if you see the GPT architecture here the input text is only
34:44
passed to the decoder see it does not have the encoder so in a sense the GPT
34:50
is a more simplified architecture that way uh but also the number of building
34:56
blocks used are huge in in the GPT there is no encoder but to give you an idea in
35:01
the original Transformer we had six encoder decoder blocks in the gpt3 architecture on the other hand we have
35:08
96 Transformer layers and 175 parameters keep this in mind we have
35:16
96 Transformer layers so if you see this if you think of this as one Transformer
35:23
layer if you see this as one Transformer layer this this there are 96 such
35:28
Transformer layers like this that's why there are 175 billion
35:34
parameters now I want to also show you the visual uh I want to show you more
35:39
visually how the next word prediction happens we already saw Here how we have input and the output but I've also
35:46
written it on a whiteboard so that you have a much more clearer idea so let's say so the way GPT works is that there
35:53
are different iterations there is iteration number one there is iteration number two and there is iteration number three let's zoom into iteration number
36:00
one to see what's going on so in iteration number one we first have only one word as the input this it it's it
36:08
goes through pre-processing which is converting it into token IDs then it goes to the decoder block then it goes
36:16
to the output layer and we predict the next word which is is so then now the output is this is so it predicted the
36:22
next word and now this entire this is now serves as an for the second
36:29
iteration so now we go into iteration two where this is is an input so is was
36:35
an output of the first iteration but now it is included in the input of the next iteration and then the same steps happen
36:42
we do the token ID preprocessing then it goes through the decoder block and then we predict the next word this is
36:49
an that's the output from the iteration two the next word which is predicted is n and now this output from iteration two
36:58
is now uh going as the input to the iteration number three so the input to
37:03
the iteration number three is this is n you see why it is called an auto
37:09
regressive model the output from the previous iteration which is n is forming the input of the next iteration so the
37:16
next iteration is called is this is n and then again it goes through the same preprocessing steps and then there is an
37:22
output layer and then the final output is this is an example so so the next
37:28
word which has been predicted is example and then similarly these iterations will actually keep on
37:34
continuing this is how the GPT architecture Works in each iteration we predict the next word and then the
37:40
prediction actually informs the input of the next iteration so that's why it's unsupervised and auto
37:47
regressive so this schematic of the GPT architecture and as you can see that uh
37:52
it only has the decoder there is no encoder if you look at iteration 1 2 and three I did not mention an encoder block
37:59
here right because encoder block is not present in the GPT architecture
38:04
schematic only the decoder block is
38:10
present okay now one last thing which I want to cover in this lecture is
Emergent behaviour
38:15
something called as emergent Behavior so what is emergent Behavior
38:21
I've already touched upon this earlier in this lecture and in the previous lectures but remember that GPT is
38:28
trained only for the next word prediction right GP is trained only for the next word prediction but it's
38:35
actually quite awesome and amazing that even though GPT is only trained to predict the next World it can perform so
38:42
many other tasks such as language translation so let me go to gp4 right
38:47
now and uh convert
38:54
breakfast into French so gp4 is not trained to do language
39:02
translation tasks it is trained to predict the next world but while the training or while the pre-training
39:08
happens it also develops other advantages it develops other capabilities and this is called as
39:15
emergent Behavior due to these other capabilities although it was not trained to do the translation tasks uh gp4 can
39:22
also do the translation tasks which I have mentioned here see you can see one more thing which I want to show you is
39:29
this uh uh McQ generator so as such when GPT
39:36
was trained it was not trained to do McQ generation right but look at this if I want the GPT to provide me uh three to
39:44
four multiple choice questions uh so I just clicked on generate right
39:50
now and you'll see uh McQ questions have been generated on gravity now
39:56
technically uh GPT was not trained really to generate these questions on gravity but
40:04
it developed these properties or it developed these capabilities uh on its own while the
40:10
pre-training was happening to predict the next World and that's why this is called as emergent Behavior actually so
40:17
many awesome things can be done because of this emergent Behavior although GPT is just train to predict the next word
40:23
it can do it can answer text questions generate worksheet sheets summarize a text create lesson plan create report
40:30
cards generate a PP grade essays there are so many wonderful things which GPT can do and in fact this was also
40:37
mentioned in one of the blogs of open AI where they say that
40:44
uh uh we noticed so this this mentioned in their blog we noticed that we can use
40:49
the underlying language model to begin to perform tasks without ever training on them this is amazing right for
40:56
example ex Le performance on tasks like picking the right answer to a multiple choice question uh steadily increases as
41:03
the steadily increases as the underlying language model improves this is an clear
41:09
example of emergent Behavior Uh so basically the formal
41:16
definition of emergent behavior is the ability of a model to perform tasks that
41:22
the model wasn't explicitly trained to perform just keep this in mind and that
41:27
was very surprising to researchers also because it was only trained to do the next door tasks then how can it develop
41:34
these many capabilities and I think this Still Still Remains an open question that how come emergent behavior is
41:40
developed by chat GPT so let me actually go to Google Scholar and search about emergent behavior I'm sure there
41:48
are many papers on this so here you can see I searched emergent behavior and
41:54
there are all of these papers which came up uh this is an area of active research such
41:59
as exploring emergent behavior in llms and I'm sure there's a lot of scope for
42:05
making more contributions here so if any of you are considering looking for research topics emergent Behavior might
42:11
be a great topic to start your research on this actually brings us to the end of
42:16
this lecture we covered several things in today's lecture so let me do a quick recap of what all we have covered so
Recap of lecture
42:23
initially before looking at zero shot and few shot learning we started with the history we saw that the first paper
42:30
which was introduced in 2017 is attention is all you need it Incorporated the Transformer
42:35
architecture then came generative pre-training GPT the architecture is a
42:41
bit different than Transformer it uses only decoder no encoder and then after
42:47
uh generative pre-training was developed as a method it shows two main things first is that it's unsupervised second
42:54
it's Auto regressive and unlabel data which which means it does not need label data for
43:00
pre-training then came gpt2 one year later in 2019 and uh in fact there were
43:06
four models of gpt2 which were released by open AI the first one had 117 million
43:11
parameters the second had 345 the third had 762 and the fourth one had about a
43:17
billion parameters but then came the big beast in 2020 that was really
43:23
gpt3 and uh this paper said that language models are few short Learners
43:29
which means that if gpt3 is actually provided some amount of supplementary data it can do amazing few short tasks
43:37
and this model used 175 billion parameters which was the largest anyone had ever seen up till that
43:44
point after looking at this history we looked at the difference between zero shot and few shot learning in particular
43:51
we saw that in zero shot learning you don't need to provide any example the model can perform the task without
43:58
example and in few short learning you can give a few supplementary examples so
44:04
when this gpt3 paper was released the authors claimed that this this was a few short model they did not say zero short
44:11
Learner in the title because although it can do zero short learning uh it's just
44:17
much better at few short learning and we actually explored this ourselves we asked gp4 are you a zero short learner
44:23
or are you a few short learner and gp4 sent that I'm a few short learner it's
44:29
it also said that it can also do zero short learning but it it's just more accurate uh at few short
44:37
learning okay that's important to keep in mind then we saw that gpt3 utilizes a
44:42
huge huge amount of data uh it it uses around 300 billion tokens in total so
44:49
just writing it down 300 billion tokens in total which is about 300 billion words approximately a token is a unit of
44:56
text which the model reads it it's not usually just one word but for now you
45:01
can think of one token as one word and then we saw that training pre-training
45:07
gpt3 costs $4.6 million why does it cost this much because we have to predict the
45:12
next word in a sentence using this architecture so sentences are broken down into training data and testing data
45:20
it's Auto regressive so one word of the sentence is used for testing or the next word and the remaining is used for
45:26
training and this has to be done for all the sentences in the billion billions of data files which we have that's why it
45:33
takes 4.6 million to train because there are 175 billion parameters in
45:40
gpt3 remember to optimize the weights for those many it would need a huge
45:45
amount of computer power access to gpus Etc that's why training process is hard
45:50
so this schematic also shows the GPT architecture remember it only has the decoder it works in each it works in
45:58
iterations and the output of one iteration is fed as an input to the next iteration that makes it auto regressive
46:05
and in each iteration the sentence itself is used to make the label which is the next word prediction that's why
46:12
it's an unsupervised learning exercise pre-training we also saw that after pre-training there is usually one more
46:18
step which is fine-tuning which is basically training on a much narrower and specific data to improve the
46:24
performance usually needed in production level tasks we also briefly looked at
46:29
the gap between the open source and the closed Source llms really closing with the introduction of Lama 3.1 which
46:36
absolutely amazing performance and it somewhat beats gp4 it has 405 billion
46:44
parameters and towards the end the last concept which we learned about today is that of emergent Behavior so emergent
46:51
behavior is the ability of the model to perform tasks that the model wasn't explicitly trained to perform
46:57
so for example tasks such as McQ uh worksheet generator McQ generator
47:04
lesson plan generator proof reading essay grader translation it's just the model was just trained to do the next
47:10
word prediction right then how come it can do so many other awesome tasks that's called emergent behavior and it's
47:18
it's actually a topic of active research so if anyone is looking to do research
47:23
paper work on llms which I really encourage all of you emergent Behavior might be a great topic in the next
47:30
lecture we'll look at stages of building an llm and then we'll start coding directly from the data
47:37
pre-processing so thank you so much everyone for sticking with me until this point we have covered five lectures so
47:43
far and in all of them I have tried to make them as detailed as possible and as much as from Basics approach as possible
47:50
uh let me know in the YouTube comment section if you have any doubts or any questions thank you so much everyone and
47:56
I I look forward to seeing you in the next video





