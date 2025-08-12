## A closer look at Generative Pre-trained Transformer (GPT) 
* Transfomer: (encoder, decoder)
* BERT: (encoder, ---)
* GPT: (---, decoder)

| Model | Parameters | Paper Link| 
|---           |---    | ---|
| Transformer (2017)  |  |       | [Transformer (2017): Attention is all you need](https://arxiv.org/abs/1706.03762) |
| GPT (2018)   |         |[GPT (2018): Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)|
| GPT-2 (2019) | 1.5 B |  [GPT-2 (2019): Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)|
| GPT-3 (2020) | 175 B | | [GPT-3 (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)|
| GPT-3.5      |       |  |
| GPT-4        |       |  |

#### [Transformer (2017) : Attention is all you need](https://arxiv.org/abs/1706.03762)
*  __Self-attention__ mechanism, where you capture the __long-range dependencies__ in a sentence.
*   A significant advancement compared to RNN (1980) and LSTM Networks (1997).

#### [GPT (2018): Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* __Generative Pre-training__ on a divrse corpus of __unlabeled__ text data.
* __Generative__ because we are generating-/predicting-the-next-word in an __unsupervised-learning__ manner.
* __Pre-training__: Text data, which is used here is not labeled. Let's say you have given a sentence right and you use that sentence itself as the training data and the next-word prediction, which is in that sentence itself as the testing data. So, everything is self-content and you don't need to provide labels.

#### [OpenAI Blog](https://openai.com/index/language-unsupervised/)
* GPT is combination of two ideas: (Transformers, unsupervised pre-training)

#### [GPT-2 (2019): Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* translation
* sentiment analysis
* answering questions
* answering MCQs
* emotional recognition, etc.

#### [GPT-3 (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

***

## Zero-Shot vs Few-Shot Learning
#### Few shot
* Ability to generalize-to-completely-unseen-tasks without any prior specific examples.
* Learning from a minimum number of examples, which the user provides as input.

*  __zero-short__ learning the model-predicts-the-answer given only a description no other Assistance or no other support.
  *  Example: The prompt can be that hey you have to translate English-to-French and take the word "cheese" and translate it into French.

* __one-shot__ learning, which means, in addition to the task description the model also sees a single-example of the task.
* Example: look at this. I tell the model that look C otter translates like this to French use this as a supporting-guide or like-a-hint, if you may and translate "cheese" into French.

* __few-short__ learning, where the model basically sees a-few-examples-of-this task.

#### difference between zero-shot, one-shot, and few-shot.
* zero shot is basically you provide no supporting examples to the model you just tell it to do that particular task such as language translation and it does it for you.
*  in one-shot the model sees a single-example of the task
* few-shot the model sees a few-examples of this task these beautiful examples.

#### Auto Regressive language model
* paper basically implied that GPT 3 was a few-short learner,
* which means that if it's given certain examples it can do that task very well although it is trained only for the next word prediction.


* gpt3 is a few-short learner what about GPT-4,
* which I'm using right now is it a zero-short learner or is it a few-short learner because it seems that I don't need to give it examples right it it just does many things on its own so let me ask gp4 itself are you a zero-short learner or are you a few short learner.
*  gp4 says that I a few short learner.
*   this means I can understand and perform tasks better with a few examples while I can handle many tasks with without prior examples which is zero-short learning providing examples helps me generate more accurate responses .

* with GPT right if you provide some examples of the output which you are looking at or how you want the output to be gp4 will do an amazing job of course it has zero-shot capabilities also uh but the two short capabilities are much more than zero-short capabilities.
*  let me ask it do you also have zero-short capabilities so when I ask this question to gp4 it says that yes I also have zero-shot capabilities.
*    so zero-shot learning is basically completing task without any example and uh few-short learning is completing the task with a few examples okay.
*    let's go to the next section which is utilizing large data Datasets for GPT pre-training ...gpt3 uses 410 billion tokens from the common craw data and this is the majority of the data it consists of 60% of the entire data set what is one token so one token can be basically you can think of it as a Subs subset of the data set so for the sake of of this lecture just assume one token is equal to one word

***

* gpt3 model think of that for a while 300 billion tokens.
*  that's huge number of gpt3 is $4.6 million
*   base models or foundational models
*    open source and closed Source language models
*      Lama 3.1 was released recently and uh the
*  Lama 3.1 llm is an open source model but it's one of the most powerful open source models which was released by meta it has 400 5 billion parameters
*   gpt3 is a scaled-up version of the original Transformer model, which was implemented on a larger data set okay so gpt3 is also a scaled up version of the model which was implemented in the 2018 paper.
*    so after the Transformers paper there was this paper as I showed which introduced generative pre-training gpt3 is a scaled up version of this paper as I already mentioned it has around uh uh 175 billion parameters.
*     so I think we are aware of this so we can move to the next point now comes the very important task of uh Auto Next word prediction regression.
*  why is GPT models called as Auto regressive models.
*   and why do they come under the category of unsupervised learning.
*    why it is called as unsupervised learning because we do not give labels.
*  difference between the corrected output and then similar to The Back propagation done in neural networks the weights of the Transformer or the GPT architecture will adapt so that the next word is predicted correctly so please keep in mind that

***

that is why it is an example of self-supervised learning uh because let's say you have a
sentence right what is done is that in the sentence itself we are divided we are dividing it into
training and we are dividing it into testing so this is the true we know the
next word this is the next word and we know its true value what we'll do is that using this as the input we'll try
to predict we'll try to predict the next word so then we'll have have something
which is called as the predicted word and then we'll train the neural
network or train the GPT architecture to minimize the difference between these two and update the
weights so these four these 175 billion parameters which you see over here are
just the weights of the neural network which we are training to predict the next word so that's why it's called as
unsupervised because the label for the next word we we do not have to externally label the data set it already
is labeled in a way because we know the true value of the next word so uh to put it in another words we
don't collect labels for the training data but use the structure of the data itself to make the labels so next word
in a sentence is used as the label and that's why it is called as the auto regressive model why is it called
Auto regressive there is one more reason for this the prev previous output is used as the input for the future
prediction so let's say let me go over this part again the previous output is
used as the input so let's say the first sentence is second law of Robotics the output of this is U right this U becomes
an input to the next sentence so now the input is second law of Robotics colon U
and then the next word prediction is robot then this robot becomes an input to the next sentence that's why the
model is also called Auto regressive model so two things are very important for you to remember here the first thing
is that GPT models are the pre-training part rather I would I
should say the pre-training part of GPT models is
unsupervised why is it unsupervised because we use the structure of the data itself to create the labels the next
word in the sentence is used as the label and the second thing which is very important is that these are
Auto these are Auto regressive models
which means that the previous outputs are used as the inputs for future predictions like I showed you over here
so it is very important to note these key things when you pre-train the GPT so
in pre-training you predict the next word you break you use the structure of the sentence itself to have training
data and labels and then you do the training you train the neural network uh which is the GPT
architecture and then you optimize the parameters the 175 billion parameters
now can you think why it takes so much compute time for pre-training because 175 billion parameters have to be
optimized so that the next word in all sentences is predicted
correctly okay now as I have mentioned to you before uh compared to the original Transformer architecture the
GPT architecture is actually simpler the GPT architecture only has the decoder
block it does not have the encoder block block so again let me show you this just for reference the original Transformer
architecture looks like this if you see it has the encoder block as well as the decoder block
right but now if you see the GPT architecture here the input text is only
passed to the decoder see it does not have the encoder so in a sense the GPT
is a more simplified architecture that way uh but also the number of building
blocks used are huge in in the GPT there is no encoder but to give you an idea in
the original Transformer we had six encoder decoder blocks in the gpt3 architecture on the other hand we have
96 Transformer layers and 175 parameters keep this in mind we have
96 Transformer layers so if you see this if you think of this as one Transformer
layer if you see this as one Transformer layer this this there are 96 such
Transformer layers like this that's why there are 175 billion
parameters now I want to also show you the visual uh I want to show you more
visually how the next word prediction happens we already saw Here how we have input and the output but I've also
written it on a whiteboard so that you have a much more clearer idea so let's say so the way GPT works is that there
are different iterations there is iteration number one there is iteration number two and there is iteration number three let's zoom into iteration number
one to see what's going on so in iteration number one we first have only one word as the input this it it's it
goes through pre-processing which is converting it into token IDs then it goes to the decoder block then it goes
to the output layer and we predict the next word which is is so then now the output is this is so it predicted the
next word and now this entire this is now serves as an for the second
iteration so now we go into iteration two where this is is an input so is was
an output of the first iteration but now it is included in the input of the next iteration and then the same steps happen
we do the token ID preprocessing then it goes through the decoder block and then we predict the next word this is
an that's the output from the iteration two the next word which is predicted is n and now this output from iteration two
is now uh going as the input to the iteration number three so the input to
the iteration number three is this is n you see why it is called an auto
regressive model the output from the previous iteration which is n is forming the input of the next iteration so the
next iteration is called is this is n and then again it goes through the same preprocessing steps and then there is an
output layer and then the final output is this is an example so so the next
word which has been predicted is example and then similarly these iterations will actually keep on
continuing this is how the GPT architecture Works in each iteration we predict the next word and then the
prediction actually informs the input of the next iteration so that's why it's unsupervised and auto
regressive so this schematic of the GPT architecture and as you can see that uh
it only has the decoder there is no encoder if you look at iteration 1 2 and three I did not mention an encoder block
here right because encoder block is not present in the GPT architecture
schematic only the decoder block is
present okay now one last thing which I want to cover in this lecture is
Emergent behaviour
something called as emergent Behavior so what is emergent Behavior
I've already touched upon this earlier in this lecture and in the previous lectures but remember that GPT is
trained only for the next word prediction right GP is trained only for the next word prediction but it's
actually quite awesome and amazing that even though GPT is only trained to predict the next World it can perform so
many other tasks such as language translation so let me go to gp4 right
now and uh convert
breakfast into French so gp4 is not trained to do language
translation tasks it is trained to predict the next world but while the training or while the pre-training
happens it also develops other advantages it develops other capabilities and this is called as
emergent Behavior due to these other capabilities although it was not trained to do the translation tasks uh gp4 can
also do the translation tasks which I have mentioned here see you can see one more thing which I want to show you is
this uh uh McQ generator so as such when GPT
was trained it was not trained to do McQ generation right but look at this if I want the GPT to provide me uh three to
four multiple choice questions uh so I just clicked on generate right
now and you'll see uh McQ questions have been generated on gravity now
technically uh GPT was not trained really to generate these questions on gravity but
it developed these properties or it developed these capabilities uh on its own while the
pre-training was happening to predict the next World and that's why this is called as emergent Behavior actually so
many awesome things can be done because of this emergent Behavior although GPT is just train to predict the next word
it can do it can answer text questions generate worksheet sheets summarize a text create lesson plan create report
cards generate a PP grade essays there are so many wonderful things which GPT can do and in fact this was also
mentioned in one of the blogs of open AI where they say that
uh uh we noticed so this this mentioned in their blog we noticed that we can use
the underlying language model to begin to perform tasks without ever training on them this is amazing right for
example ex Le performance on tasks like picking the right answer to a multiple choice question uh steadily increases as
the steadily increases as the underlying language model improves this is an clear
example of emergent Behavior Uh so basically the formal
definition of emergent behavior is the ability of a model to perform tasks that
the model wasn't explicitly trained to perform just keep this in mind and that
was very surprising to researchers also because it was only trained to do the next door tasks then how can it develop
these many capabilities and I think this Still Still Remains an open question that how come emergent behavior is
developed by chat GPT so let me actually go to Google Scholar and search about emergent behavior I'm sure there
are many papers on this so here you can see I searched emergent behavior and
there are all of these papers which came up uh this is an area of active research such
as exploring emergent behavior in llms and I'm sure there's a lot of scope for
making more contributions here so if any of you are considering looking for research topics emergent Behavior might
be a great topic to start your research on this actually brings us to the end of
this lecture we covered several things in today's lecture so let me do a quick recap of what all we have covered so
Recap of lecture
initially before looking at zero shot and few shot learning we started with the history we saw that the first paper
which was introduced in 2017 is attention is all you need it Incorporated the Transformer
architecture then came generative pre-training GPT the architecture is a
bit different than Transformer it uses only decoder no encoder and then after
uh generative pre-training was developed as a method it shows two main things first is that it's unsupervised second
it's Auto regressive and unlabel data which which means it does not need label data for
pre-training then came gpt2 one year later in 2019 and uh in fact there were
four models of gpt2 which were released by open AI the first one had 117 million
parameters the second had 345 the third had 762 and the fourth one had about a
billion parameters but then came the big beast in 2020 that was really
gpt3 and uh this paper said that language models are few short Learners
which means that if gpt3 is actually provided some amount of supplementary data it can do amazing few short tasks
and this model used 175 billion parameters which was the largest anyone had ever seen up till that
point after looking at this history we looked at the difference between zero shot and few shot learning in particular
we saw that in zero shot learning you don't need to provide any example the model can perform the task without
example and in few short learning you can give a few supplementary examples so
when this gpt3 paper was released the authors claimed that this this was a few short model they did not say zero short
Learner in the title because although it can do zero short learning uh it's just
much better at few short learning and we actually explored this ourselves we asked gp4 are you a zero short learner
or are you a few short learner and gp4 sent that I'm a few short learner it's
it also said that it can also do zero short learning but it it's just more accurate uh at few short
learning okay that's important to keep in mind then we saw that gpt3 utilizes a
huge huge amount of data uh it it uses around 300 billion tokens in total so
just writing it down 300 billion tokens in total which is about 300 billion words approximately a token is a unit of
text which the model reads it it's not usually just one word but for now you
can think of one token as one word and then we saw that training pre-training
gpt3 costs $4.6 million why does it cost this much because we have to predict the
next word in a sentence using this architecture so sentences are broken down into training data and testing data
it's Auto regressive so one word of the sentence is used for testing or the next word and the remaining is used for
training and this has to be done for all the sentences in the billion billions of data files which we have that's why it
takes 4.6 million to train because there are 175 billion parameters in
gpt3 remember to optimize the weights for those many it would need a huge
amount of computer power access to gpus Etc that's why training process is hard
so this schematic also shows the GPT architecture remember it only has the decoder it works in each it works in
iterations and the output of one iteration is fed as an input to the next iteration that makes it auto regressive
and in each iteration the sentence itself is used to make the label which is the next word prediction that's why
it's an unsupervised learning exercise pre-training we also saw that after pre-training there is usually one more
step which is fine-tuning which is basically training on a much narrower and specific data to improve the
performance usually needed in production level tasks we also briefly looked at
the gap between the open source and the closed Source llms really closing with the introduction of Lama 3.1 which
absolutely amazing performance and it somewhat beats gp4 it has 405 billion
parameters and towards the end the last concept which we learned about today is that of emergent Behavior so emergent
behavior is the ability of the model to perform tasks that the model wasn't explicitly trained to perform
so for example tasks such as McQ uh worksheet generator McQ generator
lesson plan generator proof reading essay grader translation it's just the model was just trained to do the next
word prediction right then how come it can do so many other awesome tasks that's called emergent behavior and it's
it's actually a topic of active research so if anyone is looking to do research
paper work on llms which I really encourage all of you emergent Behavior might be a great topic in the next
lecture we'll look at stages of building an llm and then we'll start coding directly from the data
pre-processing so thank you so much everyone for sticking with me until this point we have covered five lectures so
far and in all of them I have tried to make them as detailed as possible and as much as from Basics approach as possible
uh let me know in the YouTube comment section if you have any doubts or any questions thank you so much everyone and
I I look forward to seeing you in the next video
*** 




