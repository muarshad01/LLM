## A closer look at GPT 

|||
|---|---|
| Transfomer | (encoder, decoder) |
| BERT       | (encoder, ---) |
| GPT        | (---, decoder) |

***

| Model | Parameters | Tokens| Paper Link| 
|---           |---    | --- | ---|
| Transformer (2017)  |       | | [Transformer (2017): Attention is all you need](https://arxiv.org/abs/1706.03762) |
| GPT (2018)   |         | | [GPT (2018): Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)|
| GPT-2 (2019) | 1.5 B | 300 B  | [GPT-2 (2019): Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)|
| GPT-3 (2020) | 175 B |  | [GPT-3 (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)|
| GPT-3.5      |       |  | |
| GPT-4        |       |  | |

***

#### [Transformer (2017) : Attention is all you need](https://arxiv.org/abs/1706.03762)
*  __Self-attention__ mechanism, where you capture the __long-range dependencies__ in a sentence. Allowing you to do a much better job at predicting the next-word in a sentense.
*   A significant advancement compared to RNN (1980) and LSTM Networks (1997).

#### [GPT (2018): Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* __Unsupervised-learning (Unlabeled data)__
* __Generative Pre-training__ on a divrse corpus of __unlabeled__ text data.
* __Generative__ because we are generating-/predicting-the-next-word in an __unsupervised-learning__ manner.
* __Pre-training__: Text data, which is used here is not labeled. Let's say you have given a sentence right and you use that sentence itself as the training data and the next-word prediction, which is in that sentence itself as the testing data. So, everything is self-content and you don't need to provide labels.
* NLP uptill this point was supervised learning. Labeled data is scarce.

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

## Zero-Shot, One-Shot, and Few-Shot Learning

|||
| --- | --- |
| Zero-Shot ||
| One-Shot ||
| Few-Shot ||

#### Few shot
* Ability to generalize-to-completely-unseen-tasks without any prior specific examples.
* Learning from a minimum-number-of-examples, which the user provides as input.

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

* that is why it is an example of __self-supervised__ learning.
* let's say you have a sentence right what is done is that in the sentence itself we are divided we are dividing it into training and we are dividing it into testing so this is the true we know the next word this is the next word and we know its true value what we'll do is that using this as the input.
*  we'll try to predict we'll try to predict the next-word so then we'll have have something which is called as the predicted word and then we'll train the neural network or train the GPT architecture to minimize the difference between these two and update the weights so these four these 175 billion parameters.
*   which you see over here are just the weights of the neural network which we are training to predict the next word so that's why it's called as __unsupervised__ because the label for the next word we we do not have to externally label the data set.
*    it already is labeled in a way because we know the true value of the next word so uh to put it in another words we don't collect labels for the training data but use the structure of the data itself to make the labels so next word in a sentence is used as the label and that's why it is called as the
*    __auto regressive__ model why is it called Auto regressive there.
*     is one more reason for this the prev previous output is used as the input for the future prediction.
*  so two things are very important for you to remember here the first thing is that GPT models are the pre-training part rather I would I should say the __pre-training part of GPT models is unsupervised__ why is it unsupervised because we use the structure of the data itself to create the labels the next word in the sentence.
*   is used as the label and the second thing which is very important is that these are Auto these are __Auto regressive models__, which means that the previous outputs are used as the inputs for future predictions.
*    like I showed you over here so it is very important to note these key things when you pre-train the GPT so in pre-training you predict the next word you break you use the structure of the sentence itself to have training data and labels and then you do the training you train the neural network uh which is the GPT architecture and then you optimize the parameters the 175 billion parameters now can you think why it takes so much compute time for pre-training because 175 billion parameters have to be optimized so that the next word in all sentences is predicted correctly.
*  so in a sense the GPT is a more simplified architecture that way uh but also the number of building blocks used are huge in in the GPT there is no encoder but to give you an idea in the original Transformer we had six encoder decoder blocks in the gpt3 architecture on the other hand we have 96 Transformer layers and 175 parameters keep.
*   this in mind we have 96 Transformer layers
* __auto regressive model__ the output from the previous iteration ...why it's __unsupervised and auto regressive__ so this schematic of the GPT architecture ...__emergent Behavior__ so what is emergent Behavior I've already touched upon ...
* ability of a model to perform tasks that the model wasn't explicitly trained to perform ...such as exploring emergent behavior in llms.
*
*  after uh generative pre-training was developed as a method it shows two main things first is that it's __unsupervised__ second it's __Auto regressive__ and unlabel data
*
*   and gp4 sent that I'm a few short learner it's it also said that it can also do zero short learning but it it's just more accurate uh at few short learning okay that's important to keep in mind then we saw that
*   gpt3 utilizes a huge huge amount of data uh it it uses around 300 billion tokens in total so just writing it down __300 billion tokens__ 
*   training data and testing data it's __Auto regressive__ so one word of the  it only has the decoder it works in each it works in iterations and the output of
*
*
*  improve the performance usually needed in production level tasks we also briefly looked at the gap between the open source and the closed Source llms really closing with the introduction of __Lama 3.1 which absolutely amazing performance and it somewhat beats gp4 it has 405 billion parameters__
*
*   __ emergent Behavior__
*
*    in the next lecture we'll look at stages of building an llm and then we'll start coding directly from the data pre-processing

***



