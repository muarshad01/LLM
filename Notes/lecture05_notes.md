## A closer look at GPT 

|||
|---|---|
| Transfomer | (encoder, decoder) |
| BERT       | (encoder, ---) |
| GPT        | (---, decoder) |

***

| Model | Parameters/($) | Tokens| Paper Link| 
|---           |---    | --- | ---|
| Transformer (2017)  |       | | [Transformer (2017): Attention is all you need](https://arxiv.org/abs/1706.03762) |
| GPT (2018)   |         | | [GPT (2018): Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)|
| GPT-2 (2019) | 1.5 B ($4.6 M) | 300 B  | [GPT-2 (2019): Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)|
| GPT-3 (2020) | 175 B | 410 B | [GPT-3 (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)|
| GPT-3.5      |       |  | |
| GPT-4 (2023)       |       |  | |
| GPT-5 (2025)       |       |  | |

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


#### [GPT-3 (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* translation
* sentiment analysis
* answering questions
* answering MCQs
* emotional recognition, etc.

***

## Zero-shot, One-shot, and Few-shot Learning
* Ability to generalize-to-completely-unseen-tasks without-any-prior specific examples.
* Learning from a minimum-number-of-examples, which the user provides as input.

|||
| --- | --- |
| | Supporting Example(s) for Model |
| Zero-shot | No  supporting example |
| One-shot  | One supporting example |
| Few-shot (GPT-3)  | Few supporting examples|


#### Zero-Shot 
* learning the model-predicts-the-answer given only a description no-other-assistance or no-other-support.
* Example: The prompt can be that hey you have to translate English-to-French and take the word "cheese" and translate it into French.

#### Auto-regressive Language Model
* An autoregressive language model is a type of model that predicts the next-word (or token) in a sequence based on the previous words. It generates text one step at a time, where each new word depends on the ones that came before it.
* Each word is generated sequentially, and the model uses its own previous outputs to generate the next-word.
```
P(w1,w2,...,wn)=P(w1)⋅P(w2∣w1)⋅P(w3∣w1,w2)⋯P(wn∣w1,...,wn−1)
```

* GPT-3 is few-shot learner
* GPT-4 is few-shot and also zero-shot learmer

***

## Utilizing Large Datesets
* [Common Crawl](https://commoncrawl.org/)
* [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/)

* The pre-trained models are also called base/foundational models, which can be used for further __fine-tuning__.
* Many pre-trained LLMs are available as open-source.
  
|Open Source | Parameters|
|---|---|---|
|Lama 3.1| 405 B|

***

## GPT-3
* GPT-3 is a scaled-up version of the original Transformer model, which was implemented on a larger dataset. This Unsupervised learning as is done on unlabeled data.
* Error difference is computed between the output and the corrected-output and then
* Similar to the back-propagation done in NN, the weights of the Transformer / GPT architecture will adapt so that the next-word so that is predicted correctly.
* We'll try to predict the next-word so then we'll have have something, which is called as the predicted-word and then we'll train the NN or train the GPT architecture to minimize the difference between these two (prdicted-word and correct output) and update the weights to get correct answer.

*  Original Transformer, we had six encoder decoder blocks in the GPT-3 architecture on the other hand we have 96 Transformer layers and 175 parameters keep. is in mind we have 96 Transformer layers

* ...__emergent Behavior__ so what is emergent Behavior I've already touched upon ...
* ability of a model to perform tasks that the model wasn't explicitly trained to perform ...such as exploring emergent behavior in llms.
*

*   and GPT-4 sent that I'm a few-shot learner it's it also said that it can also do zero-shot learning but it it's just more accurate uh at few-shot learning.


*   __ emergent Behavior__

***


