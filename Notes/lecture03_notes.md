## Stages of building LLMs
1. __Pre-training (Unlabeled data)__: Training on a large, diverse dataset. 
2. __Fine-tuning (Labeled data)__: Refinement by trainign on a narrower dataset, specific to a particular task of domain

* __Token__: You can think of one-token as equal to one-word. There is a bit of detailing here ...discussed later.

* __Task__: Word-completion, which means you are given a set-of-words, let's say you are given this sentence the "lion is in the ---". The next word is not known and ChatGPT then predicts or LLM then predicts this word as let's say "forest".

* __Observation__: If you train the LLM for this simple-task it turns out that it can do a wide range of other-tasks as well. For example, picking the right answer MCQs steadily increases as the underlying language model improves. Other examples, language translation, answering MCQs, summarizing a text, sentiment detection, etc. 

***

#### Examples
* Airline ChatBot: ...
* Bank ChatBot: ...
* Legal Al-Tool for Attorneys: ...

#### Pre-train models (also called base/Foundational model)
* Fine-tune the pre-trend LLM on the label dataset.

***

#### Unlabeled data versus Labeled data
* Pre-training is done on unlabelled text dataset. It's an unsupervised learning task and it's also called Autoregression (AR).
* Fine-tuning is mostly done on labelled text dataset. 

### Fine-tuning Types
* First category it's called instruction fine-tuning.
* Second is called fine-tuning for classification tasks.

***

