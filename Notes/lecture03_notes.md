## Stages of building LLMs
1. __Pre-training (Unlabeled dataset)__: Training on a large, diverse-dataset. 
2. __Fine-tuning (Labeled dataset)__: Refinement-by-training on a narrower-dataset, specific to a particular task of domain.

* __Token__: You can think of one-token as equal to one-word. There is a bit of detailing here ...discussed later.

* __Task (Next-Token-Prediction)__: Word-completion, which means you are given a set-of-words, let's say you are given this sentence the `lion is in the ---`. The next word is not known and ChatGPT then predicts, or LLM then predicts, this word as let's say `forest`. 

* __Observation__: If you train the LLM for this simple-task, it turns out that it can do a wide-range of other-tasks as well.
  * For example, picking the right answer MCQs steadily increases as the underlying language model improves.
  * Other examples, language translation, answering MCQs, summarizing a text, sentiment detection, etc. 

***

#### Examples
* Airline ChatBot: ...
* Bank ChatBot: ...
* Legal Al-Tool for Attorneys: ...

#### Pre-trained (also called Base/Foundational) models
* Pre-trained models are trained on large diverse-datasets ((unlabeled dataset) using unsupervised trainging.
* In Fine-tuning, the pre-trained LLM model, we use labeled dataset.

***

#### Unlabeled dataset versus Labeled dataset
* Pre-training is done on unlabelled text dataset. It's an unsupervised learning task and it's also called __Autoregressive (AR)__.
* Fine-tuning is mostly done on labelled text dataset. 

#### Auto-regressive (AR) Language Model
* An autoregressive language model is a type of model that __predicts the next-word (or token)__ in a sequence based on the previous words. It generates text one-step at a time, where each new word depends on the ones that came before it.
* Each word is generated sequentially, and the model uses its own previous-outputs to generate the next-word.
```
P(w1,w2,...,wn) = P(w1)⋅P(w2∣w1)⋅P(w3∣w1,w2)⋯P(wn∣w1,...,wn−1)
```

***

#### Fine-tuning Types
* First category, it's called __instruction fine-tuning__.
* Second category is called fine-tuning for classification tasks.

***
