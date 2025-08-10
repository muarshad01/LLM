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

#### Unlabeled versus labeled data
* Pre-training is done on unlabelled text dataset. It's an unsupervised learning task and it's also called Autoregression (AR).
* Ffine-tuning is mostly done on labelled text dataset. 

* Example: email classification.

* When you interact with GPT you are essentially interacting with a pre-trained (also called base or foundational model)

### Fine-tuning Types
* First category it's called instruction fine-tuning
* Second is called fine-tuning for classification tasks.
*** 
