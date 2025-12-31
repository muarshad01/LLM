## Stages of Building an LLM
* 1 - Data Preparation and Sampling
  * How the data is collected from different datasets?
  * How the data is processed?
  * How the data is sampled?
* 2 - Attention Mechanism
  * What is the __Attention Score__
  * What is __Positional Encoding__
  * What is __Vector Embedding__
  * What is meant by __(key, query, value) = (K,Q,V)__
* 3 - LLM architecture
  * How to stack different-layers on top of each other?
  * Where should the __attention-head__ go?

***

## Data Preparation and Sampling

#### Tokenization

#### Vector Embedding
* Encode every word into a very __high-dimensional Vector Space__, so that the __Semantic Meaning__ between words is captured.

#### Positional Encoding
* The __order-of-words in a sentence__ is also very important.
* What is the meaning of context?
* How many words should be taken for training to predict the next output?
* How to feed-the-data in different sets-of-batches for efficient computation.
* How to construct batches (data batching sequence)?

***

## Attention Mechanism
* Allows the LLM to understand the __importance-of-words__ and not just the word-in-the-current-sentence but in-the-previous-sentences, which have come long before also.
* Allows the LLM to give access to the entire-context and select or give weightage to which words are important in predicting the next-word.
* Gives the LLM selective-access to the whole-input-sequence when generating output one-word at a time.

#### Types
1. Multi-Head Attention
2. Masked Multi-Head Attention

***

## Pre-Training
1. Training Loop
2. Model Evaluation
3. Loading pre-trained weights to build our base/foundational model.

* We'll break it down into epoch
* Compute the Gradient of the Loss (loss  validation) 
* Update the parameters
* Model evaluation
* Loading pre-trained weights

***

## Fine-tuning the LLM
* If we want to build specific applications, we will do fine-tuning.

#### Tools
*  [LangChain](https://www.langchain.com/)
*  [Ollama](https://ollama.com/)  
*  [Perplexity](https://www.perplexity.ai/)

***

## Application
#### Classifier: Spam versus No-Spam
* We cannot just use the pre-trained (base/foundational model) for this because we need to train-with-labeled-data to the pre-trained model.
* We need to use the foundational-model plus(+) the additional specific-label dataset

#### ChatBot: Answers queries
* there is an instruction
* there is an input
* there is an output

#### Emergent Properties
* LLMs are pretty generic
* If you train an LLM for predicting the next-word it turns out that it develops __emergent properties__, which means that it is not only good at predicting the next-word, but also at things like:
  * MCQs,
  * text summarization,
  * emotion classification,
  * language translation, etc.

#### Application Domains
* Airline companies,
* Restaurants,
* Banks,
* Educational companies, etc.
* Getting assistance in summarization, helping in writing a research paper, etc.

#### Auto Regressive
* so the sentence structure itself is used for creating the labels ...
* Fine-tuned LLMs can outperform only pre-trained LLMs on specific tasks

***

## Theory plus Practical
* I'll start sharing sharing Jupiter notebooks from next time onward.

***

