## Stages of Building an LLM

#### 1. Data Preparation and Sampling
   * How the data is collected from different datasets?
   * How the data is processed?
   * How the data is sampled?
#### 2. Attention Mechanism
* How to code-out the attention mechanism completely from scratch in Python
* What is meant by (key, query, value)
* What is the Attention Score
* What is Positional Encoding
* What is Vector Embedding

#### 3. LLM architecture
* How to stack different-layers on top of each other?
* Where should the attention-head go?

****

## Stage-1: Data Preparation and Sampling

#### Tokenization

#### Vector Embedding
* Every word needs to be encoded into a very high-dimensional Vector Space so that the semantic meaning between words is captured.

#### Positional Encoding
* The order in which the words appears in a sentence is also very important

*  How to construct batches?
*  What is meaning of context?
*  how many words should be taken for training to predict the next output?
*  How to basically feed the data in different sets of batches for efficient computation.

### Data Batching Sequence

***

## Attention Mechanism
* Attention mechanism gives the LLM selective access to the whole input sequence when generating output one-word at a time.
* Allows the LLM to understand the importance-of-words and not just the word-in-the-current-sentence but in-the-previous-sentences, which have come long before also.
* Context is important in predicting the next-word.
* Allows the LLM to give access to the entire context and select or give weightage to which words are important in predicting the next-word.
####        Multi-Head Attention
#### Masked Multi-Head Attention
#### Positional Encoding
#### input/output embedding 

***

## Stage-2: Pre-Training
1. Training Loop
2. Model Evaluation
3. Loading pre-trained weights to build our foundational model.

* We'll break it down into epox
* Compute the __Gradient of the Loss__ in each Epoch and we'll update the parameters towards the end
* We'll also do __model evaluation__ and loading pre-train weaps
* evaluation training and validation losses

## Stage-3: Fine-tuning the LLM
* If we want to build specific applications, we will do fine-tuning.

#### Tools
*  [LangChain](https://www.langchain.com/)
*  [Ollama](https://ollama.com/)  
*  [Perplexity](https://www.perplexity.ai/)

***

## Application
#### Classifier: Spam versus No-Spam
* We cannot just use the pre-trained (foundational model) for this because we need to train with labeled data to the pre-train model.
*  We need to use the foundational-model plus the additional specific-label dataset

#### ChatBot: Answers queries
* there is an instruction
* there is an input
* there is an output

#### Emergent Properties
* LLMs are pretty generic
* If you train an LLM for predicting the next-word it turns out that it develops __emergent properties__, which means it's not only good at predicting the next-word, but also at things like:
  * MCQs,
  * text summarization,
  * emotion classification,
  * language translation, etc.

#### Application Domains
* Airline companies,
* Restaurants,
* Banks,
* Educational companies, etc.
* Getting assistance uh in summarization, helping in writing a research paper, etc.

#### Auto Regressive
* so the sentence structure itself is used for creating the labels ...
* Fine-tuned LLMs can outperform only pre-trained LLMs on specific tasks

***

## Theory plus Practical
* I'll start sharing sharing Jupiter notebooks from next time onward.

***

