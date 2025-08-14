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

## Data Preparation and Sampling

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

#### Multi-Head Attention
#### Masked Multi-Head Attention
#### Positional Encoding
#### input embedding output embedding

***

* we'll break it down into epox and we will compute the __gradient__ of the loss in each Epoch and we'll update the parameters towards the end
*   we'll also do __model evaluation__ and loading pre-train weaps
*    evaluation training    and validation losses
*    then we'll write the LLM training function
*     Implement function to save-and-load the LLM weights
*   load pre-trained weights from open AI into our LLM.
*
#### stage-two Steps
1. training Loop plus
2. model evaluation plus
3. loading pre-trained weights to build our foundational model.
   

#### stage-three 
* Fine-tuning the LLM so if we want to build specific applications we will do fine tuning in this playlist
* 

***

#### First Application
* spam whereas hey just wanted to check if we are still on for dinner tonight let me know this will be not spam so we will build a LLM this application which classifies between spam and no spam.
* and we cannot just use the pre-trained or foundational model for this because we need to train with labeled data to the pre-train model.
*  we need to give some more data and tell it that hey this is usually spam and this is not spam can you use the foundational model plus this additional specific label data asset, which I have given to build a fine-tuned llm application for email classification.
   #### Second application 
  * ChatBot which basically answers queries.
  *  so there is an instruction there is an input and there is an output and we'll be building this chatbot after fine-tuning the LLM.
  *  [LangChain]()
  *    Tools like AMA and they directly deploy applications but they do not understand what's going on in stage-one and stage-two at all.
  *  next lecture we are going to start a bit of the Hands-On approach
  *  okay so number one LLMs have really transformed uh the field of NLP they have led to advancements in generating understanding and translating human language this is very important.
  *   uh so the field of NLP before you needed to train a separate algorith for each specific task but __LLMs are pretty generic__
  *   if you train an llm for predicting the next word it turns out that it develops __emergent properties__, which means it's not only good at predicting the next word but also at things like uh MCQs, text summarization, then emotion classification, language translation, etc.
  *   
  *    secondly all modern LLMs are trained in two main-steps
  *     first we pre-train on an unlabeled data this is called as a foundational model
  *  and for this very large datasets are needed typically billions of words and it costs a lot as we saw training pre-training GPT-3 costs $4.6 million.
  *   so you need access to huge amount of data compute power and money to pre-train such a foundational model.
  *    now if you are actually going to implement an LLM application on production level so let's say if you're an educational company building MCQs and you think ...usually airline companies, restaurants, Banks, educational companies,
  * ... __Auto regressive__ so the sentence structure itself is used for creating the labels ...
  * llm applications important thing to remember is that fine tuned LLMs can outperform only pre-trained llms on specific tasks

***

* two cases right
* in one case you only have pre-trained llms and in
* second case you have pre-trained plus fine tuned llms 
* getting assistance uh in summarization uh helping in writing a research paper Etc
* __GPT-4 perplexity__ or such API tools or such interfaces, which are available work perfectly fine.
*  but if you want to build a specific application on your dataset and take it to production level you definitely need fine-tuning.
*   okay now uh one more key thing is that the secret Source behind LLM is this Transformer architecture.
*    so uh the key idea behind Transformer architecture is the
*    __attention mechanism__ uh just to show you how the Transformer architecture looks like.
*     __attention blocks__, we'll see what they mean so no need to worry about this right now but in the nutshell
* attention mechanism gives the llm selective access to the whole input sequence when generating output one-word at a time.
*  basically attention mechanism allows the LLM to understand the importance of words and not just the word in the current sentence but in the previous sentences which have come long before also because __context__ is important in predicting the next word the current sentence is not the only one which matters.
*   attention mechanism allows the llm to give access to the entire context and select or give weightage to which words are important in predicting the next word.
* Transformer (encoder, decoder)
* GPT (---, decoder)
*  last point which is very important is that LLM are only trained for predicting the next-word right but very surprisingly they develop __emergent properties__, which means that although they are only trained to predict the next-word they show some amazing properties like ability to classify text, translate text from one language into another language, and even summarize texts.
*
* first aspect which is data preparation and sampling
* so the next lecture title will be be working with Text data and we'll be looking at the datasets
* how to load a dataset
* how to count the number of characters
* how to break the data into tokens
* I'll start sharing sharing Jupiter notebooks from next time onward.
*  I follow this approach of writing on a whiteboard and also coding um so that you understand the __details plus the code__ at the same time
*  because I believe Theory plus practical implementation both are important

***

