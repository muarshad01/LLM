## Lecture 4: Basic Introduction to Transformers
1. __Pre-training__: Training on a large-diverse dataset
2. __Fine-tuning__: Refinement-by-training on a narrower-dataset, specific to a particular task or domain.

#### Transformer Architecture
* Most of the modern LLMs rely on Transformer architecture, which was proposed in the 2017 paper.
* It's basically DNN architecture.
* The riginal Transformer was developed for machine translation, for translating English texts into German and French. 


#### [Attention is all you need](https://arxiv.org/abs/1706.03762)
*  __Machine Translation__: English-to-French and English-to-German.
*  Text completion, which is the pre-dominant role of GPT was not even in consideration here. GPT architecture, which is the foundational stone or foundational building-block of ChatGPT, originated from this paper.

* __Tokenization__
* __Embedding

#### What is Attention?
* It is actually a technical term, which is related to how attention is used in our daily life.

#### Schematic of the Transformer

1. __Input text__, which is to be translated. "This is an example".

2. __Pre-Processing__: There is process (1) breaks-down a sentence into words or tokens (called __Tokenization__) and then (2) __assigns an ID a unique number__ to each token. For simplicity, you can imagine that one-word is one-token.

3. __Encoder__: Encoder performs __Vector embedding__, which converts the words into vectorized representation. NNs are trained for this step. Giant dimension-space. This vectorized representation captures the __semantic meaning__ between the words. Basically, encode-the-information somewhere such that similar words are related to each other. The words are projected into high-dimensional __Vector space__ and the way these words are projected is such that the __semantic relationship__ or the semantic meaning between the words is captured very clearly.

4. __Decoder__: The docoder receives 1) partial-input text, and 2) the vector embeddings. It has to predict what the next word is going to be based on this information. The partial output text remember this is available to the model because. The model only generates one output word at a time.

5. Generate the translated __text one word at a time__.

6. __Final output__: Complete sentence. It's like a NN and we are training the NN. Initially it will make mistakes of course but there will be a __loss-function__ and then we will eventually train the Transformer to be better-and-better. Think of the this as a NN. __Feed-forward layers__, which means there are __weights and parameters which need to be optimized__.

***

### Transfomer
* Transformer as a neural network and you're optimizing parameters in a neural network.
* Encoder: Encodes intput text into vectors
* Decoder: Generates out text from encoded vectors

* __Attention__:  

* Main purpose of the encoder is to convert the input text into embedding vectors great and the main purpose of the decoder is to generate the output text from the embedding-vectors and from the partial-output, which it has received.

* GPT architecture is actually different than the Transformer because that came later and it does not have the encoder it only has the decoder.

* __A note on attention__: attention mechanisms have become an integral part of sequence-modeling allowing modeling-of-dependencies without regard to their distance.
* so the attention-mechanism allows you to model the dependencies between different words without regards to how close apart or how far apart the words are that is one key thing to remember.
* Then self-attention is an attention-mechanism relating different positions of a single-sequence in order to compute a representation of the sequence.
*  This is a bit difficult to understand.

### Self-attention mechanism
* Key part or transfomer
1. Allows the model to weigh the importance of different words / tokens relative to each other.
2. Enables model to capture __long-range dependencies__,

* The self-attention mechanism maintains this __attention score__, which basically tells you which word should be given more attention.
  
### Attention Blocks
* __multi-head attention__
* __mask multi-head attention__ there are these blocks, which are called as __attention blocks__.

### Attention Score
* I mentioned they they calculate an attention score, which basically it's a matrix and it tells you which words should be given more importance in relative or in relation to other words.

### Later variations of the Transformer architecture.
1. __Bi-directional Encoder Representations from Transformers (BERT)__. It predicts hidden-words in a given sentence.
2. __Generative Pre-trained Transformers (GPT)__. Generates a new word

#### BERT versus GPT
* BERT fills missing word.
* This is an ___ of how LLM perform.
* The BERT model only has the Encoder.
* Bert predicts hidden hidden words in a sentence or it predicts missing Words which are also called masked wordsso basically what this does is that word pays attention to a sentence from left side as well as from the right side because any word can be masked that's why it's called B directional encoder.


1. GPT
* This is an example of how LLM can ___.
* GPT model only has Decoder.
* GPT on the other hand just gets the data and then it predicts the next word. so it's a left to right model. basically it has data from the left hand side and then it has to predict what comes on the right or what's the next work so GPT receives incomplete text and learns to generate one word at a time.

***

### Transformers versus LLMs 
* Not ALL Transformers are LLM
* Not ALL LLMs are Transformers
* Don't use the terms Transformers and LLMs interchangeably.

* Transformers can also be used for computer vision
* LLMs can be based on convolutional architecture as well.

### CNN versus Vision Transformers(ViT)
* ViT present remarkable results compared to CNN, while obtaining substantially fewer computational resources for pre-training.
*  ViT show a generally weaker-bias. So, basically you think of only CNN when you think of image classification right

||||
|---|---|---|---|---|
|1980 | Bam        | RNN | sequence modeling & text completion| 
|1997 | Double Bam | Long Short-Term Memory (LSTM) Networks | sequence modeling & text completion| 
|2017 | Tripel Bam | Transformers | ... |

* RNN and LSTM Networks can also be called as language models
* ALL LLMs are not Transformers. LLMs can also be RNN or LSTM Networsk to give you a quick introduction.
* similarly not all llms are Transformers before Transformers came RNN and LSTM Networks and even convolutional architectures were used for text completion.

### RNN
* actually do is that RNN maintain this kind of a __feedback-loop__ so that is why we can incorporate memory into account.

### LSTM Networks
* On the other hand incorporates two separate-paths:
1. One-path about STM
2. one-path about LTM

***
