## Lecture 4: Basic Intro to Transformers
1. __Pre-training__: Training on a large-diverse dataset
2. __Fine-tuning__: Refinement-by-training on a narrower-dataset, specific to a particular task or domain.

#### Transformer Architecture
* Most of the modern LLMs rely on Transformer architecture, which was proposed in the 2017 paper [Attention is all you need](https://arxiv.org/abs/1706.03762).
* It's basically DNN architecture.
* The original Transformer was developed for __machine translation__, for translating English texts into German and French. 

#### Stages of Transformer
1. __Input text__, which is to be translated. "This is an example".

2. __Pre-Processing__: This process (1) breaks-down a sentence into words or tokens (called __Tokenization__) and then (2) __assigns an ID (a unique number)__ to each token. For simplicity, you can imagine that one-word is one-token.

3. __Encoder__: Encoder performs __Vector Embedding__, which converts the words into vectorized representation. This vectorized representation captures the __semantic meaning__ between the words. Basically, encode-the-information somewhere such that similar words are related to each other. The words are projected into high-dimensional __Vector space__ and the way these words are projected is such that the __semantic relationship__ or the semantic meaning between the words is captured very clearly. NNs are trained for this step.

4. __Decoder__: The docoder receives (1) partial-output text, and (2) the vector embeddings. It predicts the next word based on this information. The model only generates one output word at a time.

6. Generate the translated text __one word at a time__.

7. __Final output__ is complete sentence.

* We are training the NN. Initially it will make mistakes of course, but there will be a __loss-function__ and then we will eventually train the Transformer to be better-and-better. __Feed-forward layers__, which means there are __weights and parameters which need to be optimized__.

***

### Transfomer
* Transformer is basically a DNN.
1. __Encoder__: Encodes intput text into vectors
2. __Decoder__: Generates output text from encoded vectors and from the partial-output

#### Attention
* Attention-mechanisms have become an integral part of sequence-modeling allowing modeling-of-dependencies without regard-to-their-distance.
* Attention-mechanisms allow you to model the dependencies between different words without regards to how-close-or-apart the words are.

#### Self-Attention
* Key part or Transfomer
1. Self-attention is an attention-mechanism, which relates different-positions of a single-sequence in order to compute a representation-of-the-sequence.
2. Allows the model to weigh the importance of different words / tokens relative to each other.
3. Enables model to capture __long-range dependencies__.

#### Attention Score
1. The self-attention mechanism maintains this __attention score__, which basically tells you which word should be given more attention.
2. Attention score, which basically it's a matrix and it tells you which words should be given more importance in relative or in relation to other words.
  
#### Attention Blocks
* multi-head attention
* mask multi-head attention

***

## Later variations of the Transformer architecture.
1. BERT: It predicts hidden-words in a given sentence.!
2. GPT:  Generates a new word!

#### BERT
* BERT: fills missing word. It predicts hidden-words in a given sentence.!
* Example: This is an ___ of how LLM perform.
* The BERT model has only the Encoder.
* Bert predicts hidden-words (also called masked words) in a sentence. Basically, what this does is that word pays attention to a sentence from left-side as well as from the right-side. Because any word can be masked that's why it's called Bidirectional encoder.

#### GPT
* Example: This is an example of how LLM can ___.
* GPT model only has Decoder.
* GPT on the other hand just gets the data and then it predicts the next-word. So, it's a left-to-right model. Basically it has data from the left hand side and then it has to predict what comes on the right. GPT receives incomplete text and learns to generate one word at a time.

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

*  Text completion, which is the pre-dominant role of GPT was not even in consideration here. GPT architecture, which is the foundational stone or foundational building-block of ChatGPT, originated from this paper.

* __Embedding

#### What is Attention?
* It is actually a technical term, which is related to how attention is used in our daily life.

* * __Attention__:  

* GPT architecture is actually different than the Transformer because that came later and it does not have the encoder it only has the decoder.


