#### Lecture 4: Basic Intro to Transformers
1. __Pre-training__: Training on a large-diverse dataset (Unlabeled data)
2. __Fine-tuning__: Refinement-by-training on a narrower-dataset, specific to a particular task or domain (Labeled data)

***

#### Transformer Architecture
| Architecture | Paper|
|---|---|
| Transformer | [2017 - Attention is all you need](https://arxiv.org/abs/1706.03762)|

* It's basically a DNN architecture
* __Objective__: Machine translation, for translating English texts into German and French. 

***

#### Stages of Transformer
1. __Input text__: That needs to be translated. For example, `This is an example`.

2. __Pre-Processing__: This process: (1) breaks-down a sentence into words or tokens (called __Tokenization__) and then (2) __assigns an ID (a unique number)__ to each token. For simplicity, you can imagine that one-word is one-token.

3. __Encoder__: Encoder performs __Vector Embedding__, which converts the words into vectorized representation. This vectorized representation captures the __Semantic Meaning__ between the words. Basically, encode-the-information somewhere such that similar-words are related to each other. The words are projected into a high-dimensional __Vector Space__ and the way these words are projected is such that the __Semantic Relationship__ or the semantic meaning between the words is captured very clearly. NNs are trained for this step.

4. __Decoder__: The docoder receives: (1) partial-output text, and (2) the __Vector Embedding__. It predicts the next-word based on this information. 

5. Generate the translated text __ONE word (Token) at a time__. 

6. __Final output__ is complete sentence.

***

#### Loss Minimization (Optimization)
* We are training the NN.
* Initially it will make prediction-mistakes of course, but there will be a __loss-function__ to calculate the error. We'll eventually train the Transformer to be better-and-better.
* __Feed-forward layers__, which means there are __Weights and Parameters__ that need to be optimized.

***

#### Transfomer
* Transformer is basically a DNN.
1. __Encoder__: Encodes intput text into Vectors
2. __Decoder__: Generates output text from Encoded Vectors and from the partial-output.

***

#### Attention
* Attention-mechanisms have become an integral part of sequence-modeling allowing modeling-of-dependencies without regard-to-their-distance.
* Attention-mechanisms allow you to model-the-dependencies between different words without regards to how-close-or-apart the words are.

***

#### Self-Attention
* Key part or Transfomer
1. Self-attention is an attention-mechanism, which relates different-positions of a single-sequence [(1,2,3),(2,3,1), (3,1,2)] in order to compute a representation-of-the-sequence.
2. Allows the model to weigh the importance of different words / tokens relative-to-each-other.
4. Enables model to capture __long-range dependencies__.

***

#### Attention Score
1. The self-attention mechanism maintains __attention score__, which basically tells you which word should be given more attention.
2. Attention score is basically a matrix that tells you, which words should be given more importancecin relation to other words.

***
  
#### Attention Blocks
* multi-head attention
* mask multi-head attention

***

## Later Variations of the Transformer architecture.

| Model | Architecture|
|---|---|
|Transformer | (encoder, decoder)|
|BERT | (encoder, ---)|
|GPT| (---, decoder)|

#### BERT
* The BERT model has only the Encoder.
* BERT: Fills missing word. It predicts hidden-words (also called masked words) in a given sentence.
* Example: This is an ___ of how LLM perform.
* Basically, what this does is that word pays attention to a sentence from left-side as well as from the right-side. Because any word can be masked that's why it's called Bidirectional encoder.

#### GPT
* GPT model only has Decoder.
* Example: This is an example of how LLM can ___.
* GPT on the other hand just gets the data and then it predicts (generates) the next-word. So, it's a left-to-right model. Basically it has data from the left-hand-side and then it has to predict what comes on the right. GPT receives incomplete text and learns to generate one-word at a time.

***

## Transformers versus LLMs 
* Not ALL Transformers are LLM
* Not ALL LLMs are Transformers
* Don't use the terms Transformers and LLMs interchangeably.

||||||
|---|---|---|---|---|
|1980 | Bam        | RNN                                    | sequence modeling & text completion| 
|1997 | Double Bam | Long Short-Term Memory (LSTM) Networks | sequence modeling & text completion| 
|2017 | Tripel Bam | Transformers                           | ... |

#### RNN & LSTM
* RNN (1980), LSTM (1997) Networks, and and even convolutional architectures, can also be called as language models (text completion purpose).
* LLMs can also be RNN, LSTM Networsk, or convolutional architectures, to give you a quick introduction.

#### RNN
* RNN maintains a __feedback-loop__, so that is why, we can incorporate memory into account.

#### LSTM Networks
* Incorporates two separate-paths:
1. One-path about STM
2. one-path about LTM

***

## Vision Transformers(ViT)
* Transformers can also be used for Computer Vision
* ViT present remarkable results compared to CNN, while obtaining substantially fewer computational resources for pre-training.
*  ViT show a generally weaker-bias. So, basically you think of only CNN when you think of image classification right

***
















