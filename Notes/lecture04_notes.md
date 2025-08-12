## Lecture 4: Basic Introduction to Transformers
1. __Pre-training__: Training on a large-diverse dataset
2. __Fine-tuning__: Refinement-by-training on a narrower-dataset, specific to a particular task or domain.

#### Transformer Architecture
* Most of the modern LLMs rely on this architecture, which is called as Transformer architecture.
* Essentially it's a DNN architecture, where basically we're trying to optimize the parameters, which was introduced in 2017 paper.

#### [Attention is all you need](https://arxiv.org/abs/1706.03762)
*  __Translation__: English-to-French and English-to-German.
*  Text completion, which is the pre-dominant role of GPT was not even in consideration here.
*  GPT architecture, which is the foundational stone or foundational building-block of ChatGPT, originated from this paper.

#### What is attention?
* It is actually a technical term, which is related to how attention is used in our daily life.

* __Embedding tokenization__

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
* Bert receives inputs where words are randomly masked during training.
* BERT is very commonly used in __sentiment analysis__.
* The Bert model only has the Encoder.

* GPT
* This is an example of how LLM can ___.
* GPT model does not have an encoder they only have a decoder.

***

### Transformers versus LLMs 
* Not ALL Transformers are LLM
* Transformers can also be used for computer vision
* Not ALL LLMs are Transformers
* LLMs can be based or convolutional architecture as well.

||||
|---|---|---|
|1980 | Bam        | RNN |
|1997 | Double Bam | Long Short-Term Memory Networks |
|2017 | Tripel Bam | Transformers | 



* Vision Transformers(ViT)

* CNN versus ViT.

*  and viit so viit AES remarkable results compared to CNN while obtaining substantially fewer computational resources for pre-training in comparison to C CNN Vision Transformers show a generally weaker bias so basically you think of only CNN when you think of image classification right but Vision Transformers are a new method which is also gaining a lot of popularity and they can be used for image classification tasks so remember when you think of Transformers don't think of Transformers only in the context of large language models or text generation Transformers can also be used for computer vision so remember not all Transformers are llms so what about llms are all llms Transformers so that is also not true not all llms are also Transformers llms can be based on recurrent or convolutional architectures as well this is what very important point to remember

* I had made a presentation uh some time back and this image has been taken from stats Quest channel so before even Transformers came into the picture here you can see 1980 recurrent neural networks were introduced in 1997 long short-term memory networks were introduced both of them could do sequence modeling tasks and both of them could do text completion tasks so they also can be called as language models so remember that all llms are not uh Transformers right llms can also be recurrent neural networks or long shortterm memory networks to give you a quick introduction what RNN actually do is that RNN maintain this kind of a feedback loop so that is why we can incorporate memory into account uh lstm on the other hand incorporates two separate paths One path about short-term memories and one path about long-term memories that's why they are called long short-term um memory networks so One path is for long-term memories and one path is for short-term memories so basically we have one green line let's say which is shown here that represents long-term memory one line which shows the short-term memory and then basically using both we can make predictions of what comes next so even recurrent neural networks and long short-term memory Networks and even some convolutional architectures can also be large language models so as we end I just want you to remember that not all Transformers are llms this is very important to keep in mind and not all llms are Transformers also so don't use the terms Transformers and llms interchangeably they are actually very different things but not many students or not many people really understand the similarities or the differences between them one purpose of these set of lectures is for you to understand everything from Basics the way it is supposed to be that way you'll also be much more confident when you transition your career or you're sitting for an llm interview and if you don't know the difference between Transformers or llms these lectures can clarify those similarities and differences for you I'm going to go into a lot of detail in lectures like what we did right now and not assume anything so I've written number of things on the Whiteboard so that you can understand let's do a quick recap of what all we learned first we saw that most modern llms rely on the Transformer architecture which was proposed in the 2017 paper it's basically a deep neural network architecture the paper which proposed the Transformer architecture is called as attention is all you need and the original Transformer was developed for machine translation for translating English tasks or English texts into German and French we saw a simplified Transformer architecture which had eight steps we take an input example pre-process it by converting words or sentences into words and token IDs then we pass it into the encoder which converts these tokens into Vector embeddings the vector embeddings are fed to the decoder along with the vector embeddings the decoder also receives partial output text and it generates the translated sentence one word at a time this is the simplified Transformer architecture and we saw that the Transformer architecture consists of an encoder and a decoder however later we saw that GPT models do not have an encoder they only have a decoder in the middle we had a small discussion on self attention mechanism which is really the heart of why Transformers works so well and why the paper which I showed you earlier is called attention is all you need self attention allows the model to weigh the importance of different words relative to each other and it enables the model to capture long range dependencies so when we are predicting the next word from a given sentence we can look at all

* the context in the past and way the importance of which word matters more for predicting the next word you can think of also self attention as parallel attention to different parts of a paragraph or different sentences we will look into this later it's going to be one of the key aspects uh it's going to be one of the key aspects of our course as we move forward then we saw the later variations of transform Transformer architecture in particular we looked at two variations first is B which is B directional encoder representations and then we saw GPT so there is a difference between these right Bert predicts hidden hidden words in a sentence or it predicts missing Words which are also called masked words so basically what this does is that word pays attention to a sentence from left side as well as from the right side because any word can be masked that's why it's called B directional encoder and it does not have the decoder architecture it just has the encoder architecture since ber looks at a sentence from both the words both the directions it can capture the meanings of different words and how they relate to each other very well and that's why BT models are used for sentiment analysis A Lot GPT on the other hand just gets the data and then it predicts the next word so it's it's a left to right model basically it has data from the left hand side and then it has to predict what comes on the right or what's the next work so GPT receives incomplete text and learns to generate one word at a time um and main thing to remember is that GPT does not have an encoder it only has decoder great and then in the last part of the lecture we saw the difference between Transformers versus llms so remember not all Transformers are llms Transformers can also be used for computer vision tasks like image classification image segmentation Etc similarly not all llms are Transformers before Transformers came recurrent neural networks and long short-term memory networks and even convolutional architectures were used for text completion so that's why llms can be based on recurrent or convolutional architectures as well so do not use these terms Transformers and llms interchangeably though many people do it understand the similarities and differences between the two that brings us to the end of this lecture we covered a lot of we covered Five Points in today's lecture and.

* I encourage you to be proactive in the comment section ask questions ask doubts uh also make notes about these architectures as you are as you are learning that's really one of the best ways to learn about this material and as always I try to show everything on a whiteboard plus try to explain as clearly as possible so that nothing is left out and I show a lot of examples also in this process thanks a lot everyone I hope you are enjoying in this series I look forward to seeing you in the next lecture

***


