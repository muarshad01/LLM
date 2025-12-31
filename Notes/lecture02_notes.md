## Building LLMs from Scratch

#### Six major aspects 
1. What exactly is an LLM?
2. What does __Large__ mean in the LLM terminology?
3. What is difference between modern LLMs and earlier NLP models?
4. What's the secret-sauce behind LLMs? What really makes them so good?
5. LLM, GenAI, DL, ML, AI
6. Applications of LLMs?

***

#### 1. What are Large Language Models (LLMs)
* [ChatGPT](https://chatgpt.com/)
* LLM is just a neural network (NN), which is designed to __Understand, Generate and Respond to human-like text.__

* Theyare designed for text related applications, such as understanding, generating, and responding to human-like text. 

* ChatGPT the demonstration, which I just showed you is an LLM. However, many people don't know about or they don't think about LLMs is that at the core of LLM they are just NN, which are designed to do these
tasks. So, if anyone asks you what an LLM is tell them that they are deep neural networks (DNN) trained on massive amount of data, which help to do specific tasks such as understanding, generating and responding to human like text, and in many cases they also respond like humans.

***

#### 2. What does Large mean in the LLM terminology?
* By model size, I mean the number-of-parameters in the model.
* LLM typically have __Billions of Parameters__.
* That's the reason why the first-part of the terminology __Large__.
* Why are they called __language models__? That's pretty clear, if you remember the example, which I showed you over here, these models only deal with language they do not deal with other modalities like image or video. __Question answering, translation, sentiment analysis, and so many more tasks.__

***
 
#### 3. LLMs versus earlier NLP Models
* NLP models were designed for very specific tasks. For example, there is one particular NLP model, which might be designed for __language translation__. There might be one specific NLP model, which might be for __sentiment analysis__. 
* LLMs on the other hand can do a wide range of NLP tasks.

***

#### 4. What's the Secret-Sauce behind LLMs? What really makes them so good?
* For LLMs, the Secret Sauce is __Transformer__ architecture.
* [Transformer (2017): Attention is all you need](https://arxiv.org/abs/1706.03762)

***

#### 5. LLM, GenAI, DL, ML, AI
* AI > ML > DL > LLM 

* __AI__ is the broadest umbrella. Any machine, which is remotely behaves like humans or it has some sort of intelligence, comes under the bucket of AI.
* What's the difference between AI and ML?
  * Example, __rule-based ChatBot__, is an example of AI because it covers intelligence. Rule-based intelligence it's not learning-based on your responses.
* __ML__ involves neural networks (NN) plus it involves things, which are not neural networks, like __Decision Trees (DTs)__.
* __DL__ usually ONLY involves neural networks (NN).
  * Example, __predict handwritten-digit classification__. If you train a neural network (NN). I've given this neural network (NN) a bunch-of-digits to learn and the task of this neural network (NN) is whenever I give it a new digit it should identify what digit is it now?
*  __LLMs__ falls under DL llms why because deep learning and they only deal with text.
*  __GenAI = LLM + DL__ you can think of as a mixture of LLM plus deep learning, why because GenAI also deals with other modalities like image, sound, video, etc.

***

#### 6. Applications of LLMS?
1. __Content Creation__: Example, write a poem about solar system
2. __AI ChatBots / Virtual Assistant__: Examples, Airlines, Banks (How to open an account), hotel/restaurant reservation desks, Customer care representative, Movies show, all of them need ChatBots.
3. __Machine translation__: Means we can of course translate-the-text to any language.
4. __Novel Text Generation__: writing books, writing media articles, news articles, etc.
5. __Sentiment analysis__: you can give a big bunch of paragraph and ask the llm to identify whether what's the sentiment here this might be useful for hate speech detection.

***

#### In Future Lectures...
* How to write the Transformer code?
* How key-query-value works?
* What exactly is __positional encoding__?

***

