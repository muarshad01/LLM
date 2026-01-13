#### Attention Mechanism
* (Transformer, Attention Mechanism) = (Car, Engine)
* __Transformers___ is the secret sauce behind the LLM.
* So, if Transformers are like a Car then __Attention Mechanism__ is essentially the Engine, which drives the Car.

1. Why is Attention Mechanism needed?
2. What are different types of Attention Mechanisms?
3. Were this name Attention Mechanism comes?
4. What are we essentially trying to solve here?

#### Example
```
The Cat that was sitting on the Mat, which was next to the Dog, jumped.
```

* A Cat was sitting next to a Dog and the Cat was also on a Mat and then as a human when I read this I know that the Cat JUMPED. Okay, but as a LLM if you look at the sentence, you'll soon realize that this sentence is a bit confusing.

* I can very clearly see that the Cat was sitting on the Mat, if only this were the sentence, I could easily analyze that the Cat is the main subject in this sentence and the Cat was sitting and the object is the Mat, but the thing is when there are such complex sentences, which are also called __long-term dependencies__, where there is this second sentence, which is attached.
* So, then it becomes a bit difficult for the LLM because after this sentence there will be a number of other sentences right...
* The main thing which the LLM needs to really understand from this sentence is that the Cat, which is the main subject actually Jumped. So, the main action which the subject perform is jumping. The LLM should understand that when it looks at Cat the word which it should be paying the most attention to is jumped. Notice, how I use the word attention? When I look at the word Cat of course sitting is also important because the Cat was earlier sitting on the Mat, but now the Cat has jumped so there are few words in this sentence, which the LLM needs to pay the most attention in association to with Cat and if you don't introduce the attention mechanism it's very difficult for the LLM to know that the Cat is the one who has jumped.

***

* Maybe if the attention mechanism was not there the LLM would have been confused and it might think the Dog has jumped or it might think that the main part of this sentence is the Cat on a mat. So, if the attention mechanism was not there maybe the LLM would have thought that the Cat is on the Mat. I have to give a lot of attention to Jump in association with the Cat.

#### Broad-level Intuition
* This is the broad-level intuition, why we need to learn about the attention mechanism.
* It turns out that with without attention mechanism, if you use a RNN or some other NN it does not capture the long-term dependencies between sentences that's the broad level intuition.

#### 4 Types of Attention Mechanism
* The main attention mechanism, which is used in Generative Pre-train Transformer (GPT) and all the modern LLMS uses multi-head attention.

1. __Simplified Self Attention__: This is the purest and the most basic form of the attention technique that you will understand.
2. __Self Attention__: Here we will also introduce __trainable weights__, which form the basis of the actual mechanism that is used in the LLMs.
3. __Causal Attention__: This is when things really start to get interesting. We are predicting the next world right by looking at the past world. So, what causal attention does is that it's a type of self attention that allows the model to consider ONLY the previous and the current inputs in a sequence and it masks out the future inputs.

1. History of how attention came into the picture?
2. Why it is needed?
3. Why it's better than RNN, etc., and then finally, we'll move to multi-ad attention.
* So, __multi-head attention__ is just basically a bunch of causal attention heads stacked together.
* __multi-head attention__ is essentially an extension of self attention and causal attention that enables the model to simultaneously attend to information from different representation subspaces.
* __multi-head attention__ allows the LLM to look at input data and and then process many parts of that input data in parallel.
* __multi-ad attention__ allows the LLM to have let's say ...one attention head looks at this part... one attention head looks at this part ...one attention head looks at this part, etc. This is just a crude description so that you get an understanding of what do you mean by multi-head attention.

***

#### (2) - The problem with modeling long-sequences?
* Word-by-word transalation doesn't work.
* The transaltion process requires:
1. __Contextual understanding__
2. __Grammatical alignment__

* Problems with modeling long-sequences start 
* We need contextual understanding and grammar alignment

* Whenever you are developing a model, which translates one sequence to another sequence or tries to find the meaning of a sequence or makes the next word prediction from a sequence you need to really understand the context you need to understand how different words relate with each other what's the grammar of that particular language and only then will you be able to process sequences or only then you'll be able to model long-sequences of textual information.

* We cannot just use a normal NN because if you have a normal NN it does not have memory.

* sequence translation models need to have a memory to stor the past

```
Harry Potter went to Station Number 9 3x4. He did this, etc.
```
* and then when I come to a sentenc,e which is three to four sentences after the first sentence, which is Harry Potter came to station 9 3x4. I should not forget what came before because the station number 9 3x4 is very important for me to know even if I come at the end of the paragraph.

* They augmented a NN with two subm modules:
1. the first sub-module is an encoder and
2. the second sub-module is a decoder

***

* __Context Vector__ essentially captures meaning so it has memory and it captures meaning. Okay, instead of just word by word translation. What does this sentence represent and the encoder processes.

* How RNNs work?
* Mechanism which really employed the encoder decoder blocks successfully is called RNN

* (c) - so before really Transformers came into the picture RNN were was that architecture, which was extremely popular for language translation and it really employed
the (encoder, decoder) architecture. it was implemented in
the 1980s.

* The input text is passed to encoder. What the encoder will do is that at every step it will take the input and it will maintain something which is called as the __hidden state__. This hidden state was the biggest innovation in the RNN.
* This hidden State essentially captures the __memory__ so imagine the first input, which is the first German word comes the encoder augments it or the encoder maintains a hidden State then you go to the next iteration then the second input world comes then the hidden State also gets updated so as the hidden state gets updated it receives more and more memory of what has come previously and the hidden state gets updated at each step and then there is a __final hidden State__. The final hidden state is basically the encoder output. What we saw the __context Vector__ over here. So when we looked at the context Vector, which is passed from the encoder to the decoder.
* Here you see a context Vector that is passed from the encoder to the decoder. This context Vector is the final hidden State this is basically the encoder telling the decoder that hey I have looked at the input text ...here's the meaning of this text ...here's how I encoded it ...here's the context Vector ...take this final hidden State and try to decode it and then the decoder uses this final hidden state to generate the translated sentence and it generates the translatedsentence one word at a time .

* __final hidden State__ essentially contains the __accumulation of all previous hidden state__ . It contains or encapsulates memory this is how memory is incorporated, which was missing earlier with just a normal NN. So, this is the final hidden State and then this final hidden State essentially memorizes the entire input and then this hidden state is passed to to the decoder and then the decoder produces the final output which is the translated English.

***

20:00

#### Big problem with RNN
* The model the decoder has essentially no access to the previous hidden States.
* The reason it's a big problem is because when we have to process long-sequences, if the decoder just relies on one final hidden State that's a lot of pressure on the decoder.
* Essentially one final hidden State needs to have the entire information and for long-sequences it usually fails because it's very hard for one final hidden state.
* so as we saw the encoder processes the entire input text into one final hidden state, which is the memory cell and then decoder takes this this hidden State. decoder takes this hidden state to essentially produce an output great now here's the biggest issue with RNN. 

* __RNN Limitations__: The biggest issue with the RNN is that a RNN cannot directly access earlier hidden States.

* so the RNN can't directly access earlier hidden States from the encoder during the decoding phase it relies only on the current hidden state which is the final hidden State and this leads to a loss or this leads to a __loss of context__. Especially in complex sentences where dependencies might span long distances.

*  okay so let me actually explain this further what does it mean loss of context right uh so as we saw the encoder compresses the entire input sequence into a single __hidden State Vector__.

*   Let's say if the sentence if the input sentence is very long if the input sentence is very long it really becomes very difficult for the RNN to capture all of that information in __one single final hidden state__ that becomes very difficult and this is the main drawback of the RNN.

***

* 25:00
  

* if you are putting all the pressure on one final hidden state to capture all this context especially in Long sequences so uh the key action which is jumped depends on the subject which is cat but also an understanding longer depend dependencies so jumped depends on cat but we also need to understand that the cat was sitting on the mat and the cat was also sitting next to the dog because the dog also might be referred somewhere else in the big text so we need to understand many things from this sentence and these are also called as longer dependencies so jumped the action jumped of course depends on the subject cat but we Al to understand this we also need to understand longer dependencies that the cat was sitting next to the dog and the cat was also sitting on the mat so to capture these longer dependencies or to capture this uh longer context or difficult context the

* RNN decoder struggles with this because it just has one final hidden State.  to get all the information from this is called __loss of context__, which was one of the biggest issues because of which RNN was not as good as the GPT which exist right now which is based on the attention mechanism.


* Bahdanau Attention Mechanism dependencies is where attention mechanism actually comes into the picture. Okay, we will capture long range dependencies with attention mechanisms and let's see how
* (a) - RNN work fine for translating short-sentences. They don't work for long-text because they don't have direct access to previous words in the input.
* (b) - One of the major shortcoming in the RNN is that: RNN must remember the entire encoded input in a single hidden State before passing it to the decoder.
* [Neural machine translation by jointly learning to align and translate -- Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=kukA0LcAAAAJ&citation_for_view=kukA0LcAAAAJ:__bU50VfleQC)
*  `Bahdanau attention mechanism for RNN`
*   difies the encoder decoder RNN such that the decoder can selectively access different parts of the input sequence at each decoding step.

***

30:00

* The text generating decoder part of the network can access all the input tokens selectively.
* This means that some input tokens are more important than others for generating a given output token.
* This importance is determined by the __attention weights__.

* Only 3 years later researchers found that RNN architectures are not required for building DNN for NLP and this is when the researchers proposed the Transformer architecture. The main core of the Transformer architecture was the Bahdanau attention mechanism.


* [Attention is all you need - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, Illia Polosukhin](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=oR9sCGYAAAAJ&citation_for_view=oR9sCGYAAAAJ:zYLM7Y9cAGgC)

***

35:00

* So at each decoding step the model can look back at the entire input sequence and decide, which parts are most relevant to generate the current word.

* For example, let's say we are predicting this word `saute`, which is the French translation for jumped. So, when the decoder is predicting this French translation the attention mechanism allows the decoder to focus on the part of the input that corresponds to jump. So, we can selectively look at which part of the input to give maximum attention to uh

* This __Dynamic focus__ on different parts of the input sequence allows modes to learn long range dependencies more effectively

* So Dynamic Focus, which means that for every decoding step, we can selectively choose which inputs to focus on and how much attention to give to it each input. That's why it's called Dynamic Focus. This Dynamic focus on different parts of the input sequence helps us to learn long-range dependencies more effectively and that's why the attention mechanism uh actually works so well.

***

40:00

| Model | Year|
|---|---|
| RNN |1980|
| LSTM |1997|
| Attention |2014|
| Transformers |2017|

* RNNS had this problem which is called __Vanishing gradients__
*  when you stack multiple feedback loops together it leads to the vanishing gradient problem.
*  LSTM solved this.  So we had a long-term memory route and a short-term memory route and that's how the LSTM actually operates.
* both of these had problems with respect to __longer context__,  which was solved 2014 and 2017 papers.

#### __self attention__: 
* So self attention is basically a mechanism that allows each position of the input sequence to attend to all positions in the same sequence. So, we are not looking at different sequences.
* Self attention is basically different. Now instead of giving attention to another sequence all the attention is directed inwards.
* So, we we are just looking within a particular sequence. We are looking at different tokens within a sequence and see how these tokens are related to each other.

***

45:00

* (f) - Self attention is a key component of contemporary LLMs bassed on the Transformer architecture, such as GPT series.
* so LLM remember are predicting the next word in a given sentence right they can of course do language translation tasks as well but they were predominantly trained for predicting the next World and they are able to do translation tasks
which is also called as __emergent Behavior___ but they were trained to predict the next word and that's why they have to so.

* self attention module is a key component of contemporary LLMs

* In self attention the self really refers to the attention mechanism's ability to __compute attention weights__ by relating different positions in a single input sequence
*  so as I mentioned, we are looking at just one input sequence and we are looking at the attention between different tokens of that sequence.
*  so it learns the __relationship between various parts of the input itself__
* so this is different than traditional attention mechanisms like the translation task, which we saw where the focus is on __relationships between elements of two different sequences__.
* For example, if we want to translate from English to German OR German to French OR English to Hindi, etc.. We have two sequences right ...so traditional attention mechanisms look at one part of the sequence and another part of the sequence and how they are related to each other.
* In self attention, we basically learn the relationship between various parts of the input itself

* 
* __RNN have a Major Short-coming__: and that shortcoming is that they have to remember the entire encoded input in a single hidden state before passing it to the decoder and that's a problem when dealing with long sentences.
*  This leads to __context loss__, because the decoder does not have access to the previous inputs. This is exactly the problem, which is solved by attention mechanisms.
*  In attention mechanisms when you are decoding a particular part like
* which parts of the output sequence are more related to which parts of the input sequence this is __traditional attention__.
* In self attention you just look at one sequence and you look at different parts of that same sequence and how they are related with respect to each other.

***
