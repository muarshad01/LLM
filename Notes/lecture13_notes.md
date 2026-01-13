#### Attention Mechanism
* (Transformer, Attention Mechanism) = (Car, Engine)
* __Transformers___ as the secret sayce behind the LLM.
* So, if Transformers is like a Car then __Attention Mechanism__ is essentially the Engine, which drives the Car.

 
* Why is attention mechanism needed?
* What are different types of attention mechanisms?
* Why this name Attention Mechanism comes?
* What are we essentially trying to solve here?

#### Example
```
The Cat that was sitting on the Mat, which was next to the Dog, jumped.
```

* A Cat was sitting next to a dog and the Cat was also on a Mat and then as a human when I read this I know that the Cat JUMPED. Okay, but as a LLM if you look at the sentence you'll soon realize that this sentence is a bit confusing.

* I can very clearly see that the Cat was sitting on the Mat, if only
this were the sentence, I could easily analyze that the Cat is the main subject
in this sentence and the Cat was sitting and the object is the Mat, but the thing
is when there are such complex sentences, which are also called __long-term dependencies__ where there is this second sentence, which is attached so then it becomes a bit difficult for the LLM because so after this sentence there will be a number of other sentences right but.
* The main thing which the LLM needs to really understand from this sentence is that the cat, which is the main subject actually jumped so the main the action which thesubject perform is jumping. So the LLM should understand that when it looks at Cat the word which it should be paying the most attention to is jumped. Notice, how I use the word attention so when I look at the word Cat of course sitting is also important because the Cat was earlier sitting on the Mat but now the cat has jumped so there are few words in this sentence which the LLM needs to pay the most attention to in association with cat and if you don't introduce the attention mechanism it's very difficult for the LLLM to know that the Cat is the one who has jumped.

***

Maybe if the attention mechanism was not there the LLM would have been confused and it might think the dog has jumped or it might think that the main part of this sentence is the cat is on a mat so if the attention mechanism was not there maybe the LLM would have thought that the cat is on the mat that's it would not know that. I have to give a lot of attention to jump in association with the cat this is the broad level intuition why we need to learn about the attention mechanism.

* it turns out that with without attention mechanism if you used a recurrent neural network or some other neural network it does not capture the longterm dependencies between sentences that's the broad level intuition.

#### 4 Types of Attention Mechanism
* The main attention mechanism, which was which is used in GPT uh Generative Pre-train Transformer (GPT) and all the modern LLMS use this multi-head attention.


1. __Simplified Self Attention__: This is the pureest and the most basic form of the attention technique that you will understand.
2. __Self Attention__: Here we will also introduce __trainable weights__, which form the basis of the actual mechanism that is used in the LLMs.
3. __Causal Attention__: This is when things really start to get interesting. We are predicting the next world right by looking at the past world. So what causal attention does is that it's a type of self attention that allows the model to consider ONLY the previous and the current inputs in a sequence and it masks out the future inputs.

* History of how attention came into the picture?
* Why it is needed?
* Why it's better than RNN,  etc., and then finally, we'll move to multi-ad attention only when you have understood causal attention and self attention and simplified self attention.
* You will be able to understand multi-head attention this is the main concept which is actually used in building GPT.
* So, __multi-head attention__ is just basically a bunch of causal attention heads stacked together.
* __multi-head attention__ is essentially an extension of self attention and causal attention that enables the model to simultaneously attend to information from different representation subspaces.
* __multi-head attention__ allows the LLM to look at input data and and then process many parts of that input data in parallel.
* The multi-ad attention allows the LLM to have let's say ...one attention head looks at this part... one attention head looks at this part ...one attention head looks at this part, etc. This is just a crude description so that you get an understanding of what do you mean by multi-head attention.

***

#### (2) - The problem with modeling long-sequences?
* Word-by-word transalation doesn't work.
* The transaltion process requires  contextual understanding and grammatical alignment.

* Problems with modeling long sequences start where we are modeling.


* you need contextual understanding and grammar alignment

* so whenever you are developing a model let's say which translates one sequence to another sequence or tries to find the meaning of a sequence or makes the next word prediction from a sequence you need to really understand the
* context you need to understand
* how different words relate with each other what's the grammar of that particular language
* and only then will you be able to uh process sequences or only then you'll be able to model long sequences of textual information that's understanding number one okay with this understanding what

* We cannot just use a normal neural network because if you have a normal neural network it does not have memory so we are going to use this word memory a lot just like

* sequence translation the models need to have a memory the models need to know what has come in the
past why because let's say I have a sentence that
```
Harry Potter went to Station Number 9 3x4 uh he did this he did this Etc.
```
* and then when I come to a for a sentence which is uh three to four sentences after the first sentence which is Harry Potter came to station 9 3x4 I should not forget what came before because the station number 9 3x4 is very important for me to know even if I come at the end of the paragraph.


* so they augmented a neural network with two subm modules:
1. the first subm module is an encoder and
2. the second sub module is a decoder

***

* __Context Vector__ essentially captures meaning so it has memory and it captures meaning of okay instead of just word by. word translation what does this sentence represent and uh the encoder processes.

* case the output is the transl translated English text okay this is how the encoder decoder blocks work and uh the

* How RNNs work? mechanism which really employed the encoder decoder blocks successfully is called RNN
* (c) - so before really Transformers came into the picture RNN were was that architecture which was extremely popular for language translation and it really uh employed
the (encoder, decoder) architecture uh it was implemented in
the 1980s.

* The input text is passed to encoder what the encoder will do is that at every step it will take the input andit will maintain something which is called as the __hidden State__ this hidden state was the biggest innovation in the RNN. This hidden State essentially captures the memory so imagine the uh first input which is the first German word comes the encoder augments it or the encoder maintains a hidden State then you go to the next iteration then the second input world comes then the hidden State also gets updated so as the hidden state gets updated it receives more and more memory of what has come previously and the hidden state gets updated at each step and then there is a __final hidden State__ the final hidden state is basically the encoder output what we saw the __context Vector__ over here so when we looked at the context Vector which is passed from the encoder to the decoder. let's see over here yeah so here you see a context Vector is passed from the encoder to the decoder this context Vector is the final hidden State this is basically the encoder telling the decoder that hey I have looked at the input text uh here's the meaning of this this text here's how I encoded it here's the context Vector take this final hidden State and try to decode it and then the decoder uses this final hidden state to generate the translated sentence and it generates the translated
sentence one word at a time uh.

* so here's a schematic which actually explains this pretty well so I have an input text here so this is the first word in German the second word in German the third word in German and the fourth word in German what the encoder block will do is that it will take each input sequentially and it will maintain a different hidden state so for the first input it has the first hidden State then we move to the next iteration then the second hidden State then the third hidden State and then finally when we have the last input we have this final hidden State the

* __final hidden State__ essentially contains the __accumulation of all previous hidden state__ so it contains or encapsulates memory this is how memory is incorporated which was missing earlier with just a normal neural network so this is the final hidden State and then
this final hidden State essentially memorizes the entire input and then this uh hidden state is passed to to the decoder and then the decoder produces the final output which is the translated English.

***

20:00

* big problem with the RNN and that problem happens because the model the decoder has essentially no access to the previous hidden States.
*  Why is this a big problem um the reason it's a big problem is because when we have to process long-sequences if the decoder just relies on one final hidden State that's a lot of pressure on the decoder to essentially that one final hidden State needs to have the entire information and for long sequences it usually fails because it's very hard for one final hidden state.

* so as we saw the encoder processes the entire input text the encoder processes the entire input text into one final hidden state, which is the memory cell and then decoder takes this this hidden State. decoder takes this hidden state to essentially produce an output great now here's the biggest issue with RNN and please play pay very close
* __RNN Limitations__: attention to this point because if you understand this you will understand why attention mechanisms were needed the biggest issue with the RNN is that a RNN cannot directly access earlier hidden States.

* so the RNN can't directly access earlier hidden States from the encoder during the decoding phase it relies only on the current hidden state which is the final hidden State and this leads to a loss or this leads to a __loss of context__especially in complex sentences where dependencies might span long distances um.

*  okay so let me actually explain this further what does it mean loss of context right uh so as we saw the encoder compresses the entire input sequence into a single __hidden State Vector__.

*   Let's say if the sentence if the input sentence is very long if the input sentence is very long it really becomes very difficult for the RNN to capture all of that information in __one single final hidden state__ that becomes very difficult and this is the main drawback of the RNN.

***

* 25:00
  
* as I mentioned to you before this English sequence is pretty long uh the RNN or the encoder
really needs to capture the dependencies very well so the final hidden State needs to capture that the cat is the subject here and the cat is the one who has jumped and this information this context needs to be captured by the final hidden State and that is very hard.

* if you are putting all the pressure on one final hidden state to capture all this context especially in Long sequences so uh the key action which is jumped depends on the subject which is cat but also an understanding longer depend dependencies so jumped depends on cat but we also need to understand that the cat was sitting on the mat and the cat was also sitting next to the dog because the dog also might be referred somewhere else in the big text so we need to understand many things from this sentence and these are also called as longer dependencies so jumped the action jumped of course depends on the subject cat but we Al to understand this we also need to understand longer dependencies that the cat was sitting next to the dog and the cat was also sitting on the mat so to capture these longer dependencies or to capture this uh longer context or difficult context the
* RNN decoder struggles with this because it just has one final hidden State uh to get all the information from this is called __loss of context__ and loss of context was one of the biggest issues because of which RNN was not as good as the GPT which exist right now
which is based on the attention mechanism.

* okay so these are the issues with RNN uh the decoder cannot access
27:06
the hidden states of the input which came in earlier so we cannot capture long range
Bahdanau Attention Mechanism
27:12
dependencies this is where attention mechanism actually comes into the picture okay we will capture long range
27:18
dependencies with attention mechanisms and let's see how so
* (a) - RNN work fine for translating short-sentences they don't work for long-text because they don't have direct access to previous words in the input.
* (b) - One of the major shortcoming in the RNN is that: RNN must remember the entire encoded input in a single hidden State before passing it to the decoder.
*
* [Neural machine translation by jointly learning to align and translate -- Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=kukA0LcAAAAJ&citation_for_view=kukA0LcAAAAJ:__bU50VfleQC)

*  `Bahdanau attention mechanism for RNN`
*   difies the encoder decoder RNN such that the decoder can selectively access different parts of the input sequence at each decoding step.

***

30:00

* The text generating decoder part of the network can access all the input tokens selectively.
* This means that some input tokens are more important than others for generating a given output token.
* This importance is determined by the attention weights.

* Only 3 years later researchers found that RNN architectures are not required for building DNN for NLP and uh this is when the researchers proposed the Transformer architecture and at the main core of the
Transformer architecture was the Bahdanau attention mechanism.

 
 so it was called self


***

35:00

* [Attention is all you need - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, Illia Polosukhin](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=oR9sCGYAAAAJ&citation_for_view=oR9sCGYAAAAJ:zYLM7Y9cAGgC)

***

35:00

* At each decoding step so at each decoding step the model can look back at the entire input sequence and decide which parts are most relevant to generate the current word.

* so for example uh let's say we are predicting this word saute which is the French translation for jumped so when the decoder is predicting this French translation s the attention mechanism allows the decoder to focus on the part of the input that corresponds to jump so we can selectively look at which part of the input to give maximum attention to uh

* This __Dynamic focus__ on different parts of the input sequence allows modes to learn long range dependencies more effectively

* Dynamic Focus so Dynamic Focus which means that for every decoder for every decoding step we can selectively choose which inputs to focus on and how much attention to give to it each input that's why it's called Dynamic Focus so this Dynamic focus on different parts of the input sequence helps us to learn long-range dependencies more effectively and that's why the attention mechanism uh actually works so well.

***

40:00


|||
|---|---|
| RNN |1980|
| LSTM |1997|
| Attention |2014|
| Transformers |2017|


* RNNS had this problem which is called __Vanishing gradients__

*  when you stack multiple feedback loops together it leads to the vanishing gradient problem LSTM solved this so we had a longterm memory route and a shortterm memory route and that's how the lstm actually operates.
  
* both of these had problems with respect to __longer context__,  which was solved 2014 and 2017 papers.

#### __self attention__: 
* is a bit different so self attention um is basically amechanism that allows each position of the input sequence to attend to all positions in the same sequence so we are not looking at different sequences.
* self attention is basically different now instead of giving attention to another sequence all the attention is directed inwards so we we are just looking within a particular sequence so we are looking at different tokens within a sequence and see how these tokens are related to

***


45:00

* (f) - Self attention is a key component of contemporary LLMs bassed on the transformer architecture, such as GPT series.
*
*
*
* so large language models remember are predicting the next word in a given sentence right they can of course do language translation tasks as well but they were predominantly trained for predicting the next World and uh they are able to do translation tasks
which is also called as __emergent Behavior___ but they were trained to predict the next word and that's why they have to so.

* self attention module is a key component of contemporary LLMs

* self attention the self really refers to the attention mechanism's ability to compute attention weights by by relating different positions in a single input sequence
*  so as I mentioned we are looking at uh just one input sequence and we are we are looking at the attention between different tokens of that sequence
*  so it learns the relationship between various parts of the input itself
*   so this is different than traditional attention mechanisms like the translation task which we saw where the focus is on relationships between elements of two different sequences
*    so for example if we want to translate from English to German German to French uh English to Hindi Etc we have two sequences right so traditional attention mechanisms look at one part of the sequence and another part of the sequence and how they are related to each other
*     in self attention we basically learn the relationship between various parts of the input itself

* Lecture recap this particular lecture where we have covered so many things we covered the basic intuition of attention but before that I explain to you in a lot of detail about the shortcomings in recurrent neural networks so just one thing if you take away from this lecture is that
* __RNN have a major shortcoming__ and that shortcoming is that they have to remember the entire encoded input in a single hidden state before passing it to the decoder and that's a problem when dealing with long sentences.
*  it leads to __context loss__ because the decoder does not have access to the previous inputs this is exactly the problem which is solved by attention mechanisms.
*   in attention mechanisms when you are decoding a particular part like
*
* which parts of the output sequence are more related to which parts of the input sequence this is traditional attention
*  in self attention you just look at one sequence and you look at different parts of that same sequence and how they are related with respect to each other.

***
