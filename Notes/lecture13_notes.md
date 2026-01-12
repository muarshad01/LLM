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
*
*
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

30:21
 uh let me repeat that remember in
30:27
the original and and the decoder only had access to the final hidden state but the bhano attention mechanism says that
30:34
what if the decoder now can selectively access different parts of the input
30:40
sequence at each decoding step and let me explain to you what this means by simple figure so let's say we are at
30:47
this decoding step where uh we want to so let's say the word is Mir which is
30:52
the uh German word and we want to decode this so we are at this hidden state
30:59
right now and we want to decode this okay if we use the original RNN we just had access to this final hidden state
31:06
but now what we say is that when you are uh decoding this uh hidden State what if
31:13
you have access to all of the input tokens so follow this orange curve which I'm drawing here so what if you have
31:20
access to this token what if you have access to this token what if you have access to this token and what if you
31:27
have access access to this token so let's say when you're decoding you have access to all the tokens and you can
31:33
decide how much attention to pay to each token so for example I think the German
31:38
uh translation for you is do which is do so I know that the maximum attention needs to be paid to this token so that's
31:46
why you I have marked this with a thick line over here for all the other tokens well we pay less amount of
31:53
attention what this does is that it allows the decoder to access all the to tokens so if we are dealing with long
31:59
sentences we can access all the tokens even in long sentences and decide which token we want to pay more attention to
32:06
so for example let's take this longer sentence which we have looked at a lot in today's lecture yeah so let's
32:14
say and we'll look at it once more again uh yeah so let's say we have look we are
32:20
looking at this sentence the the cat that was sitting on the mat
32:25
which was next to the dog jumped let's say we looking at this sentence now in the decoder so let's say you are on the
32:32
decoder part and you are decoding for jumped and you translating it into French what we will do is that instead
32:38
of just looking at the final hidden State we will have access to all of the words let's say we have access to all of
32:44
the input words and not just that we have access to all of the input words and we can also decide how much
32:50
attention to pay to each of the input word so now I will say that okay I want to
32:56
translate jump right so of course a lot of attention should be paid to jumped but I will also pay a lot of attention
33:02
to cat because the cat is the uh one who has really jumped so I can decide which
33:09
tokens to pay the maximum attention to so when I'm translating jumped I will pay attention to jump and I will pay
33:15
attention to cat also because now I can access the token for cat this access itself was not possible in RNN because
33:22
we could not access the previous input tokens in an RNN and this was the main problem which was solved by the
33:27
introduction of the attention mechanism so this figure actually explains the general idea behind the
33:34
bhan attention mechanism um let me Zoom onto this figure once more yeah so if you look at
33:41
this figure here uh when you are translating from do to English you pay
33:46
attention to all the input tokens and then you can have these attention weights which basically describe how
33:51
much you want to pay attention to each input token and that solves the problem of the loss of context which was present
33:58
in the RNN so essentially using an attention mechanism using an attention mechanism
34:05
the text generating decoder part of the network can access all the input tokens
34:12
selectively uh and as I mentioned we can decide how much weight to give to each input token so this means that some
34:19
input tokens are more important than others for generating a given output token this importance is determined by
34:26
the attention weights so this is where the attention weights and the attention score comes into the picture which we
34:31
will learn about later now uh so this paper was introduced in 2014 right this paper
34:39
which is which introduced the bhan attention mechanism only three years later the main Transformers paper came
34:45
out and that came out in 2017 so only 3 years later researchers
34:50
found that RNN architectures are not required for building deep neural networks for natural language processing
34:58
and uh this is when the researchers proposed the Transformer architecture and at the main core of the
35:04
Transformer architecture was the badano attention mechanism so it was called self


***

35:10
attention mechanism but it was really inspired a lot from the bhano attention mechanism we'll come to what self
35:17
attention means bhano attention did not introduce the term self attention but the Transformers paper which really
35:23
changed everything for llms that paper came in 2017 uh it introduced the Transformer
35:29
architecture with a self attention mechanism that was completely based on the badano attention mechanism this is
35:35
that paper which came out in 2017 and here you'll see uh attention
35:41
attention basically the paper itself is titled attention is all you need um so
35:46
this is how attention really came to be the core building block of large language models I just want to explain
35:53
to you this once more so that I drive this concept home about what why attention mechanism helps so let's say
36:00
again the same sentence this is the sentence the cat that was sitting on the mat which was next to the dog jumped and
36:06
this is the French translation uh for this particular sequence so what the
36:12
attention mechanism does is that at each decoding step so at each decoding step
36:18
the model can look back at the entire input sequence and decide which parts are most
36:25
relevant to generate the current word so for example uh let's say we are
36:33
predicting this word sa which is the French translation for jumped so when the decoder is predicting this French
36:40
translation s the attention mechanism allows the decoder to focus on the part
36:45
of the input that corresponds to jump so we can selectively look at which
36:51
part of the input to give maximum attention to uh and this Dynamic focus on
36:57
different parts of the input sequence is what helps the attention mechanism to learn long range dependencies more
37:03
effectively so remember we looked at how RNN fail to understand or even learn
37:08
longrange dependencies uh one main or key thing of the attention mechanism is this word
37:15
which is called as Dynamic Focus so Dynamic Focus which means that
37:20
for every decoder for every decoding step we can selectively choose which inputs to focus on and how much
37:26
attention to give to it each input that's why it's called Dynamic Focus so this Dynamic focus on different parts of
37:32
the input sequence helps us to learn long range dependencies more effectively and that's why the attention
37:40
mechanism uh actually works so well there is another animation which I want to show you uh so here we saw the RNN
37:49
right let me show you again so this is the recurrent neural network here you will see that just the final hidden
37:55
state which is the hidden state number three is actually passed to the decoding stage nothing else is passed but now let
38:02
me show you the modification of this with with the attention mechanism so now
38:07
uh if we do this with attention added what happens is that you'll see that hidden state number one is generated
38:13
hidden state number two is generated hidden state number three is generated but all of these hidden states are
38:18
passed to the decoder which means that the decoder at every step has access to all of the Hidden States not just the
38:25
final hidden state number three but even hidden States number one and hidden States number two this is an extremely
38:30
important point to remember uh one more thing which I want to show you here is that uh yeah this
38:38
part so let's say uh uh yeah so let's say we are
38:44
translating from French to English over here I think let me play this
38:56
again yeah yeah so let's say we are translating from French to English here again let's see what is happening here
39:02
so if we are at the decoder stage we want to generate this word I right for Jo so this when we are at the decoder
39:09
State the decoder has access to all of these hidden States 1 2 and three when it's generating the English translation
39:15
for J but what it does is that it gives more preference to the hidden state number one because it realize that that
39:22
Jo is more important for translation to I similarly when we want to translate sui s u i s it actually translates to M
39:30
so when the decoder is translating this to English it has access to Hidden State 1 2 and three but it learns that the
39:36
hidden state two is most important for this translation similarly when it's translating Aon with the attention
39:43
mechanism it has translate it has access to one two and three hidden state but it learns that the hidden state number
39:48
three is the most important uh Etc so this is how it actually works so at
39:53
every decoding step we have access to all the different uh hidden States and one more key thing to mention
40:00
here is that the model is isn't just mindlessly aligning the first word at the output with the first word at the
40:06
input it actually learns from the training phase how to align the words in the language pair so remember at the
40:13
start we saw that you cannot blindly do word to word translation that's not what's happening here in the training
40:20
phase the attention mechanism actually learns which word to align with which word so which word in the output should
40:25
be aligned with which word in the input so this this graph is actually presented
40:31
in the 2014 paper and I think this graph is very interesting so you can also take a look at this uh this is figure number
40:38
three in the original the badana attention mechanism paper so here you can see that let's see European economic
40:45
area so let's say we want to translate European economic area to French it actually translates to uh European
40:53
economic zone so here it says that in French the order of this word is reversed as compared to English so let's
41:00
see how the model actually learns so European so it learns this European happens over here then economic is here
41:07
and area is here so basically the point is that uh when the attention mechanism
41:13
learns that European maps to this European economic maps to economic and
41:19
area maps to Zone it does it's it's not just blindly looking at the order because if you see the order European
41:25
comes at 1 2 3 4 fth but European which is the French translation comes at 1 2 3
41:32
4 5 6 7th so order is different it's not blindly copying the order or the number
41:39
uh at which the word occurs it's actually learning from the training uh it's learning the dependencies between
41:45
different words and that's how it does does the translation um I'll be sharing the link
41:51
to this paper in the information section the 2014 badano attention paper and I'll
41:57
also be sharing the link to the 2017 attention is all you need paper before we move to self attention I actually
History of RNNs, LSTMs, Attention and Transformers
42:03
want to take a bit of time to tell you about the history of the attention and
42:09
Transformers so that you have a timeline in mind so uh it all started in 1980 1980s
42:16
is when recurrent neural networks came into the picture and recurrent neural networks introduced this hidden layer
42:23
the hidden state that was the key Innovation here in 1997 came long short-term memory networks and they were
42:30
an advancements compared to lnn uh compared to RNN so recurrent neural networks had this problem which is
42:36
called Vanishing gradients so when you stack multiple feedback loops together it leads to the vanishing gradient
42:42
problem long short-term memory solved this so we had a longterm memory route
42:48
and a shortterm memory route and that's how the lstm actually operates but still both of these had problems with respect
42:54
to longer context which was solved by the B Z Now attention mechanism in 2014
42:59
where the decoder can can have access to each of the input State when it's performing the decoding operation and it
43:06
can selectively decide which input to give more attention to then came the
43:12
Transformers architecture in 2017 and the Transformers architecture really
43:17
used the attention mechanism which was proposed in 2014 in this paper badana
43:24
attention mechanism so this is the history we we are living in the age right now where people only know about attention and Transformers right but
43:31
work has been going on in this area for the past 43 to 45 years or even more uh
43:38
so please keep this particular uh figure in mind or historical mind map so that
43:45
you are aware of how lucky we are to be living in these times where all of the research work has been accumulated for
43:51
50 years and now we are reaping the benefits of these scientific advancements these advancements have not
43:57
just happened in a period of one year or two years or 3 years they have taken time they researchers have worked hard
44:03
over the past half a century to get us to this statee okay so I hope until now you have
Self attention
44:10
all understood a flavor and an intuition behind the attention mechanism itself um
44:16
what the Transformers paper did is that it introduced another terminology which is actually called as self attention so
44:22
self attention is a bit different so self attention um is basically a
44:28
mechanism that allows each position of the input sequence to attend to all
44:33
positions in the same sequence so we are not looking at
44:38
different sequences now when we looked at language translation we looked at converting from English to French or
44:44
German to English right so we are looking at one sequence and we are looking at another sequence self attention is basically different now
44:51
instead of giving attention to another sequence all the attention is directed inwards so we we are just looking within
44:58
a particular sequence so we are looking at different tokens within a sequence and see how these tokens are related to
45:04
each other that's called as self attention so self attention is a key
45:09
component of contemporary uh large language models so large language models
45:15
remember are predicting the next word in a given sentence right they can of course do language translation tasks as
45:21
well but they were predominantly trained for predicting the next World and uh they are able to do translation tasks
45:28
which is also called as emergent Behavior but they were trained to predict the next word and that's why
45:34
they have to so let's say when you want to predict the next word in a sentence you need to know that how different
45:41
words are related to each other so let's say the cat jumped on a wall next to the dog the dog also jumped something like
45:48
that you need to know when you look at cat which are the other words in this sentence which you should pay the most
45:53
attention to for dog which are the words you should pay the most attention to So within a sentence itself you are
46:00
deciding how different words of the sentence are related to each other and which word you should pay more attention
46:06
to when looking at a particular word that's called self attention and self attention module is a
46:13
key component of contemporary large language models so if you look at uh how
46:18
this Series has progressed up till now we have looked at pre-processing steps like uh tokenization word embedding
46:25
positional embedding but now we have started looking at the llm architecture itself right and one of the key points
46:32
here is the self attention module without really understanding attention and self attention we cannot proceed to
46:38
the llm training which comes in uh phase number two of building a large language
46:44
model okay so one last thing to mention is that uh in self attention the self
46:51
really refers to the attention mechanism's ability to compute attention weights by by relating different
46:57
positions in a single input sequence so as I mentioned we are looking at uh just
47:03
one input sequence and we are we are looking at the attention between different tokens of that
47:09
sequence so it learns the relationship between various parts of the input
47:14
itself so this is different than traditional attention mechanisms like the translation task which we saw where
47:21
the focus is on relationships between elements of two different sequences so for example if we want to translate from
47:27
English to German German to French uh English to Hindi Etc we have two sequences right so traditional attention
47:34
mechanisms look at one part of the sequence and another part of the sequence and how they are related to each other in self attention we
47:41
basically learn the relationship between various parts of the input
47:47
itself so here we learn the relationship between various parts of the input itself uh and that's the difference
47:54
between traditional attention mechanisms and self attention ention mechanisms okay so this is the end of
Lecture recap
48:02
this particular lecture where we have covered so many things we covered the basic intuition of attention but before
48:08
that I explain to you in a lot of detail about the shortcomings in recurrent neural networks so just one thing if you
48:16
take away from this lecture is that recurrent neural networks have a major shortcoming and that shortcoming is that
48:23
they have to remember the entire encoded input in a single hidden state before passing it to the decoder and
48:29
that's a problem when dealing with long sentences it leads to context loss because the decoder does not have
48:36
access to the previous inputs this is exactly the problem which is solved by attention mechanisms in attention
48:42
mechanisms when you are decoding a particular part like let's say when you are translating do which is German for
48:49
you into English the decoder has access to all of the tokens all of the inputs
48:55
and then it decides how much attention to give to each input this small Insight
49:00
leads to a revolution in GPT and as you can see chat GPT is so awesome the small
49:05
Insight is at the heart is the engine of all that awesomeness now what I've demonstrated
49:11
here is an example of traditional attention because you are looking at one sequence you looking at this sequence
49:17
which is in German and then you are looking at this sequence and then you are seeing which parts of the output
49:23
sequence are more related to which parts of the input sequence this is traditional attention in self attention
49:29
you just look at one sequence and you look at different parts of that same sequence and how they are related with
49:34
respect to each other uh as I already mentioned to you before I have planned three to four more
49:40
lectures which are coming up in the attention series and what we are going to do in this lectures is that we are
49:46
going to uh first start in the next lecture with a simplified self attention uh we'll start with the
49:53
simplified self attention and we'll code it out in Python then we will move to self attention we'll code it in Python
50:00
then we'll move to causal attention we'll code it out in Python and then we'll move to multi-head attention and
50:05
we'll code it out uh and we'll code this also in Python so all of the different
50:11
modules I'll be showing the mathematical formulations on the Whiteboard and the rest will code it in Python so in the
50:18
next class let me show you where we'll begin with in the next class we'll begin with this this part which is
50:24
implementing self implementing a simplified attention mechanism uh so I also have whiteboards whiteboard notes
50:31
for this ready and then I'll take you through code remember this is a very serious lecture Series where I I don't
50:37
just want to give you the code directly I want to teach you the theory make your foundation strong and build everything
50:44
from scratch in code we can always use a package for this Lang chain is there so
50:49
many other packages are there we don't even need to uh go into the attention mechanism details if we directly use the
50:56
package but I don't think that's a good way to learn about large language models the best way is to learn from Basics to
51:02
understand the nuts and bolts of how things work and to build things out from scratch
51:08
yourself uh I hope you all are enjoying this series um thanks a lot everyone there are number of exciting lectures
51:14
which are planned ahead and I look forward to seeing you all in those lectures

***









