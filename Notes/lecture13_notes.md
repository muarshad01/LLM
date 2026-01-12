
* Transformers as the secret Source behind the llm so if Transformers is like a car then
* attention mechanism is essentially the engine which drives the car
this is the mechanism.

 
* introduction to attention mechanism what it is why it is really needed and the types of attention mechanism

* why this name attention mechanism comes and uh what are we essentially trying to solve here
so let's look at this example

```
The cat that was sitting on the mat which was next to the dog jumped.
```

a cat here the cat was sitting next to a dog and the cat was also on a mat and then uh as a human
when I read this I know that the cat jumped okay but as a LLM if you look at the sentence you'll
soon realize that this sentence is a bit confusing.


I can very clearly see that the cat was sitting on the mat if only
this were the sentence I could easily analyze that the cat is the main subject
in this sentence and the cat was sitting and the object is the mat but the thing
is when there are such complex sentences which are also called long-term dependencies where there is this second sentence which is attached so then it becomes a bit difficult for the LLM
because uh so after this sentence there will be a number of other sentences right but the
main thing which the LLM needs to really understand from this sentence is that the cat which is
the main subject that subject actually jumped so the main the action which the
subject perform is jumping so the LLM should understand that when it looks at cat the word which it should be paying the most attention to is jumped notice how I use the word attention so when I look at the word cat
um of course sitting is also important because the cat was earlier sitting on the mat but now the cat has jumped so there are few words in this sentence which the LLM needs to pay the most attention to in association with cat and if you don't introduce the
attention me mechanism it's very difficult for the LLLM to know that the cat is the one who has jumped uh maybe

***

5:07
if the attention was attention mechanism was not there the llm would have been confused and it might think oh the dog
5:13
has jumped or it might think that the main main part of this sentence is the cat is on a mat so if the attention
5:20
mechanism was not there maybe the llm would have thought that the cat is on the mat that's it it would not know that
5:27
I have to give a lot of atten ention to jump in association with the cat this is
5:33
the broad level intuition why we need to learn about the attention mechanism when
5:38
you have sentences such as this and then there is a big story after this the LM needs to analyze this sentence and it
5:45
needs to process in relation to a particular word let's say in relation to cat which other word should I pay the
5:52
most attention to and that's where the attention mechanism comes into the picture it turns out that with without
5:59
attention mechanism if you used a recurrent neural network uh or some other neural network it does not capture
6:07
the longterm dependencies between sentences that's the broad level intuition now let's dive deeper into
6:15
what all we will be covering about attention in the subsequent lectures so if you look at the attention
4 types of attention mechanism
6:21
mechanism itself there are essentially four types of attention mechanism uh the main attention
6:28
mechanism which was which is used in GPT uh generative pre-train Transformer and
6:33
all the modern llms is this multi-head attention and many YouTube videos and um
6:40
courses all many courses just directly start with multi-head attention it's a very difficult concept to understand if
6:47
you directly start learning this so you have to go in a sequential manner so what I'll be covering in this SE in the
6:53
series of lectures is first I'll start with something called simplified self attention so this is the pure EST and
7:00
the most basic form of the attention technique so that you understand what is attention then we will move to self
7:07
attention so here we will also introduce train trainable weights which form the
7:13
basis of the actual mechanism which is used in the llms until this part we are still not at the actual mechanism but we
7:20
are building up slowly after I cover self attention the next thing which I'll move to is causal
7:26
attention this is when things really start to get interesting we are predicting the next World right
7:32
by looking at the past world so what causal attention does is that it's a type of self attention uh that allows
7:39
the model to consider only the previous and the current inputs in a sequence and it masks out the future inputs no need
7:46
to uh pay too much attention to this right now I'm just giving you a broad overview of what all I'll cover in the
7:52
subsequent lectures when we look at attention today we are not going to cover all of these today we are just
7:58
going to look at uh more details about the history of how attention came into
8:03
the picture why it is needed why it's better than RNN Etc and then finally we'll move to
8:10
multi-ad attention only when you have understood causal attention and self attention and simplified self attention
8:17
you will be able to understand multi-head attention this is the main concept which is actually used in
8:22
building GPD so multi-head attention is just basically a bunch of causal attention
8:28
heads stacked together and we'll code out this multi-head attention fully from scratch I'll show
8:34
you the dimensions how they work etc all of that is planned in the subsequent
8:39
lectures so this multi-head attention is essentially an extension of self attention and causal attention that
8:46
enables the model to simultaneously attend to information from different
8:51
representation subspaces don't worry about this just remember that the multi-ad attention allows the llm to
8:57
look at input data and and then process many parts of that input data in parallel so for example if this is the
9:04
sentence the multi-ad attention allows the llm to have let's say one attention
9:09
head looks at this part one attention head looks at this part one attention head looks at this part Etc this is just
9:15
a crude description so that you get an understanding of what do you mean by multihead
9:21
attention so I just wanted to show you this overview so that you get an idea of how these four to five lectures are
9:27
actually planned um it is is impossible as I mentioned to cover all of this in one lecture and that's why I will follow
9:33
a very comprehensive approach I'll show everything on the Whiteboard and then I have this uh Google collab notebook
9:40
where everything has already been implemented and we'll go through this entire notebook see hiding future words
9:46
with causal attention and then I also have a section on U essentially multi-head attention yeah
9:52
see so at the end of these four to five lectures we'll be implementing this multi-head attention in Python and code
9:57
it out from scratch okay for now let's continue with today's lecture which is an introduction to the


***

10:04
attention mechanism and uh how researchers got to discovering attention
10:10
so let's go back in time a bit because to always appreciate something new we need to know about the history of how of
10:17
how we came to this uh Innovation so let's go right at the
Problems with modeling long sequences
10:24
start where we are modeling let's say long sequences so we have one sequence in English and uh let's say we want to
10:30
translate it to the German language so what's the problem in modeling long
10:36
sequences so let's look at this question what is the problem with architectures
10:41
without the attention mechanism which came before the llms um so for reference we'll start
10:48
with the language translation model so let's look at this this figure here um
10:53
so I have words in the English language so can you uh or let's say I have the
10:58
German inut sentence which I want to translate to English so this is the
11:03
input sentence I have the first word the second word then I have the third word Etc uh and I want to translate this into
11:11
English Okay so uh let's say we do a word by word translation if I translate
11:16
the first word to English it's can if I translate the second word to English it's you if I translate the third word
11:22
it's me the fourth word help so if you translate every German word word by word
11:29
the translation comes out to be can you me help this sentence to translate uh that's obviously not
11:36
correct right so the main takeaway here is that the word by word translation does not work and uh you can also see
11:44
this in Hindi so if the main text is in English so can you help me and if you
11:49
want to translate it in Hindi uh so the Hindi translation
11:55
is so is associated with can that's fine you is associated with tum but Mary is
12:01
the third word in h in the Hindi translation right but it's actually the fourth word in the English translation
12:08
similarly madat is the fourth word in the Hindi translation but help is
12:14
actually the third word in the English so the main point here is that
12:20
uh word by word translation does not work in this case and uh that was a
12:25
major realization when people started modeling long sequence es and this is a
12:31
general problem when you deal with sequences you cannot just do word by word translation you need contextual
12:37
understanding and grammar alignment so whenever you are developing a model let's say which translates one
12:44
sequence to another sequence or tries to find the meaning of a sequence or makes the next word prediction from a sequence
12:51
you need to really understand the context you need to understand how different words relate with each other
12:57
what's the grammar of that particular language and only then will you be able to uh process
13:04
sequences or only then you'll be able to model long sequences of textual
13:10
information that's understanding number one okay with this understanding what
13:15
people realized is that we cannot just use a normal neural network uh because if you have a normal neural network it
13:22
does not have memory so we are going to use this word memory a lot just like
13:27
humans have memory we store information about the past uh in order to do a good
13:32
job in sequence to sequence translation the models need to have a memory the models need to know what has come in the
13:39
past why because let's say I have a sentence that Harry Potter went to Station Number 9
13:44
3x4 uh he did this he did this Etc and then when I come to a for a sentence
13:51
which is uh three to four sentences after the first sentence which is Harry
13:57
Potter came to station 9 3x4 I should not forget what came before because the station number 9 3x4 is very important
14:04
for me to know even if I come at the end of the paragraph So if I'm making some prediction at the end of the paragraph
14:10
and the word station comes over there I need to go back to the start I need to have memory of what came at the start
14:17
that it was the station number 9 3x4 and this happens a lot with textual data if you want to have meaningful
14:25
outcomes in terms of text summarization next word prediction language translation you definitely need to have
14:32
understanding of the meaning and for that you need the model to retain the memory so uh to address this issue that
14:40
word by word translation does not work in this particular case of translation uh people realize that a normal neural
14:47
network will not work so they augmented a neural network with two subm modules the first subm module is an encoder and
14:54
the second sub module is a decoder so what the encoder does is that uh in in
14:59
the example which we saw it will receive the German text and it will read and process the
15:05
German text and then it will pass it to the decoder and then the decoder will translate the German text back into
15:12
English this is the simplest explanation of the encoder decoder and there's a nice animation here which actually shows
15:20
uh how the encoder decoder works so here you can see the input sequence comes in the German language it goes to the
15:26
encoder uh a context is generated by the encoder it's called as a context Vector
15:32
the context Vector essentially captures meaning so it has memory and it captures meaning of okay instead of just word by
15:40
word translation what does this sentence represent and uh the encoder processes
15:47
the entire input sequence and sends the context over to the decoder so let me play this again the input sequence comes
15:53
to the encoder it generates a context Vector which basically encodes meaning
15:58
and then the encoder transfers the context Vector to the decoder and the decoder generates the output in this
16:04
case the output is the transl translated English text okay this is how the encoder decoder blocks work and uh the
How RNNs work
16:12
mechanism which really employed the encoder decoder blocks successfully is called recurrent neural networks so
16:20
before really Transformers came into the picture recurrent neural networks were was that architecture which was
16:27
extremely popular for language translation and it really uh employed
16:32
the encoder decoder architecture uh it was implemented in
16:37
the 1980s so let's look a bit more at how the RNN actually works because when
16:43
we if we understand how RNN works that's when we'll understand the limitations of recurrent neural networks and that's
16:49
when we will really appreciate why the attention mechanism needed to be discovered so here's how the encoder
16:57
decoder in the RNN actually works what happens is that you first receive an input text okay uh and that let's say
17:04
is the German uh German text the input text is passed to the decod to the
17:09
encoder what the encoder will do is that at every step it will take the input and
17:15
it will maintain something which is called as the hidden State this hidden state was the biggest innovation in the
17:21
recurrent neural networks this hidden State essentially captures the memory so imagine the uh first input
17:28
which is the first German word comes the encoder augments it or the encoder
17:34
maintains a hidden State then you go to the next iteration then the second input world comes then the hidden State also
17:40
gets updated so as the hidden state gets updated it receives more and more memory of what has come
17:47
previously and the hidden state gets updated at each step and then there is a final hidden
17:52
State the final hidden state is basically the encoder output what we saw the context Vector over here so when we
18:00
looked at the context Vector which is passed from the encoder to the decoder let's see over here yeah so here you see
18:05
a context Vector is passed from the encoder to the decoder this context Vector is the final hidden State this is
18:12
basically the encoder telling the decoder that hey I have looked at the input text uh here's the meaning of this
18:18
this text here's how I encoded it here's the context Vector take this final hidden State and try to decode it and
18:25
then the decoder uses this final hidden state to generate the translated sentence and it generates the translated
18:31
sentence one word at a time uh so here's a schematic which actually explains this pretty well so I
18:38
have an input text here so this is the first word in German the second word in German the third word in German and the
18:45
fourth word in German what the encoder block will do is that it will take each
18:50
input sequentially and it will maintain a different hidden state so for the first input it has the first hidden
18:56
State then we move to the next iteration then the second hidden State then the third hidden State and then finally when
19:02
we have the last input we have this final hidden State the final hidden State essentially contains the
19:07
accumulation of all previous hidden state so it contains or encapsulates memory this is how memory is
19:14
incorporated which was missing earlier with just a normal neural network so this is the final hidden State and then
19:21
this final hidden State essentially memorizes the entire input and then this
19:26
uh hidden state is passed to to the decoder and then the decoder produces
19:32
the final output which is the translated English sentence so I want to show you another animation of this so that you
19:38
understand it much better um so here's how the RNN actually works right so see
19:44
the input Vector number one is the first word in German which needs to be translated so here you will see in this
19:51
animation how the hidden state gets updated so the first word of German comes then the RNN maintains the hidden
19:58
state zero and here you see uh the hidden state is the hidden
20:04
state zero and the input one is used to produce the output one and then we also
20:09
have a hidden State one then as we move further we have the hidden state two hidden state three hidden State four and
20:15
final hidden State when the last word needs to be processed uh actually I can show you
20:22
this again here so this is from a French to English translation using the recurrent neural network look look here
20:28
so we have French input which is coming here justu then uh yeah so the first word
20:34
goes into the encoder see now we have hidden so let me expand it so and play
20:39
from the start okay so now the first word of French which is J goes into the encoder it has went into the encoder
20:46
right now and the first hidden state is generated see in the orange color great now this hidden state number one and the
20:53
second input is used again look at this animation again the hidden state state
20:58
number one and the second input which is sui sis s is used U and then we have the
21:04
hidden state number two then the hidden state number two and the input three which is will be used to produce the
21:10
final hidden State great this final hidden State hidden state number three essentially contains all the information
21:17
in the given sequence plus it also contains some memory or some
21:23
context regarding uh what came in the past and now this final hidden state is
21:29
then passed to the decoder and the decoder produces the output in English one word at a time this is exactly how
21:35
the recurrent neural network works okay now you might think that awesome right this is already doing sequence to
21:42
sequence translations and we are translating from one language to another language so why do we need attention
21:47
mechanism memory is being encoded here and we are passing in the context which
21:53
means that we will be able to identify how different words of the sequence are related to each other so why do we need
21:59
attention well there is a big problem with the recurrent neural network and
22:05
that problem happens because um the model the decoder has
22:11
essentially no access to the previous hidden States so if you look at this video you'll see that uh the decoder has
22:19
access to only the final hidden state so hidden State one hidden State 2 hidden
22:24
state three and then hidden state three is passed to the decoder C the the decoder has no access to the previous
22:30
hidden States now why is this a big problem um
22:37
the reason it's a big problem is because when we have to process long
22:43
sequences if the decoder just relies on one final hidden State that's a lot of
22:49
pressure on the decoder to essentially that one final hidden State needs to
22:54
have the entire information and for long sequences it usually fails because it's very hard for one
23:01
final hidden state to have the entire information let me explain this bit more
23:07
so as we saw the encoder let me change my color here I think let me change it to
23:15
Green yeah so as we saw the encoder processes the entire input text the
23:21
encoder processes the entire input text into one final hidden state which is the memory cell and then decoder takes this
23:29
this hidden State decoder takes this hidden state to essentially produce an output
23:34
great now here's the biggest issue with RNN and please play pay very close
RNN Limitations
23:39
attention to this point because if you understand this you will understand why attention mechanisms were needed the
23:46
biggest issue with the RNN is that a recurrent neural network cannot directly access earlier hidden States as we saw
23:52
in the video it only accesses the final hidden state so the RNN can't directly
23:58
access earlier hidden States from the encoder during the decoding phase it relies only on the current
24:04
hidden state which is the final hidden State and this leads to a loss or this leads to a loss of context especially in
24:12
complex sentences where dependencies might span long distances um okay so let me actually
24:19
explain this further what does it mean loss of context right uh so as we saw
24:25
the encoder compresses the entire input sequence into a single hidden State
24:31
Vector I hope you have understood up till this point now the problem happens let's say if the sentence if the input
24:37
sentence is very long if the input sentence is very long it really becomes very difficult for the recurrent neural
24:43
network to capture all of that information in one single final hidden state that becomes very difficult and
24:50
this is the main drawback of the RNN so for example let's take a practical case
24:56
so let's say uh we take the example which we looked at at the start of the lecture which is the cat that was
25:01
sitting on the mat which was next to the dog jumped and let's say we want to convert this English into a French
25:08
translation okay so the French translation will be lat whatever I cannot spell this out fully but this
25:14
will be the French translation for this English sequence now as I mentioned to you before this English sequence is
25:20
pretty long uh the RNN or the encoder
25:25
really needs to capture the dependencies very well so the final hidden State needs to capture that the cat is the
25:32
subject here and the cat is the one who has jumped and this information this
25:37
context needs to be captured by the final hidden State and that is very hard if you are putting all the pressure on
25:44
one final hidden state to capture all this context especially in Long sequences so uh the key action which is
25:52
jumped depends on the subject which is cat but also an understanding longer
25:57
depend dependencies so jumped depends on cat but we also need to understand that the cat was sitting on the mat and the
26:04
cat was also sitting next to the dog because the dog also might be referred somewhere else in the big text so we
26:10
need to understand many things from this sentence and these are also called as longer
26:16
dependencies so jumped the action jumped of course depends on the subject cat but we Al to
26:24
understand this we also need to understand longer dependencies that the cat was sitting next to the dog and the
26:29
cat was also sitting on the mat so to capture these longer dependencies or to capture
26:36
this uh longer context or difficult context the RNN decoder struggles with
26:41
this because it just has one final hidden State uh to get all the
26:46
information from this is called loss of context and loss of context was one of the biggest
26:53
issues because of which RNN was not as good as the GPT which exist right now
26:59
which is based on the attention mechanism okay so these are the issues with RNN uh the decoder cannot access
27:06
the hidden states of the input which came in earlier so we cannot capture long range
Bahdanau Attention Mechanism
27:12
dependencies this is where attention mechanism actually comes into the picture okay we will capture long range
27:18
dependencies with attention mechanisms and let's see how so RNN work fine for
27:24
translating short sentences and they did work amazingly actually for quite a while for short sentences but
27:31
researchers soon discovered that they don't work for long text because they don't have direct access to previous
27:37
words in the input so when an RNN decoder only receives the final hidden state right
27:45
they don't even have access to the the decoder does not have access to all the prior Words which came in the input so
27:51
let's say I'm decoding uh uh let's say I'm looking at this word
27:57
um jumped right let's say I'm looking at the word jumped cat is a word which has come way prior in the sequence so when I
28:04
am looking at the word jumped I need to give a lot of attention to the word cat but an RNN gets the entire encoded
28:11
version of this sentence so how would the RNN know that jump actually if you're looking at jumped you should pay
28:16
a lot of attention to the word cat it does not even have access to this input Vector for cat this is where attention
28:23
mechanism actually comes into the picture uh okay so as I said one of the
28:30
major shortcoming in the RNN is that the RNN must remember the entire encoded input in a single hidden
28:37
State before it passes the encoded input to the decoder the RNN has to remember
28:44
the entire encoded input in a single hidden state I'm repeating this again
28:49
because unless you understand this you won't understand why we are learning about attention and it's very hard for
28:56
the RNN to encode the entire or to remember the entire encoded input in a single hidden
29:02
State this is when the researchers started looking at other mechanisms and that's when in 2014 researchers
29:09
developed the so-called bhano attention mechanism for RNN so when people think of attention
29:15
mechanism they always think of the 2017 paper right attention is all you need
29:20
but actually attention was introduced in a paper in 2014 which was called neural machine translation by jointly learning
29:28
to align and translate um sadly many people don't remember this now everyone just
29:34
remembers attention is all you need but remember that this these authors badano
29:40
Benjo and Kung yuno they were the ones who worked on the first proposition of
29:46
the attention mechanism so I just want to give uh credit here and this attention mechanism
29:53
was called badan attention mechanism because the author of this paper was was last name was
29:59
Bano so what was the main idea behind this what the attention mechanism
30:04
basically uh prescribed is that okay let's take
30:09
the encoder decoder RNN and let's modify the encoder decoder RNN so that the
30:16
decoder can selectively access different parts of the input sequence at each
30:21
decoding step uh let me repeat that remember in
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

