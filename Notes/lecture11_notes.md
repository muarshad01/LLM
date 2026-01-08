#### Positional Encoding
* three modules
* positional encoding layer
* along with the token embedding layer

* (4) It is helpful to inject additional position information to LLM.
* (5) There are two types of positional embeddings:
* (5.1) Absolute positional embeddings
* For each position in the input sequence, a unique embedding is added to the token's embedding to convey its exact location.
* (5.2) Relative positional embeddings
* In this type of embedding the emphasis is on the relative position or the distance between the tokens. Th model essentially learns the relationships in terms of "how far apart" rather than at which "exact position".
* Advantage: model can generalize better to sequence of varing lengths, even if it has not seen such lengths during training.

***

* They enable the large language models to understand the order and relationship between the tokens and
this actually ensures more accurate and context aware predictions.
* The choice between the two types of uh positional embedding really depends on the specific application and the nature of the data being processed.

* Absolute: Suitable when fixed order of tokens is crucial, such as for sequence generation.
* Relative: Suitable for tasks like language modeling over long sequences, where the same phrase can appear in different parts of the sequence.

* (7) OpenAI's a GPT models use absolute positional embeddings. This optimization is part of the training model itself.

* sinusoidal and cosine formula over here uh so you can read a bit about what they have written
here since our model contains no recurrence in order for the model to make use of the order of the sequence we must inject some information about the relative or absolute position of the to tokens so they have added absolute

***

20:00

***

25:00

***

30:00

* 8 X 4 X 256 dimensional tensor
* Generate positional embeddings
* similar to the  token embedding, we also have to create another embedding layer for the positional encoding.
* 4 X 256 dimensional tensor

***

35:00

* we only need four positional vectors four positional embedding vectors and then each Vector will of course have size 256 so then the
* positional embedding Vector Matrix which we have will be a 4 X 256 Matrix

***

40:00

***

46:00
uh so what we added here was yeah positional embeddings and then
46:06
this resulted into input embeddings so token embeddings plus positional
46:11
embeddings is input embeddings and then these are the ones which are actually used as input to the GPT so essentially
46:18
what we did in token embedding and positional embedding is we try to exploit as many things in textual
46:23
language as possible the first thing which we exploited semantic meaning when we did token embeddings when we did
46:29
positional embeddings we exploited the fact that the different positions also mean something until now we have not
46:36
seen how exactly to obtain the values in the positional embedding see even in today's lecture we just randomly
46:42
initialized uh the embedding Matrix right like if you see in the code this positional embedding layer is randomly
46:49
initialized so what what tor. nn. embedding does is that it initializes a matrix with number of rows as context
46:56
length number of columns as output Dimension and it puts random values in this so then how do we know the actual
47:03
values so that is actually a part of the llm training process and it's exactly similar for
47:09
the uh token embeddings as well so we need to optimize the values in the token embedding layer and we need to optimize
47:16
the value in the positional embedding layer however for us to reach the optimization stage stage first it's very
47:22
important for you to understand what exactly is positional embedding what exactly is token embedding and that was
47:29
the whole purpose of today's lecture uh this brings us to the end of
47:35
today's lecture thank you so much everyone I hope you understood a lot I hope you understood the difference
47:40
between absolute um absolute and positional
47:46
embedding yeah absolute and relative sorry absolute and relative positional embedding I hope you understood uh why
47:54
positional embeddings are added and most importantly I hope you understood the dimensions ultimately I feel it all
48:00
comes down to Dimensions people who understand the dimensions really don't feel scared or intimidated by the
48:05
subject so if someone understands where this 8X 4X 256 is actually coming from
48:11
right this 8x4 by 256 then I feel they will have a much stronger grasp on the subject so that's why today I spent a
48:18
lot of time on explaining these Dimensions um thank you so much everyone
48:23
and if you have any doubts or questions please put it in the comment section the lectures are getting bit more involved
48:28
and detailed now so I'll be happy to interact in the comment section and solve any doubts or questions also let
48:35
me know if you're liking this teaching style which is a mix of the Whiteboard lectures uh plus the Hands-On coding
48:41
I'll of course be sharing the code file with all of you thanks everyone and I'll see you in the next lecture

***
