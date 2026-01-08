#### positional encoding
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

35:19
embedding layer size will have one 2 3 4 rows and the number of columns will be
35:25
the vector Dimension which is 256 that is fine because for every position we are going to have a 256 dimensional
35:32
Vector remember we have to add this Vector to the Token embedding Vector so the size should be same the size here
35:39
was 256 right every uh token which was embedded had a size 256 so here also for
35:46
every position we need a 256 dimensional Vector but there are only four positions right it can either be the first token
35:52
the second token third token or fourth token so the number of rows when we create the embedding layer for
35:58
positional encoding is going to be four which is the context length so now let us write that in the code so remember
36:05
when we created the token embedding the number of rows was the vocabulary size
36:10
but here the number of rows is going to be equal to the context length so now we
36:16
are creating a embedding layer for the positional embedding number of rows is equal to the context length and the
36:22
number of columns is output Dimension which is 256 because each Vector needs to have a size of 256 great so we have
36:29
created the positional embedding layer right now and now I'm going to uh visually try to explain how we are going
36:37
to add or how we are going to essentially create the positional embedding vectors so let's look at our
36:43
input Matrix again here the batch size is eight so we have eight rows and the context length is four which that's why
36:49
we have four columns so if you look at each input sequence let's look at the first row if you look at the first row
36:55
it has four token IDs let's say those token IDs are 10 8 20 and 21 each of
37:02
these token ID is now a 256 uh Vector length Vector because we
37:07
have done the embedding we have done the token embedding so each token ID is a 256 dimensional Vector so this is the
37:14
first batch of input you can see that there are four positions here maybe these words are the cat sat on now what
37:21
we need to do is that we need to add one one uh positional encoding Vector for
37:27
each of these to this uh to the Token embedding for the 10 for the token ID 10
37:33
we have to add a positional uh embedding Vector to a token ID of 8 we have to add
37:38
another Vector to a token ID of 20 we need to add positional Vector to a token ID of 21 we need to add another
37:45
positional embedding Vector so uh we need to add one position
37:50
Vector to each of these four token embeddings and remember that the same
37:56
positional embeddings are applied uh because there are only four positions
38:01
right so uh the positional embedding we just need to do it uh once
38:08
and then for every token or for every input sequence the same four positional
38:14
embeddings can be applied so for example this is the batch one right if you look at this
38:19
batch uh which is row number three let's say there are some input IDs like one um
38:26
two five and six let's say these are the token IDs now whatever positional
38:33
positional embedding we added to the first input the same positional
38:39
embedding can be added to this input because the positions are the same either it's position one 2 3 or four we
38:45
just need to encode the different positions right so that's why uh the positional encoding size or the
38:52
positional embedding size uh has to be 4X 250 56 we only need four positional
38:59
vectors four positional embedding vectors and then each Vector will of course have size
39:05
256 so then the positional embedding Vector Matrix which we have will be a 4X
39:10
256 Matrix why do we only need four why not have a separate positional uh
39:17
embedding for each of the inputs here because for each of these inputs we only want to encode whether the token ID is
39:24
in the first position second third or fourth so we only need the positional uh
39:30
we only need four positional embedding vectors one one for the first position
39:36
one for the second one for the third one for the fourth and then we can add the same four to basically all the input
39:42
sequences in a given batch that's what we are going to do so in the next step which is the step number 13 we are going
39:49
to generate the four positional embedding vectors from the positional embedding Matrix so as I told you all
39:56
embedding Matrix are essentially lookup tables so to generate the embedding vectors we just need to pass these

***

40:01
positions 0 1 2 and 3 and then it will generate the corresponding vectors
40:08
according to that so how to pass the positions you just use tor. arranged max
40:13
length what torch. arrange max length will do is that it will create 0 1 and 1
40:19
2 and three so max length is equal to four so torch. arrange will create a sequence of number 0 1 up to Max input
40:27
length minus 1 so this will be 0 1 2 3 so essentially it will create uh the token ID 0 1 2 and 3 and then we can
40:34
just look up the positional embedding table and generate these four positional embedding
40:39
vectors so then what we do is that as I said we just look up the positional embedding layer
40:45
Matrix uh which is a lookup table and just pass in these four arguments 0 1 2
40:50
3 so then it generates four vectors each of size 256 these are the four
40:56
positional embedding vectors which we need one is for position number one second is for position number two third
41:02
is for position number three and fourth is for position number four so as we can see the positional
41:08
embedding tensor consists of four 256 dimensional vectors we can now add this
41:14
directly to the Token embedding uh so let's see how that is done uh let's look at one batch for now
41:21
and then uh let's see how to add the token embeddings with the position embeddings
41:27
so we have generated these four positional embedding vectors from the positional embedding Matrix great so we have completed step number 13 and so
Add token and positional embeddings
41:35
finally we come to step number 14 this is the last step where we have to add the position embeddings to the Token
41:40
embeddings so if you look at the token embedding matrix it's 8X 4X 256 as we had already seen before so for
41:48
each token ID there is a 256 dimensional vector and how many token IDs are there
41:54
there are eight batches and each batch has four token IDs so 8x4 that's why the size is 8x4 by 256 and then the
42:02
positional embedding we just have four 256 256 256 256 so this is just 4X
42:07
256 so for the first position we have a 256 Dimension Vector for the second
42:13
position we have a 256 Dimension Vector for the third position we have a 256 Dimension vector and for the fourth
42:19
position we have a 256 Dimension Vector so now we are adding the token embeddings with the positional
42:25
embeddings so you must be thinking this is 8x 4X 256 this is 4X 256 how does
42:32
python really add them so when you add such matrixes matrices what happens is a
42:37
broadcasting operation so what python does is that it converts this 4X 256 to
42:42
8x4 by 256 by duplicating these same values eight times so then what is
42:49
essentially happening is that to the first row these four values are added to
42:54
to the second row these same four values are added to the third row these same four values are added similarly to the
43:01
eighth Row the same four values are added so finally the input embeddings
43:06
which are the result of the token embeddings plus the positional embeddings have the size of 8X 4X 256
43:13
and these are the input embeddings which then will be the final training input to the
43:19
llms so we did so many things to reach this stage right but I hope you have understood this part I just wanted to
Lecture recap
43:25
show you for gpt2 we had a vocabulary size of around 50,000 and I showed you a
43:30
vector length of 256 so first I showed you how to create the 8X 4X 256 token
43:37
embedding uh Matrix in the first place so our initial task was for you to
43:42
understand uh how is this token embedding itself created so every token ID in a batch is converted into a vector
43:49
of size 256 and then for each position we add uh a positional uh positional
43:56
embedding so first we need to see how many positions are there and for that the parameter which becomes important is
44:02
the context length because that's the maximum length of the input to be fed at any time to the llm so only those many
44:09
positions are important so the context length is four so we then create so we
44:14
only need four positional embedding vectors one for position one second for position two third for position three
44:21
and fourth for the position number four and then we add it to the toer embedding Matrix how do we add it python does a
44:29
broadcasting so even if the token embedding Matrix is 8x 4X 256 and even
44:34
if the positional embedding is just 4X 256 it just copies these uh these four
44:40
values eight times uh so essentially what happens is that to each row of the
44:45
token embedding the same positional embedding four vectors are added and that's how we get the input embeddings
44:52
8X 4X 256 I hope you have understood these Dimensions so eight because in
44:58
each batch we have eight input sequences and in each input sequence we have four tokens that's why four and why 256
45:05
because each token ID or each token is essentially a 256 length
45:11
Vector I hope everyone has understood this lecture on positional positional embedding now let me go back to the
45:18
start where we looked at what all we have covered and what needs to be the input to the
45:23
llm so look at this diagram um
45:29
so in today's lecture we actually looked at one more step which is I would say
45:34
maybe step 3.5 in that case and that is adding positional embeddings to the token embeddings and that finally uh
45:41
leads to uh input embeddings so these input embeddings which we obtained so actually
45:48
let me add write here positional embedding so what we actually added to the Token embeddings was positional
45:54
embeddings so



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




