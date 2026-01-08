Lecture agenda
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from
0:11
scratch Series today we are going to be looking at a very important topic and
0:17
that is called as positional encoding so I've divided today's lecture
0:23
into two to three modules initially we'll look at what all we have covered so far then we will try to understand
0:30
why is positional encoding really needed and then finally we will do a Hands-On
0:36
coding exercise in Python where we'll actually add positional encoding layer
0:42
along with the token embedding layer which we have already created so far so
0:47
let's get started if you have not seen the previous lecture on token encoding or token embedding I would highly
0:54
encourage you to see that lecture because positional encoding Builds on top of token and coding if you are
1:01
coming to this lecture series for the first time welcome I have designed the lectur so so that they are good to watch
1:07
even as Standalone lectures but of course if you go through the previous lectures your understanding of the
1:13
current one will be much better so let's get started until now we have looked at
1:20
something which is called as token embedding and what is meant by token embedding so token embedding is the step
1:27
number three in the llm train training process so first we have we start with
1:32
an input text then we convert that text into tokens so for example if the text
1:38
is this is an example the tokens can be this is an example this what I'm showing
1:45
right now is an example of a word based tokenizer but there are other tokenizers
1:51
such as subword based tokenizer or bite pair encoding tokenizer which actually GPT uses so remember that one word is
1:59
generally not equal to one token you can even have subwords or characters as tokens for the case of
2:06
Simplicity I'm just showing here the token X text to be individual words so
2:11
that's step number one step number two is maintaining a vocabulary of the tokens sorting them in ascending order
2:18
and assigning a token ID to each token that's step number two but even then we
2:24
are not ready to give the token IDs as input for the GPT or for the LS M
2:30
training there is a very crucial step number three which we looked at in the previous lecture and that's called token
2:36
embeddings basically the token IDs which we obtained are converted into
2:42
vectors uh so that the semantic meaning between the different words is retained
2:47
if you directly use the token IDs as inputs to the large language model we do not retain the meaning between words so
2:54
for example dog and puppy are related to each other cat and kitten are related to to each other if we don't convert token
3:02
IDs into token embeddings this semantic relationship or this meaning between the
3:08
words is lost that's why it's very important to have step number three so if you have a
3:14
vocabulary uh with GPT uses let's say of around 50,000 words gpt2 was trained on
3:20
a vocabulary of 50,000 Words which means of 50,000 tokens which means we have
3:25
50,000 token IDs and then there is a vector corresponding to each token ID that is uh called as token embeddings
3:34
every token is converted into a vector in a higher dimensional space so now you
3:40
must be thinking okay so now I have these vectors which capture the semantic relationship uh between words and now I
3:46
can feed these vectors as inputs to the large language model right almost we are
3:51
almost there I promise there is just one last step remaining in the data pre-processing part and then in the next
3:59
lecture on will come to training the llm we'll see the attention mechanism Etc and the last step is called as
4:05
positional embedding so why is this needed so let's say we have a sentence
4:11
the cat sat on the mat and then another sentence on the mat the cat sat so if
4:16
you look at both of these sentences the cat cat word appears in both of these sentences right and in both of these
4:23
sentences cat is a token that will be assigned a token ID and uh and that will be converted
4:30
into a vector so the vector for CAT will be the same in both of these sentences
4:37
although the position of the cat in these two sentences is different so until now if we just use the vector
4:43
embedding we do not have any information about the position of the particular
4:49
word so in the embedding layer in the token embedding layer whatever we have seen up now the same token ID gets
4:56
mapped to the same Vector representation so the token ID for CAT will get mapped
5:03
to the same Vector regardless of where the token ID is positioned in the input
5:08
sequence so for example cat the position of a cat is different in these two sentences right but if we don't do
5:16
anything about the position if we don't encode the position and if we just use the embedding layer we learned
5:21
previously both of these tokens cat in both of the sentences will have the
5:27
exact same Vector representation so in the embedding layer which we have seen so far the position is not
5:34
Incorporated there is a figure which demonstrates this pretty nicely so uh
5:40
let's say this is the embedding layer weight Matrix which is basically we have all of these tokens in our vocabulary
5:46
those are converted into token IDs and for every token ID there is a vector in a higher dimensional space so each row
5:53
correspond corresponds to one such Vector for the token ID now if you want to embed these token IDs which means if
6:01
you want to convert these token IDs into Vector representations you will see that
6:06
these the five the token five comes here and the token five comes here also and
6:11
these two tokens clearly appear at different positions in the sentence but if you look at the embedded
6:18
vectors for these two tokens they are exactly the same if you look at the first row here that's the embedding
6:23
Vector embedded Vector for the token ID 5 the first token ID 5 and for the
6:29
second token id5 you'll see that the embedded Vector is the third row so if you look at the first row and the third
6:35
row they are actually exactly the same so the main point here is that the same token ID result in the same embedding
6:43
vectors which says that we are essentially not exploiting the maximum information present in sentences in
6:50
sentences there are there is meaning between different words which we already captured through Vector embeddings but
6:56
the position also matters a lot right cat appears at different positions here and that completely that may change the
7:03
meaning of the sentence entirely so it's very important to also encode the information related to tokens and the
7:11
position at which the token appears in the given sentence so it is very helpful to inject
7:19
additional position information to the large language model along with capturing the semantic meaning it is
7:26
extremely important to inject this additional information about the position to the llm so now let us talk a
7:32
bit about how do we encode this information about the position so let's say if I give a word and I convert it
7:39
into token ID and I convert it into a vector how do I also give information
7:44
about the position of the world let's look at that so there are essentially two types of positional
Absolute vs Relative positional embeddings
7:51
embeddings the first type of positional embedding is absolute and this is the more commonly used positional embedding
7:58
and the second type of positional embedding is relative positional embedding so as you must have guessed
8:05
from the name itself these two positional embeddings are different let's look at Absolute positional
8:10
embedding initially in absolute positional embedding for each in for each position in the input sequence a
8:17
unique embedding is added to the tokens embedding to convey its exact location
8:25
let me repeat that in absolute in absolute positional embed
8:30
for each position in the input sequence a unique embedding is added to the tokens embedding so for example if you
8:37
look at these two sentences cat sat on the mat cat sat on the mat the token embedding for cat in both is the same
8:43
right but as I said for each position there will be a different positional en encoding which will be added to the
8:50
Token encoding so let's say if the token encoding is X for sentence so let me say
8:57
so let's say this is sentence number one and let's say this is sentence number
9:03
two so let's say the token encoding for cat is x in sentence one and for sentence two also it's X because it's
9:09
the same word but now the positional encoding is different because in the first sentence the cat appears in
9:16
position number two so there will be some different positional encoding for this position and for uh for the second
9:24
sentence the cat appears in position number five so there will be some different positional encoding so the
9:29
final embedding uh will be the addition of the token embedding plus the positional
9:36
embedding so for sentence one the final Vector embedding for CAT will be x + y
9:41
for sentence two the final Vector embedding for CAT will be X Plus Zed so they are different now see they are not
9:48
the same so cat won't be embedded in the same manner in both the sentences and so
9:53
some information about its Position will be incorporated that's exactly what happens in absolute position embedding


***

10:00
and there is a nice figure to Showcase this so let's say uh we have four tokens
10:06
in the input sentence so let's say the first token is the the second is
10:14
cat uh the third is sat and the fourth is
10:19
mat so what I've have shown over here is that these are the token embeddings so for the word the this is a
10:25
three-dimensional token embedding for cat it's the same threedimensional token embedding for sat it's the same and for
10:31
mat it's the same ideally the vector embeddings for every word will be different but I'm just showing this for
10:37
the sake of Simplicity remember the vector embeddings for each word are ideally are actually different but I
10:44
just want to uh show a visual demonstration here so these are the vector embeddings now what we will do is
10:50
that for each Vector embedding We'll add a positional embedding and these positional embeddings are different
10:56
based on the position so for example the comes in position number one right so the positional embeddings are 1.1 1.2
11:04
and 1.3 cat comes in the second position so the positional embeddings are 2.1 2.2
11:10
2.3 sat comes in the third position so the positional embeddings are 3.1 3.2
11:15
3.3 Matt comes in the fourth position so the positional embeddings are 4.1 4.2
11:21
and 4.3 again these numbers I'm just showing for representative purpose now
11:27
if you add the token embedding to the position embedding you'll see that although the token embeddings are the
11:32
same for these words the final embeddings which are also called as the input embeddings which will be fed as
11:38
input to the llm they are different because the positional embeddings which are added to each token embedding is
11:44
different so for the word the the final input embedding is 2.1 2.2 2.3 for the
11:50
word Cat the final input em embedding is 3.1 3.2 3.3 and similarly for ma the
11:56
final input embedding is 5.1 5.2 and 5.3 this is the intuitive idea behind
12:03
absolute positional embeddings we have token embeddings and then we add
12:08
positional embeddings for different positions so finally we get an input embedding which encodes positional
12:14
information that's it uh one more key thing to remember
12:19
here is that the positional vectors have the same Dimension as the original token embeddings can you think why this is the
12:26
case why should the positional embeddings have the same Dimension as the token embeddings pause for a moment
12:33
to think about this okay so the reason the positional embeddings should have the same
12:39
Dimension as the original token embeddings is because we want to add them together right and this addition
12:44
would be difficult if the dimension of the positional embedding is different so for example here if this positional
12:49
embedding was a four dimension thing adding this four dimensional Vector to a three-dimensional Vector would not be
12:55
possible so the in absolute positional embedding the position vectors have the same Dimension as the original token
13:03
embeddings uh so this is uh absolute positional embedding the second type of
13:09
positional embedding is called as relative positional embedding and this is also a very interesting way of
13:16
encoding positional information so in this type of embedding the emphasis is
13:21
on the relative position or the distance between the tokens so the model
13:26
essentially learns the relationships in terms of how far apart rather than at
13:31
which exact position so that's a bit different than the absolute embedding right for each position in absolute
13:38
embedding there is a separate positional embedding but in relative positional embedding what we care about is how far
13:46
apart different words are rather than their exact position now you must be thinking where
13:52
exactly is relative uh positional encoding important so relative positional encoding is actually very
13:59
important because the model in this case can generalize better to sequence of varing lens even if it has not seen such
14:06
lenss during training so for example in in absolute positional in coding if you train with a sequence length of five and
14:14
if in the test you have a sequence length of six it's very difficult for the absolute position encoding to know
14:19
what to do because it has trained for five positions and five positional embeddings but in this case of relative
14:27
positional embedding uh the model can generalize better to sequence of varing lens because even if it gets some random
14:34
length during uh the testing phase the absolute position does not matter
14:40
anyways all that matters is the relative position between different words can you try to think of an example
14:46
where relative positioning might be more important than absolute
14:55
positioning okay so it turns out that relative positionings are actually better if the sequences are very long
15:01
and if longer paragraphs or longer input sequences need to be analyzed because then we need to know the relationship
15:08
between how different words are connected rather than the exact specific position I'll come to the uh advantages
15:16
and disadvantages of both of these in a moment okay so in the next uh in the
15:23
next section which is section number five I want to discuss about uh just these two types of of encodings and
15:30
which one to use in practice which one does GPT uses Etc so both of these type
15:35
of positional embeddings which are absolute embedding and Rel relative positional embedding are very good
15:41
because they enable the llms let me uh switch color they enable the large
15:48
language models to understand the order and relationship between the tokens and
15:53
this actually ensures more accurate and context aware predictions so whichever
15:58
position encoding you use it is actually much better than not using any positional encoding because uh it makes
16:06
the llm more aware of the order and relationship between tokens and that actually leads to better
16:12
predictions um now let's come to point number six so the choice between the two
16:17
types of uh positional embedding really depends on the specific application and
16:23
the nature of the data being processed so for example generally uh abs abolute
16:29
positional encoding is preferred when the fixed order of tokens is crucial such as for sequence generation so GPT
16:37
was trained using an absolute positional encoding and the original Transformer
16:42
paper was also trained using absolute positional encoding relative positional encoding on
16:48
the other hand is suitable for tasks like language modeling or long sequences where the same phrase can appear in
16:55
different parts of the sequence so generally relative positional encoding is useful if you are analyzing long
17:02
sequences and where the same phrase can repeat over and over again for all practical purposes I would
17:08
say absolute positional encoding is the one which is used more commonly in fact as I mentioned uh open a GPT models so
17:18
gpt3 GPT 4 Etc use absolute positional embeddings that are optimized during the
17:24
training process so one thing to remember here is that similar to the embedding vectors we do not know what
17:31
values of the positional embedding to be used so for example here I randomly showed these values 1.1 1.2 and 1.3
17:39
right ideally as I mentioned in the previous lecture the vector embeddings need to be optimized right we need to
17:45
know the weights for each Vector which what's the value for each Vector
17:50
similarly uh when GPT was developed even the values of the positional embedding
17:57
vectors need to be op optimized and these positional embedding vectors
18:02
Vector values are actually optimized during the training process this optimization is actually a part of the
18:08
training process itself so when you look at the model training for GPT we also have to optimize for the token
18:15
embeddings and we also have to optimize for the positional embeddings we do not know about these values
18:22
before uh remember I mentioned that along with GPT the original Transformer paper which is called as attention is
18:29
all you need so let me show you this paper yeah this paper so they also used an absolute positional encoding and in
18:36
this case they actually propos some formula for how to encode the different positions so they use sinusoidal and
18:42
cosine formula over here uh so you can read a bit about what they have written
18:48
here since our model contains no recurrence in order for the model to make use of the order of the sequence we
18:55
must inject some information about the relative or absolute position of the to tokens so they have added absolute
19:01
oppositional embedding and they have used some formula for how to actually calculate the positional embedding for
19:08
each token uh on the contrary when GPT was trained no such formula was used and uh
19:16
the positional embedding Vector values were actually optimized during the training process itself I hope everyone
19:24
is with me until this point because now we are going to jump into a handson
19:29
demonstration of uh looking at a very uh real life example such as gpt2 looking
19:37
at the vocabulary transforming that vocabulary into token embeddings adding positional embeddings to those token
19:44
embeddings and then generating the input embeddings which are the final input to the llm training so you might be
19:51
noticing that I'm using the words encoding and embedding interchangeably so sometimes positional encodings are
19:57
also called positional embed edings so these words are used interchangeably all you need to remember is that when


***

20:03
someone says encoding or embedding it's usually a vector in higher dimensional
Hands on Python implementation
20:08
space okay so now let's come to the next part in the next part what we are going
20:13
to do is we are going to implement positional embeddings in a Hands-On manner so I'm going to take you to the
20:19
Jupiter notebook now very similar to how we have been doing in the previous
20:24
lectures okay so previously uh especially in the last lecture we focused on very small embedding sizes in
20:31
this chapter for illustration purposes now we are going to consider much more realistic and useful embedding sizes and
20:39
encode the input tokens into a 256 dimensional Vector space so remember I
20:44
mentioned that in token embedding you have every word and that word is projected into higher dimensional Vector
20:50
space usually that Dimension is very high so gpt3 I think was trained on a
20:55
vector space of around 256 or even more Dimensions uh so we are going to
21:00
consider a vector space of that size right now for demonstration uh as is written here this
21:06
is smaller than what the original gpt3 model used so in gpt3 the embedding size
21:12
is actually 1 12288 Dimensions so it's nowhere close to 256 but it's fine let
21:18
me actually ask G uh Chad GPT what is the um Vector embedding size for gpt2
21:26
for gpt2 for one of their smallest model I think it is around 256 so let me ask
21:32
chat GPT what is the vector embedding size for
21:41
gpt2 so if I ask this to chat GPT you will see that uh uh for the for all
21:47
their models it actually starts from 768 so the value of 256 we are using is three times smaller but it's fine it's
21:53
at least the same order of magnitude so okay we are going to encode the input tokens into 256 dimensional Vector
22:01
representation and then we are going to assume that the token IDs were created by the bite pair encoder tokenizer which
22:08
has a vocabulary size of 50257 let's check this
22:14
uh what's the
22:19
vocabulary size of gpt2 pre trining so let's see the
22:26
vocabulary size is 502 25 7 using the bite pair encoder this is exactly what
22:31
we are going to use right now in today's Hands-On example so as I mentioned before to actually create the embedding
22:38
layer uh or this is also called as the uh token embedding Matrix you need two
22:45
you need two quantities or two variables the first you need the vector Dimension so basically every token ID in the
22:51
vocabulary will be converted to a vector of these many dimensions and right now we are going to use uh 25 6 over
23:01
here uh 256 and the second thing which we need
23:06
is the vocabulary size and the vocabulary size in our case is as I've already mentioned here it's
23:13
50257 this means that there are 50257 token IDs and each of these token IDs
23:18
will be transformed into a 256 dimensional Vector awesome so now what
23:25
we are going to do is that we have defined the vocabulary size to be 0257 we have defined the output
23:31
Dimension which is the vector size as 256 and then we are going to create a token embedding layer using torch. nn.
23:38
embedding so torch. nn. embedding actually creates this kind of an embedding layer provide provided we have
23:45
those two inputs which is the vocabulary size and second is the vector length
23:50
which we want so I I'll put the link of this in the chat or in the information
23:56
section of the YouTube video as well so when you run this you will see that the token embedding layer has been created
24:01
and it takes two inputs the vocabulary size and the output Dimension awesome so now once the token embedding
24:08
layer is created we need to uh remember the token embedding layer is is
24:14
essentially a lookup Matrix where if you give it uh the IDS so if you have a
24:21
token embedding layer and if you provide the token embedding layer with input IDs which you want to look look for it will
24:27
give you the responding embedding Vector so for example uh this is the token
24:32
embedding Matrix right if you and this is a lookup table why is it a lookup table because if you pass in the input
24:38
ID it looks up that particular row and it gives you the vector associated with
24:44
that so now we need to create the input IDs so that we can generate the vector
24:50
embeddings or the token embeddings for those inputs and to create the input IDs
Creating input batches using DataLoader
24:55
we are going to use something called as data loader so we looked at data loader in the in one of the previous lectures
25:02
so what we are going to do here is that uh we are going to have a batch and that batch size will be equal to 8 and we are
25:09
going to use a context size of four which means that the maximum input length is four which means four tokens
25:15
at a time can be passed as inputs that's also called as the context size so uh let me actually show this to
25:23
you in pictorial format so that it's easier for you to visualize okay okay so
25:28
this is the input which we are going to create uh we are going to divide the data into batches and the batch size
25:35
will be equal to 8 so the parameters will be updated after processing every
25:40
eight batches right now we are not looking at parameter updation at all I just want you to be aware of the
25:46
dimensions so the input which we will be looking at one time is a batch of eight so there are eight rows over here and
25:53
each row corresponds to one input sequence so if you look at the first row
25:59
these will be four tokens if you look at the second row these will be four tokens so the first row when I say four tokens
26:05
these are four token IDs so remember the goal is to look at these four token IDs
26:10
and to predict the next word so you can think of each row as an input to the llm
26:17
and each row consists of four token IDs now our goal is to transform each of
26:23
these token IDs into a uh 256 dimensional vector
26:28
right so if you look at the first row the first row has uh four token IDs and
26:34
we want to transform each into a 256 dimensional Vector so before coming to that let me
26:41
show you how the inputs are initialized and how from the data loader so you so I
26:46
have already defined a data loader in this jupyter notebook and you will have access to that when I share the code
26:51
file with you if you have not seen the data loader lecture I highly encourage you to go to that but if not it's fine
26:58
I'll try to give you an intuition of what's Happening Here essentially what we are doing here is that we are looking
27:03
at the raw text and this raw text is actually the verdict so this is a book called The
27:10
Verdict and uh let me actually refresh this so that it appears in a better format yeah so this is actually a book
27:17
called The Verdict and this is the main text which we are using as sample text for uh for these set of lectures so
27:25
what's happening is that we are taking this raw text and uh we are chunking it into batches so each each batch is of
27:32
size eight and uh for each batch we are looking at a max length of four so only
27:39
four input tokens will be used to predict the next World so think of this Matrix which I showed you over
27:45
here so we are going to create a data loader which takes in the Raw text which is a batch size of eight which is a max
27:52
length which are the number of essentially columns which is the context length which is equal to four
27:58
and then a stride of four and Shuffle equal to false so stride basically means that let's say if you look at this text
28:06
right and if you want to uh create inputs so the first input will be I had
28:12
always thought because the context size is four now since stride is equal to four the next input will be Jack gisburn
28:19
rather a if the stride was equal to one the next input would be had always thought Jack but now the stride is equal
28:26
to four so after one in input we'll Skip One 2 3 four times and then give the next input so what we are essentially
28:33
doing is creating inputs so the batch size is eight right so the first batch will have this as the first row of input
28:40
this as the second row of input this as the third row of input and so on up till eight eight batches and each of these
28:47
tokens uh are converted into token IDs and then what we want to do is map
28:53
each of those token IDs into vectors that's what we are doing exactly through
28:58
this data loader data loader just helps us to uh manage the task of inputting
29:04
the data batching the data creating different batches parallel processing much easier so it's highly recommended
29:11
to use data loader I'll actually just show you the data loader so it's this
29:17
link data sets and data loaders in Python uh it's actually highly useful
29:22
when dealing with large language models okay so once we Define a data loader like this we just iterate through
29:29
the data loader and we can get inputs and targets so if you print out the
29:34
token IDs so this is the input of a batch and if you see this exactly similar to what we had written on the
29:40
Whiteboard so if you look at one batch it will have uh uh eight input sequences and each
29:48
input sequence has four token IDs or four tokens and using these we want to
29:53
predict the next word for each input sequence so this is the batch of inputs
29:58
which we have received and now what we actually want to do is for each of these input token IDs we want to convert each

***

30:05
of these into a 256 uh dimensional Vector
30:11
representation but first let's look at this token ID tensor and it's a 8x4
30:16
tensor because it has eight rows and four columns uh the data badge consists of eight text samples with four tokens
30:23
each awesome now what we are going to do is that we are going to uh convert each
Generate token embeddings
30:28
of these token IDs into a 256 dimensional uh Vector using this
30:34
embedding layer so as I told you what this embedding layer is is it's actually a
30:40
lookup table right so if you look at this embedding layer and if you give the
30:45
token ID it will generate the or it will fetch the corresponding Vector representation for you that's exactly
30:51
what we are going to do over here so these are the inputs right now let's say I'm looking at one batch uh which has
30:59
eight rows and which has four columns and I have given the input IDs as randomly assigned over here right
31:05
now but let's say these are the input IDs so for the first first batch the input IDs are 10 8 uh 20 and uh
31:15
21 now what is done is that these input IDs are then mapped to the embedding
31:22
Vector Matrix so we we know the corresponding row so if you look at the first row it's 10 8 10 and 20 these are
31:29
the input IDs right so then here you look for the input ID of 10 and then if
31:34
you find then you find the corresponding Vector for it then you look for the input ID of8 and then you find the
31:41
corresponding Vector for it then you look at the input ID of 20 and then you find the corresponding
31:48
Vector for it and then you look at the input ID of 21 and then you find the
31:53
corresponding Vector for it so basically for each token in this input uh input
32:00
batch one embedding Vector of size 256 is generated for each token in the input
32:06
so I just constructed another visualization for you all to see it
32:11
further so if this is the input batch for each token ID here it's converted
32:17
into a 256 dimensional Vector so if the size of this original input batch was
32:24
8x4 when we generate the embedding vectors for each of this the we will get
32:30
a tensor which is 8x 4 by 256 so it's a threedimensional tensor now why 256
32:36
because for each of these 8x4 32 values for each of these 32 values we have a 256 dimensional tensor so for each of
32:44
these you can think of like a vector which has basically 256 Dimensions so
32:51
there is a vector here Vector here Vector here Vector here Etc I cannot show the three-dimensional structure
32:56
right now but it's basically Ally 8X 4X 256 uh okay so that's what we are going
33:04
to do next so we have this inputs right which is the batch of inputs generated
33:09
from the data loader we are just going to pass it as an argument to the lookup table to the embedding layer and then we
33:15
are going to get this tensor which is 8x 4X 256 which is exactly the tensor shape
33:21
which I was showing to you over here that's the 8x4 by 256 tensor after we
33:26
pass the input ID to this uh lookup table of the embedding layer basically what this lookup table
33:33
does is that it generates the 256 dimensional Vector for each of these token IDs up till now what we have done
33:40
is that we have broken down the input text into batches so let's say we so this is the first first batch this this
33:47
is the first this is the uh first input of the first batch remember each batch
33:53
has eight inputs so this is the first input we converted into token IDs and then each of the token ID is have has a
34:00
256 or 256 Vector length uh embedding that's what we have done until now it's
34:07
as simple as that I hope everyone is following until now I'm trying to explain this vectorial and tensorial
34:13
notation as uh in detail as possible but of course sometimes it gets a bit
34:19
difficult so if you have any questions you can ask it in the comment section so uh here I have written that
34:26
as we can tell based on the 8X 4X 256 dimensional tensor output each token ID
34:32
is now embedded as a 256 dimensional Vector awesome and now we have what we have to do is that we have to add a
Generate positional embeddings
34:38
positional embedding uh positional embedding Vector to each of these vectors similar to the uh token
34:46
embedding we also have to create another embedding layer for the positional encoding so remember that at each time
34:54
only four vectors need to be processed right because if you look at the context
34:59
size that is equal to four at one time in the input only four tokens are going to be given so at one time only the
35:07
input the maximum input size is four which means the llm is going to predict the next word based on a maximum of four
35:13
tokens so actually we need to encode only four positions in this case so the
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
