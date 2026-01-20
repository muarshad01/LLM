 view of GPT-2 architecture
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from
0:10
scratch Series today what we are going to do is we are going to construct the
0:16
entire GPT architecture or put in other words we are going to code the entire
0:21
GPT model we have been building up to this lecture for a pretty long time now
0:26
so let me just quickly recap what all we'll be covering in today today's lecture and what we have completed
0:32
previously so in the previous set of lectures which started around four to five lectures back in this GPT
0:40
architecture or llm architecture lecture series we started out with a dummy GPT
0:45
model class so when we started around four to five lectures back we did not
0:50
have H an understanding of the building blocks of the llm architecture or the
0:56
GPT architecture slowly we worked our way towards it initially we learned
1:02
about layer normalization we wrote code for it then we wrote a code for the feed
1:07
forward neural network along with the J activation then we learned about shortcut connections and wrote code for
1:14
that and in the previous lecture we also wrote the code for the entire Transformer Block in which the layer
1:20
normalization the J activation the feed forward neural network and the shortcut connections all come together today what
1:27
we are going to do is we are going to write code for this final piece of the puzzle which is how Point number one
1:34
point number two 3 4 5 and six essentially come together to form the
1:39
entire GPT architecture throughout this these four to five lectures on the GPT architecture
1:46
we have this or we should have this visual figure in our mind so here is
1:51
what happens in the GPT architecture we have text which gets tokenized we have
1:59
the dropout layer over here and then the input embeddings are essentially passed
2:05
to this blue colored block which is called as the Transformer Block in the previous lecture we coded out all of
2:12
these individual Elements which I'm marking in yellow right now in the Transformer block and we saw that when
2:18
you give an input to the Transformer how to code out all of these blocks and how to stack them all together so that we
2:24
get the output today we are also going to learn about these final two steps which is
2:30
another layer normalization layer and another linear output layer towards the end and we are going to see how do we
2:38
postprocess the output from the Transformer to get an output from the entire GPT model that's the main goal
2:46
today so if you if you look at the overall picture the mass multihead
2:51
attention is the key component of the Transformer block and the Transformer
2:57
block is the key component of the entire GPT model we have coded out the Transformer block
3:02
right that's fine but now we have to code out the entire GPT model so we'll start from the we'll start from the
3:09
bottom we'll start from uh uh this input
3:14
IDs or rather input text we'll tokenize it then we will uh integrate the Dropout
3:22
and then we'll integrate the Transformer building block and then we'll also integrate these final two building
3:27
blocks so you can think of this Le as an assembly lecture where different moving pieces of the GPT architecture will all
3:34
come together and ultimately you will have a system in which or ultimately you'll develop a model which receives an
3:41
input so the input is in the form of text tokens and the output from the model looks something like this right
3:47
now in the next lecture we are going to see how to decode this output to predict the next word so the main goal of the
3:54
GPT model is to take an input text such as every effort moves you and to predict
3:59
the next word so we are training the model for the next word prediction task today we are going to take today I'm
4:06
going to take you all to this stage where we get this final output and in the next lecture we are going to see how
4:13
to get the next word from this final tensor I'll assure you that the final tensor which we will obtain today can be
4:20
easily used to decode the next word so the main task today is how to go from
4:26
the Transformer output to this final output tensor from the GP G PT model and we are also going to take an Hands-On
4:33
example and show you how the operations Stack Up throughout this entire GPT
4:38
model okay so when we started this lecture series we actually started with
4:43
this dummy GPT model which I'm going to show you right now so we started this lecture series with this dummy GPT model
4:51
class right over here and at that time we had left several aspects blank so we
4:57
had left the Transformer Transformer block blank we had also left the layer normalization class
5:04
blank we had coded out some aspects of the forward method but most of this
5:09
these classes were blank um but that's fine now we are ready to fill up these different aspects so that we can return
5:17
the output so ultimately we are going to see
5:22
this entire workflow to today we are going to have the input IDs which will be converted into token embeddings then
5:29
we'll add positional embeddings We'll add the Dropout layer we'll pass the output of the Dropout through the
5:34
Transformer block then the output from the Transformer block will be passed through another layer normalization
5:40
layer and then we'll finally have an output head layer and we'll return the Logics don't worry about understanding
5:48
all of this right now I'm going to sequentially take you through each of this step by step uh and explain to you
5:54
what exactly happens in each of these uh these code lines okay okay so what we are
6:00
essentially going to do is that since we coded this Transformer block last time we can replace this with the actual
6:05
Transformer code we are also coded this layer normalization Block in one of the previous lectures so we'll replace this
6:12
with the layer normalization code okay so the goal now we are ready
6:17
to achieve the goal of today's lecture which is to assemble a fully working
6:23
version of the original uh 124 million parameter version of gpt2
6:31
so we are going to do a pretty awesome thing today we are going to take input texts and then we are going to pass them
6:38
through this entire gpt2 architecture which by the way has 124 million parameters and then we'll get the output
6:45
answer and all of this we'll be doing on our local computer I'll be sharing the code with you so you'll be able to
6:51
execute it on your own end that's pretty awesome right you'll be probably running
6:56
a large scale large language model for the first time on your local machine so you at this moment if you have followed
7:03
the previous lectures we are going to assemble several components here so people who have followed the previous
7:10
lectures this lecture is going to be very enriching for you if you have come to this lecture for the first time I've
7:16
designed it so that you can follow it along but please understand that the value which you will derive from this
7:22
lecture will be significantly higher once you have also gone through the previous
7:28
lectures okay uh I hope all of you have this visual map in
7:33
mind we are going to do all of these steps which are shown in this visual map today and with actual parameters from
7:40
gpt2 we are going to use around 124 million parameters today before I dive
Token, positional and input embeddings
7:46
into code I want to first show you everything on this whiteboard especially in terms of Dimensions so that you get a
7:53
clear understanding of what is exactly happening when we are going to move to code I have seen that many students who
7:59
learn about large language models they are very unclear about how the dimensions work out so what I've
8:05
actually done is that I've have made this uh flow map over here so I'm just zooming out here right now we are going
8:11
to going to go through all of this in just a moment but this is the flow map which I'm going to teach you right now
8:18
and the blocks which you see on the screen they are mostly there to represent the dimensions of the input
8:23
the dimensions of the output so my goal here is to visually convince you of what
8:28
exactly is happening in the GPT model so that the code becomes significantly easier the code for this GPT model is
8:36
actually pretty simple and straightforward the only difficulty which students face is that many tensors
8:41
come into the picture many dimensions come into the picture and students get confused as to they cannot visualize
8:47
what's going on and I've have not found too many too much good material out there which takes students through every
8:53
single step in the GPT model like this uh even in the previous lecture we went
8:58
through the entire code but we did not see this Hands-On example of let's say if you take a specific input sequence
9:04
how does that input sequence flow through the different blocks of the GPT model and how do we get the output let's
9:11
dive into every single detail remember the name of this whole playlist is building llms from scratch we are not
9:19
going to assume anything I want to teach you the nuts and bolts of how every single line of code works and that's why
9:25
I have made this effort to construct this visual flowchart okay so let's say
9:30
the input is every effort moves you right and then we have to make the output prediction which is the
9:37
prediction of the next word and the next word is forward every effort moves you forward so let's go through a sequence
9:44
of steps of what exactly happens in the GPT model when this input is given I'm going to switch color right now to a
9:50
darker color so that you all will see what I'm writing on the board okay so ideally inputs come in batches so what
9:58
we are going to do is is that let's say when we go to code we'll see that we have two batches and in each batch there
10:04
are four tokens so the first batch has every effort moves you and the four
10:10
token IDs corresponding to that and the second batch has token IDs corresponding
10:15
to the second sentence for the sake of Simplicity right now I'm just going to
10:21
analyze the first batch and then the same learnings which I'm going to show you can be applied to the second batch
10:27
as well uh so I'm going I'm getting rid of one dimension which is the batch dimension for now for the sake of
10:33
simplicity so we are going to focus only on these four words every effort moves you and prediction of the next World the
10:40
first step is that remember that we have a vocabulary right even when gp22 model
10:45
was constructed we have vocabulary and the vocabulary size is five U
10:52
5 I think it is 50257 so 50257 is the vocabulary size
10:58
for GP 2 so the first step is you take every word and you map it to a token ID
11:04
in the vocabulary every token is mapped to a token ID so these are the four
11:09
tokens for the sake of Simplicity think of every one token equal to one word that's not at all what's happening in
11:15
the gpt2 because gpt2 uses bite pair encoding which is a subword tokenizer we
11:21
have a separate lecture for that uh where you can see that even characters and small subwords can be tokens but
11:29
just for the sake of Simplicity I'm going to use one token equal to one word interchangeably in today's
11:35
lecture okay so we are looking at this first batch which has four tokens or four words that's the first step to
11:42
convert these tokens into these four token IDs awesome the next step is actually to take these token IDs and to
11:48
convert them into token embedding vectors so this is a key Point here computers can't understand words right
11:55
and it does not make sense to just have token IDs because because we need to capture the meaning between words dog
12:02
and puppy are close to each other cat and kitten are closer to each other it turns out that representing words in a
12:09
vectorial format can help preserve the semantic meaning between words so the
12:14
first step is to convert every input token into this token embedding vector and to decide an embedding size so we
12:22
are using an embedding size of 768 because that was the embedding size which was used for the smallest gpt2
12:29
model when it came out so if you see every which is the first token is now
12:34
encoded as a 768 Dimension Vector over here effort which is the second token is
12:40
also encoded as a 768 dimensional Vector moves which is the third token is
12:45
encoded as a 768 dimensional vector and U which is the fourth token is also
12:50
encoded as a 768 dimensional Vector one point which I want to mention here is that this encoding we do not know what's
12:57
the best encoding from the start we are initially going to project these vectors randomly in the 768 dimensional space
13:05
and then we are also going to learn the token embedding parameters in GPT models
13:10
along with everything else the token embedding all the embedding parameters are learned so right now let's say I
13:16
told you about cat and kitten right when we start out the vector for cat and the vector for kitten is initialized in
13:22
random directions but when the model is trained when the embedding vectors are trained they will be closer together
13:27
they'll be more aligned so right now for every U for the token every for the
13:34
token effort for the token moves and for the token U we randomly initialize
13:39
vectors in the 768 dimensional space that's the first step token embedding
13:44
the second step is that remember along with the semantic meaning of words what's also important to capture is
13:51
where the word comes in the particular sentence so every comes so every effort moves you and every comes in position
13:57
one effort comes in position two moves comes in position three and U comes in
14:02
position number four so along with representing the words themselves as an embedding Vector we also represent every
14:09
position as an embedding Vector so we are considering four positions here right which also becomes the context
14:15
size remember the context size is the maximum number of words which can be used to predict the next word in our
14:22
case the context size is equal to four which means only four positions matter so that's why in the positional
14:28
embedding we we are going to look at the embedding vectors for four positions position number one has again a 768
14:35
dimensional Vector position number two has a 768 dimensional Vector position number three
14:42
has a 768 dimensional vector and position number four has a 768 dimensional Vector similar to token
14:49
embedding we actually do not know the embedding values in each of these uh um
14:55
in each of these positional embedding vectors these embedding values are initialized randomly initially we do not know what
15:02
these embedding values represent these will be trained as the during the training procedure but the important
15:08
thing to note is the embedding size the embedding size for every Vector in the positional embedding is 768 and this is
15:15
the same size as the uh token embedding and the reason for this is in the third
15:22
step what we are going to do is that we are going to add the token embedding for each token along with the positional
15:28
embedding so if you go to step number three which is seen on the screen right now let me zoom in further for the first
15:36
word which is every it's in position one so we are going to take the token embedding for every and we are going to
15:42
add it uh with the positional embedding for the first position and the result is
15:48
the input embedding for the first token so since the token embedding and positional embedding have the same
15:54
dimensions the input embedding also has the dimension of 768 so this is the input embedding for position then effort
16:03
so when you come to effort which is the second word it's in second position so we take the token embedding uh for the
16:09
second word and we add it with the second positional embedding for the second position and we get the input
16:15
embedding for the second token which is effort and here again the embedding size is equal to
16:22
768 you can see over here which I'm marking right now in purple color the
16:27
embedding size for the uh second position or the second the
16:33
input embedding size for the second token is 768 Now we move to the third token which is moves this is in position
16:40
three so we'll take the token and add the positional embedding Vector for position three and that leads to input
16:47
embedding Vector for moves which has an embedding size 768 similarly for the
16:52
fourth position which is U we take the token embedding and add the positional embedding and we get the input embedding
16:58
which has the size of 768 this is Step number three remember this is a very important step token
17:05
embeddings plus positional embeddings leads to the input embeddings which this formula I have also written over here
17:11
input embedding equal to token embedding uh yeah input embedding equal
17:18
to token embedding plus the positional embedding okay so that's the step number
17:25
three and after these steps are completed we move to step number four step number four introduces Dropout so
Dropout layer
17:31
what happens in Dropout is that until now we have input embeddings for every word right we have input embeddings for
17:38
every effort uh moves and U and the embedding size is a
17:45
vector so which means that for every token we have uh the input embedding of 768 sized Vector in Dropout what happens
17:53
is that we randomly turn off some elements of every uh every input embedding to zero and that's specified
18:00
by the dropout rate so if the dropout rate is 50% from every embedding
18:05
randomly 50% of the elements are turned off to zero so let's say this might be turned off this might be turned off so
18:13
50% of 768 is around 384 right so around 384 elements of each input embedding are
18:21
turned off to zero and this is done for every token so here I'm just showing the random Elements which are turned off to
18:27
zero remember this is probabilistic so when I say 50% not exactly half of the
18:33
embeddings will be turned off to zero on an average 50% of the input embeddings will be turned to zero okay so why is
18:41
Dropout implemented the main reason why Dropout is implemented is to prevent overfitting improve
18:48
generalization uh and this generally helps a lot the main Dropout technique was initially implemented in neural
18:54
networks to prevent some neurons from being lazy so during training sometimes what happens is that some neurons don't
19:01
learn anything and they depend on other neurons and that leads to problems in generalization so what people do is that
19:08
they Implement Dropout layers where neurons are turned off randomly so the neurons which were lazy earlier have no
19:15
choice but to learn something and that improves the generalization performance because every neuron is generally trying
19:21
to learn something and that's the similar case for year also wherever Dropout is implemented the main reason
19:27
for implementing Dropout is to prevent overfitting or to improve generalization
19:32
performance okay so we have seen four steps up till now let's recap them the first step is uh token embedding uh
19:41
which we saw the second step is positional embedding which we again saw the third step is uh input embedding
19:49
which is essentially adding the token embedding plus the positional embedding and the fourth step is implementing
19:55
Dropout now remember up till here the Transformer block has not been introduced at all we have still we are
20:01
still outside the Transformer block so let's go to this overall structure of this Transformer block again to see what
20:07
all we have seen up till now so if you look at this structure until now um let
20:13
me zoom in further yeah if you look at the structure until now we have seen the four steps which come before here so we
20:19
tokenize the text into input IDs then we add the to then we have the token embeddings which was the first step then
20:26
we add the positional embeddings which was the second and third steps we get the input embedding and then we apply
20:31
the Dropout so we have seen these four steps until now we are at this point and
20:36
after the Dropout now we will enter the Transformer Block in which all of these steps will be performed so let's go to
20:42
step number five right now where we'll be looking at the Transformer block okay so when we reach the
The 8 steps of the transformer block
20:49
Transformer this these are the input embeddings which we have so these are
20:54
the input embeddings with Dropout one thing which I would like you to see is that the dimensions are being preserved
21:00
so when we started this at the first step these were the same dimensions right every token had the dimen
21:05
embedding dimension of 768 and when we enter the enter the Transformer block every is still a 768
21:13
dimensional Vector effort is still a 7608 dimensional Vector moves is still a
21:18
7608 dimensional vector and U is still a 7608 dimensional Vector the one thing
21:24
which has not happened is that until now every Vector only contains meaning about itself but we do not know let's say when we are
21:30
looking at every how much information how much attention should be we give to effort moves and you to predict the next
21:37
world that's very important right along with capturing the semantic meaning which Vector embedding does it does not
21:44
capture the meaning of how every how let's say each word is related to other words so let's say when we look at
21:51
effort and we want to see when we are looking at effort how much attention should we pay to every moves and you
21:58
when predicting the next word that information is not captured and that will be done through the attention Block
22:04
in the Transformer module let's see when we get to that so now we are at the stage where we have the input embeddings
22:11
with Dropout then we apply the first layer in the Transformer block which is the layer normalization so this is also
22:17
called as the layer Norm what this layer normalization will do is that it will look at every uh every token so let's
22:23
say I'm looking at this token every uh and then U it will look at all
22:29
of these values here and it will normalize the values so that the mean of these values is equal to zero and the
22:36
variance of these values is equal to one and this will be done for every single token so after we do it for the token
22:42
every we then move to effort so we'll take the take all of these embedding values and we will normalize them so
22:49
we'll subtract the mean from every value and divide by uh the square root of variance and this same procedure
22:56
normalization procedure will be done to move and to you so after the layer normalization is done when we look at
23:02
every token and when we look at the values present in the embedding we'll see that the mean of those embedding
23:08
values is equal to zero and the variance of those embedding values is equal to one for every single token layer
23:14
normalization is performed to improve the stability during the training procedure okay after layer normalization
23:21
is performed the most important step which is actually the engine of the Transformer block which is why llms work
23:27
so well is this Mass multi-ad attention what is done in this mod in this step is
23:33
that we conver we take the embedding vectors and we convert them into context vectors so if you look at the output of
23:40
the M multi attention the size of the output is same so for the word effort let's say it we still have an embedding
23:47
of 768 Dimensions but now this is called as the context Vector embedding the reason is called context Vector is that
23:54
along with capturing the semantic meaning of effort which the Vector embedding already did this context
24:00
Vector which exists for effort also captures the meaning of how much attention should we give to every how
24:07
much attention should we give to moves and how much attention should we give to you when we are looking at effort so the
24:15
context Vector for every token captures the meaning of how much attention should be given to all the other tokens in the
24:22
sentence that's why it's called attention this is by far the most
24:27
important step in the entire GPT model without this part it would llms would not perform as well this part really
24:34
tells us that when we want to predict the next word which are the important words to look at what is the meaning
24:40
between different words how do different words attend to each other so the context Vector which I have written here
24:47
it looks very simple right now but we have devoted five lectures of 1 Hour 1 and a half hour each to understand this
24:53
one step uh which appears in the whole GPT model because here is where the
24:58
magic happens here is where we transfer U Vector embeddings into
25:04
context Vector embeddings so we we contain we capture meaning we capture context as to how different tokens or
25:11
different words are related with each other awesome so until this step here
25:17
again you can see the dimensionality is mentioned is preserved throughout that's what I like about the GPT model so for
25:25
throughout every step you'll see that we still have four tokens and the dimension of each is 768 this really makes the GPT
25:32
model scalable it's much easier to add multiple modules together because addition of modules does not change the
25:39
dimensionality then after multi-head attention we again have a Dropout layer which randomly uh turns off
25:47
certain um certain context Vector values to zero so here I have shown the color
25:53
red where the values of every we look at every token and weite randomly switch
25:58
off certain values to zero it's the same Dropout layer which we had seen earlier then we actually have a shortcut
26:04
connection so wherever shortcut connections are mentioned it means that the output of this the output of this
26:12
layer the output of um the Dropout layer is added back to the input which we
26:18
started with to the Transformer uh this input the output is added back and the reason it's done is
26:25
because it provides another route for the gradient to flow and it prevents The Vanishing gradient problem which means
26:32
that the training proceeds in a much smoother manner so after the shortcut connection is applied that does not
26:38
change the dimension at all we again apply one more round of layer normalization so again we look at every
26:45
Row the mean is the mean is uh changed to zero the variance is changed to one
26:52
so that's how every values is normalized we subtract the mean from every value divide by the standard divide by the
26:58
square root of the variance so that finally when we look at the values together their mean will be zero and
27:03
their variance will be one this is done for every single token and then after the layer normalization layer we have a
27:09
feed forward neural network so when you zoom into this network you will see that this is kind of like an expansion
27:16
contraction uh Network where let's say you have the inputs right so we process every input step by step here if you
27:23
first look at the first token which is every it's a four dimension it's a 768 dimens token right um if you see the
27:32
input which enters the speed forward neural network is that every token has a dimension or an embedding size of
27:39
768 so what happens in this neural network is that we look at each token sequentially so let's say we are looking
27:45
at the first token which is every let me zoom into this neural network a bit uh yeah so first we project this
27:53
input into a higher dimensional space in fact uh this neural network has one hidden layer and the number of neurons
27:59
in this hidden layer is 4 * 768 which is four * the embedding Dimension and then
28:06
we contract back to the original Dimension so the output from this neural network is the same Dimension as the
28:13
input but there is this expansion and contraction which is happening and that allows a much richer exploration of
28:19
parameters that makes our llm much better at prediction of the next word
28:24
since it captures the meaning in a much better manner so if you look at the output from this neural network the
28:29
output from this neural network is the same size as the input which entered it we have four tokens and the embedding
28:35
size is equal to 768 uh but this expansion contraction
28:40
which you see over here the arrow which I'm showing here is the expansion the arrow which I'm showing now is the
28:47
contraction that is the key because since the middle layer has huge number of neurons it is four times the
28:52
embedding Dimension neurons which is 4 * 768 the number of parameters are huge in this feed forward neural network that
28:59
allows for a richer exploration space after this feed forward neural network
29:05
we have another Dropout layer which randomly uh switches off certain values to be equal to zero which I have just
29:11
shown here again by the red um by the red color so in every token random
29:18
values are set to zero typically one dropout rate is mentioned at the start and that's applied everywhere where
29:23
these Dropout layers are implemented then we have another shortcut connection where we basically add the output
29:31
uh after this Dropout to the input when we entered here so the this is the input
29:38
in this second row so this so the input which was there here is
29:44
added uh to the output of the Dropout layer which we are seeing over here and that's why it's a shortcut connection
29:51
and then the output from the shortcut connection is our Transformer block output over here so right now the
29:58
Transformer block output which I'm highlighting is the output after so many steps in the Transformer block so if you
30:05
look at the dimensions of this output you'll see that every token is again a 768 dimensional Vector over here so the
30:11
dimensions are exactly the same as the input to the Transformer block so let me
30:16
zoom out a bit here and you can now try to appreciate the number of steps which are involved in the Transformer so if I
30:22
zoom out here and if you look at the right side of the screen along with me uh let me actually move it to the
30:29
center yeah so these are all the steps which are involved in the Transformer right now whatever you are seeing on the
30:36
screen U let me minimize this
30:47
color yeah I hope uh you can see the entire Transformer workf flow on the screen right now we have we start with
30:55
the layer normalization so I'll show show this with a different color probably yeah we start with the layer
31:01
normalization then we move to the uh mask multihead attention then we then we
31:07
have a Dropout layer uh then we have a shortcut connection over
31:13
here let me Mark this
31:21
line yeah we have a shortcut connection over here then we have another layer
31:26
normalization layer we we have a feed forward neural network we have a Dropout layer then we again have a final
31:32
shortcut layer which actually takes us to the Transformer block output um now let us actually go to
31:40
the uh visual diagram again to see whether we indeed have implemented all the steps which were mentioned in the
31:47
Transformer block so I'm zooming in over here right now and if you see there are actually 1 2 3 4 5 6 7even eight
31:54
different steps right we start out with the again let me switch to a different color
32:00
here
32:07
yeah yeah we start out with the layer Norm we go to the multi-ad attention we go to the Dropout we have the shortcut
32:13
connection that's the first block then we have again the layer normalization the feed forward neural network the
32:19
Dropout and then the last shortcut so there are eight different steps here which were the same eight different steps we implemented in the uh visual
32:27
flow map and now there are the final two steps which are remaining so let's look at the last two steps of the GPT model
32:34
in the visual flow map so the fifth step was the Transformer step which we implemented
Post transformer layer normalisation
32:40
within the Transformer also there were eight different steps now we go to the next block which is
32:48
uh uh the next block is another layer of normalization so the Transformer block
32:53
itself has two normalization layers but we have another normalization layer after the Transformer block output so
33:00
here we have the Transformer block output and here you can focus on the dimensions again every vector or every
33:06
token has an embedding size of 768 right again the dimensionality is preserved
33:12
and then we apply the layer normalization so same thing as what the near layer normalization does is done
33:18
here also in every token we normalize the embedding values so that the mean of
33:25
the resultant embedding values is zero and the variance is equal to one and this is done for all the four
33:32
tokens so again the layer normalization does not change the dimensions and then the last layer which we have is the
Output layer
33:38
output head so the output head is a neural network at the final stage of the GPT model so the input to the output
33:46
head is this input which has four tokens and each token has a dimension of 768
33:51
then what we do is that we pass this through a neural network whose size is the embedding dimension multiplied by
33:58
the vocabulary Dimension and I'll tell you why just in a moment but if you take this input input tensor and if you pass
34:06
in through the neural network which has uh which takes in inputs of 768 dimensions and the output is
34:13
50257 the resultant uh output which you get it's called as the logits Matrix and
34:20
this logits Matrix actually has four rows because we have four tokens every
34:26
so every f effort moves you and the number of columns here is not equal to 768 the number of columns is equal to
34:33
the vocabulary size because we are passing in passing the input through a neural network whose final output size
34:41
is 50257 so for every token we have this logits which is a
34:46
50257 dimensional Vector so for every there is a 50257 dimensional Vector for
34:53
effort there is a 50257 dimensional Vector similarly for move and U there
34:58
are 5 257 dimensional vectors and the reason here is because we want to predict the next word right
35:06
based on the input so when every is the input we want to predict what's the next word and that should be effort so we
35:12
look at the vocabulary and we look at that that token or that column which has
35:18
the highest value and that column should ideally correspond to effort when every is the input similarly when every effort
35:26
is the input we we will look at the column which has the maximum value and that column corresponds to certain word
35:32
in the vocabulary and that word should be moves because when every and effort are the input moves is the output
35:39
similarly when we look at every effort moves as the input we will predict the next word and that should be U so we
35:45
look at that column which has the highest value and that should hopefully be U and finally we look at every effort
35:51
moves you that's the final prediction task and then uh when every effort moves
35:56
you is the input we look at the final output and that should hopefully be the
36:02
token which corresponds to the next word which is forward every effort moves you forward we'll see how to make the next
36:09
word prediction in the next class but for now I just want you to appreciate that this last step is the only step
36:15
where the dimensions of the input change the dimensions of the input all along where four which were the number of
36:21
tokens multiplied by 768 which was the embedding Dimension but now the final
36:26
output size four which were the number of tokens multiplied by 50257 and based on this final output
36:34
we'll get the next word prediction now when you look at every effort moves you the context size is
36:40
four whenever we have a context size we have that many prediction tasks so
36:45
remember here there are four tokens right so we don't only have one prediction task you might think that the
36:51
only prediction task is every effort moves you is the input and we have to predict the next word no there are four
36:56
prediction tasks the first prediction task is every is the input then effort should be the output every effort is the
37:02
input movees should be the output every effort moves should be the input U should be the output and only the fourth
37:08
prediction task is every effort moves you what's the output so that's why we have this four rows here because there
37:15
are four prediction tasks we'll see in the next uh next class how to predict
37:20
the next word from this output tensor which we have obtained now one thing to mention is that uh so here if you you
37:27
see the output tensor value will actually be four so let me write this
37:32
down over here the output tensor value will be four multiplied
37:40
by uh the vocabulary size which is uh
37:48
five so it's 5 0
37:54
2 5 and 7 so that is the fin final tensor value uh remember one thing which
38:00
I have considered here is only one batch if you have two batches this will be the final tensor generated for both those
38:07
batches so similar similar to every effort moves you if you have another sentence of four words such as I like uh
38:15
movies and and you have to predict the next word that's the second batch so then the output tensor will also include
38:20
the batch size so then it will be 2 multiplied by uh 4 by 5 2 57 why this
38:29
initial two because the batch size is equal to two so the First Dimension is always the batch size the second
38:35
dimension is the number of tokens which we have or the context size uh and the third dimension is the
38:43
50257 which is the vocabulary size so this is the output tensor Dimension
38:49
format which we have now let me zoom out here and show you the entire flow map which we have
38:54
seen it was a pretty long flow map but I I hope you all have understood what we
38:59
are trying to do over here the reason I went through this entire flow map starting from token embedding positional
39:05
embedding to input embedding to drop out to the Transformer block uh then we went
39:10
to layer normalization then finally we went to Output head the reason I showed you all these things is just so that you
39:16
understand dimensions and what's going on with Dimensions I did not just want to show you the code because when you
39:21
see the code you cannot visualize the dimensions but now once you have seen this lecture and when whenever you let's
39:28
say you're looking at the layer normalization part of the Transformer right you can just visualize what's
39:34
happening uh what are the dimensions of the input to a particular block what are the dimensions of the output and so so
39:42
that will make your learning process much easier so now actually uh on the
39:47
Whiteboard we have seen how the building block stack up for the entire GPT model
39:52
we started with tokenization input embedding positional we went to Transformers and then we saw the final
39:58
layer nor norm and then we also saw the final linear output layer okay so now once your intuition is
40:06
clear once your visual understanding is clear we are pretty much ready to move into code so now let us dive into code
40:13
and uh see how these different blocks can be arranged together to code the entire GPT model
Coding the entire GPT-2 architecture in Python
40:20
all right so let's jump into code right now here's the GPT configuration it's
40:26
the same configuration is gpt2 which we are going to uh use when we are going to
40:32
look at this entire coding module so here you will see that the vocabulary size is equal to
40:38
50257 this was actually the vocabulary size which was implemented when gpt2 was
40:44
trained the context length is equal to 1024 the vector embedding Dimension
40:50
which we also saw on the Whiteboard that is equal to 768 the number of attention heads is equal to 12 the number of
40:58
Transformers U which are there which we are going to implement are equal to 12 the drop rate or the dropout rate is
41:05
equal to 0.1 and the query key value bias is false this is for setting or
41:11
initializing the weight metrices for the queries the keys and the values we don't need the bias term in that for now you
41:19
can focus on a couple of things the first is the embedding Dimension uh which is going to stay 7608 throughout
41:25
the second is the 502 spice and vocabulary size these are the same dimensions which we just saw on the
41:31
Whiteboard so you might be able to relate to them uh 12 is the number of Dropout uh 12 is the number of
41:38
Transformer blocks so in gpt2 when it was trained they had 12 Transformer blocks we are also going to use a
41:45
similar number of Transformer blocks then the number of attention heads so every Transformer block has a multi-ad
41:51
attention module right and that module has a certain number of attention heads we are going to use that equal to 12
41:57
dropout rate is 0.1 I think this much should be enough to understand the entire code now as we
42:04
are going through this code I want you to just keep this diagram in mind and the dimensions which we just saw in the
42:11
visual flow map everything will follow from this particular diagram since we are exactly going to uh code it in a
42:18
similar manner okay so we started out this
42:23
lecture series four to five lectures back with this GPT model class where we
42:29
had the forward method but the Transformer block was not implemented so this was a dummy Transformer block the
42:36
layer normalization was not implemented this was a dummy layer normalization
42:42
what we did over the last three to four lectures is that we coded the layer normalization class feed forward neural
42:47
network class and also the Transformer class so here's the layer normalization class which we had coded given a
42:55
particular uh layer now you can think of when whenever you see layer normalization remember we have seen the
43:02
visual visual flow map now right so you can uh try to visualize what happens in
43:08
the layer normalization layer we have seen that already so see this is exactly what happens in a layer normalization
43:14
layer we look at every token that has 768 Dimensions right then what we are
43:19
doing is that we are going to subtract the mean and divide by the square root of the variance so that the resulting
43:26
values have a mean of zero and standard deviation or variance of one we also have a scale and a shift here which are
43:32
trainable parameters then we have the feed forward block and remember where the feed forward block comes in the
43:39
Transformer architecture it's this uh it's this feed forward block um let
43:46
me actually change my pen color to it's this feed forward neural network which I have shown you over here the expansion
43:52
contraction feed forward neural network this is that feed forward block so you'll see that the input is the
43:58
embedding Dimension the layer is four times the embedding Dimension size and the output is the embedding Dimension
44:04
once you are able to visualize this you'll be able to easily understand this code the input is the embedding
44:09
Dimension the hidden layer is four times the embedding Dimension that's the expansion layer then there is a
44:15
contraction layer from the four times embedding Dimension to the embedding Dimension and there is a Jou activation
44:22
function we spent a whole lecture on the J activation because it's a bit different than the
44:27
it's smoother on the negative side I think I also have a plot for the J activation over here so let me quickly
44:33
show you that um if I can find it yeah so on the right hand side you can see
44:40
railu it's zero for negative inputs for the J it's not zero and also it's
44:45
differentiable at x equal to Z that's why in llm training generally the J activation uh is used so we coded the
44:53
layer normalization class the j class and the feed forward class and then in
44:58
the last lecture we coded the entire Transformer block class itself so here we had the so the eight steps in the
45:06
Transformer if you remember take a look at these eight steps which we saw um let
45:12
me zoom out here further so these are the eight steps which we saw in the Transformer we start with input
45:18
embeddings add a Dropout layer normalization then uh we go to the mass
45:24
multi attention then drop out and then the shortcut connections that's the first of these eight steps in the that's
45:30
the first four of these eight steps in the Transformer block then what we do is that then we again have a layer
45:37
normalization feed forward neural network with J then Dropout and then
45:42
another set of shortcut connections these are the last four steps so overall there are these eight steps which are
45:47
happening in the Transformer block architecture and it's the same eight steps which we wrote down when we wrote
45:53
the code for the Transformer block class these are the first four steps and it
45:59
ends with the shortcut connection and these are the last four steps and here again it ends with the shortcut
46:05
connection the main point which we also recognized when we implemented the Transformer block class is that the
46:11
dimensions are preserved the dimensions of the input to the Transformer are the same as the dimensions of the output
46:18
from the Transformer block awesome now we have all the knowledge and now we are
46:23
ready to code the entire GPT architecture so first first let's look at the forward method um which takes in
46:30
the input and remember the input looks something like this let me show you how the input actually looks like the input
46:37
might look something like this where here we are showing two batches and each batch has let's say four token IDs uh so
46:44
the number of rows here in this tensor are equal to the number of batches and the number of columns are the sequence
46:50
length or the number of tokens which we are considering that's why if you see the number of rows here in the input
46:56
shape are size number of columns are essentially the sequence length or the number of
47:02
tokens whenever now I'm explaining this next whole part you can keep that flow
47:07
map in mind which I showed you on the Whiteboard and think about the flow which we had for every effort moves you
47:13
so first we convert the tokens into token IDs and then we convert the token
47:19
IDs into embedding vectors that's what's done in this token embedding uh so we initialize an
47:26
embedding layer from the pytorch module and then from this embedding layer we
47:33
get the embedding token embedding vectors for the input IDs and then similarly we get the positional
47:39
embedding vectors for the four positions so see sequence length is equal to four so we get the positional embedding
47:45
vectors for the four positions and we add the token embedding to the positional embeddings so whenever you
47:51
try to make sense of Dimensions here always think of one batch at a time when you think of one batch things will be
47:57
simplified because one batch only has four tokens let's say so when you look at this step the output of this step uh
48:04
you can visualize the output exactly from the flow map which we had seen so let me take you to that yeah look at the
48:12
first step over here which was titled as positional embedding when you look at this code the token embeddings these are
48:20
essentially uh these are essentially this kind of a
48:26
tensor where for every token you have the 768 dimensional Vector similarly you
48:31
can think of the same visualization for positional embeddings and when you look at this x equal to token embeddings plus
48:38
positional embeddings you can uh you can try to visualize this this part which we
48:43
saw we added the token embeddings to the positional embeddings right over
48:48
here so for every token we added the token embeddings and the positional embeddings to get the input embeddings
48:55
that's exactly what I'm I've written in the code over here and then after this part we go to the next steps which were
49:02
which were the same in the visual flow so the next step Next Step
49:08
was addition of the Dropout which was step number four over here I'm showing that with a star right now that was the
49:15
step number four so Dropout layer followed by the Transformer followed by
49:22
the Transformer block and then the last two layers were another layer normalization layer which which was step
49:28
number six here this was the second last layer which was another layer of
49:33
normalization and then the final layer was output head so after you get the input
49:39
embeddings there are four things you apply Dropout layer you apply the Transformer blocks you have the another
49:46
layer of normalization and then you apply the output head that's it now if you see the Transformer blocks
49:54
we are chaining different Transformer blocks together together based on the number of layers so in the configuration
50:00
we have seen that the number of layers is equal to 12 right in this configuration we have seen that the
50:05
number of layers is equal to 12 so uh we are actually chaining 12 Transformer
50:11
blocks together using NN do sequential this NN do sequential is a tensor flow or pytorch module which allows us to
50:18
chain um different neural network blocks together so we have chained 12
50:23
Transformer blocks so when you see the cell trf blocks it just looks like one line
50:29
of code right but there are actually 12 Transformer blocks chain together here and in each Transformer block there are
50:36
eight different steps which are being performed so it's a huge number of operations being performed in this one
50:41
simple line of code that's it actually if you think of the building blocks if you have
50:47
understood the Transformer block if you have understood the layer normalization class uh if you have understood the
50:53
dimensions with respect to the Token embeddings positional embeddings that's all what
50:58
the um Transformer or what the GPT model
51:03
class is actually doing it takes in the input and then the outputs are the logits which we saw on the Whiteboard so
51:11
the dimensions of the logits if you remember let me take you to the Whiteboard once more so if you go to the
51:16
Whiteboard and if I zoom into this part of the if I zoom into this part of the Whiteboard right now you'll see that the
51:23
output of the GPT model is this output Logics and the shape of these logits is for each batch we have the number of
51:30
tokens uh in this case those were four tokens and the number of columns are equal to the vocabulary size and then
51:37
the First Dimension is the number of batches which we have okay so now what we can actually do
Testing the GPT model class on a simple example
51:43
is that we can take a simple input batch and then we can pass in through this entire GPT
51:49
model uh so what we are doing here is that we are taking an input batch which has every which has two batches and each
51:56
batch has two tokens what we are doing is that first we create an instance of the GPT model and pass in the model
52:02
configuration and then we create this object so we create an instance of model
52:08
uh which Returns the output so if you print out the input batch you'll see this and if you print out the output
52:14
batch let's look at one batch currently if you look at one batch you will see that there are four tokens and each
52:19
token has number of columns equal to the vocabulary size which is 50257 this is exactly what we had seen
52:25
in the output logic right so if you see in one batch there are four tokens four rows and there are 50257 columns this is
52:33
exactly the same thing which is happening for the second batch also there are four tokens here and there are
52:39
50257 columns because the vocabulary size is 50257 so as we can see the output tensor
52:46
has the shape uh two batches four tokens in each batch and 50257 columns since we
52:53
passed in two input text and four tokens each the last Dimension 50257 corresponds to the vocabulary size
53:00
of the tokenizer in the next class we are going to see how to convert these 50257 dimensional output token vectors
53:07
back to tokens and how to predict the next word but for now you can see that
53:12
the trans the GPT model class which we have constructed is working fine and there are a huge number of parameters
53:19
which we are actually dealing with here this small piece of code over here has more than 100 million parameters can you
53:25
imagine that there are just a huge number of parameters because there are 12 Transformer blocks chained together
53:31
there are eight steps in each Transformer block then uh and remember there is that expansion contraction
53:38
layer in each Transformer block that that layer itself has a huge number of parameters we have parameters for token
53:45
embedding positional embedding uh Etc so totally all of the parameters add up we
Parameter and memory calculations
53:51
can actually print out the number of parameters so what we can do is that we can have have total params and we can do
53:58
p. numl what p. numl will do is that it will actually print out uh so using the
54:04
numl method short for number of elements we can collect the total number of parameters in the model's parameter
54:09
tensors so if you print out the total number of parameters you will see that the number of parameters is equal to 163
54:16
million huge number of parameters right and all of this is running on our local
54:22
machine so you might be thinking earlier we spoke of initializing a 12 4 million parameter GPT model right so then why is
54:29
the actual number of parameters 163 million why is it so higher so this actually relates back to
54:37
a concept which is called weight time that is used in the original gpt2 architecture so which means that the
54:43
original gpt2 architecture is reusing the weights from the token embedding layer in its output layer so if you go
54:51
back to the Whiteboard right now yeah in the schematic which we saw over here there is an output layer right there is
54:57
an output layer towards the end and there is a token embedding layer over here when GPT model when gpt2 model was
55:04
constructed the parameters which were used for the token embedding layer were the same parameters which were used in
55:10
the linear output layer and that's why uh the total number
55:16
of parameters was less in our case right now we did not reuse the parameters so when we print out the parameters they
55:22
say 163 million uh but in gpt2 architecture they are reusing the weight
55:27
so we can actually print this out so in our case what we can do is that we can print out the embedding layer shape of
55:33
the token embedding layer and we can see that um the embedding layer shape is 50257 comma 768 and we can also print
55:42
out the output layer shape and these both have exactly the same shape so that's why in gpt2 model they just
55:48
reused this these parameters um so what we can actually do
55:54
is that we can um let we can remove we can remove the output layer parameter
55:59
count from the total parameter count and check the number of parameters so in the output layer parameter parameters are
56:06
removed the total number of parameters comes up to be exactly 124 million parameters which is the same size of the
56:12
gpt2 model so weight Ty actually reduces the overall memory footprint from 163
56:19
million we go to 124 million and it also reduces the computational complexity of the model however using separate token
56:26
embedding and output layers results in better training and model performance that's why when in our code which we are
56:32
writing we are using separate layers in our GPT model implementation and this is true for modern llms as well weight time
56:40
is good for reducing the memory footprint and reducing the computational
56:45
time and complexity but to get a better model training and performance it's good
56:50
not to reuse the parameters um we can also print out the space which is taken by our model
56:58
so we can compute the memory requirements of the 163 million parameters so the total size of the
57:04
model it turns out it's around 600 megabytes of space so in conclusion by
57:09
calculating the memory requirements for the 163 million million parameters in our gpt2 model um and assuming that each
57:17
parameter is a 32-bit float taking up to four bytes that's why we have multiplied by four over here we have assumed that
57:23
each parameter takes around four bytes so we multiplied the total number of parameters by four to get the total
57:29
size that 620 MB which illustrates the relatively large storage capacity
57:34
required to accommodate even relatively small llms so gpt2 is relatively small
57:39
120 million compared to there are now models which have more than b billion parameters imagine how much space such a
57:47
model will take on your device it's impossible to run extremely large models on local devices and that's why we need
57:53
GPU access so in this lecture today we implemented the GPT model architecture and saw that
Conclusion and summary
58:01
it outputs numerical tensors right so you might be curious how do we go from that numerical tensor output into the
58:08
next word prediction task uh so that we'll be doing next in the next lecture we are going to
58:14
generate text from the output tensor which we have obtained today today what I want to what I want you to appreciate
58:20
is just within 10 lines of or even 15 20 lines of code we have actually run a
58:25
very power powerful large language model completely on our laptop and we buil this model from scratch this model had
58:32
more than 160 million parameters it took 600 megabytes of space on our machine
58:38
indicating the size required by these models and I want you to appreciate that once you understood the visual flow map
58:44
I showed you on the Whiteboard it's just eight lines of code but to understand these eight lines of codes we need to
58:50
put in a lot of effort because there is a lot of theory and lot of intuition behind every single line of code over
58:57
here but once you get that it just these uh stacking of the token embeddings
59:02
positional embeddings Dropout layer Transformer blocks another layer normalization and the output output head
59:08
that's it and then we directly get the Logics um I hope you are following these
59:14
lectures I know this lecture is also become a bit long but it is essential to show you the theory as well as the code
59:21
in today's lecture I could have directly showed you this code which was only this much but then at every step you would
59:27
not have visualized the dimensions what is exactly happening can we take a specific example and see what is
59:32
happening for that example and now I've given you a visual understanding intuitive understanding as well as a
59:38
coding understanding in the code I written uh the description of what what
59:43
is happening in every layer for example here you will see that the forward method takes a batch of input tokens
59:50
computes their embeddings applies the positional embeddings passes the sequence through the Transformer block
59:56
normalizes the final output and then computes the logits representing the next tokens unnormalized
1:00:04
probabilities we will convert these Logics into tokens and text outputs in
1:00:10
the next class but whenever this when you access this code file Ive also written a number of such comments and
1:00:17
paragraphs here which you can read and try to understand the code for yourself
1:00:22
one thing which I will very very highly suggest is that whatever I written on the Whiteboard
1:00:28
right when I was I was myself writing the flow map of the Transformer on the Whiteboard this this flow map I
1:00:35
understood that my understanding also became very clear if I just ran the code
1:00:40
my understanding was not that strong but when I wrote every single step of the code on a whiteboard um and when I
1:00:48
created a visual flow map uh which I demonstrated to you in today's lecture
1:00:54
such kind of visual flow map which I'm showing right now on the screen it really strengthened my understanding and
1:00:59
it made me confident about this subject if you can also write this similar flow map on your notebook or on the
1:01:06
Whiteboard which you are using it will tremendously improve your understanding thanks a lot everyone I hope you are
1:01:12
enjoying from this lectures next lecture will probably be the last lecture in the GPT architecture series uh where we'll
1:01:19
generate text from the final output tensor which we have obtained today thanks a lot everyone and I look forward
1:01:24
uh to seeing you in the next lecture

***
