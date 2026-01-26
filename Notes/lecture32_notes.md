etrained weights
0:00
[Music]
0:05
hello everyone and welcome to this lecture in the build large language models from scratch Series today I am
0:13
very excited for this lecture because we are going to look at how to load pre-trained open AI weights into the GPT
0:21
model which we have constructed so that the text generation
0:26
is coherent this lecture will serve as a culmination of all the hard work which
0:32
we have put in in the previous lecture previous lectures U because we are going
0:37
to use the exact GPT architecture which we have built ourselves using this
0:43
schematic which I'm showing you right now and the GPT model class which we have defined in this lecture series and
0:50
we are going to integrate this model class with the gpt2 open AI weights
0:56
which they have publicly released so let's get started with today's
1:02
lecture first let me cover what all we have completed so far until now what we
1:08
saw in this pre-training series is that we first saw how to define the losses for a large language model how the cross
1:16
entropy loss comes into the picture to find the loss between the
1:21
llm predicted output and the target sentence then we looked at the llm
1:26
pre-training Loop itself and we generated output text using our pre-training Loop the text which was
1:33
generated from our Loop was not not very coherent so let me just show you uh the
1:39
text which was generated using our training so this was the training Loop which we had and here you see that the
1:45
input was every effort moves you and the output was not very coherent it was not
1:50
making too much sense and that is understandable because we just we had just trained this on one small book with
1:58
a which was a very small data set and we had run it for 10ox later what we saw was we integrated decoding strategies to
2:05
reduce overfitting the first strategy we looked at was temperature scaling and
2:11
then the second strategy which we looked at was topk sampling introduction of
2:16
these two strategies definitely reduced overfitting a bit but even then the next sentence which was generated as you can
2:22
see over here did not really make too much sense and now we have come to today's lecture where we are going to
Open AI GPT-2 weights
2:29
load the pre-trained weights from open AI in in an in in the hope that with the
2:35
pre-trained weights from openai the next tokens which are generated or the next Tok or the next sentences which are
2:41
generated they start making a lot more sense we are going to spend a lot of
2:47
time in today's lecture in understanding first of all how to download these weights from open a second of all how to
2:53
integrate these weights with our code which we have written already so let's
2:58
Dive Right into today to lecture first I would like to thank open for releasing
3:04
these weights for gpt2 the training itself might have taken around millions of dollars because the training has been
3:11
done on a huge amount of data set and these gpt2 weights are now publicly available even on platforms such as
3:18
gagle so without access to this weights today's lecture would not have been possible and I would just like to take
3:26
this opportunity to thank all of the llm company who are in the open source
3:31
domain especially and who are publicly making available all the weights which are used by these
3:37
models okay so I'm going to take you to directly to code right now and let me
Loading libraries
3:42
start explaining to you the different steps in the code so previously for
3:48
educational purposes we trained a small gpt2 model right which used a very limited data set it was just a book this
3:56
approach allowed us to focus on the fundamentals but we did not get coherent text as the our prediction what we are
4:04
going to do is that open a has fortunately shared their weights publicly so we are going to load these
4:09
weights into our GPT model class and we are going to use this model class itself
4:15
for the next text Generation Um one thing to note before
4:20
we get started is that open originally saved the gpt2 weights via tensor flow
4:25
whereas all of the coding which we have done so far in this lecture series is using pytorch so we will have to install
4:32
tensor flow and one more Library which we are going to install is called tqdm to track the download progress so here
4:40
you can see that in this line of code I'm installing tensor flow and I'm also installing the tqdm library the
4:46
tensorflow version should be greater than or equal to 2.15 and the tqdm
4:51
library version should be greater than or equal to 4.66 so you can install both of these
4:57
two libraries and then import them so here I printed the tensorflow version and the tqdm version on my
5:04
system once you install this these two libraries you can you can pause the video for the time being and then you
5:10
can continue with the rest of this video now comes the very important step of
Gpt2 weights downloading introduction
5:15
downloading the gpt2 weights and their parameters this is a step which still


***


5:22
might be complex to so many students and Engineers so I want to explain this in a lot of detail first of all when you go
5:29
to platforms like kagle you will find files to download so these are the seven
5:34
files um which contain all of the information basically and you'll see that the file size is about 500
5:40
megabytes but after you download this there is a number of pre-processing steps which need to be done before the
5:47
weights can be integrated with our GPT architecture and I'm going to explain
5:52
those steps to you right now so let me take you through to the vs code interface where we are going to
6:00
look at this uh GPT download 3 function and or this file this code file which
6:07
will help us download um these seven files and not just download we are going to extract
6:13
the parameters from these seven files and then we are going to store them in a very specific format all right so here
6:20
you can see the GPT download 3. py file and first of all you'll see that when
6:26
you open this file it has three functions it has the down load and load gpt2 function which is the main function
6:32
which we are going to look at then this function utilizes two helper functions
6:38
the first is called download file and the second is load gpt2 params from the TF checkpoint so as the file names
6:45
itself suggest what this main function is going to do is that it's going to do two things first it's going to download
6:52
the seven files these seven files which I just showed you on kagle also it's going to download those files and it's
6:58
going to save them on my Lo loal computer that's what it's going to do as the step number one and then in the
7:04
Second Step what we are going to do is that we are going to take the downloaded file and then we are going to call this
7:10
load gpt2 params from uh TF checkpoint
7:16
and uh what this function is going to do is that this function is going to store
7:22
the parameters in the dictionary called as params and this dictionary has a very
7:27
specific format which we are going to see just in in a moment so just downloading the data from kaggle is not
7:33
enough you need to understand the rest of the code and the format in which this params dictionary is returned so let's
7:40
start understanding this code sequentially before that I just want to take a moment and explain these
7:45
downloaded files so once you run this code you will see that you'll get these files which are downloaded on your local
7:51
machine without even running this code you can even go to kaggle and download these seven files now if a student has
Understanding the gpt downloaded files
7:58
not been through this llm lecture series which we have developed they will not understand these files and they might
8:04
seem a bit complicated these files but for us for those students who have
8:10
followed this lecture series I want to show you that now these files are very easy to understand so I want to explain
8:16
each of these files sequentially the first thing is checkpoint so if you go to this file named checkpoint you will
8:23
see this thing called Model checkpoint path what this essentially means is that this is the path where all the current
8:30
parameters the weights of the gpt2 model are stored so you'll see that model. CPT
8:36
we have three files named model. CPT the most important file is model. cp. dat
8:42
this is where all the weights of the gpt2 model are stored so this checkpoint
8:47
is just going to indicate the path at which the model weights are stored the second file is encoder do Json so let me
8:55
come to the top of this file this file as a whole looks complex but it's actually a vocabulary it's a vocabulary
9:01
of keys uh and token IDs so we have
9:07
tokens here and corresponding to every token there is a token ID that's the vocabulary which we are using and if you
9:13
remember the vocabulary size it's 50257 so it starts from zero and then the end of text is 50256 so there are
9:20
50257 tokens in our vocabulary then the second file is w.
9:26
bpe remember if you have followed this lecture series we learned about the bite pair encoder it's a subword tokenization
9:33
scheme and the way this encoder works is that it looks for pairs of tokens which are occurring the most frequently and
9:39
then it merges this pair and that becomes a token in itself that's why it's called subord tokenizer so this W
9:46
cap. BP gives a list of the tokens which have been merged with the highest
9:51
probability at the top so all the tokens which have been merged have been mentioned in this list and the tokens
9:57
which have which come with the maximum probability they at the top of this list so you might be thinking what is this g
10:03
dot right this g dot is a special convention uh which basically indicates
10:08
that one token has ended and a new token has begin so you can think of it as a space and which marks the beginning of a
10:15
new token so it's a special convention used by open AI to indicate the end of one token and the start of a new
10:22
token then the third file which I want to show you is ham. Json this is the
10:27
file which essentially contains all the the settings of our model the vocabulary size which we are using is
10:33
50257 the context length which we have is 1024 the embedding Dimension is 768
10:40
what is the embedding Dimension essentially all the token IDs which you see over here if you see any any token
10:47
every token has a token ID right and every token ID will be converted converted into a 768 dimensional Vector
10:54
space uh these vectors are not trained and our training will involve parameters


***



10:59
corresponding to these as well then n heads so n heads is basically the number
11:05
of attention heads present in each Transformer block and N layer is
11:11
essentially the number of number of Transformer blocks itself now remember
11:17
these are the these parameters the setting values which I'm showing you
11:22
they are for the smallest uh gpt2 model when open made the GP B2 weights public
11:30
they actually made the weights of their larger models also public so 124 million 355 million
11:37
774 million and 1558 million the weights of all of these models were made public
11:43
but for the sake of Simplicity we are going to look at 124 million model right now which had 12 Transformer blocks as
11:49
you increase in the complexity of the gpt2 model the number of Transformer blocks also increase remember that the
11:56
same code which I'm going to show you today can be run for these larger architectures as well okay so I hope you
12:03
have understood the meanings of all of these files now let us start going through uh this download and load gpt2
Understanding the gpt-2 download code
12:10
code step by step and understand every single sentence so the first thing is that we
12:16
mention the URL so we'll make a call to this API and we'll download these files
12:21
download the seven files which are present on my local computer right now so the download file function will be
12:27
called for this and I'm not going to go through this in detail because this is just downloading the file onto the local
12:32
machine the real interest which I have is showing you what happens after the file is downloaded so after the file is
12:40
downloaded first we have this PF checkpoint path which is uh the latest
12:45
checkpoint directory where all the model parameters are stored so what this function will do is that tf. train so
12:52
tensorflow do tr. latest checkpoint model directory so it will go to our directory and it will look for this
12:58
checkpoint and it will look for this model checkpoint path which is model. CPT so it will know where the model
13:04
parameters are stored awesome the next thing is that settings we are going to maintain
13:10
another dictionary which is called as settings and what the settings dictionary is that it will exactly
13:16
contain the same thing what is present in the ham. Json so it will contain the vocabulary size it will contain the
13:22
context length the embedding Dimension the number of attention heads and the number of Transformer blocks
13:30
uh so let me go back to the code again right so this is the settings dictionary
13:35
and then in this last line of the code this is where all the magic is essentially happening and you really
13:41
need to go into depth to understand what happens in this one line of code so here what we are doing is that uh we are
13:47
looking at the model path which is given by this TF ckpt checkpoint answer flow
13:53
checkpoint path and we are going to take the parameters from that path and we are
13:58
going to use the settings dictionary and then the we'll return a params dictionary and the code which does that
14:05
is this load gpt2 params from TF checkpoint so from the TF checkpoint parameters we are going to load the
14:12
parameters into a special dictionary called as params before explaining to you what is
14:18
happening in this code I first want to show you what the params dictionary looks like all right so here is how the
Understanding the gpt-2 parameter dictionary
14:24
params dictionary actually looks like the dictionary will have five Keys which I'm going to show to you right now what
14:31
are these five keys and what do they exactly mean the first key which this dictionary is going to have is wte the
14:38
second key is WP the third key is blocks the fourth key is the final
14:44
normalization scale and the fifth key is final normalization shift to really
14:50
understand these five keys and why do we need these five Keys you need to understand the GPT
14:55
architecture um and we have spent a lot of time on this I just just want to summarize it quickly so that you can
15:01
relate to these keys so whenever a token comes in whenever sentence comes in rather we have to first convert the
15:08
input token IDs into token embeddings so this will require weights this will require parameters which need
15:14
to be trained then we will need to add positional embeddings these are also parameters which need to be trained the
15:21
result of token embeddings plus positional embeddings is input embeddings and these pass on to the
15:26
Transformer block so already you understood the purposes of the first two keys this is for the token embeddings
15:32
parameters the wp is for the positional embedding parameters now after we get
15:38
the input embedding we have the Dropout layer Dropout layer has no trainable weights so we don't have any Keys
15:44
corresponding to that then we move into this Transformer block the Transformer block has several places where trainable
15:51
weights exist the first place is the multi-head attention layer we have queries keys and the values weight
15:57
matrices over here right and these metries consists of parameters which we don't know so this is one scope
16:05
for trainable parameters the second scope for trainable parameters is the speed forward neural network remember
16:11
this speed forward neural network has an input so there's an input which comes in then it passes through hidden layer the
16:18
the number of dimensions in the hidden layer are four times the input and then we have a final output layer which
16:24
projects it back to the same size so this first layer
16:29
uh so this first layer of the neural network is is the fully connected layer so I'm going to name it as FC and the



***



16:37
second is the output projection layer so I'm going to call it as P both of these layers will have trainable weights so
16:43
that's the second scope for trainable weights now if you look at the layer normalization one and layer
16:48
normalization 2 let me highlight them with a different color layer normalization um let me choose purple
16:55
color so if you look at layer normalization 2 and if you look at layer normalization one normally layer
17:01
normalization does not have any weights because we just subtract the mean and divide by the square root of variance
17:07
right but the way we have defined layer normalization is that after we do the scaling after we subtract the mean and
17:13
divide by the square root of variance we also multiply with a trainable parameter
17:19
called scale and we add a trainable parameter called shift it turns out that these two parameters actually make a big
17:26
difference so that's why we have uh layer normalization 2 and layer
17:31
normalization one also which come into the trainable weight parameters category and then we have the final output layer
17:38
the final normalization layer which is another trainable weight Matrix category where wherever we have parameters to
17:45
train those are Keys corresponding to this parameter dictionary so you already
17:50
saw token embeddings and positional embeddings blocks corresponds to all the
17:55
trainable weights and parameters in this Transformer block in this blue color Transformer block and then G and B
18:02
correspond to the trainable weight parameters in this final layer Norm which is indicated as four right
18:08
now so uh token embeddings right I hope you remember what what are token
18:14
embeddings we have a vocabulary size of 50257 right and corresponding to every
18:21
token we have a vector whose Dimension is 768 so the size of this is 50257 rows
18:27
and 768 columns these are token embeddings now positional embeddings are governed by the context size in gpt2 the
18:35
smallest model which we are considering the contact size is 1024 because maximum we can look at 1024
18:42
positions and then make the next token prediction the one 25th position does not matter to us so we only need for we
18:50
only need embeddings corresponding to position number one position number two up till position number
18:56
1024 and these embeddings each of these embeddings also has a dimension of 768 because we need to add the token
19:03
embeddings to the positional embeddings the vector dimension of both of these need to be exactly the same and that's
19:09
there because it's 768 in both of these cases now so this WT key will have all
19:16
these values corresponding to the Token embedding Matrix the wp key will have all the values corresponding to the
19:23
positional embeddings Matrix now let's come to the blocks when you come to the blocks there are several things which
19:28
which are happening in the block so let me just rub this part a bit so that
19:33
things are a little more clearer over here great first we are going to look at the Mast multi-ad
19:40
attention right this has the query this has the key and this as the
19:46
value in GPT 2 when they release the weight they fuse this into one single
19:52
big Matrix so we need all the weights corresponding to this large Matrix so that later we can split it into query
19:58
key and value trainable Matrix so the way it is done in this blocks dictionary
20:04
is that there will be a dictionary called or there will be a dictionary called parameter within that there will be a keys blocks this will link to
20:11
another dictionary which has the keys Transformer this will link to another dictionary which is the keys H not why H
20:18
not because there are 11 such 12 such Transformer blocks remember gpt2 has 12
20:24
Transformer blocks so whatever I'm showed you right now that will be replicated 12 times right so there are 12 Transformer blocks
20:31
the first one is h0 then there will be a key called ATN why because we are looking at the attention mechanism right
20:39
and then and this attention mechanism also has the output projection so we
20:44
need to specify in the attention mechanism what we are looking for in the attention mechanism currently we are
20:51
looking for CN what is CN it is the fused Matrix of query key and value and



***



20:58
within the this fuse Matrix we are looking at the weights So within this fuse Matrix we are looking at the
21:03
weights so you see that's how we access the weight Matrix of the um attention
21:08
layer so we look at the blocks key within that Transformers so let me show it to you here now we look at the blocks
21:14
key within that we look at the Transformers within that we look at H not then attention then C attention
21:20
which is the fused query key value and then we look at the weights similarly we do all this for the
21:26
biases of the qu key quy and value this is only for one Transformer Block H we
21:32
are going to repeat this for H1 H2 up till h11 since there are 12 such Transformer blocks so this is how you
21:39
access the weights in the attention layers of the Transformer blocks you access this is how you access the
21:44
weights and biases of the attention layer in the Transformer block uh and when I say attention layer I I mean the
21:51
query key and the value weight matrices and the biases corresponding to each of these right so that's the first
21:58
component which we looked at the second component is this feed forward neural network now here if you look closely
22:05
there are two layers there's a fully connected layer and there is a projection layer so we'll have weights and biases corresponding to this first
22:12
layer as well as the second layer and that is clearly mentioned here if you look at the fully connect feed forward
22:19
neural network you have the Transformer Keys within that you have the H not which is the first Transformer block
22:25
within that there is a key called MLP multi-layer perceptron because we are looking at that expansion contraction
22:30
neural network currently we are accessing the fully connected layer and the weights of that layer fully
22:36
connected layer the biases of that layer so what I'm highlighting right now corresponds to the first the fully
22:42
connected layer and its weights and biases similarly to access the weights and biases of the projection layer we
22:48
just have the key as ccore PJ and we access the weights and the biases so this is how we access the
22:55
weights and the biases of the feed forward neural network work within the Transformer block okay and then usually
23:02
at the end of the Transformer it's not shown here but we we actually have this output projection
23:07
head this output this output
23:13
projection and there are weights associated with that also so to access the C projection in the OR to access the
23:21
output projection layer we just do Transformers then we go to hn the first
23:26
block then go to ATN c pro which is the output projection layer and we get the weights and similarly we do this for H1
23:33
H2 up to h11 so the 12 Transformer blocks so this is how you get the weights and the biases of the attention
23:40
layers of the feed forward neural network and the output projection layer
23:45
in the Transformer block but remember the Transformer block has two additional things it has the layer Norm one and
23:50
layer Norm two and both of these have trainable scale and shift parameters
23:56
right so we need to access those so the last thing which we are going to access in the Transformer block keys so these
24:02
blocks Keys is that we are going to access Transformer slh not/ layer
24:08
normalization one then the scaling which is denoted by G and the shifting which is denoted by B this is the scale
24:15
parameter and B is the shift parameter similarly with respect to layer normalization 2 we will get the scale
24:23
parameter denoted by G and we'll get the shift parameter denoted by B remember
24:28
that all of these are sub dictionaries within the blocks dictionary and within the subd dictionaries ultimately we
24:34
access the parameters so if you look at the blocks blocks Keys within the block Keys we are
24:41
actually getting four things we are getting the attention layer weights query key value weights we are getting
24:47
the feed forward neural network weights we are getting the output projection layer weights and we are getting the
24:52
layer normalization weights so essentially with this we get all the trainable parameters within the trans
24:58
Transformer block itself awesome but our task is not yet over because when we come outside of the Transformer block
25:05
there is this final layer normalization right and actually let me Mark it with a different color there is this final
25:11
layer normalization and similar to the layer normalizations earlier it will have the scale and the shift now
25:17
remember that there are entirely different keys for the final normalization layer scale and that's the
25:23
key named G and for the final normalization shift there is an entirely new key which is is called B so the
25:29
params is the dictionary and then when you do params B params B you'll get the
25:35
final Norm shift parameters and if you do params of G you'll get the final Norm
25:43
scale parameters but to access let's say the attention head what you'll need to do is that I'm just showing a part of
25:49
the code which lies ahead but I think it is important to get the attention head you'll do params blocks so you'll access
25:56
the blocks Keys then you'll specify that block number 1 2 12 Etc then you will go
26:02
to the ATN Keys what we mentioned over here ATN
26:07
Keys then we'll go to the C ATN keys and then we'll go to the W key this is how
26:12
we'll access the weights of the query key and the value uh matrices in the attention block attention layers so
26:20
that's why we need all of these five Keys which are returned by the parameter dictionary so the whole goal
26:26
of this GP load gpt2 params from the tensorflow checkpoint is to get the
26:32
parameter values uh from this checkpoint and then convert the parameter values
26:38
into this params dictionary so here you see we Define the params dictionary which is empty currently and we first
26:44
only Define the blocks keys and then we fill the blocks keys with every attention layer the feed forward neural
26:50
network the output projection head all of these are already present in the model checkpoint but we just need to put
26:56
them in the appropriate values I'm not going to explain this part of the code but because the main learning lies in
27:02
you understanding these five keys so this is how the blocks Keys is filled up
27:08
and similarly all the other Keys wte WP uh G and B they are already present
27:14
in this uh model checkpoint path so we just augment the params dictionary with
27:20
all of those keys so when you finally execute the load gpt2 params from TF
27:26
checkpoint which is mentioned over here the params dictionary will have those five Keys which I mentioned to you on
27:32
the Whiteboard this is what is happening in this piece of code and then when you finish this function you return return
27:38
two things you return return the settings dictionary which consists of the vocabulary size context length
27:44
embeding Dimension number of attention heads and number of Transformer blocks and you also return the params which is
27:50
the params dictionary consisting of the five Keys which I just showed to you on the Whiteboard I could have just skipped
27:57
this part but then I really wanted to show you the nuts and bols of how the downloading is done if you want to do
28:03
research in large language models it is very likely that you will need to download the pre-trained weights to do
28:09
uh some testing or some training and for that you really need to understand the format in which gpt2 releases these
28:16
weights if you don't understand the format and if you don't understand how to convert this format into this
28:23
parameter dictionary it will be difficult to do novel research so I hope you have understood this this part uh
28:29
now let me move back to the go uh to the Jupiter notebook and uh until now we
28:36
have reached this stage where uh right up till here where we will now what we'll be doing is that from this GPT
Downloading gpt-2 weights into Python code
28:43
download 3py this python file we'll import the download and load gpt2 function it this
28:49
function the download and load gpt2 function and then what we are going to do is that we are just going to run this
28:55
function we have to pass two things we have to pass the model size because remember that the model size can be 355
29:01
million 774 million and 1558 million also and I encourage you to experiment
29:07
with this after today's lecture is over so we put in this model size and we specify the directory so I have
29:13
specified the directory to be gpt2 so here you can see in the folder name gpt2 all of my files have been stored and
29:20
then what you can do is that you can run this piece of code so then settings dictionary as we saw we'll get the
29:26
settings dictionary and we'll get the par dictionary when you run this piece of code now when you run this as I told
29:32
you this total size of all of this is around 500 megabytes initially when I ran this code it took a very long time
29:38
on my laptop because my laptop kept crashing it was not in a good internet
29:44
area and then I moved to another place where the internet connectivity was a bit strong so here you can see I was
29:50
getting speeds of 5 225 mb per second and then this entire loading took around
29:55
5 to 10 minutes so I encourage you to SA sit in a place with a good internet connectivity and don't restart your
30:01
session or close your laptop during this time because once this is loaded the rest of the code proceeds in a very
30:07
smooth manner this is the most time consuming part of the code until this point now let's say this code is
30:14
executed after it's executed since we have loaded this tqdm Library we'll see the progress which is happening so here
30:20
you can see that I've have reached 100% in all of the different steps uh so after this code has been completed you
30:27
can inspect things you can inspect the settings dictionary and you can inspect the parameter dictionary keys so if you
30:33
print out the settings dictionary you'll see that it has keys like n vocab nctx n embed n head and N layer now as I
30:41
mentioned this this is exactly the same as the ham. Json file here it's just
30:47
being converted into a dictionary now uh and if you print the parameter dictionary Keys you will see blocks b g
30:54
WP and wte we learned about this EXA ly on the Whiteboard where we saw that the
31:00
parameter dictionary will have these five Keys wte WP blocks G and
31:06
B awesome so I hope you have understood until this part where we have actually loaded uh the gpt2 architecture right
31:15
now we have loaded all of the parameters into our laptop and the
31:21
parameters seem to be loaded correctly what we can also do is that uh we could have printed the
31:28
um parameter weight contents but that would take a lot of screen space hence we only printed the parameter dictionary
31:34
keys not its values but we can go a step ahead and look at the params dictionary
31:39
and print out the wte which is the key corresponding to the Token embedding
31:45
vector and we saw that the dimension should be 50257 rows 768 columns let's
31:51
just see if the dimensions make sense so if you if you access the params dictionary with the key wte you get this
31:58
tensor whose shapee is 50257 and 768 at least the dimensions seem to be
32:03
making sense great so these values which you see on the screen right now they are
32:09
optimized values which means that for every token the token embedding weight
32:14
Dimension encodes some semantic meaning again we should be thankful to open a for releasing the weights publicly
32:20
because they would have spent about a million dollars or even more for this pre-training awesome so as a I told you
32:28
we could have also downloaded the 355 million 774 million or 1.5 billion parameter which is this release which
32:35
gpt2 had made and you can feel free to experiment with that but we have loaded the 124 million parameter now before
32:42
moving forward one change which we'll need to do is until now when we use the GPT configuration in this lecture series
32:49
we used a GPT we used this thing called GPT config 124 million and the
32:54
configuration was almost exactly same as what's actually used in gpt2 except that
32:59
we used a context size of 256 whereas the actual context size is 1024 so we'll
33:05
need to change that so what we are going to do is that we are going to say that the new configuration is the same as our
33:11
old configuration but we'll update the context length to be 10 to4 and the
33:16
second thing which we are going to update is the query key value bias so when we trained the attention mechanism
33:23
and when we run our own llm before we have put this query key value bias to false but in gpt2 this was actually put
33:30
to true so we are also going to put this to True uh here I have added a small note that uh bias vectors are not
33:38
commonly used in llms anymore because they don't improve the modeling performance and they are not that
33:44
necessary however since we are working with pre-trained weights we need to match the settings for consistency and
33:50
that's why what we are going to do is we are going to enable the query key value bias to be equal to true and we are
33:56
going to use the context l to be 1024 so then we uh create an instance of
34:01
the GPT model class with this new configuration I just want to show you the GPT model class which we have so
34:08
that it is on the screen in case you have you coming to this lecture for the first time we have
34:15
developed a GPT model class which looks something like this yeah this is our GPT
34:20
model class uh and now the main goal which we have is how are we going to
34:26
integrate the weights which we have downloaded with the GPT model class which we have defined so let's learn about that a bit
Integrating the gpt-2 weights with our LLM architecture
34:34
so there is a specific way in which we are going to do this integration so look at the GPT model class what we are doing
34:40
currently is that we are just initializing the token embedding matrices the positional embedding matrices the Transformer blocks weights
34:48
we are initializing them to random values but now our main goal is that the
34:53
weights which we have downloaded from gpt2 and which are currently stored in this params dictionary which we have
34:59
returned we need to somehow make sure that these weights are integrated with our GPT model class and instead of these
35:06
random initializations using NM do nn. embedding we actually make the
35:11
initializations uh from the downloaded gpt2 parameters so for that we first
35:16
need to look at the Transformer block and I want to show you a couple of things in this uh Transformer block so I
35:24
just control F here and searched for the Transformer block um yeah so here's the Transformer block
35:30
okay what we are going to do in the code is that here you can see that there is a object called attention so that's a
35:37
instance of the multi-ad attention class what we are going to do is that we are going to take this object and we are
35:42
going to make sure that when you define this at object the query key and the
35:47
value matrices are assigned to the query key and the value matrices which are obtained from the parameters
35:54
dictionary uh from this dictionary over here this the attention layers from this
36:00
dictionary similarly when we look at the feed forward neural network FF object we
36:05
are going to make sure that this feed forward neural network receives values from this feed forward neural network
36:12
weights dictionary which we have in the parameters dictionary so let me again take you back
36:19
to the current code it's a bit down below but let me scroll down below so that um you understand what's really
36:27
going on one awesome so now what we are going to do as I said is that we are going to link our GPT model class with
36:33
the downloaded weights from open AI gpt2 so the way we are going to do this is
36:39
that first let's take a look at the attention block right let's take a look at the attention block and let's take a
36:45
look at the queries the keys and the values so what we are doing here is that first let's access the queries keys and
36:52
the values downloaded from open a gpt2 and the way to access it as we have already seen is that you go to the
36:59
params dictionary you go to the blocks Keys then you go to the Transformer sub Keys the hn the ATN the C ATN and the W
37:08
this is exactly how we are accessing these weights but remember these weights are Fusion of queries keys and the
37:13
values so we are going to split these along the columns and then we'll get the queries weight Matrix the keys weight
37:19
Matrix and the values weight Matrix as I told you before we are going to get the at object remember I showed you in the
37:26
Transformer block class the at object and then in that object I'm going to assign the queries the key and the value
37:34
weight equal to the qore W the Kore W and the Vore W which has been obtained
37:41
from open a gp22 that's it it's as simple as that this right here is the assignment
37:47
step and the A and assign is the function which we have defined here what this assign does is that it takes left
37:53
and right and uh it will first check whether these two values the shape is matching and if the shape is matching we
38:01
just return the right values which means the left is just assigned the value equal to the right and then we return it
38:06
if the shape does not match we it will give us an error and that means that we are not loading the gpt2 weights
38:13
correctly and not assigning them correctly so this is the part where the trans uh where the attention block query
38:20
key and the value weight matrices are updated similarly in this part the bias
38:26
is updated so it's the same as the earlier part but then W is replaced with b um to update the bias
38:34
terms now if you look at the Transformer block there are other things also there is this output projection layer which is
38:40
accessible to trans through Transformer dh- at and- c-w so what we are going to do is that
38:48
again we are going to access this output projection layer weights and we are going to assign these weights downloaded
38:54
from open a to the at. output projection weight so we are going to look at the at
38:59
object again and then we are going to assign output projection weights equal to what we have downloaded and this is
39:05
the same for weights as well as biases right then we are going to look at the feed forward neural network what I'm
39:11
highlighting on the screen right now is for the first layer which is the fully connected layer as I've shown over here
39:17
the feed forward neural network has two layers the fully connected layer and the projection layer so in the fully
39:23
connected layer what we are doing here is that we are accessing ing the weights and the biases of the fully connected
39:29
layer from the gpt2 downloaded values and then we are assigning these weights
39:35
and biases to the FF object which we saw in the Transformer block so that way the
39:40
neural network the fully connected layer weights and biases are equal to the gpt2 downloaded weights and biases now this
39:47
same thing is done for the second layer which is the projection or the output layer of the multi-layer perceptron or
39:53
the feed forward neural network and then finally we come to the last uh puzzle or
39:58
the last building block of the Transformers rather and that is the layer normalization so there are two
40:05
layer normalization the layer normalization one and Layer normalization Two and both have scale as
40:10
well as shift right so this is what's Happening Here what I'm highlighting right now we
40:16
are accessing the uh shift and the scale parameters from the gpt2 downloaded and
40:22
then we're assigning those parameters to our GPT model class and what I'm
40:28
highlighting on the screen right now is the similar process done for the second normalization layer which comes after
40:34
the attention mechanism in the Transformer block okay now when we come
40:39
out of the Transformer block you see there is another normalization layer right the final normalization layer and
40:45
it just accessible through G and the B keys so what we are doing is that we are accessing the param G and the params B
40:52
and then we are assigning our GPT model class scale and shift values to whatever is down loed from
40:58
gpt2 now you must be thinking that okay there is if I look at the architecture closely there is this final layer there
41:05
is this linear output layer and careful readers might remember that this linear output layer also is a
41:12
neural network and where do we get the dimensions where do we get the weights of these we did not download this from
41:18
GPT right so the way gpt2 was designed is that it uses this concept of weight
41:24
tying which means that the token embedding weights are used for
41:31
constructing this output head so the same token embedding weights are used for this output head layer so we don't
41:38
have to Define any new weights for this layer we recycle the same weights what's used in the token embedding weight tying
41:44
is not used these days too much but it was used in the gpt2 architecture and so we are also using the concept of weight
41:50
tying that actually brings the total parameters from 162 million to 124 million if you don't do weight time the
41:57
number of parameters will be 164 million awesome right so what we have
42:03
done here is that until now this this piece of code over here load weights into GPT this code what it does is that
42:10
it takes two values the first it takes the instance of the GPT model class and the second it takes the params
42:16
dictionary so it takes this dictionary which essentially contains this dictionary which essentially contains
42:21
all the weights of gpt2 and then it just assigns all of these weights into the GP
42:27
G PT model that's what this load weights into GPT is doing and now if you see above we have already created an
42:33
instance of the GPT model class and if you scroll even above we have already got the we have already got the params
42:40
dictionary awesome right so now we just need to call this function and that's exactly what we are doing here we are
42:46
calling this function load weights into GPT what this function does is that it takes the params values and then it uh
42:53
which are downloaded from gpt2 and then it loads into our GPT model instance
42:59
which means that our own GPT model which we have constructed from scratch is now fully ready it's fully functional to be
43:06
tested so now let's go ahead and do the most exciting thing in this lecture which you all have been waiting for let
43:12
us test our model which we used our own architecture with the gpt2 pre-train
43:19
weights and let's see what the output is great so now let's go ahead and test our model uh so here you can see that here's
Testing the pre-trained GPT2 model
43:27
the the generate function which basically generates new tokens and we are passing the model which we have
43:32
defined and that this model now had has the weights which have been downloaded from gpt2 the input token IDs are every
43:40
effort moves you and the maximum number of new tokens is 25 I have defined the temperature not too high 1.5 and the top
43:48
K is 50 which means that 50 tokens will have the chance or the opportunity to be in the among the maximum new tokens or
43:56
to be in the generated token so before we see the output for this let me show
44:01
you our performance without using gpt2 weights and here you can see that if we do not
44:08
use gpt2 weights we were getting something like this which did not make any sense at all and now what we are
44:14
going to do is that we are going to run this right now so I'm going to run this live and just to show you that once you
44:21
load the weights the running actually this code does not take too much time because we have already
44:27
we already have pre-trained weights so here you can see with the star symbol that it's running right now and now it
44:34
generated so every effort moves you toward finding an ideal new way to
44:39
practice something what makes us want to be on top of that this is incredible right this output sentence is much more
44:46
coherent than what we had obtained earlier so right now in this lecture we have built our own GPT from scratch and
44:53
it seems to be working that's incredible all all the other students which have not been through this course or who have
45:00
not seen these lectures would be just using chat GPT but now we have built our
45:05
own GPT from scratch isn't that incredible it took us a long time to get here and the code length is also um the
45:14
code length is also pretty large so you can scroll above and see how long the
45:20
code has become but it's all worth it because we have learned how to build GPT from scratch now what you can do is you
Research explorations and next steps
45:26
can go ahead and do your own research so for example if you want to change the temperature value to 10 I know that this
45:33
is not good but I want to see the effect of a higher temperature value we cannot do this using chat GPT right but now
45:39
since we have built our own GPT we can explore with so many things so here you see I've increased the temperature value
45:45
to 10 and let's see the output text which is coming right now ideally it should be a bit random because
45:51
increasing the temperature increases the entropy now see the output every effort moves you towards finding an ideal new
45:57
set piece but only at times or for hours I was working on my first game called G
46:03
so as expected increasing the temperature has given me random outputs but you see now this opens the door to
46:09
so much more creativity hyperparameter tuning even you can do research such as small language models what if you want
46:16
to change the architecture you can now easily go ahead and change the architecture right all
46:21
you need to do is that you need to go to this GPT model uh once you have understood the the code once you have
46:27
seen the previous lectures you need to go to the GPT model which we have defined so here's the GPT model right
46:34
over here and then you can add or subtract a few layers what if you we don't need 12 Transformer blocks what if
46:40
you want to test with a smaller language model all of these experiments now remain open to you so this lecture
46:46
series is also the pathway for you to become a large language model researcher or a machine learning researcher because
46:52
now you have something which works on your local computer and runs in a fast manner you can do iterations you can
46:58
test so if you want to test the effect of top K if you want to test the effect of Maximum new tokens if you want to
47:04
vary the number of attention heads the number of Transformer blocks the optimizer size you can even vary the
47:09
optimizer step size so for example we have used Adam rate with the learning
47:15
rate of 5 into 10us 4 weight DK of 0.1 you can even change this and check the
47:22
output all of this is now accessible to you so that's why I believe that we have achieved a significant Milestone
47:28
completing this lecture if you have come to this lecture for the first time without watching the other videos in
47:34
this lecture series uh it's amazing but now please go back and try to watch all
47:40
the other videos to master your Concepts to become a very powerful llm engineer
47:46
who is creating new Norms who is really doing Cutting Edge research you need to know the nuts and bolts of how the
47:52
modeling process works and I hope that through this whiteboard approach
47:57
and through this coding approach you are getting exposed to those nuts and bolts I've not seen any other content out
48:04
there like this currently that's why it's very hard for researchers to even download or load publicly available
48:10
weights so I'm trying my level best to teach you every single thing by not making short videos but by making longer
48:17
format videos like this one and now we are confident that we have loaded the weights correctly why
48:23
because the model is producing coherent text so again let me switch the temperature to
48:28
0.1 um and let me run this again so now I'm running this and as you see earlier
48:34
when the temperature was I think it was not 0.1 it was 1.4 we need to run this again for 1.4
48:41
but if you see for 0.1 uh again the model is quite good but
48:47
even I'll do for 1.4 because we started with that condition earlier so we are confident
48:53
that the model weights are uh loaded correctly because the model can produce coherent
48:59
text at least the words make sense a tiny mistake in this process would cause the model to fail now in the next
49:05
chapters what we are going to see is that now that we have mastered pre-training we are going to look at fine tuning so the llm subject does not
49:12
end at pre pre-training after getting this model let's say if you want to build a text classifier let's say if you
49:19
want to build an educational quiz app which is a very specific application how can you use the pre-train weights how
49:25
can you fine tune these weights so that it's specific to the application which you are building we are going to learn
49:31
about this in the next lecture so all the next lectures are going to be very interesting since they are going to be
49:36
application oriented and for all of those we are going to use this GPT model which we ourselves have built not
49:42
relying on any other GPT model and that gives us a lot of confidence as a large language model or a machine learning
49:49
engineer so thanks everyone for this lecture this was a a bit of a long lecture and also a dense lecture but I
49:55
hope you understood everything which I was trying to teach uh please uh try to
50:01
reach out or comment if you have any doubts or any questions and I'll be happy to discuss this further and I'll
50:07
also be happy to see what all research you have worked on by using this code file which I'll share with you thanks a
50:13
lot everyone and I look forward to seeing you in the next lecture

***

