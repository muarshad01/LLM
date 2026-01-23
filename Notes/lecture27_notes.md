

* 20,000 characters
* 5,000 tokens

* [OpenAI - tiktoken](https://github.com/openai/tiktoken)
Lecture Objectives
0:00
[Music]
0:05
hello everyone and welcome to this lecture in the build large language models from scratch
0:11
series let me recap the stage of building the llm in which we are at
0:18
currently in terms of a diagram okay so we are in stage number two right now
0:24
based on the schematic which is mentioned in front of you and in the stage number two we are essentially
0:29
learning how to build this foundational model or rather we are going to learn
0:34
how to train or pre-train a large language model in this stage in the previous stage in the all the previous
0:41
set of lectures we have looked at data preparation attention mechanism and llm architecture in fact in this stage we
0:49
even looked at um how to calculate the loss in terms of a large language model
0:56
we looked at how to define the cross entropy loss based on the input and
1:02
targets of a large language model and that was the previous lecture which we had now we will be looking at a much
1:10
larger data set in fact this is going to be a very interesting Hands-On lecture in which we are going to collect data
1:16
from a story book and we are going toh make predictions using the large
1:22
language model which we have built in the first stage and then we are going to
1:27
measure the loss of our large language model model based on the actual data which we
1:33
need so in the previous lecture what we did is we looked at some simple examples we looked at just two inputs the first
1:40
input which we looked at was every effort moves and the second input was I really like we looked at these two
1:47
inputs and we predicted outputs from our large language model and then we calculated the loss between those
1:53
outputs and the target values which we need now what we are going to do is we
1:58
are going to look at an actual data set and on that data set we are going to find the training loss and we are going
2:04
to find the validation loss so if you zoom into stage two of building an llm further in the previous
2:11
lecture we have looked at text generation and text evaluation so we saw the cross entropy loss in the previous
2:18
lecture and how to calculate it for just two simple input text today we are going
2:23
to calculate the training and validation losses on the entire data set we'll also
2:29
split the data set into training set and validation set so let's get started it
2:34
will be a completely Hands-On lecture and after this lecture actually you will be equipped to take any story book of
2:41
your choice or for that matter any data set and uh train the large language
2:48
model not really train it because we are not training the parameters in this lecture do a forward pass of the large
2:54
language model get the outputs and then get the loss based on
3:00
the True Values which you need and you will get this loss for any data set which you use you'll just have to follow
3:06
the same code which I'll provide you at the end of this lecture we have not yet started doing the training procedure but
3:13
once you get this loss for the data set in the next lecture we are going to look at the llm training function where we'll
3:20
also dive dive deep into back propagation so at the end of today's
3:25
lecture you will have a very cool result which is applicable to a wide range of data sets
3:30
so let's get started the data set which I'm going to consider is called the verdict it's a book written by an author
Understanding the dataset
3:37
named edit Warton and this book was published in 1906 I think here's how the
3:43
book looks like and uh the book is publicly available to download I'll
3:48
share the link you can download the book from this link and it's not a very long
3:53
book it's a pretty short book in fact but we are using this just because I want to demonstrate uh an example which
4:00
runs very fast on my laptop and it will run very fast on your laptop also in
4:06
fact if you take a look at this data set and if you count the number of characters you'll see that the number of
4:13
characters in this data set is 20,000 characters what we'll do uh on the data
4:21
set first is that we'll first convert this data set into tokens so we'll use a
4:26
tokenization scheme for that which is called as bite pair en code B B pair
4:31
encoding so bite pair encoding is the tokenization scheme which we are going to use and what this will do is that in
4:39
bite pair encoding one word is not one token it's a subword based tokenization
4:44
scheme where even characters can be tokens just uh pairs of letters such as
4:51
T and H this can be one token just T can be one token Etc so if you use bite pair
4:58
encoding you'll find that this data asset has 5,000 tokens don't worry I'm going to show all of this in the code
5:04
but just if you search tick token you'll see that uh this is the bite pair



***


5:10
encoder which we are using and it's the same encoder which open AI actually used
5:16
and using this encoder we can encode our data uh data set into
5:22
tokens great so this is the data set which I'm considering and remember you can use any data set which you want on
5:28
publicly available books now the next step what we are going to do is that uh we are going to
5:35
divide the data set into training and validation remember this is the same thing what we do for all machine
5:40
learning problems because the training loss is not the one which really matters what matters is how our large language
5:47
model is doing on text which it has not seen before so we'll do a simple thing over here let's say this is the entire
5:54
data set we are going to use a train test ratio of90 and we we are just going
6:00
to do a simple split the training data will be the first 90% of the input and
6:05
the validation data will be the latter half so the remaining 10%
6:12
right um and that's how we are going to split the training and the validation set so if you imagine the entire data to
6:18
be like this what I'm going to do is that I'm going to reserve the first 90%
6:25
so this is 90% And this is 10% so I'm going to reserve this for training
6:30
purposes and I'm going to reserve the 10% for validation or testing
6:36
purpose now what I'm going to do after this is pretty interesting usually in
6:41
other machine learning problems getting the inputs and the outputs is pretty easy right you just have images of cats
6:47
and dogs and the output is whether it's a cat or whether it's a dog so it's pretty simple you have inputs which are
6:53
images and you have outputs which are the labels you don't have to do anything special to create these input output
6:58
pairs but it's not as simple in the case of a large language model because large language models are Auto regressive
7:05
models we don't uh label anything beforehand but from the text itself we construct the inputs and the output and
7:13
so we'll need to understand the process through which we conu construct these input output
How to construct LLM input-target pairs?
7:19
pairs so in the code we are going to use the data loader to chunk the training
7:24
and the validation data into input and output data sets or input and output pairs and let me show you how I'm going
7:31
to do that if you understand this part it will be much easier for you to visualize what comes next right so the
7:37
first thing which we have to decide is what is the context size which I'm going to use in other words what is the
7:43
maximum number of tokens the llm can see before it predicts the next token so I'm
7:48
going to show you how to construct the input output data pairs on this data set using a context size of four so that's
7:55
the first thing which I need to decide context size equal to four
8:01
okay great so now let me look at this data set and here's how I create the input output pairs my first input is
8:08
this I had always thought and since the context size is
8:14
four I'm looking at four tokens at a time and although one word is not one token I'm just assuming it here for the
8:20
sake of demonstration so this is my first input and I'm going to label it let me use the same color here and I'm
8:26
going to label this as X1 right this is my first input now what's the first
8:32
output the first output is just this input shifted by one so then it will be
8:37
this had always thought Jack this is y1 let me write this down
8:44
over here so uh ultimately the input will be a tensor X and I'm currently I'm
8:50
just writing the first row of this so this will be I had always thought that's my first input
9:00
put and let me also collect the output tensor over here with a different color
9:05
of course and uh this will be had
9:11
always thought Jack right now let us
9:17
focus on this first input output pair for just a moment the first thing to notice is that of course the output is
9:23
just the input shifted to the right by one but another thing to notice is that this one input output pair essentially
9:29
has four prediction tasks what are the four prediction tasks the first is that when I is the input had is the output so
9:38
the index corresponds that way when I had is the input always is the output
9:43
when I had always is the input so when these three are the input thought is the output and when I had always thought is
9:51
the input then Jack is the output right so essentially when X is given an input
9:58
when we pass it through an llm it will produce these four tokens as the output uh and then the output produced
10:06
by llm will be compared to this which is the actual output which we want for this input let me repeat that again if this
10:14
is an input I had always thought the actual output which we want is had always thought Jack but you'll see that
10:20
when you pass these four tokens through the large language model it's not trained currently right so it will
10:25
predict some random words over here and then we have to find the loss between those random tokens and what we want
10:32
that's how you get the loss between the first input pair and the second first input and the first output right now
10:39
let's move ahead a bit and let let us see how to construct the second uh second input so I'm going to





***



10:46
rub this output pair right now uh okay so we have the first input now we have
10:51
an option of how we are going to actually construct the second input and let me show you uh how we are going to
10:58
do that okay so the first option is that you just move it to the right so then X1 is
11:05
I had always thought right X2 will be had always thought Jack now this movement which we are
11:11
going to do is also called as stride and that's the second parameter
11:17
which you have to decide along with the context size so if you put stride equal to one like I have done in this case the
11:23
second input will have will be X2 and that has lot of overlap with the first input right had always thought is
11:28
overlap um so you can also do this but usually What's Done in models such as GPT is
11:36
that the stride which uh we are going to Define right now is is fixed to be equal to the
11:43
context size and that is equal to four so we are going to use a stride of four
11:49
what this will do is that when X1 if X1 is this input we are going to have 1 2 3
11:55
and four so then X2 will start from here so then X2 will will be Jack gisburn
12:00
rather a this will be X2 you see what stride equal to 4 does it it makes sure
12:06
that we don't have any overlap but it will make sure that we also don't skip any token as the input when you look at
12:12
X3 this will be the um this will be X3 which will be the third input when you look at
12:19
X4 uh this will be X4 which is the fourth input right uh this is how the
12:26
inputs are created and you don't skip anything but also you make sure there are no
12:31
overlaps between inputs so now if you look at the input tensor the first is I had always thought right the second will
12:38
be Jack gizan rather a jack
12:45
gisburn rather o that's the second input which is X2 the third input would be
12:51
cheap genius dash dash do so the third will be cheap genius dash dash do
12:59
so that's how this entire input tensor Matrix will be created sorry this input tensor will be created like this we'll
13:06
stride through the entire data set and we'll collect these pairs so then the first row will be X1 the second row will
13:12
be X2 the third row will be X3 and we'll keep on accumulating these rows until we reach the end and how are the outputs
13:19
created once the inputs are created the output is just the input shifted by one
13:24
so for the first input I had always thought the output was had always thought Jack right now Jack gisburn rather a if this
13:32
is the input the output would be gisburn gisburn rather a
13:40
cheap so this will be Y2 the first row will be y1 and similarly we'll construct
13:47
all the other outputs output pairs till we reach the end of the data set so that is how input output pairs
13:54
are created in the case of a uh large language model so if you have a data set
13:59
like this what's very important is this X and this y so the x is the input and
14:05
the Y is the target which is the actual value so let me write this here x is the
14:10
input and Y is the Target now the loss will be between the llm output which we
How to get the LLM loss?
14:16
have not seen yet but we'll also get llm output so this input will be passed into
14:22
this input will be passed into our llm model and then we'll get the output from
14:29
the llm right and that will also be a tensor with the same format as this target tensor and then what we are going
14:37
to do is that we are going to then find the loss between the we are going to
14:42
find the loss between the llm output and the Target and this is the loss which we are going to find in today's lecture and
14:49
this is the loss which we eventually want to minimize okay I hope you have
14:54
understood this this part and I deliberately wanted to show you visually because students are generally quite
14:59
unclear regarding how input output pairs are created in the context of large language models now here I showed you
15:07
the input input Target pairs I should call them here I showed you the input Target pairs for the training data right
15:13
these are the input and let me call them targets because outputs will be what I refer to as the llm
15:20
predictions so I created the input and the target pairs for the training data
15:25
similarly we'll have the input and the target pairs for the validation dat data and that will give us the validation
15:31
loss the input and Target pairs will give us the train loss for the training
15:37
data and the similar tensors will give us the validation loss for the validation data now let me explain to
15:44
you the rest of the process and then we'll I'll take you through code but first I want you to understand how
15:50
exactly is the loss function calculated so that uh the code becomes so much more



***




15:56
easier to understand now let's say you create the input and Target pairs like this which is the first step first you
16:02
look at the first row of the input and the first row of the target uh and then
16:08
let's see how do we get the loss so let's say let's look at the first input so I'm going to look at the first input
16:14
now which is I had always thought which is the first row over here U and
How to get the LLM output?
16:19
remember that based on the input we have to get the llm model output right what is our model predicting we need to know
16:25
that because I I have the actual output corresponding to this input so if the input is this I know that the output is
16:32
had always thought Jack that is my output so let me even write that here the output or rather the target I should
16:38
call it the target value for this input is had had
16:45
always thought Jack right that's the target output which I need but of course
16:50
my llm is not going to uh give me this at the first uh first shot because we have not trained it yet so I need to
16:56
first know what my llm is predicting and for that we are going to go through the entire uh not right now but uh I'm just
17:04
going to show you the schematic of what we are going to put the input through so the input which we have that will go
17:10
through this entire GPT architecture or the entire llm architecture which we have trained in stage number one uh so
17:17
in the previous lectures we have actually built this entire architecture from scratch and you can see that there
17:23
are so many different modules in this architecture and let me quickly explain to you what we are actually doing so we
17:29
take in the input imagine those four tokens I had always thought we tokenize them using the tick token or the bite
17:36
pair encoder then we convert them into token embeddings which are vector representations in higher dimensional
17:42
spaces we add positional embeddings to it we add the Dropout layer at this
17:48
stage uh our input passes through the Transformer block so the Blue Block which I've shown over here this is the
17:54
Transformer block and this is the main engine of the GPT architecture Ure the
17:59
main thing which happens in the Transformer block is that the input embedding vectors are transformed into
18:05
something which is called as context vectors now what are context vectors and
18:10
how do they differ from input embedding vectors context vectors are more richer
18:15
so if you look at one word let's say if you look at effort here if you look at the input embedding Vector for effort it
18:21
just encodes santic meaning about effort it does not contain any information about how effort relates to the other to
18:28
tokens but the context Vector is much richer because the context Vector for effort not only contains semantic
18:35
meaning about effort but it also contains information about when effort when we are looking at effort how much
18:42
information should we pay to every moves and youu so in other in other words it pays attention the context Vector
18:49
includes attention which is given to the other tokens when we look at a particular token and that's what gives
18:55
all the power to the large language model so this multi-ad attention is the
19:01
main driver behind the Transformer block so if you look at the whole GPT architecture the Transformer block is
19:07
the engine of the uh GPT architecture and within the Transformer block the
19:13
multi-head attention is what allows us to convert these input embedding vectors into context vectors which encodes
19:19
information about how tokens relate to other tokens that's how the llms capture meaning and that's how they do so well
19:26
so GPT does so well because it has this multi-ad attention mechanism so you can see that the
19:32
Transformer has a number of building blocks we have the layer normalization multi-ad attention Drop Out shortcut
19:37
connection feed forward neural network Etc and then finally when the input
19:43
comes out of the uh out of the GPT model architecture we get something which is
19:49
called as the logits uh we get something which is called as the logit sensor and it's very important to understand what
19:56
the logit sensor is and what's the dimensions of this logic sensor so let's say the input is I had always thought
20:02
and when we pass it through the GPT model which is when it passes through the entire architecture which I just showed you we get this logic sensor now
20:11
I had always thought uh those are the tokens and when you look at the logits tensor corresponding to every token
20:18
there are these logits there is a logits vector who whose dimensions are equal to
20:23
the vocabulary size so if you look at I the Logics for I are 0257 because that's
20:30
our vocabulary size if you look at had the logits for had are 50257 because
20:35
that's the vocabulary if you look at always the logits for always are 50257 because that's the vocabulary size and
20:41
similarly for thought now when the logits tensor come out of the GPT architecture they are not
20:47
normalized so if you look at the logits for I they don't add up to one so the next step is to convert this logic
20:53
tensor into a probability tensor and that makes sure that when you look at every token the logits or the




***



21:00
probabilities add up to one so now you can see that every logic essentially has a meaning because it adds up to one the
21:07
way to predict the output now is that you look at I and you look at that logic which has the highest value okay and
21:14
let's say it corresponds to index number 50 in this vocabulary of 50257 or maybe
21:19
5,000 and then you look at the token corresponding to that index and maybe it's something random like a c h now you
21:28
look at had and you look at the logic which has the highest value you get the index corresponding to this maybe it is
21:34
31 1 01 and then you find the token corresponding to this index maybe it's
21:39
something completely random similarly for always you get the index which has the highest probability and you get the
21:48
outputs like a am m m o let's say uh am
21:54
and then you look at thought and you find the index corresponding to the highest probability and let's say this
21:59
is 611 6111 and then you get the uh
22:05
outputs so these are the output tokens now which are llm is predicting and these are the output tokens which we
22:11
need to make sure that uh they need to be as close to our Target as possible
Finding the loss between the target and LLM output
22:17
okay I hope you have understood the workflow which we are trying to follow here we first have a logic sensor we
22:22
convert it into a soft Max sensor uh which which is a probability sensor and
22:27
I just gave you the intuition that based on this tensor how do you make the output for this input right so what's
22:35
done now is that after this point let's say we have this probability tensor right then what we actually do is that
22:42
we look at the targets uh this is the actual value which we want and we look
22:47
at the index which each token in the Target corresponds to so had has index
22:53
23 in the vocabulary always has index 3881 in our vocabulary thought has IND
22:58
index 1 1 2 2 3 in our vocabulary Jack has index 15 in the vocabulary I'm
23:03
assigning these random values for now just for demonstration purposes now let me show you one thing here so first let
23:09
me rub these uh rub these colors Okay now what's done next is that based on these
23:16
based on these IND indexes uh let's say we look at I and we look at index number
23:22
23 let's say this is index number 23 and we take its uh value that will be P1
23:29
then we look at had and then we uh that so the target for had is always and its
23:36
index is 3881 let's say we look at that index and find the probability corresponding to that index that's P2
23:42
then we look at the third row and find index number 1 1 22 3 and find the probability corresponding to that that's
23:48
P3 and then we uh find uh the index
23:54
number 15 in the last row and then we find the probability corresponding to to that so then that will be P4 ideally if
24:01
in an Ideal World if the llm is trained perfectly these probabilities will be close to
24:07
one and if these probabilities are close to one It means that our llm is also predicting these values right um as the
24:15
output but when the llm is not trained at all these probabilities will not be close to one at all they might be very
24:21
low which means that our llm does not think that this target needs to be the output because it has not been trained
24:28
so the whole goal is that to make sure that these probabilities get as close to one as possible and that's why we uh
24:35
employ the categorical or I should call it the cross entropy loss so at first take the logarithm of all these values I
24:42
add up these log values then I take the mean and then I take the negative this is also called as the negative log
24:50
likelihood uh this is called as the negative log likelihood and the whole goal of training the llm is to make sure
24:57
that this this negative log likelihood which I'm calling n LL and if you plot
25:02
nnl of X as a function of X so it's negative of logarithm right so it will
25:07
look something like uh it will look something like this since it's the negative it will it will look something
25:14
like this and uh our whole goal is to make sure that the loss comes down and
25:20
it comes down to zero as much as possible so now today what we are going to do is that today we are not going to
25:26
train the llm we are just going to see this starting point of this loss uh which which will be very high value but
25:33
then in the subsequent lecture we are going to train the large language model so that this loss comes as down as
25:38
possible so this is the workflow which I showed you for one input right I had always thought uh and then how do we get
25:45
the loss now remember that we don't just have one input we have all these inputs
25:52
uh which are stacked together in a batch so remember that this data loader processes inputs in a batch
25:58
so each batch has an accumulation of inputs right based on the size of the batch so now let me show you how
Finding loss for multiple batches
26:05
multiple inputs in a batch are processed so let's say we have a batch which has two inputs together so let's say this is
26:11
a batch whose batch size is equal to two which means that there are two inputs
26:17
together uh in a batch at a time right so this is one batch and this has two
26:23
inputs I had always thought Jack gpan rather so this is X1 and this is X2 and
26:29
this is y1 and these are this is Y2 these are the target outputs so for the first input X1 my output should be had
26:35
always thought Jack which we also saw in the previous example where just one input was there and my second input is
26:42
Jack gpan rather a the output should be gizan rather or cheap that's what I want these are the targets right now similar
26:49
to the similar to what we saw for one input the steps are pretty similar for the case of batches as well I'm trying
26:56
to see a color which would work the best here yeah so what we'll do is that we'll first take the input and we'll pass it
27:01
through the entire GPT architecture in this case what we'll get is that we'll get two logic sensors the first logic
27:08
sensor is for the first bat first input the second logic sensor is for the second input so if you look at the size
27:15
of this tensor now we have two batches here and in each batch there are four rows and in each row there are 5 to 57
27:22
columns in the previous case where there was just one input the size was just 4 into 50257 but now we have two such
27:29
batches right two such input so this is input number one and here is input number two right so in the code what
27:37
we'll do is that when we get this logic sensor we'll flatten this out we'll flatten the batch Dimension out which
27:42
means we'll merge both of these together uh we'll merge both of these together so that now the my cumulative logic sensor
27:50
looks something like this this is my first input this is my second input all merged together the size of this now
27:56
will be eight rows multiplied by 50257 that will be the size of this okay
28:02
and the next steps are pretty similar we add we apply soft Max we convert this into a tensor of probabilities and then
28:09
we look at the Target we look at the Target tokens and we stack this target tokens also so for the first input these
28:16
four are the target outputs for the first for the second input these four are the target outputs and we get the
28:22
token IDs corresponding to these Target outputs we stack them together and then we find for each row we find the value
28:30
corresponding to these token IDs and then these values are noted down as P11
28:35
p12 p13 p14 this for input 1 p21 P22 p23 p24 that's for input number two and then
28:42
similarly we get the cross entropy loss so this is the loss for the first input
28:47
this is the loss for the second input and then we add add it together and ultimately it will also look something
28:53
like this uh and the whole goal is that in this case also we want to minimize
28:58
this loss and bring it as close to zero as possible so now I hope you see that even for a batch of two inputs the
29:05
process of getting the loss is pretty similar as what the process was for just one input right uh and I want you to
29:13
keep this visual workflow in mind so that when we come to the code you will really understand what is happening over
29:18
here so in the code there is going to come a time when uh we are going to apply the data loader to the training
29:24
and validation set and the training data set is going to look like this after it passes the data loader and I want you to
29:32
analyze this right I told you that the input is processed into batches right so here you can see that uh let's look at
29:41
first the each row for now right each row corresponds to one batch so this is
29:46
X and this is y so what this 2 comma 256 means is that the so let's look at our
29:53
case right now let's look at the case which we took for the batch uh we looked at X1 we looked at X and Y right this is
30:01
the first batch and the size here was 2A 4 because there were two uh there were
30:07
two inputs and each input had four tokens and here also it was 2A 4 because
30:12
there are two outputs and four tokens similarly what I want you to note over here is that when you look at this
30:18
training data loader it's exactly similar to the example which we saw so let's look at the first row over
30:25
here let's look at the first row over here uh it has two inputs but it has 256
30:31
tokens because that's the context size we are going to use when we go to code so that's X the first input and it has
30:39
two uh it has two inputs so it's X1 comma X2 um and that's exactly what we
30:46
saw right why there is two here because there is X1 uh so if you zoom this in
30:51
further let me show you how it looks like it will be um
30:56
I uh I had always this will be 256 now
31:03
not four like what I had shown you because the context size is 256 this will be X1 and then X2 will be another
31:09
input batch this will be X1 X2 and then similarly we have y1 Y2 which are
31:14
shifted by one I already told you right how to construct the y1 and Y2 they are the inputs just shifted by one so these
31:21
are y1 and Y2 and they will also have the size of 256 this is just the first badge the
31:28
first row is just the first batch uh where each batch had two inputs of 256
31:34
tokens each similarly here we can see that there are nine batches so there are nine training set batches each batch has
31:41
two samples and each sample has 256 tokens each so I don't want you to be scared when you see this in the code
31:48
similarly in the validation we just have one batch because only 10% is used each batch has h two samples and each sample
31:56
has 256 tokens each okay I hope you have understood this visual workflow which I have constructed
32:02
over here I could have directly taken you through the code but then it would have been very difficult to understand the different steps in the code right
32:10
now I'm going to take you through the code and we are going to see how to calculate the loss for this verdict
32:15
short story step by step but please keep this intuition of this workflow in mind then everything will be very clear for
32:22
you okay so now we are at the code before we get started I want to clarify that we are actually using a relatively
32:29
small data set and that is because we want to run the code in a few minutes on our laptop computer I could have used a
32:36
larger data set but it would have taken a huge amount of time uh just to give you a sense of how
32:42
much time it takes to run big data sets Lama 7 billion the that model required
32:49
84320 GPU hours um and was trained on two trillion tokens and training this llm would cost
32:56
about $700,000 that's why I cannot show it to you on the full scale data set but I'm showing
33:02
it to you on a smaller data set and the whole logic is completely scalable for larger data as well okay so here's the
Coding: Loading the dataset
33:09
code the first thing what we are doing is that we are getting this data set link from here uh and I'll share this
33:16
link with you in the YouTube description and I'm importing this data set over here so here you can see that uh um I'm
33:24
reading I'm reading the data set and I'm storing the all the information all the
33:30
text in this variable called Text data and let's check whether the text is loaded fine by printing out the first
33:37
100 words so here I'm printing the first 100 characters uh and here you can see
33:42
that the first 100 characters have been printed and they look very closely matching with what was actually there in
33:48
my text I had always thought Jack gispan rather a cheap genius and here also I
33:54
had always thought Jack is rather a cheap genius awesome let let's print the last 100 characters it for me the stoud
34:01
Strand alone and let's see whether that also
34:08
matches okay I think this also matches the whole data set is not being displayed over here but now we are
34:14
pretty sure that the data has been loaded fine let's move to the next step in the next step what I want to show is
34:19
that we can print out the total number of characters in this data and it's 20479 that's fine but remember I told
34:26
you about the bite pair encode encoder what we are going to do is that we are going to use the bite pair encoder tokenizer to encode this text data and
34:34
we had already defined the tokenizer in the previous lecture but I'm going to do it once more here so that
34:41
uh um yeah so that everything is from scratch so what we are actually going to
34:47
do is that we are going to import tick token uh and we we are going to Define this tokenizer from The Tick token
34:54
Library so let me write it down here
35:00
okay yeah so we are importing this tick token Library which is the same Library
35:06
open AI uses for their tokenization and we are going to get the encoding from tick token which is a bite pair encoder
35:13
character and subord level encoder basically and we are going to encode the
35:18
entire text data using this encoder and we are going to print out the number of tokens right so the number of tokens are
35:26
5145 so with 5145 tokens the text is very short for training and llm but
35:31
again it's for educational purposes uh okay the next step is that
Coding: Implementing the dataloader class
35:36
we are going to divide the data set into training and validation data and we are going to use the data loader exactly
35:42
what I told you over here right so we have loaded the data set now and we have to div divide it into training and
35:48
validation okay let's do that before that what I'm going to show you is that remember I told you what our data loader
35:55
does our data loader we have to specify this uh context size and we have to
36:01
specify the stride our data loader Loops over this entire data set and creates this input output pairs that's what I've
36:08
implemented in the code right now uh so here you can see that max length is the
36:13
context size and stride is the uh how many steps we want to leave before creating the next input so first so we
36:21
create two tensors input and the target so these are the X and Y tensors which I showed and then we Loop over the entire
36:28
we Loop over the entire data set we create the input Chunk we create the target chunk which is based on the
36:34
context length and we append it to the tensor so uh the first row here will be
36:40
the first input chunk the first row in the Target will be the first Target chunk then we move over in the second
36:45
Loop then we fill the second row of the input and the target similarly we Loop over the entire data set and fill the
36:51
input tensor and the target tensor uh that's what this GPT d data
36:57
set version one creates and then we use the create data loader function it takes
37:03
the input output data sets which which have been created in the GPT data set V1 class and then U we create this data
37:11
loader instance so data loader is already um provided by pytor so let me
37:17
show you this I'll also add the link to this so data sets and data loaders right they are very useful for processing data
37:24
also in batches remember we want to use batches over here so using a data loader like this just makes uh processing the
37:31
batches much more convenient so we we create an instance of this data loader and we feed in the data set which we
37:38
created the input output data set which I mentioned over here input and targets and then here we specify the batch size
37:45
uh we specify Shuffle uh these uh arguments are not useful in the current context right now
37:52
but I'll also explain them to you later when we are going to train the llm uh see the output EX ET but remember that
37:59
for now the only important aspect here is that we are going to create an instance of this data loader feed in
38:05
this data set and Define the batch size if you want to do parallel processing you can also set the number of workers
38:11
Etc okay so now an instance of the data loader is created and let me actually
38:16
take some time to explain the shuffle and the drop last so what this suff
38:21
Shuffle essentially does is that it shuffles the data set order when batches are created that's sometimes useful for
38:27
generalization what this drop last actually does is that uh if the last batch size is very small and uh some
38:35
very small data is left at the last batch and it's not equal to the full batch size then it just drops that last
38:42
last batch so here we are setting the drop last equal to true and see the thing which I want to
38:48
mention here is that max length equal to 256 which means that the context size which we are going to use is 256 and
38:55
we'll also see that later uh we are going to use a context size of 256 over
39:02
here and that set by the GPT the GPT configuration which we are
39:08
going to provide so I'll also mention it over here okay so for now I hope you have
39:15
understood the GPT data set version one class and this create data loader function which basically creates the
39:21
input and the output data Pairs and then it also specifies the batch size one
39:27
more thing I want to mention before we create the training and the validation data set is that this is the configuration which we are going to use
39:33
so look at the context length that's 256 which means that uh we are going to look
39:39
at 256 tokens at one time whenever I showed you this example here I showed
39:44
four tokens I showed the context length of four because that's easier to demonstrate so when you try to
39:50
understand the code always try to think of four as being replaced with 256 rest all the workflow remains exactly the
39:57
same okay the next thing what we are going to do is that we are going to split the
Coding: Creating training and validation dataloaders
40:02
data set uh so we are going to use a train test split of 90% the first 90% of
40:07
the data is a training data the remaining 10% is the validation data and here's the main part where magic happens
40:15
so we are going to create a data loader based on the training data what this does is that it uh it splits the
40:21
training data into the input and the target tensor pairs which we had seen uh over here
40:28
and we are also going to create a validation data loader which splits the validation data into input and the target pairs because we also need the
40:34
validation loss so here you see the train data loader is an object so we
40:40
create we uh we create the train data loader based on this create data loader version one function and uh we specify
40:48
that batch size equal to two maximum length is GPT config context length so that's 256 that's the context size
40:56
stride equal to the context size remember I had mentioned to you that generally when these llm architectures
41:02
are run the stride and the context size are um matched because we make sure that
41:08
no word is lost but at the same time there is no overlap between consecutive inputs awesome right and then drop last
41:14
equal to True Shuffle equal to true and we are not doing parallel processing so I I'm putting number of workers equal to
41:21
zero similarly we construct the validation loader with a batch size of
41:26
two MA X length which is the context length of 1024 and the stride sorry context length of 256 and the stride of
41:33
256 when gpt2 smallest version was trained they actually used a context length of 1024 and you can even do that
41:40
but it just takes a long time uh all you need to do is just replace this with 1024 and just run the same code which
41:47
I'll be providing to you but please be patient when you run the code on your end it might take some time we can do
41:54
some sanity check so ideally the number of uh tokens which we want in our
41:59
training data set should not be less than our back context length right because then we don't have enough tokens
42:05
to predict the next word so here I have just written that if this is the case if our number of training tokens is less
42:11
than our context length then print an error similarly if the number of validation tokens is less than the
42:17
context length print an error it does not print an error which means we are good to go uh one more thing I want to
42:25
mention here is that we are using a batch size of two in large language models in training GPT level models they
42:31
usually use a pretty large batch size but we use a relatively small batch size to reduce the computational resource
42:37
demand and because the data set is also very small to begin with to give you a
42:42
context Lama 2 7 billion was trained with a batch size of 1024 here we are using batch size of two because I want
42:49
to run it very quickly on my laptop one more check we can do to make
42:54
sure that the data is loaded correctly is that remember both in the training and the validation there are now X and Y
43:00
pairs input and Target pairs uh so the training has inputs and targets and the validation is inputs and targets let's
43:08
actually print out the shape of these inputs and targets so the training loader has this if you print out the X
43:14
and the y shape in the training loader it will look like this and if you print out the X and Y shape in the validation
43:20
loader it will look like this so if you look at the train loader let's look at the first row this is the X and what I'm
43:27
highlighting now is the Y what this represents is that the input um so in one batch so this is one
43:35
batch so first row corresponds to the first batch the First Column of the first row is the input the second column
43:40
of the first row is the output if you look at the first batch input you'll see that there are
43:47
two samples each sample has 256 tokens similarly if you look at the first batch
43:54
output you'll see that there are two samples and two 56 tokens this is the target which we want and this is the
44:00
input which is there similarly uh since 256 tokens are exhausted um in the input
44:08
and we have to Loop over the entire data set there are it turns out that there are nine such batches which are created
44:14
uh in the training data and there is one batch which is created in the validation data similar to the training data in the
44:21
validation data you'll see that the batch has two samples each sample has 256 tokens and I also printed the length
44:28
of the training loader here and you can even print the length of the validation loader and you will get
44:35
that uh the length of the training loader is equal to 9 because there are nine batches each batch has two samples
44:42
and the length of the validation loader is equal to one and uh there's just one batch with two samples I hope now this
44:49
part is clear to you to to make sure you understand this part that is why I
44:54
actually went through this entire whiteboard demonstration to show you that towards the end we are going to get
45:00
something like this in the in the code and remember I spent some time to explain these sizes and these Dimensions
45:08
I hope you are following along and if I directly went through the code and when you reach this part it it would have
45:14
been impossible for you to understand this that's why it was very important for me to go through this entire whiteboard demonstration so that you
45:21
understand the dimensions of what's really going on so up till now what we have created is that we have created Ed
45:27
the we have the input and the targets and we have badged them into the input and the uh Target data but we have still
Coding: implementing the LLM architecture
45:34
not got the output predictions right we have still not um got the GPT model
45:40
predictions so that's what we are going to do next uh one more thing before going next
45:46
is that we can print out the training tokens and validation tokens uh uh just for the sake of Sanity so
45:54
this makes sure that the data is now loaded correctly now we can actually go to the next part which is getting the
46:00
llm model outputs so in one of the previous lectures we have defined this GPT model class what this GPT model
46:07
class does is that uh it essentially implements every single thing what I've shown in this figure it takes the inputs
46:14
it takes the inputs and then it returns a loged sensor remember the logic sensor
46:20
as it is returned does not encode probabilities we need to convert it to a probability tensor using the soft Max so
46:27
the GPT model class which we have constructed Returns the logic sensor and we have several lectures on this for now
46:34
you can just um keep in mind that okay first the inputs are converted into token embeddings we add the positional
46:40
embeddings then we add a Dropout layer then we pass the output of the Dropout
46:46
layer to through this Transformer block this Transformer block which I highlighted right now that has the
46:51
multi-head attention mechanism which is the main engine behind the llm power
46:56
after coming out of the Transformer we have another layer normalization layer followed by output neural network which
47:02
gives us this loged sensor then we create an instance of this GPT model class and we call it
47:08
model and we are using the same GPT model config 124 million parameters which I had defined over here we have to
47:16
specify the vocabulary size context length embedding Dimension number of attention heads number of Transformer
47:22
blocks dropout rate and whether the query key value bias is set to false in this lecture I'm not going to explain
47:29
all of these parameters because that was the subject of previous lectures uh if you don't understand what those
47:34
parameters mean I encourage you to check out the previous lectures in a lot of detail we have around six lectures on
47:40
that and uh six lectures explaining how we constructed this GPT model class for
47:46
now just remember that we have got the output Logics and we have constructed an instance of the GPT model class so when
47:53
you pass an instance when you pass an input to this model it will give you the logits now we are actually ready to
Coding: LLM Loss function implementation
48:00
implement the loss because we have the uh we have the targets over here we have the targets over here and we also have
48:07
the GPT model output and now we are actually going to implement the exact same steps over here remember First We
48:13
Take the soft Max then we index with the probabilities uh then we index these tokens based on the target tokens and
48:21
then we get the cross entropy loss right using the negative log likelihood and we
48:26
did the same thing for this batch over here so now I want you to keep this in mind when we had a batch remember what
48:32
we did first when we had a batch we first flatten the logits right this is exactly what we are going to do when we
48:38
calculate the loss so there is a function called calculate loss batch which takes the input batch and the
48:44
target batch right what this means is that it's exactly like what I've shown
48:49
over here this is the input batch X and this is the target batch y it just that instead of four tokens there will be 256
48:57
then what we are going to do in the code is that uh we are going to pass the input batch through the model the GPT
49:03
model and gets the logit tensor so until now in the code we are at this stage where we have got the logit tensor then
49:10
we are going to flatten the logits um 0 comma 1 so see we are going to flatten the logit 0 comma 1 uh and we get this
49:19
now remember up till now we have not implemented soft Max we have not indexed this uh probability tensor with the
49:26
Target index indices and we have not got the negative log likelihood it turns out that with just one line of code nn.
49:33
functional doc cross entropy we can do all of these three steps so when you do the nn. functional. cross entropy on the
49:40
flats logic tensor and the flatten Target batch so remember the flatten Target batch is this is this tensor over
49:48
here this is the flatten targets batch so what the nn. functional doc cross
49:55
entropy does is that it first applies soft Max to the logic
50:00
uh tensor because that's the first argument it first applies softmax to this first argument uh which is also
50:07
shown in this white board and then it takes the uh values corresponding to the
50:12
indices in the second argument so then it takes the values in this corresponding to the indices in this
50:18
argument so it it then gets this P11 p12 Etc this Matrix and then it also gets
50:25
the negative log likelihood it calculat the negative log likelihood so in one line of code we actually get the loss
50:32
and this is an awesome function which is a very powerful function in pytorch you
50:37
can take a look at this I'll also share the link to this uh this uh torch P
50:43
torch function with you awesome so this is how we calculate the loss between an input batch and a Target batch but now
50:50
remember that we have to calculate the loss for all of the batches right and that's why we are defining a function
50:57
called calculate loss loader which calculates the loss from all of the batches together the main function in
51:02
this the main part in this is that you get the input and Target batch for the
51:08
entire data loader which means that uh so here we just looked at one input and
51:13
one output batch right one target batch but you will see here there are many input and Out target batches uh so Row
51:20
one row one of input and Row one of Target is the first batch row two is the second batch so there are multiple batches and we have to miate the loss
51:27
for all of those right so similarly you get the input and Target batch uh and then you Loop over so when you're
51:33
looking at one batch you just run this earlier function and then you just aggregate the losses together so when
51:39
you uh run the loss for one batch you'll get the loss then you add it with the loss for the second batch and similarly
51:45
you get the total loss and then ultimately you just divide the total loss with the number of batches which
51:51
will give you a mean loss per batch the different parts of the code which are added before ensure that if
51:57
the length of the data loader is zero it we return that the loss is not a number
52:03
because length of data loader is zero does not make sense both our uh training
52:08
and the validation data loaders currently training data loader is of length nine because there are nine batches validation data loader is of
52:15
length one because there is one batch if the length of the data loader is itself zero which means that there are no
52:20
batches and there is nothing to compute similarly when we uh when we call this
52:26
Cal caloss loader function and if we don't specify the number of batches so if by
52:31
default it's none we set the number of batches equal to the length of the data loader so for the training data loader
52:37
that will be equal to 9 for the validation data loader that will be equal to one now if someone specifies
52:43
the number of batches here which are more than the number of batches in the data loader we set the actual number of
52:50
batches to be minimum of those two uh right so the number of batches
52:56
equal to minimum of number of batches set here and remember that in the data loader also there is a provision to set
53:01
the number of batches so the ultimate batch size which is used for computation will be minimum of those two that's it
53:08
and then we take the one input and one target at a time we find the loss according to this scal loss batch
53:15
function which implements the uh functional cross nn. functional. cross entropy loss and then we actually add
53:23
all of the losses together from every input Target batch and then we just divide by the number of batches and this
53:29
is how we got get the average uh cross entropy loss per batch this the output
53:36
of this function is the loss of our large language model on this book The Verdict data set which we considered in
Coding: Finding LLM Loss on our dataset
53:43
today's lecture now let's actually run uh let's actually call this function on the data
53:49
which we have and let's see the output which we get right okay so what I'm going to do now
53:54
is that uh I'm going to call this Cal Closs loader and I'm going to uh input
54:01
the train loader the model and the device and here you see tor. device if
54:07
tor. Qi is available else it will run on CPU so uh my code is running on my CPU
54:13
right now and I'm calling this scal Closs loader function for both the train loss and for the validation loss so when
54:19
I call it for the train loss I I input the train loader here when I call it for the validation loss I input the
54:25
validation loader and the model is essentially uh the instance which we
54:30
have already created here this is the model an instance of the GPT model class
54:35
and uh that's the second argument the third argument is the device so uh if if you want to run on
54:43
CPU it can even run on CPU like I'm showing right now so I uncommented I've
54:48
commented these lines right now uncommenting these lines will allow the code to run on Apple silicon chips if
54:54
available which is approximately 2X faster than on Apple CPU so right now my
54:59
code is running on Apple CPU you can also run it on Cuda if Cuda is available
55:05
or if you have GPU access you can even run it on GPU so right now what I'm going to do is that I'm just going to
55:10
click on this run and I'm going to show you live how much time it is taking for me to run this uh I just want to show
55:17
you that uh here what we have essentially done is that we have loaded
55:23
this entire data set we have converted this into input output pairs we have passed the uh data set into the GPT
55:32
architecture block so we have passed the data set into this llm architecture block which looks like this uh and then
55:40
we have got the llm outputs and then we have compared the loss with the targets and with these outputs and then we have
55:46
collected an aggregate matric of this loss so here you can see I've got the training loss and I've got the
55:51
validation loss and I got it live in less than 30 seconds I would say now you you can take this code and you can do
Next steps
55:58
whatever you want you can increase the context size to 1024 all you need to do
56:03
is go here and change the context size to 1024 to mimic conditions more closely
56:09
to gpt2 you can even go to internet and search uh Harry
56:16
Potter book download um you can download the Harry Potter book there's an ebook series here
56:23
just make sure the um just make sure about the copyright versions similarly
56:28
you can go ahead and download any data set which you want and just train the large language or just run this code on
56:34
the data set which you are considering it will be truly awesome for you to use your own data set and get this training
56:41
and validation loss because once you have obtained the training and validation loss that really opens up the
56:46
door for us to to back propagate so in the next lecture what we are going to do is that we are actually going to Define
56:52
an llm training function which implements the back propagation and which tries to minimize the training and
56:57
the validation loss so then it will make sure that the outputs being generated are very coherent and then even if you
57:06
run this code on another data set even in the next code when we do the pre-training you can do the same pre-
57:12
trining on your custom data set so the code which we have developed today is pretty generalizable and uh I hope you
57:21
you understood what we are trying to demonstrate today we are trying to demonstrate through a real hand on
57:26
example how we can actually take a data set from the internet and we can divide it into input Target pairs we can run
57:34
the data set through a large language model which we ourselves have developed if you have not been through the previous lectures this this model I have
57:42
not taken it from anywhere we have developed it live we have developed it from scratch without any single Library
57:49
like Lang chain or any other Library we have coded this from the basic building blocks and that has been used to produce
57:55
this output that's even more satisfying uh okay students so that brings me to the end of this lecture I
58:02
deliberately wanted you to I wanted to give you a feel of the Whiteboard teaching uh so that you understand the
58:08
intuition Theory and also the coding which is my main goal in every lecture
58:13
which I conduct in the next lecture we are going to look at llm pre-training I'll be sharing this code file with you
58:19
if you can run it before the next lecture it's awesome if not it's fine I'll try to make the next lecture so
58:25
that it's selfcontain thank you so much everyone and I look forward to seeing you in the next lecture

***

