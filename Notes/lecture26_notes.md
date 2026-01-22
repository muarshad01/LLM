

* $$L_{BCE}=\frac{1}{N}\sum_{i=1}^{N}y_i.\log(p(x_i))))+(1-y_i).\log(1-p(x_i))$$

s of lecture
0:00
[Music]
0:07
hello everyone and uh welcome to this lecture in the build large language models from scratch Series today we are
0:15
going to look at a very important topic and that is regarding loss functions
0:20
when you look at machine learning and traditional machine learning models such as regression and classification the
0:27
loss function for these type of models are pretty well defined for regression you generally use the least Square type
0:34
of a loss mean square error for classification you use cross entropy cross entropy loss hinge loss
0:42
Etc however the question is that when we come to large language models how do we Define the loss function how do we
0:49
measure whe whether the large language model is doing a good job or whether it's not performing well let's find out
0:56
in today's lecture so up till now in this series we have covered stage one of building a
1:03
large language model from scratch in stage one we essentially covered three subm modules data preparation and
1:10
sampling attention mechanism and llm architecture in each of these
1:15
subcomponents we divide we devoted a huge number of lectures and whiteboard
1:21
notes and also coding so that by the time you have reached this stage I hope the building blocks of how to construct
1:29
a large language models are clear to you in particular till the last lecture we
1:34
have seen this GPT or the llm architecture which we have built from
1:39
scratch and up till now we we are at a stage where the GPT architecture which
1:45
we have built and whose code we have written takes an input it takes an input text and it returns an output we also
1:53
saw how this output is converted into the next word prediction task so our GPT
1:59
model up till this stage or our large language model we which we have set out to build is at a level where it takes an
2:07
input as a text and it can predict next tokens or it can predict next words
2:12
that's awesome right but now we want to make it better and better and better in
2:17
particular if you remember uh where we ended the previous lecture on llm architecture we got this kind of the so
2:25
we gave the input as hello I am and the next tokens which we got are or something completely
2:31
random so now from this lecture onwards what we'll be doing is that we'll be
2:36
looking at stage two stage two of building a large language model just centers around one
2:43
word and I'm just going to write that one word here and that is called as
2:49
training we need to train our large language model so that the next token which it predicts makes
2:56
sense the first step of the training procedure is that you you need to ask yourself okay let's say my large
3:03
language model is has is giving me an output and I know it does not look good
3:10
how do I capture this qualitative intuition of not of knowing that it does not look good into a quantitative matric
3:18
and that quantitative matric is generally the loss
3:24
function so if we can define a loss function which quantifies how good or
3:29
how bad our llm performance is we can aim to minimize the loss so that the llm
3:35
can do better and better and better and once we construct a loss function it opens the door for integrating gradient
3:43
descent based back propagation algorithms which we have already learned in neural networks so then we convert
3:50
the problem of training a large language model into a problem which we have previously solved
3:55
before so what we are going to do in these set of lectures we are going to look at this entire pipeline of seven
4:02
steps which I have shown over here in today's lecture we are going to look at the first two steps here we are going to
4:08
look at how text is generated which we have already seen before but I want to quickly recap it in case some of you
4:16
have come to this lecture directly and then we are also going to look at text evaluation which means we are going to
4:22
define a loss function in this lecture itself and we are going to see how that loss function quantifies the loss L
4:30
between what our llm has generated and the good output which we actually want
4:36
in the subsequent lectures we'll be looking at taking an entire data set
4:41
feeding it into the llm getting the output and finding the training and the validation losses will then generate the
4:48
llm training function that's the back propagation I was talking about and then towards the end of this series on
4:54
training llms we'll also load pre-trained weights from open AI all that will come next for now let's just
5:01
start with these initial two goals for today's lecture and that is text generation using llms and text


***



5:09
evaluation first I want to quickly recap how can we use the llm or the gpt2
5:15
architecture which we have trained so far to generate text okay so the way it


***

5:20
works is that we have to specify a GPT configuration uh and this GPT
5:26
configuration or the llm configuration contains many parameters first is our vocabulary size this is the length of
5:33
the number of tokens we have in our vocabulary so we are using the same tokenizer which gpt2 and in fact all
5:41
open AI models used and that's called as tick token so this is thck token and what it
5:47
does is that it uses a bite pair encoder and it creates a vocabulary of tokens like this the vocabulary size which gpt2
5:55
had is 50257 and that's the same size which we have when we conr constructed our large
6:00
language model we have to specify a context length gpt2 used a context
6:05
length of 1024 but for the purposes of easy calculation which you can train on your
6:12
own local machine in less than 2 to 3 minutes we are using a context length of 256 keep in mind that you can just use
6:20
the same code for a longer context length as well but just make sure that you have enough compute power and memory
6:27
to execute the code then we have the embedding Dimension which is which basically means that the token
6:33
embeddings need to be projected into higher dimensional space to capture the semantic
6:38
meaning then we have the number of attention heads so these are the number of self attention mechanism blocks we
6:44
have within one Transformer why do I say within one Transformer because there is not one Transformer there are many
6:51
Transformers and the end layer specifies how many Transformer blocks we have
6:56
dropout rate is can be set to zero also this is the Dropout layer parameter and
7:01
query key value bias is basically uh when we initialize the weight metries for the query key and value we don't
7:08
want the bias term if any of these terms are looking unfamiliar to you right now
7:14
please revise the previous lectures we have had on uh llm Basics attention
7:21
mechanism and GPT architecture awesome so the GPT model which we built earlier looks something
7:28
like this what this model model does is that it takes in the input tokens it converts them into token embeddings adds
7:34
positional embeddings to it then we have a Dropout layer this output upti layer
7:40
passes through the Transformer block after we come out of the Transformer block we have a final normalization
7:45
layer and then we return the logits now these logits which are returned consists
7:52
of the tokens and for each token we have uh the number of tokens equal to vocabulary size sorry for each token we
7:59
have the number of columns equal to the vocabulary size I'll come to this part later and I'll explain to you again what
8:06
we have done in this part of the code uh so that it's revised for you okay great
Inputs and targets
8:12
now let's start looking at the first section of today's lecture okay so let's start with the first part of today's
8:18
lecture remember that to to get the loss function we need the input from the input we'll get the
8:26
predicted values and we should already know the True Values which I'm going to call as Target values in today's lecture
8:33
whenever I use the word Target remember that it stands for the True Values and ultimately the loss function will be
8:39
determined by the predicted and the target values and how close they are so in the first section let's look at the
8:46
inputs and let's look at the targets which are the True Values okay so the input to the large language model which
8:53
we are building comes in a format like this the input is a tensor and uh the
8:58
number of Ro rows here correspond to the number of batches so I have two batches over here and that's why there are two
9:04
rows uh so two batches corresponding to the two rows I'm writing two over
9:10
here the first row corresponds to the first batch the second row corresponds to the second batch now you'll see that
9:16
every row has three tokens here which is usually set by our context size and I'm just assuming context size equal to
9:22
three for this example so for the first batch the input tokens are every effort
9:28
moves for the second badge the input tokens are I really like now the thing
9:33
is based on these inputs our task is to predict the next next word right and uh
9:40
so for targets you must be thinking that the target should only be two token IDs
9:45
for the first batch we need a token ID for the second batch we need a token ID but now look at this target tensor over
9:52
here it has two rows corresponding to the two batches that's fine but why does the first row have three values and why
9:59
does the second row have three values the reason is because whenever you look at the input like this there
10:05
are actually three there are actually three uh prediction tasks which are happening here when every is the input


***


10:13
effort should be the output that's why the first Target is 3626 which corresponds to effort when every effort




***


10:20
is the input which means that 16833 and 3626 are input 61 0 is the output which
10:27
is moves and and then finally when every effort moves is an input the output
10:32
should be U so the correct answer which we want is U over here similarly for I
10:39
really like there are three prediction tasks when I is the input uh really is
10:44
the output when I really is the input like is the output and when I really
10:50
like is the input chocolate is the output so we want the output to be chocolate in this case but then there
10:56
are three prediction tasks here and that's why the target tensor consist of two rows first row corresponds to the
11:01
first batch second row corresponds to the second batch and then we have three values I hope you have understood why
11:07
there are three values in the Target tensor because that will be very important to keep in mind as we move
11:13
forward ultimately we are just concerned about the last word right which is this 345 that is the token ID corresponding
11:20
to U uh token ID corresponding to U and
11:25
11311 that's the token ID essentially corresponding to Chocolate but but the first two are also important for the
11:31
initial prediction tasks in this input okay so the these are the True Values which we want so this tensor are the
11:38
True Values true prediction values if our large language model is doing an amazing job for these two inputs the
11:46
prediction should be the first tensor of the output should look like this the second of the out the second row of the
11:52
output should look like this but of course initially when the large language model is initialized randomly it won't
12:00
produce these outputs which we want we'll need to train it okay so we have
12:05
looked at the input and we have looked at the Target now let's see the outputs which are the predicted values of our
LLM Model Outputs
12:13
llm so this is the input which we saw over here right uh 16833 3626
12:20
610 then that's the first batch the second batch is 40 1107 and
12:26
588 so the this input will go to GPT model and what is the GPT model the GPT
12:32
model looks something like this so the input which I told you goes over here and then it passes through all of
12:39
these steps and then we have an output tensor which is also called as the logit
12:45
tensor okay so the first step as to what happens in the GPT model is that the tokenizer is converts the tokens into
12:52
token IDs then the token IDs pass through all of these steps and we get the logic sensor and finally the logits
13:00
are converted back into token IDs which are the next predictions uh of our large language
13:06
model this entire sequence is encoded in this figure here which I want to explain thoroughly first we start with a
13:12
vocabulary and for the sake of Simplicity let's assume that the vocabulary just has seven elements
13:18
remember in the actual example which we are going to consider the vocabulary actually has 50257
13:25
elements but I'm just illustrating this diagram for the sake of Simplicity where there are seven elements in the
13:31
vocabulary right and the input text let's say is just one batch for now and every effort moves that's the input
13:38
right the first step is to use our vocabulary to map the input text to token IDs so every is token ID number
13:46
two effort is token ID number one and moves is token ID number four so First
13:52
We Take These tokens and map them into token IDs 2 1 and four the second step
13:58
is that we'll pass all of this through the GPT model and the GPT model will give uh a
14:07
logit sensor and to the logit sensor we apply the soft Max distribution and
14:13
we'll get this output Matrix look at this output Matrix here I'll just write it Dimensions here so this output Matrix
14:21
essentially uh it has uh let me use a different color it has
14:27
three rows the first row corresponds to every the second row corresponds to effort and the third row corresponds to
14:33
moves now you'll see that the number of columns are equal to 7 1 2 3 4 5 6 and
14:39
7even and we have some values see every row has some values and there are seven
14:45
columns corresponding to every row basically the values in each row corresponds to the probability of what
14:51
the next token will be so let's look at the first row values and I'll just write them over here for your reference 0.1 6
14:59
2 0.05 0 0.02 and 0.01 these are the values corresponding
15:06
to every now what these values means is that when every is the input what is the probability for the
15:13
next token being the output and 1 2 3 4 5 6 7 corresponds to a effort every
15:19
forward move Z and Zoo so let me write the a
15:26
effort every
15:32
forward moves U and Zoo this is the vocabulary
15:37
right now every value here in this row corresponds to the probability of what
15:43
the next token will be if every is an input so let's Analyze This probability if every is an input the probability
15:50
that the next token is a is just 10% or 0.1 there is 60% chance of the next
15:56
token being effort 20% chance of the next token me every Etc so then what we
16:01
do is that we look at the index with the highest probability which is this 6 over here and so we know that the next token
16:08
is equal to effort now when every effort is the input we again look at the second row so
16:14
we again look at the second row and find the index which has the highest probability and that corresponds to
16:20
moves so this is index number four and that corresponds to moves over here then
16:25
we look at the third row so every effort moves when that's the input what's the output and this is the main prediction
16:32
task because it actually predicts the next token and if you look at the third row you look at the index which has the
16:38
maximum probability and that's index number five which is here and we know that index number five corresponds to
16:44
you so we know that the next uh token is you so remember when I told you that
16:50
when the input has three tokens like this there are three prediction tasks for the first prediction task the output
16:56
is index number one because that corresponds to the highest probability and the token which it corresponds to is
17:03
effort for index number two or sorry for the prediction task number two the input
17:08
is every effort and the output is index four which corresponds to moves and for
17:13
the third prediction tasks the input is every effort moves and the output is five which corresponds to U so when you
17:21
give this input every effort moves to the uh large language model the output
17:29
is a sequence of indexes 1 4 and five what this output means is that one so if
17:35
you look at one over here it means that when every is an input effort is the output when every effort is the input
17:43
moves is the output and when every effort moves is the input U is the output so this is what is actually
17:49
happening within the GPT model itself I just gave you a five minute crash course of what we learned in four to five hours
17:56
in the previous set of videos so if any any of this you're finding hard I would
18:01
again highly encourage you to go through all the previous videos so that you understand this better for now even if
18:08
you get an intuition of what's going on here it's fine and you can follow along till the next part right so here what I
18:15
want to illustrate to you is that this process which I've shown here we have the input when it goes to a GPT model we
18:21
have the output token IDs and uh I've told you how we get the output token
18:26
IDs so if you have understood the process let's say these are the output token IDs which we have obtained
18:33
remember for actually gpt2 the vocabulary size is 5257 right so the
18:38
range of the output token ID is is 0 to 50257 here the vocabulary size was 7 so
18:45
the range was 0 to 7 so if you look at batch number one uh the output tokens
18:50
are 16657 339 and 42826 U and let's look at the inputs
18:56
right every effort moves so
19:02
every effort and moves what this output essentially conveys is that if every is an input the
19:09
token corresponding to the Token ID of 16657 is the output when every effort is
19:15
the input the token corresponding to token ID 339 is the output and when every effort moves is the input the
19:21
token corresponding to the Token ID 42826 is the output and what do we want these token IDs to be we want these
19:28
token IDs to be as close to the Target token IDs which we saw look at these Target token
19:34
IDs the token IDs which we have obtained right now are not the actual token IDs because the training has not yet
19:41
happened so these are the output token IDs and we want these output token IDs to be as close to the Target token IDs
19:48
as possible that's the whole goal of today's lecture and so we are going to then Define the loss function between
19:53
the outputs and the targets so these are the output token IDs for batch number number one and here you can see that
20:00
these are the three output token IDs for batch number two where the input is uh I
20:06
really like so I really like and for these outputs we
20:13
have to compare these outputs to these True Values for the second batch I hope you have you have
20:20
understood the goal of today's lecture um up till now we have understood what
20:25
are the inputs the shape in which inputs will be given to the model what are the
20:30
True Values which I'm also calling as targets and we understood that what are the output uh output token IDs for batch
20:39
number one and batch number two and to get these output token IDs the inputs have to go through a huge Transformer
20:46
block and it has to go through this whole GPT architecture then we get the output Logics we apply soft Max and we
20:52
get this tensor like this this one tensor essentially contains all the information you take a look at at this
20:58
tensor and then you extract the indexes with the highest probability in each row
21:04
that is essentially your output okay and now what we are going to
21:10
do next is that we are going to then find the loss between the targets and the output before coming to the next
Coding the LLM Model Outputs
21:15
step I want to go to code and really code everything what we have learned so far so that the understanding is clear
21:22
and so that you also understand uh what is really going on in the
21:27
code I already explained to you the configuration which we are going to be using right vocabulary size 50257 the
21:34
context length of 256 uh embedding dimension of
21:40
768 number of attention heads 12 number of Transformer blocks equal to 12
21:45
dropout rate 0.1 and the query key value bias equal to false awesome now the next thing what we
21:53
are going to do is that we are going to uh construct our model what this model
21:58
is basically it's an instance of the GPT model class and this is the GPT model class which we have defined previously
22:05
uh to get a visual representation of what this class does is that it takes in the input and then it converts the
22:11
inputs into this logic sensor that's so all of these steps
22:16
which have been written here are actually encoded visually in this Transformer block or in this GPT model
22:22
architecture which I've seen the blue color which I'm highlighting right now with the orange pen that's the transform
22:28
for block and it's the heart of the entire GPT architecture so here we are creating an instance of this model so
22:35
that when we pass an input to this model we get the logic sensor as the output one note is that we reduce the
22:43
context length of only 256 tokens here although we know that the gpt2 model
22:48
uses 1024 tokens the reason is because you all will be able to follow and execute the code on your laptop computer
22:56
and we reduce the computational resource requirements that way okay there are two more functions which I want to define
23:03
the first is text to token IDs what this will do is that whenever we get any text
23:08
which is the input we'll just convert it into token IDs and this uses this tick
23:13
token tokenizer to convert the text into token IDs and there is also token IDs to
23:19
text which is the reverse function so basically when we get any token ID we'll convert it back into
23:25
text so uh if every effort moves you is the text and when you pass it through
23:30
the tokenizer you'll get the token ID is corresponding to this and then you can even decode the token IDs and it will
23:37
get you back the same text just as a proof that our model is working what I've done over here is that uh I have
23:44
use this generate text simple which is another function we have defined in the previous lecture no need to worry about
23:49
this right now but what this function does is that it takes in our model and we predict the maximum number of new
23:55
tokens the number of input tokens is four every effort moves you and then it generates 10 new
24:03
tokens um so here you can see the we have printed the output text and this shows that the output text consists of
24:09
four input tokens and 10 output tokens remember one word is not necessarily equal to one token because we are using
24:16
the bite pair encoder where we even have subwords and characters which can be individual tokens so up till now as we
24:23
see the model does not produce good text because it has not been trained yet and
24:28
uh so what we'll be doing is that how to measure what what good text is so that's
24:33
why we have to define the loss function right now let's start with the main implementation in today's lecture and
24:40
these are the inputs so the first input as as I told you on the Whiteboard is every effort moves these are the token
24:46
IDs corresponding to the input text and the second input is I really like and
24:51
the token IDs which are corresponding to the second input text are 40 1107 and
24:57
588 great and the targets are essentially 3626 610 345 these are the True Values
25:05
for the this for the first batch for the second batch the targets are 1107 588
25:10
and 11311 take a note that the targets are the input shifted by one so if you take
25:17
the inputs and shift it or just take the last two values here and add a new value
25:22
that's the targets this is because there are three input output pairs okay now now once we get the
25:29
inputs what we are going to do is that we have the inputs and we have the targets but we have not yet generated
25:34
the outputs to generate the outputs what we'll do now is that we'll pass in the input through this GPT model we'll pass
25:41
the input through this GPT model block and generate the output token IDs in this in this kind of a format which I've
25:47
shown you okay so the first step is that remember that when you pass the input
25:52
through the GPT block it returns this logits and which are not converted into
25:58
normalized 0 to one format we have not applied soft Max so the first St step is
26:03
that to get these Logics and pass them through a soft Max uh always keep an eye out for Dimensions why is it 2A 3A
26:12
5257 the reason is because everything can be explained through this
26:18
diagram look when the vocabulary is so let me first rub many things here just
26:23
so that Things become a bit easier to understand
26:30
okay so look when the vocabulary size was equal to 7 over here take a look at the form of this logic sensor so if the
26:37
vocabulary size was equal to seven for one batch the logic sensor size was three rows and seven columns right now
26:45
if there were two batches the size would have been 2 comma 3 comma 7 that's the
26:51
general size of this logic sensor so it's batch size batch size the second is the number
26:58
of tokens and the third is the vocabulary Dimension so in this case the
27:06
vocabulary Dimension was seven so there are seven columns in our case there are 50257 columns so this third dimension
27:13
will be 5257 so here we see that the probab the
27:18
logic tensor is converted into a tensor of probabilities whose shape is batch size number of tokens and vocabulary
27:25
size so that will be 2 comma 3A 50257 great and now what we'll do as I
27:32
mentioned to you the next step is that we extract from each row the index which
27:38
has the maximum value and that will be done using the AR Max function and these
27:43
indexes with the maximum value that's our output from each given batch so what
27:49
we are now going to do is that we are going to do torch. AR Max you can even go to the P Tor documentation and see
27:56
what argmax does Arc Max basically Returns the indexes of the maximum value
28:02
so we are doing tor. AR Max Dimension equal to minus one which is along the column so what this will do is that it
28:08
will look at all the values in the columns and give the index of the value which is the give the index which has
28:14
the maximum value and so here we see for the first batch there are this is the
28:20
first tokens output second tokens output third tokens output index remember these
28:25
are token IDs for the second batch this is the first tokens token ID output this
28:32
is the second output token ID this is the third output token ID and these output token IDs are the ones which we
28:38
want as close as possible to these Target token IDs that's the main goal and for this we are going to construct a
28:45
loss function so now actually let's decode these tokens remember we have
28:50
already written a function which decodes uh um here we have written a function
28:55
which converts token IDs into text right let's decode these tokens in the output and let's see what they mean so we have
29:02
passed our input into a GPT model and we have got these outputs and let's decode these now so if you decode for the first batch
29:12
the output the input is effort moves you and uh the output is armed H Netflix
29:19
completely random output which does not make sense we want this output to be every effort moves
29:26
you uh every effort moves you yeah so input so the target is every effort
29:32
moves you sorry effort moves you correct this is the true value so we want the target to be effort moves you but the
29:38
actual output which we have got is armed H Netflix so if you look at the first
29:44
batch you can see that there's a huge difference between the targets and the outputs right which makes sense
29:51
because well uh we have not yet optimized we have not even defined the
29:57
loss function we have not try to minimize the loss function so ideally now what we'll do is that we'll Define a
30:02
loss function between the targets and the outputs from both the batches so that the target output and the so that
30:10
the target text and the output text will be as close as possible to each
30:16
other great now that actually motivates us to go to our next section in today's
Cross entropy loss in LLMs
30:22
lecture which is actually defining the loss between the targets and the output so now let let's see what's going on
30:29
here so we have the inputs and this is the probability tensor which indicates
30:35
what's the probability of the next token for every input token what I've marked in blue over here
30:41
is the target indexes so if we want the actual answer to be that for every it
30:47
should be effort for every effort it should be moves and for every effort moves it should be U the the correct
30:53
answer of the indices which we want is 1 four and five these are the true values uh so the token IDs here are 3626
31:04
610 and 345 but uh in the sample code which I've shown over here we just has
31:11
seven uh we just has seven vocabulary size the token IDs which are 1 four and five those are the correct ones so now
31:18
one thing which I would like to mention is that let's say these indexes which are marked with the star right now these
31:24
are the actual indexes which we want right but if you see the values corresponding
31:30
to those those values are not the highest because our llm is not trained yet so what I'm going to do now is that
31:37
I'm going to collect the probabilities at these indexes so I'm going to take this index I'm going to take this index
31:43
and I'm going to take this index and I'm going to find the probabilities at these these indexes or these indices rather
31:50
and I'm going to collect these together so for example for the first batch what I collect can be 144
31:58
and3 these are the set of probabilities and for the second batch I'll collect another set of three probabilities like
32:04
this this is exactly what we are doing in the actual problem in the actual problem we know that the target indexes
32:11
are these so what I'll do is that I look at my probability tensor and I'll get
32:16
the probability which is corresponding to these indexes I know it will not be maximum because my llm is not optimized
32:23
but I'll just write down these probabilities for now so uh I I have these Target indices and let me call
32:29
them i11 i12 I13 for the first batch and i21 i22 and i23 for the second
32:37
batch so what I'm uh so what I'll be doing now is that I'll be uh looking at
32:43
batch one and I'll be looking at badge two and I'll find the value corresponding to these indices in the
32:48
probabilities tensor so in this in in the probability tensor which look which
32:54
will look something like this for the 50257 vocabulary size also I'll find the
32:59
probabilities corresponding to i11 i12 and I13 for batch number one and I'll find the probabilities corresponding to
33:07
i21 i22 and i23 for batch number two so for batch number one I'll aggregate
33:12
these probabilities together and they look like this P11 p12 p13 for batch two I'll aggregate these probabilities and
33:19
they look like p21 P22 and p23 remember these are the probabilities which are
33:24
not maximum right now the goal of training is that to get all of these
33:29
values as close to one as possible and why do we want all these values close to one as possible because then we'll make
33:36
sure that the output indexes which have the maximum probabilities will be closer
33:41
to i11 i12 I13 i21 i22 and i23 which are
33:46
my targets so the goal of the llm Performing better now is reduced to this
33:53
problem that I want these probabilities to be as close to one as possible I want all six probabilities to be as close to
33:59
one why are there six probabilities because there are two batches each batch has three tokens so there are three
34:05
prediction tasks so that's why there are three probabilities for the first batch three probabilities for the second
34:11
batch so let me merge these probabilities together and then this will be P11 p12
34:18
p13 and this will be p21 P22 and p23 now what I want is I want all of
34:25
these P11 p12 p13 p21 P22 and p23 I want all of these to be as close to one as
34:32
possible how do I enforce this mathematically first let's see the workflow so we had the logit tensor we
34:39
converted it into probabilities through soft Max and then we had the target probabilities tensor what this target
34:47
probabilities means is that we have the IND indices corresponding to the Target values so we have these indexes i11 i12
34:55
I13 i21 i22 i23 these Target probabilities are just the merge merge
35:01
Target probabilities which is P11 p12 dot dot dot up to p23 these are
35:07
the six probabilities and we want all of these to be as close to one as possible
35:13
to all of you students who have studied uh classification and the loss this problem would be familiar since we are
35:20
dealing with probabilities it's natural that logarithms and cross entropy will come into the picture so instead of
35:26
directly dealing with this number numbers it's much better mathematically and from an optimization P perspective
35:31
to just take the logarithm of these values and it comes out to be this and then we take the average of all of these
35:38
logarithm values and it comes out to be this and then we take the negative of this average so that's 10.77%
35:58
Target probabilities and we find the mean of so we take the then the log of this so we take log P11 log p12 and the
36:07
last is Log p23 and then what we do is that we find the mean of this
36:14
mean and uh so then that will be Sigma which will be the summation of log 1 one
36:21
summation of log P11 DOT log p23 divided 6 and then I'll just
36:30
take the negative of this and that's called the negative log
36:38
likelihood so then that will be negative of summation of log of P11 dot dot dot up till log of
36:47
p23 divided 6 so now see we want uh we want P11 to be close to uh one we
36:56
want p all the probabilities to be close to one so we essentially want this negative log likelihood to be as less as
37:03
possible so this cross entropy loss we want to be as low as possible and as close to zero as possible so uh why are
37:12
we taking the negative log likelihood the reason we take the negative log likelihood is
37:19
because it just makes more sense if you if you don't take the negative value then uh the loss function would look
37:27
reverse of this the loss function would look reverse of this and then we'll have to reframe the problem as trying to
37:33
maximize the loss instead it just makes more intuitive sense if the negative
37:40
looks like this and now the whole
37:45
goal instead it makes more physical and intuitive sense if the log likelihood
37:50
looks like this and now our whole goal is to bring down this loss as low as
37:56
possible what will happen if this loss is brought down to let's say zero if the loss is brought down to zero which means
38:02
that the mean which means this value is Clos to one which means that there is a higher chance that P11 p12 p13 up till
38:10
p23 are equal to one that is exactly what we want the goal of training is to get these values as close to one as
38:18
possible so that is the main uh main hope now so initially let's say when we
38:24
start the training procedure our value uh the x value whose log we are going to
38:30
take that will not be close to one it will be somewhere maybe here it will be somewhere 0 to one and somewhere here
38:36
the goal is to go here slowly so that the mean of the probabilities will be equal to
38:44
1 okay so this is the negative log likelihood and it's also called as
38:51
the uh it's also called as the cross entropy loss and uh we want to minimize
38:57
this cross cross entropy loss as much as possible that will ensure that the indexes at which our
39:03
tokens uh indexes at which we predict the next token that matches with the target indexes so then the output and
39:10
the target indexes will match closely to each other and then we know that the
39:16
large language model is actually doing a very good job so this is exactly what we are going to do right now in the code
39:22
one thing which I would like to mention about the cross entropy loss is that the cross entropy loss essentially measures
39:29
the difference between two probability distributions so here what we are actually doing is that we are just
39:34
adding discrete probabilities but in a sense this can also be called as the categorical or I should call it the
39:40
cross entropy loss so now let's see the sequence of steps which we are going to implement
39:46
the logit tensor which is a tensor of probabilities uh has the shape of 2 into 3x
39:53
50257 why because it looks something like this so every effort moves and then
39:58
we have 50257 uh so this is the first batch and
40:03
this is the second batch what we are going to do is that we are going to flatten this so that these two first
40:10
batch and second batch are merged together right so these are so this is
40:17
my output tensor right now every effort moves you I really like it's a merging of these two batches and my target so my
40:24
target is 2x3 why is it 2 by three because for the first batch this is the
40:30
target index of the first input every this is the target index of every effort and this is the target IND index of
40:37
every effort moves 107 is the target index of I so it will be really 588 is
40:43
the target index of I really and 11311 is the target index of I really like so
40:49
then chocolate then what we are going to do is we are going to flatten this out so then this will be 3626
40:55
610 uh 345 107 588 and
41:00
11311 now ideally what we need is that we are going to look at every year and
41:06
we are going to look at the index corresponding to 3626 so let's say if this is the index corresponding to 3626
41:13
this will be P11 for effort we are going to look at the index corresponding to 610 let's say it's this one so that will
41:20
be p12 similarly for like which is the last we going to look at the index
41:26
corresponding to 1 311 so let's say this is this so this is p23 so what we'll get is that we we'll
41:32
get P11 p12 p13 p21 P22 p23 we'll take the log of them we'll add take the mean
41:40
and then do the negative this whole process is encapsulated in this just one line of code tor. nn. functional cross
41:48
entropy logits so this is the logits flat I'm calling this logic flat tensor
41:54
and the second argument here is the target flat what this uh this code will
42:00
do is that tor. nl. function cross entropy logit flat comma targets flat
42:05
first it will convert this logits tensor into a tensor of probabilities until now the logits is is not does not represent
42:12
probabilities so it this function will apply soft Max to the Lo logits and then
42:17
what it will do is that as I told you before it will find the negative log likelihood so first it
42:24
will uh find the probabilities P11 p12 p13 p21 P22 and p23 in this logic flat
42:31
tensor corresponding to the index indices in the Target flat tensor it will take the negative log and then it
42:38
will find the mean this one line of code is going to do all of the steps for us that's why python is so powerful in one
42:45
line of code we'll not only convert this Logics into uh a tensor of probabilities
42:51
through soft Max but we will also get the negative log likelihood and that's the final loss function which we are
42:57
looking for so this one line of code is going to give us the loss between the output and the Target and that's why
43:04
it's a very important piece of code uh to do all of these steps in just one
43:10
line of code I think it's pretty amazing but what I want you all to focus on is what we really are doing over here what
43:17
we are essentially doing is that we have inputs and those inputs are every effort moves and I really like we
43:24
are passing them through the GPT architecture and then we get this logits we get this logit tensor what we are
43:30
going to do with this logit tensor is that we'll first merge them merge this tensor merge the batches and call it
43:37
logits flat then what we'll do is that corresponding to every token we are going to look at the indexes which
43:44
corresponds to the Target and then we are going to find the probabilities now in an Ideal World these probabilities
43:51
will be close to one because that would mean that the output and the in uh Target will match
43:58
but since the llm is not optimized these probabilities which correspond to the Target tensor indices will be probably
44:05
be very small and our goal is to bring all of these probabilities close to one as possible so that's why we'll uh take
44:12
the negative log of these probabilities and take the mean that's the cross entropy loss and through one line of
44:18
code cross entropy tor. nn. functional doc cross entropy between these two
44:24
tensors we are going to implement this exact same operation which I just described in words right now so you can
44:30
even check this and let me type it torch.nn do functional cross
44:37
entropy and you can check it finds the cross entropy loss between the input
44:43
Logics and the target this is exactly um what we did right
44:49
now okay now I'm going to implement this same thing in code we are going to find the cross entropy loss in code okay so
Coding the LLM cross entropy loss
44:56
for first what we are going to do is that uh we are going to find the P1 P2
45:01
P3 so we are going to find P11 p12 p13 and p21 P22 and p23 that's the first
45:10
step so remember here in the sequence of steps over here first we find these Target probabilities right uh and to
45:17
find these Target probabilities what we'll do is that we will uh we'll take the target indexes which
45:25
are uh the indexes corresponding to the True Values and what we are going to do
45:30
is that we are going to uh take the probas so let me see what the probab probas is so this probas is the
45:38
output uh this probas is the output probability tensor and what we are going to do is that we are going to find the
45:45
token probabilities corresponding to the Target indexes and to do that what we are going to do is that we are going to
45:51
use this line of code and what this will just do is that it will take the target indices and it will index or it will
45:59
look at this probas which is the tensor of output probabilities and then it will look for those particular indexes so
46:05
basically what it will be doing is that for zero it that's P11 that will be this
46:11
for one that will be P1 two and for two that will be p13 similarly when we look
46:17
at the second batch over here um what we are doing over here is
46:22
that uh that this is one which means that it's going to look at
46:28
the second batch so one way to understand what's going on here is to look at the probas and try to see the
46:34
dimensions so uh let me write it here again for reference just so that your understanding is clear so the probas is
46:41
actually it will look something like this first let me write it for the first
46:49
every effort moves right and then
46:55
uh this is 50257 the number of
47:02
columns and then second is I really
47:09
like and the size of this so this whole thing is the probas tensor which is the
47:15
probability tensor and the size of this is so we have two batches we have three tokens and
47:22
50257 so what we are essentially doing here is that
47:27
uh let me scroll down below yeah what we are doing here is that in this line text idx
47:34
is equal to zero which means that we are first looking at the first batch so we are looking at this batch and then what
47:42
we are doing is that we are looking at row 0o row one and row two and from row
47:48
0o we'll get the um we'll get the so let's look at the Target answer we'll
47:54
get the index corresponding we'll get the value corresponding to 3626 index from Row one we'll get the value
48:00
corresponding to 61 0 and from row two we'll get the value corresponding to
48:05
345 now similarly here what we are doing here is that we are looking at so text
48:10
idx equal to 1 which means we are looking at the second batch we are looking at the second batch and then uh
48:17
row zero Row one and row two so then we'll look at the targets tensor again
48:22
and for row zero we'll take the value corresponding to index 1107 for Row one we'll take the value corresponding to
48:28
index 588 and for row two we'll take the value corresponding to index
48:33
11311 so that is essentially uh p21 P22 and p23 so these
48:40
three values are the P11 p12 P2 p13 and these three values are p21 P22 and p23
48:48
the whole goal is to get these values as close to one as possible right then what we'll do in the Second Step as I told
48:54
you uh over here once we get these P11 p12 p13 p21 P22 p23 we are going to
49:01
merge these together right so that's what's written here we are going to concatenate these six values together
49:07
and you can see that they appear like this the next step is that we are going to take the
49:13
log uh actually after concatenation yeah we take the log over here and then we print these log values then we'll take
49:20
the mean of these log values and then we are going to do the negative of the log likelihood
49:27
uh so as I told you we can also not do the log likelihood and just do positive log likelihood but then that would mean
49:35
maximizing the loss that does not make sense so in deep learning it's conventional to use negative log lik Le
49:40
so that we minimize the loss right so this is now the negative log likelihood loss value which we ultimately need to
49:47
minimize now as I told you there's a much simpler way of doing this using the
49:53
torch.nn do functional cross entropy the first step of this is to flatten the logits and the targets and to use this
49:59
one line of code so that's what I'm going to do now I'm going to take the logits which was earlier 2A 3A 5257 and
50:08
I'm going to flatten it I'm going to flatten the these first two Dimensions together so that it's 6A
50:14
5257 this is exactly what we saw over here look at this it's 6A
50:20
5257 batch one and batch two are merged together then what I'm going to do is
50:25
that I'm going to look at the Target stenor and I'm going to flatten it so if the target stenor is this two rows and
50:31
three columns I'm going to merge these two rows into six values and then I'm going to just write
50:38
one line of code this one line of code will first convert these Logics into a soft Max and then it will find the
50:45
values corresponding to the Target indexes then it will take the log of these the mean and then the negative log
50:51
likelihood all of this will be done and I will get my categorical cross entropy loss of 10 remember I want to take this
50:57
loss as close to zero as possible that's the goal so when we train the large language models later the goal is to
51:03
bring this categorical cross entropy loss to as low as possible before we end the lecture I
Perplexity loss measure
51:09
want to show you one last thing there is another major of loss and uh that's like
51:15
cross entropy itself but a minor modification and that's called as perplexity so perplexity actually
51:22
measures how well the probability distribution predicted by by the model matches the actual distribution of words
51:29
in the data set and so in that sense it is more interpretable and it's more
51:34
interpretable way of understanding model uncertainty in predicting the next token and I'll show you why in a minute
51:41
remember that lower per perplexity score also means better predictions so the formula for
51:47
perplexity is just to find the exponent of the loss and surprisingly this simple uh
51:54
modification leads to a lot more intuition so if the exponent of the loss
51:59
in our case is 48725 it means that the model is roughly
52:04
as uncertain as if it had to choose the next token randomly from about 48725
52:10
tokens in the vocabulary so the number of tokens in the vocabulary is 50 uh 257
52:16
I think now what this means is that if the input is uh
52:22
every effort moves right every effort moves is the input currently our llm is
52:29
at a stage that to predict the next token that's as as if we have to choose
52:35
between 48725 tokens that's pretty pretty bad right it means that there is
52:40
a lot of uncertainty in predicting the next token if the perplexity score was equal to two that's pretty good which
52:46
means that we just need to predict between two tokens so that means our llm
52:51
is very accurate but in this case the perplexity score is 48725 48725 which means that 4 48,000 tokens
53:00
are kind of equally likely to become the next token which means that our model is not good at all uh you can see how it is
53:07
more interpretable than getting a categorical cross entropy loss of equal to 10 when I get a categorical cross
53:13
entropy loss of 10 I don't really know how it relates to my vocabulary size also but if you get a uh entropy or a
53:21
perplexity score of 48725 you can kind of relate it to your vocabulary size and make interpretable
53:28
predictions like these that the 48,000 next token 48,000 tokens in our
53:33
vocabulary of 50,000 are equally likely to be the next token and that's pretty
53:38
bad because our llm is not yet trained so this brings us to the end of today's lecture where basically what we
Recap and next steps
53:46
did is we took at we uh we first started with the inputs then we started with the True
53:53
Values which were known as the targets and then what what we did was we found the output values this was kind of a
53:59
revision we got the Logics tensor we converted it into a tensor of probabilities and then we got these
54:05
output tokens so then we Tred to find the loss between the targets and the
54:11
output using the cross entropy loss and we ultimately
54:16
saw that using one single line of code torge nn. functional. cross entropy we can find the loss between the loged
54:23
sensor and the target tensor as the lecture concluded we even looked
54:28
at another way of measuring loss which was called as perplexity which is much more intuitive and the way perplexity is
54:35
calculated it just e rais to loss which is exponent of loss and I also mentioned
54:40
the perplexity concept over here in our case the loss was 10.79 and the
54:45
perplexity value was 48725 and this is usually more
54:50
interpretable awesome in the next lecture what we are going to do is that we are going to take an actual
54:56
data set from a book which is called the verdict and first we'll tokenize this data set we'll divide it into input
55:04
output input output Pairs and then we are going to get the llm output and we are going to find the loss function or
55:11
the loss value for this entire uh data set until now in this lecture we just
55:17
looked at uh in the code we just looked at two sample inputs right we looked at
55:23
um let me yeah we looked at these two inputs in the next lecture we'll be scaling this up and look at an entire
55:29
text data set so next lecture will be a lot of fun and uh we'll run the entire
55:34
architecture on this um Hands-On example you can replace this example with any
55:40
book which you like Harry Potter book any other book so next lecture is going to be a lot of fun I hope you all are
55:46
liking these lectures we have now started a new module which is on U the
55:52
training large language model and we are slowly making a lot of progress on in building the large language models we
55:59
have already finished stage one now we are on stage two and rapidly moving towards completion thanks a lot everyone
56:06
and I look forward to seeing you in the next lecture

***





