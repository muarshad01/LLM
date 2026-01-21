nction recap
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from scratch Series today what we are going
0:12
to do is we are going to train our large language model until this point in
0:19
the uh llm pre-training stage what we have actually done is that we have uh
0:27
found out or rather we have understood the method which which is used to calculate the training and the
0:32
validation losses for a large language model and we covered that extensively in the last lecture where we saw that let's
0:39
say if you have an llm model and if you feed an input to the model you get an output how can you compare the output
0:46
with the target value and how can you get the loss function now that we have
0:51
calculated the loss function in the previous lecture the door is open for us to implement back propagation and to try
0:58
to minimize this loss as much as possible before we start learning about the llm training function in today's
1:05
lecture let me quickly recap how we calculate the loss between the llm
1:11
output and the target values so in last lecture and as is the case with today's
1:16
lecture the data set which we are going to use is the book which is called the verdict it's a book which is written in
1:24
196 and you can download this data set completely uh it's an open source dat
1:29
data set and uh it's not too big I think it has around 5,000 or 5,500 tokens so
1:37
it has around 20,000 characters and it has 5,000 tokens that's the data set we are going to use the first step we do is
1:44
that divide this data set into training and validation we use a training ratio of 0.9 which means that we use the
1:50
initial 90% of this data set as training and we use the remaining 10% of the data
1:56
set as the validation data great now once we have the training data there is uh we first need to divide
2:04
the training data itself into input and Target pairs remember we have not even come to the llm output for now since
2:11
llms are Auto regressive models we don't have labels so there is a special way in which the training data and the
2:17
validation data itself needs to be divided into input and targets within the data itself and we do that using
2:24
something which is called as the data loader so what the data loader does is that it looks at a data set and it
2:30
creates these input and Target pairs so let me give you an example of how these input and Target pairs actually look
2:36
like so here's one batch of input and Target pairs so this is one batch and
2:42
here you can see that the input consists of two samples and the output consists of two and the target consists of two
2:48
samples this is from our data set itself and in each sample here you can
2:54
see that there are four tokens but actually when we code the number of tokens is pretty big I think it's around
2:59
250 phic that's the context size or how many tokens you are going to see before
3:04
predicting the next token right so this is the first sample of the input and
3:09
here you can see this is the first sample of the output now the output or I should call
3:16
this the target rather I shouldn't call it the output because the target is the true value the actual value which we
3:21
want our llm to approximate so if you look at the first row of the input and the first row of
3:27
the target you'll see that the target Target is just the first row shifted to the right by one and similarly for the
3:33
second row of the input and the second row of the target that's how the input Target pairs are constructed so we we go
3:41
we first take this data set and uh what we do is that based on the context size let's say the context size is four so
3:48
the first input is these four tokens that is X1 here I'm assuming one token is equal to one word but that's not the
3:54
case because we use a bite pair encoder so I'm just showing this for illustration purposes here so this is
4:00
the first input that's X1 the first four tokens then the next four tokens that's the second input X2 and here there is
4:07
one more important variable which is called as stride so if you see between the input one and input two there is no
4:13
overlap right because the stride is equal to four so the stride is usually
4:19
equal to the context size so X1 X2 and similarly we go through the entire data set like this and we get the input and
4:26
similarly corresponding with each input we shift the input to the right right and we get the target so throughout this
4:32
entire lecture when we are going to refer to input and Target pairs we keep this visual in mind so this is a batch
4:40
each batch has here two samples so here's the input batch and here's one
4:45
output batch similarly sorry one input batch and one target batch similarly there will be huge number of input and
4:52
Target batches once we split all of the data set into input and Target pairs
4:58
right so that's the the first step and if you want to understand the details about how this input and Target pairs
5:03
are created we have done that in a lot of detail in the previous lecture so I'm not going to cover that right now so
5:10
these are the input and the target pairs right that's the first step then we need to get the llm output so what we do is
5:17
that we take the input value so we take the first batch take the second batch take the third batch Etc and pass all of
5:24
these batches into this llm architecture which we have built this is a pretty big architecture
5:30
where we first do the tokenization then uh do the embedding then add the positional embedding add the Dropout
5:37
layer and the output from this Dropout layer passes through the Transformer block so this blue color this blue color
5:43
block here which I'm highlighting right now that's the Transformer block here where all the magic happens the
5:49
Transformer has multiple layers such as normalization multi-ad attention Dropout shortcut connection feed forward Network
5:55
Dropout another shortcut connection Etc then we come out of the transform former block there is another normalization
6:01
layer followed by a output head or output neural network and then we get
6:07
this output tensor which is called as the logit tensor so the shape of the logit tensor
6:13
is pretty interesting to note over here let's say we are looking at the first batch right now and the first batch uh
6:20
has two samples right this is the first sample I had always thought which I mentioned here and this is the second
6:25
sample this is how the logic sensor looks like each of the now in each sample every token let's say I has the
6:33
number of columns which is equal to the vocabulary size that is 50257 uh so the size of this logic
6:39
tensor is 2 into 4 into 50257 in this case because there are two batches four
6:45
tokens in each batch and 50257 columns then we flatten this logic stenor which means we merge the first uh sample in
6:54
this batch and the second sample in this batch and we create this unified tensor which is eight and 50257
7:01
columns now when we get the logit sensor as the llm output every entry in the
7:07
logic stenor does not correspond to probabilities all the entries corresponding corresponding to one token
7:12
do not even add up to one so we need to implement the soft Max and convert this into Vector of probabilities so what
7:19
this means is that when you look at the first row I each value here corresponds to how much probability is there for
7:25
that token to be the next token so if the input is I uh every value here corresponds to
7:32
what's the probability for that token to be after I so if the llm is trained the
7:38
token index which has the maximum probability should correspond correspond to had because when I is the input had
7:43
is the output similarly when you look at the second row I had is the input if the llm is trained the token which has the
7:50
maximum probability should correspond to always because when I had is the input always is the output similarly when I
7:57
had always thought is the input uh the token I the token here which
8:04
corresponds or the index rather which corresponds to the maximum probability should correspond to uh the next word
8:10
which is I had always uh thought or I had always thought Jack
8:18
so let's see yeah I had always thought Jack so here the token ID which
8:23
corresponds to the maximum value in this case this index should correspond to the
8:28
token for Jack when the llm is actually trained now initially when we get this
8:34
tensor of probabilities uh the llm is not trained right so what we will do is
8:39
that we know the target we know the target values right we know the target which we want and we know the token in
8:45
indexes of this target so for example for I is the input had is the output and that has a certain token Index right so
8:52
let's say if I is the input and had is the output I want this token index to have the maximum value of the probab
8:59
ility when I had is the input always is the output so let's say I want this token index so what we do here is that
9:05
we just collect this list of targets and their token token IDs and we find the
9:10
probabilities on in each row corresponding to these token IDs now the whole aim of this uh exercise of
9:19
training llms is to make sure that these probabilities are as close to one as possible because then we'll make sure
9:24
that the llm outputs are matching the target values which means that the next token which our llm is predicting uh is
9:32
similar to the Target values Target value tokens but initially when the llm is not trained these probabilities will
9:38
be not close to one at all so we employ the cross entropy loss and we find this
9:43
negative log likelihood and the whole aim uh when we train the llm today is to
9:49
make sure that this loss function decreases and it becomes close as close to zero as
9:54
possible where the uh loss function is minimized so today what we are going to
10:00
do is that all the parameters in the GPT architecture here all of the parameters we'll take a look at them shortly we are
10:07
going to optimize them so that when the input inputs are fed into this model and when we take the loss function in the
10:13
way I have outlined today the loss function is minimized uh I went through this in a fast manner because we had one hour
10:20
lecture previously detailing this entire process okay now I hope you have
LLM Pretraining loop
10:26
solidified your understanding of how the loss function is calculated now that we have the loss function we are ready to
10:32
do the llm pre-training so here what it means is that we are going to try to minimize the
10:38
loss function as much as possible so we want the output the llm outputs to be as
10:44
close as possible to the uh Target to the Target value tensor and for that
10:50
we'll be doing back propagation so here's the pre-training loop schematic if you have worked with neural networks
10:56
before if you have done machine learning or deep learning it's extremely simple to understand
11:02
because once we have written down the entire loss function all we need to do is just do the backward pass and I I'll
11:09
explain to you what this means so what we are going to do is that uh we are going to do multiple epochs so let's
11:16
start understanding this step by step we are going to do multiple epochs one Epoch is going through the entire
11:22
training set once now in each Epoch we have multiple batches right uh because
11:28
the training set is divided into batches as I showed you we are going to uh look at one batch in one Loop and then in
11:36
that batch what we are going to do is we are going to find the loss function in the same way which I have described to
11:42
you right now and we even wrote a code for this in the last lecture I'm going to find the cross entropy loss function
11:48
for the entire batch and then I'm going to do this step which is the backward path to calculate the loss gradients in
11:55
this entire schematic this is the most important step the backward pass allows us to calculate the gradients of the
12:01
loss and what the gradients of the loss will enable us to do is to update the model weights and parameters so let's
12:07
say one particular parameter has a value of P old in one iteration in the in the
12:13
next iteration its value will be something like this its value will change uh sorry this
12:22
should be the other way around so this should be P new the new value is equal
12:28
to the value minus the step size into the loss gradient so the loss gradient
12:34
once we get these loss gradients they will enable us to Value all the parameters in our GPT architecture and
12:40
if your question is what are these parameters I'm going to come to that in just a moment and then after we update
12:47
the parameters we are going to print the training and validation losses we are going to see what the llm output is
12:52
there just for visual inspection and then we are going to go to the next next batch when we go to the next batch we
12:59
are going to reset the loss gradients from the previous batch to zero and then do the same process all over again for
13:05
all the batches and we'll do this until we finish one training EPO Epoch and
13:10
then we'll do this entire procedure for multiple training epochs just so that the training uh
13:16
proceeds as many time as possible and the parameter values are updated as much as possible this is the main algorithm
13:23
for pre-training the large language model and it looks pretty simplified right now but what we have done so far
13:29
in all of the lectures we have conducted is that unless we had a way to calculate
13:34
the loss function getting to this step getting the backward pass would have been impossible and to get this loss
13:40
function right now to get this loss function we first needed to understand how this logic sensors are obtained how
13:46
the llm output is obtained to get to how the llm output is obtained we need to understand this entire GPT architecture
13:53
itself to understand this GPT architecture we need to understand tokenization positional embedding multi-ad attention Dropout layer feed
14:00
forward networks Etc so without doing all of this it would have been impossible to calculate the loss
14:06
function which I explained to you today in 5 minutes so these 5 minutes have required a hard work of 20 to 25
14:14
lectures which have been covered previously so this is the training Loop schematic in in short we are we are
14:20
finding the loss we are doing the backward pass to get the loss gradient in every iteration and we are updating
14:26
the parameters based on these loss gradient values and our hope is that as we do the
14:32
updating let's say the loss function landscape is this it will not be this at all because it's a huge multi-dimensional landscape uh if this
14:39
is the loss function landscape and if we start from somewhere here our goal is to make the optimization so that at the end
14:46
we reach this Minima where the loss function is minimized great so in this entire code
14:52
the main step is finding the loss gradients and we are going to do that in Python through this method called loss.
14:58
backward I'm going to show you how elegant this pre-training code is in just one line python essentially
15:04
computes U or tensor flow pytorch essentially computes the entire backward
15:10
pass in just one line of command that's pretty awesome but you need to understand why it is so awesome the
15:16
reason it's so cool is because the sheer scale of the number of parameters we are dealing with so what we are essentially
15:23
doing is that we have the inputs we pass them through the GPT model we get the logits we apply soft Max then we get the
15:30
cross entropy loss between the output and the target this whole operation is differentiable which means that this
15:36
gives us so once the loss is obtained we can do the back propagation and find the partial derivative of the loss with
15:42
respect to all the parameters which come in in the model and mostly the parameters come in this step in fact I
15:48
think all of the parameters come in this step in the GPT model itself so the back propagation needs us to understand what
15:55
makes the GPT model differentiable so if you look at the GP model you will see that this is the workflow and
16:03
differentiability is ensured and maintained at every single step of this workflow so if you have the loss
16:08
function over here you can back propagate it and find the partial derivative of the loss with respect to
16:13
all of the parameters which come within this within all of these layers and if you add all of these parameters together
16:20
the number of parameters will be around 161 million parameters I'm going to show
16:25
show you how it comes to 161 million but through this one line of code through loss. backwards we are essentially
16:32
finding the gradient of the loss with respect to all of these parameter values and then updating all of these parameter
16:38
values isn't that pretty awesome the sheer scale of these operations would have been impossible to do 50 years back
16:44
when compute resources were not available but now I can do this backward pass and this optimization on my laptop
16:52
and uh I did this training in 6 minutes uh which I think even on your laptop you
16:57
can do it in five to 6 minutes we optimized 161 million parameters in this much amount of time and I'll show you
17:04
how we can do that first let me uh draw your attention to what makes the number of parameters to be this high so uh if
Measuring LLM parameters (~160 M)
17:12
you have followed the GPT model the GPT model consists of the embedding parameters the token embeddings
17:18
positional embeddings it consists of the Transformer block parameters and it also consists of the final layer before we
17:24
get the logits so the embedding parameters have the token embeddings whose size is equal to uh the vocabulary
17:31
size multiplied by the embedding Dimension and the positional embedding which is defined by the context size
17:38
multiplied by the embedding Dimension if you add the these two parameters this is 38.4 million these parameters are not
17:46
known to us we are even optimizing for the token and positional embedding parameters now if you look at the
17:52
multi-ad attention we have the query key and value trainable weights so there are three matrices here and each M each each
17:58
matrice has dimension of 768 which is the input embedding and the output embed embedding which are generally same in
18:05
this multi- tension and multiplied by three because we have query key and value three matrices so this is 1.77
18:12
million parameters and then there is also an output head um whose number of parameters are equal to 59 million the
18:20
total number of parameters in the multi-ad attention block that itself is equal to uh 2.36 million so 2. 36
18:29
million parameters here and then we have a feed forward neural network in the Transformer block and the feed forward
18:35
neural network is like this expansion contraction type of a setting where we have the input equal to the embedding
18:41
Dimension that's projected into an hidden layer which whose dimensions are four times the embedding Dimension and
18:48
then we compress it back to the original Dimension so uh the number of parameters
18:54
here are 768 which is the embedding Dimension then 4 into 7 68 which are the
18:59
number of parameters here so these are the parameters in this expansion layer and these are the number of parameters
19:05
in the contraction layer both are the same and they add up to 4.72 million now if you add up the parameters in the
19:12
multi-ad attention the feed forward neural network and the output head it comes out to be 2.36 + 4.72 million this
19:20
these are the number of parameters in one Transformer block and we have 12 Transformer blocks so the total number
19:26
of parameters which are coming from the Transformer block itself is 85.2 million
19:31
parameters and then there is a final layer which is a soft Max uh which gets us the logic tensor
19:38
and the number of parameters here are the embedding Dimension multiplied by the vocabulary size and that's 38.4
19:43
million parameters so if you add up this number of parameters together the embedding has 38.4 million Transformer
19:50
has 85.2 million and then the final layer is 38.4 million so the total
19:55
number of parameters if you add up are 162 million gpt2 on the other hand the smallest
20:01
model is 124 million right so the reason between this discrepancy is that they use something called weight time so in
20:07
the output projection layer they recycle the uh same number of parameters in the
20:13
embedding layer so hence we we reduce those many number of parameters and that's why the
20:21
number of parameters in gpt2 smallest model comes out to be 124 million in any ways I want to illustrate here the scale
20:27
of this operation and I want you to understand how the number of parameters comes to be of the order of 100 million
20:33
so when we do back propagation what we are doing is that for each of these parameters we'll first get this gradient
20:40
and that's obtained in this step you see uh backward pass to calculate gradients and once we get the gradients we are
20:46
going to update all of the model weights uh so the way to update the model weights so this is just a simple
20:52
gradient descent which I have shown in the actual code we'll use a version which is called Adam which is a bit more
20:57
complex but at the underlying core the operation is similar we get the
21:02
gradients which is the partial derivative of loss with respect to the weights and then we update all the parameter values based on this gradient
21:09
in the backward pass awesome right so this is how the parameters are going to be updated in
21:16
each uh in each iteration and we hope that as the parameters are updated the loss function value goes on decreasing
21:23
like I have shown over here and it reaches some sort of a Minima that is what the go is so let's see whether this
21:30
goal is satisfied now what I'm going to do is I'm going to take you through code and we are going to implement the
21:35
pre-training loop for the large language model so the loop looks like this I I
Coding the LLM Pretraining loop
21:40
want to draw your attention to uh something like so I want to draw your attention first to the most important
21:46
part what we are going to do is that remember I mentioned that we have a loader a data loader for the training
21:52
and the validation set correct so first what we are going to do is that we are going to look at the training loader
21:58
first and we are going to divide it into the input batch and the target batch this is similar to what I explained to
22:04
you right now over here see when I say input batch and Target batch always keep this this in
22:10
mind so the data set is divided into input batch and Target batches so there are multiple batches like this and at
22:16
one time I'm processing one such batch so at one time I'll look at one input and one target batch and I'll find the
22:22
loss between the input and the target batch how do I find the loss using this same workflow using the category cross
22:29
entropy towards the end I find the loss that is this step and then this this
22:35
step right here is the most important step in this whole code we are doing loss do backward what the loss. backward
22:41
does is that it calculates the backward pass which means that it will calculate the gradient of the loss with respect to
22:47
all of the parameters all of the 160 million parameters this one step is going to find the gradient of the loss
22:54
then we are going to do Optimizer dot step what this Optimizer do step does is that it essentially looks at
23:00
uh it essentially looks at this update Rule and it will update the parameter values based on the gradient values
23:07
which are obtained so this is the update model weights part and what I'm doing is that I'm just
23:15
uh mentioning how many tokens the model has seen and I'll show you where this
23:20
number comes into the picture but let's say we are looking at the input and Target batch and the cont context size
23:27
is uh 256 so we are looking at 256 tokens in one sample 256 in the second sample so we
23:33
are looking at 512 tokens in one batch in the one input batch at a time so the tokens seen will just return the number
23:40
of tokens which we are seeing at a particular time and you see we are doing so it will get added up as the as we
23:47
Loop through different number of batches so this Loop is for going through each batch in the training data set and this
23:54
Loop is for the number of epoch so after we go through all of the batches we we
23:59
will do the same thing again uh so the number of epo is specified by numor EPO
24:05
right now what we are doing is that this step is actually an evaluation step so you can even get rid of this but it's
24:11
important for us to see the validation loss so in this step what we are doing is that we are going to define a
24:18
function called evaluate model and we are going to get the training loss and the validation loss and we are going to
24:24
print these out as the training proceeds so let me show you what the evaluate model function looks like the evaluate
24:30
model function is actually pretty simple we calculate the loss U the Cal loss
24:36
loader is we calculate the loss for the training loader and we calculate the loss for the validation loader for the
24:41
entire data set and we have defined these functions before see in the previous lecture we used the same two
24:47
functions Cal Closs loader for the training and the validation and it Returns the training and the validation lots for the entire data set at that
24:55
particular uh at when that particular batch is being processed so what we'll
25:00
do is that after the first batch is processed we will run this part which means that after the first batch is
25:06
processed we will evaluate the model and we will get the training and the validation loss correct but we are going
25:13
to show it we are going to show it only uh only at a particular evaluation
25:19
frequency so what I'm showing is that here I'm set I will set the evaluation iteration and the evaluation frequency
25:26
to be equal to five which means means that after every five batches are processed only then I will show the
25:33
training and the validation loss so remember what is happening here when one batch is processed Global step will be
25:38
one when the second batch is processed Global step will be two now if the evaluation frequency is five this will
25:44
be zero only when Global step is equal to five right uh that is when the remainder will
25:51
be zero and only at that step only at that when I reach batch number five I am going to calculate the training and the
25:57
validation loss and I'm going to print it out that's it when I come to batch number 10 I'll do the same I'll
26:02
calculate the training and validation loss I'll print it out and I'll do this for every single Epoch so in this step
26:09
what I'm doing is when I reach at a particular batch number when the llm has processed that particular batch I will
26:15
print out the training and validation loss that's all we are doing up till now and then just for the sake of uh
26:21
visualization and understanding after one batch is processed so after uh one
26:28
batch is processed what I'm going to do is that I'm also going to print out a sample text or here I should say that
26:34
after each Epoch is processed sorry not after each batch is processed we'll first make sure we go through all of the
26:40
batches for one Epoch so after each Epoch is processed what I'm going to print is okay here's what the next
26:46
tokens my llm is generating right now so uh this generate and print sample is
26:52
another function which we have defined over here and what this will do is that it will print out the next 50 tokens
26:58
which our large language model is predicting we have already seen this generate Tex simple function before what
27:03
this function does is that it takes the model at its current stage so if we are at Epoch number five the parameters are
27:09
optimized maybe they are not very correctly optimized but we are at certain stage and we want to see what
27:14
the output the llm is predicting for the input right so this function is
27:20
essentially going to generate the tokens for us and it will generate 50 new topens for us to visualize and we can
27:25
see right whether the output is correct or not whether the output of the llm is making sense or
27:30
not so this will be a lot of fun we are going to print this out at every single Epoch so the main step in this code is
27:38
this loss. backward what we are doing after this point is just printing the training and the validation loss after
27:43
every five batches and we are going to print the Tex sample after each epox and
27:49
we are returning the training losses validation losses and we are also tracking the token scene see we are
27:54
tracking this token scene which is the number of tokens the input batch is using uh that will give us a sense of
28:01
how many tokens the um so it Returns the elements how many tokens have been
28:07
consumed until that particular point in the model and we'll also plot the number of tokens in the output so remember one
28:14
one Epoch will go through the entire data set once right and if we look at our data set again um the data set which
28:21
we are considering I think it has around 5,000 tokens so if we go through the if
28:27
the the number of epo are 10 so which means that if you go through the entire data set 10 times it actually means that
28:34
the number of tokens seen should be 5,000 multiplied by 10 so it should be of the order of magnitude of 50,000 just
28:41
keep this in mind so yeah this is the code and it's pretty simple it just I think around 15
28:47
to 20 lines of code and the reason it's made so simple in Python is because of this loss. backward method it's pretty
28:54
awesome and it does the gradient updates uh sorry it calculates the loss gradients and then we just do the
28:59
optimizer do step we have not yet defined the optimizer but we'll be doing it
29:04
shortly okay now what we are going to do is that uh I'm just going to explain the evaluate model function and the generate
29:11
and print sample so that it's more clear for you so the evaluate model function calculates the loss over the training
29:18
and the validation set and we ensure that the mod model is in evaluation mode uh with gradient tracking and
29:25
Dropout disabled remember that when the model when we evaluating the model when we are printing the training and
29:30
validation loss we don't need to keep track of the gradient updates and we can even disable the Dropout because we are
29:37
just calculating the loss we are just doing the forward pass and then this generate and print sample right what we
29:42
are doing here is that it's a convenience function that we use to track whether the model improves during training because we will be able to see
29:50
what text is being generated the generate and print sample function takes a text snippet which is
29:55
called start context as an input and converts it into token IDs feeds it to the llm to generate a text sample at
30:04
that particular point in the training and we are going to print the text sample after every Epoch right now let's
Pretraining the LLM on our dataset
30:11
see all of this in action by training a GPT model instance for 10 EPO using an
30:17
admw Optimizer now two things I would like to clarify here when I say we are
30:22
training a GPT model instance the model which we are defining is an instance of the GPT model class which we have
30:28
already defined before so this is the GPT model class what this class does is that it essentially performs all the
30:35
operations which we saw in the uh GPT model architecture schematic yeah all of
30:41
these operations so it will con it will create create or it will give us the llm output at the
30:47
end uh okay so that's what uh that's what this GPT model class
30:53
is actually doing great now what we are what I also want to show you is the model
30:59
configuration which we are using for this particular code so this is the model configuration we are using vocabulary size of
31:05
50257 a context length of 256 remember gpt2 smallest model originally used a
31:11
context size of 1024 but I'm showing 256 here because I want the code to run on
31:17
your machine in small amount of time and using small amount of resources you can change it to one24 and the code will not
31:23
change significantly the vector embedding Dimension which we we are using is 768
31:29
because the inputs will be projected into that much Dimension space the number of attention heads is 12 the
31:35
number of Transformers we are using is 12 and the dropout rate is 0.1 and the key query value query key value bias uh
31:44
term bias is false because when we initialize the weight matrices for query key and values we don't need the bias
31:50
term so this is the GPT configuration which I'm using and I thought it's important for you to know that when we
31:57
uh create an instance of the GPT model class using this configuration right so we create an instance of the GPT model
32:04
class and uh the second thing I want to mention is the optimizer so we are using
32:10
this Optimizer called adamw uh adamw is a variation of Adam which uses weight
32:15
Decay if you are not familiar with Adam or adamw that's totally fine just know
32:20
for right now that for all modern machine learning algorithms for classification regression Adam has now
32:26
become the go-to optimizer of choice for all of these algorithms because it works very well it avoids local Minima and it
32:33
leads to faster convergence as well Adam W is another version of Adam where we
32:39
specify the learning rate and we specify the weight Decay now these are the parameters which you can play around
32:44
with these are generally called hyper parameters because we need to tune them there are some other variables which we
32:51
are going to Define before we run the pre-training code we Define the number of epoch to be equal to 10 so here I
32:58
told you right we are going to go through the entire data set uh based on
33:04
what we set in the number of epoch so if we set the number of epo equal to 10 we are going to repeat this entire process
33:09
10 times which means we are going to print the generated sample after every one Epoch for 10
33:16
times okay now uh one more thing is that evaluation frequency and evaluation
33:21
iteration is five which means that after every five batches I'm going to print the training and the validation LW and
33:27
and the initial text which I have given is every effort moves you because the generate and the print sample requires
33:33
us to give an initial text then it will print out what the llm is predicting for this initial
33:39
text awesome so what I've done here is that I've also uh recorded the start time at at which I start running this
33:45
code and the end time because remember the number of parameters which we are using here are huge they are of the
33:51
order of more than 100 million parameters and I just want to record the time it takes so I'm running my code on
33:56
a MacBook Air right now I think it takes similar time on I5 or i7 um computers as
34:02
well as MacBook even the smallest or the earliest MacBook model should run this
34:09
code in in a short amount of time so now uh here you can see that I've already run the training process before looking
Analyzing pretraining results
34:16
at the output the first thing which I want to show you is training completed in 6.6 minutes and I continue to be
34:22
amazed by this because I ran a llm architecture code on on my laptop which
34:29
had this code was optimizing 160 million parameters and it was doing it 10 times
34:36
or 10 EPO and the compute power which my laptop had made it so that in 6.6 minutes this entire code was run and uh
34:44
that's pretty awesome you can run it on your own machine and then when you see this you'll feel a lot of satisfaction because to get to this point we needed
34:51
to understand so many things we needed to understand about the llm data set how
34:56
the data set is pre process the data pre-processing pipeline then the llm architecture itself multi-ad attention
35:03
Dropout layers M multi-ad attention causal attention then we needed to understand how to define the loss
35:08
function after all of this effort we have reached the stage where we are able to train our own llm from scratch so
35:15
let's look at the training and validation losses which have been which are being printed after every five batches so here you can see that the
35:22
training loss if I if I see towards the end the training loss started from 9.78 one
35:28
and you will see that the training loss has decreased to 39 what awesome so as
35:33
we can see the training loss improves drastically which means it has started with
35:38
9.58 and it has reduced to a very small value that's actually awesome right in our case actually the training law
35:45
started from 9.78 1 and it reduced to 391 so let me change it uh the training
35:51
law started from 9 781 and it it reduced to 391 when I
35:59
had run it earlier these were the values I had obtained but for now the values are even better awesome let's look at
36:05
the validation loss the validation loss as you see started from 9.93 3 and it did not decrease that much it stay it
36:13
stagnated at around 6.4 6.3 6.2 this is a classic sign of overfitting we'll come
36:19
to that in a bit but let's look at what the llm has predicted and does it make sense so the remember we are printing
36:27
the generator text after every Epoch so after the first Epoch the next so we are
36:32
printing out 50 tokens and so the llm is printing out comma comma comma comma comma it has not understood anything
36:39
after the second EPO the LM is printing U comma and and and and and still not understanding anything after the third
36:45
Epoch it printed and I had been after the four fourth Epoch it printed you know the I had the donkey and I had the
36:53
then let's see after Epoch number seven it printed every effort moves you know was one of the picture for nothing I
36:59
told Mrs now you see that it started to use some of the words from our text and
37:05
after Epoch number nine you see every effort moves you question mark yes quite insensible to the irony she wanted him a
37:12
Vindicated and by me now here if you if you go to the training data set and
37:18
search irony you'll see yes quite insensible to the irony she wanted him indicated and by me so here you see the
37:25
llm is predicting something which does does make sense but it is directly recycling text from the data another
37:31
classic sign of overfitting so the final text which we obtain at the end of 10 epoxes every
37:37
effort moves that was our input youo was one of the xmc laid down across the SE
37:43
and silver of an exquisitely appointed luncheon table I had run over from Monte Carlo and Mrs J this is the output which
37:51
has been printed and uh you'll see that the language skills of the llm have improved
37:57
during this training first it started with comma comma comma comma comma and then you'll see that this is the output
38:03
which it is predicting now so the language skills have improved a lot in the beginning the model is only able to
38:09
append commas uh at the end of the training it can generate grammatically correct text like was one of the exams
38:15
Etc that itself is a huge win for us because it's a positive sign the llm is not generating something completely
38:22
random it had so many options to generate completely random things right but due to training process it is at
38:28
least generating Words which makes sense okay so similar to the training set loss the validation loss starts high
38:36
and decreases during the training however it never becomes as small as the training loss and it stagnates at 6.37
38:42
to after the 10th EPO what we can do is that we can even create a plot which shows the training and the validation
38:48
loss so you'll see that the training loss continuously goes on decreasing as shown by the Blue Line the validation
38:54
loss on the other hand you'll see that the validation loss decreases and then remain stagnant so here you see we are
39:00
also tracking the number of tokens and as I told you each Epoch is going through the data set once the data set
39:07
has around 5,000 tokens so when we do 10 EPO we should roughly see 10,000 tokens which is what we are seeing right now so
39:14
as the number of tokens seen increases you'll see that the training loss decreases a lot but the validation loss
39:19
stagnates so both of the training and validation loss improve after the first Epoch however the losses start to
LLM Overfitting
39:26
diverge past the second Depo see over here after the second Depot the losses have started to diverge this Divergence
39:32
and the fact that the validation loss is much larger than the training loss indicate that the model is actually
39:38
overfitting to the training data and we can confirm that the model memorizes the training data because
39:45
quite insensible to the irony remember I showed you this after uh the epoch
39:50
number n quite insensible to the irony and this is exactly taken from this data
39:56
set so it's memorizing it basically so it's memorizing what is already present in the data set and memorization is a
40:02
classic sign of overfitting so this memorization is expected since we are working with a very very small training
40:09
data set and training the model for multiple epochs okay and uh usually it's common
40:17
to train the model on a much much larger data set for only one Epoch so what
40:22
other if actually in real life practice what's done is that the data set set which is used is huge we are using a
40:29
data set which is quite small this data set only has 5,000 tokens and 20,000 characters usually people train such a
40:37
model which we have developed this large language model on extremely large number of data set which consist of millions of
40:42
tokens and at that time the model does not overfit too much because the data itself has so much variability our in
40:48
our case the model is overfitting on the data because the data itself is small so it it gets away by memorizing pieces of
40:55
data I actually encourage you to uh vary various or change various parameters
Next steps
41:02
here such as learning rate weight Decay you can change the number of epoch you can even change the training and the
41:07
validation loss percentage uh change the maximum number of tokens although this might not lead to too many changes if
41:13
you want to see changes in the code I encourage you to change these hyper parameters like learning learning rate
41:19
weight DK number of epo Etc and convince yourself that our model might be overfitting but at least we have tried
41:25
to reduce the loss function as much as possible and we have set up this Loop where the loss can be minimized and our
41:31
large language model is learning that's pretty awesome and we have reached this stage completely from scratch we have
41:36
not used any Library such as Lang chain Etc okay so this is where we are at
41:42
right now and what we'll do in next lecture is that we'll make sure that the model does not overfit too much and
41:49
these are called as decoding strategies so we'll make sure that the randomness is controlled so that in the models
41:55
prediction we will uh make sure that the model is predicting new words and not
42:00
just memorizing the text and there in come strategies such as temperature
42:06
scaling uh Etc and I'll explain all of those to you in the next lecture which will also be a very interesting lecture
42:12
like this one okay so that brings us to the end of today's lecture I think today's lecture was a very very
42:18
important lecture for us in this course because we actually trained a large
42:24
language model we got its lost to minimize as as much as possible we reached into some errors towards the end
42:31
like overfitting but that is good I would say because now we are at a stage where we can reduce overfitting and
42:36
improve the performance of the model and uh it took us several lectures
42:42
to reach this stage but I hope you are following along and you liking these lectures because I don't think anywhere
42:47
else these lectures are covered in this much depth and in this much detail my aim is to always show you these
42:53
explanations on a whiteboard and then also take you to the code so that along with the Whiteboard
42:59
explanations you can also do the coding on your own so thank you so much everyone I look forward to seeing you in
43:05
the next lecture where we will be covering decoding strategies to make sure that the llm output is more
43:11
coherent and more robust thanks so much and I'll see you in the next lecture

***
