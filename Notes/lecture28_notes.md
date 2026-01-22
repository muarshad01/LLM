#### 
1. Text Generation
2. Text Evaluation
3. Training & Validation Losses
4. LLM Training Function

***

* 5:00

***

* 10:00

***

* 15:00

* __Main step__: Find loss gradients using loss backward

* Input --> GPT model --> Logits --> Softmax --> Cross Entropy Loss (Output, Target)

* 161M parameters
  
***

* 20:00

***

* 25:00
* ADAM

***

* 30:00

***


30:11
see all of this in action by training a GPT model instance for 10 EPO using an
30:17
admw Optimizer now two things I would like to clarify here when I say we are
30:22
training a GPT model instance the model which we are defining is an instance of the GPT model class which we have


***


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

