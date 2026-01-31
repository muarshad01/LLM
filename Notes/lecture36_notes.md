## Calculating the classification loss and the accuracy 

* [Welcome to the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
* Training batch = 8 samples
* 8 x 120
* (batch_size, num_tokens, num_classes)

***

* 5:00

```python
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())
```

***

* 10:00

***

* 15:00

* $$\text{Loss} = -\sum_{i}y_i\log(p_i)$$

* 
defined then what we'll do is that we'll simply take the partial derivative of the loss function with respect to all my
16:00
trainable weights we'll calculate the gradient with respect to the trainable weights
16:06
and then we'll just uh update so weight new is equal to weight old minus the
16:14
partial derivative of loss with respect to that weight so we'll use a variation of this simple gradient descent called
16:20
Adam or Adam W and so then we'll just continue updating these parameters until the loss function is minimized So
16:27
currently so let's say the loss function looks like this it of course won't be as simple as this but I'm taking a
16:33
simplified example initially we start out with this where the loss is not that
16:38
low and then we move down this loss function and hopefully we'll Reach This Global Minima where the loss is
16:45
minimized and once loss is minimized then we'll make sure that the accuracy is also higher automatically so then
16:52
comes the question of how do you define the loss function and what loss function to use if you have studied neural
16:57
networks and machine learning learning before we know that if we have uh if we
17:03
have targets um which are Pi or let's say if the targets are
17:10
Yi and if my predictions are Pi then the loss function which is used in this case
17:16
is the categorical cross entropy loss and is defined by negative of Sigma which is adding over all the class
17:22
labels and minus Yi into log of
17:27
Pi let me illustrate with a simple example here let's say if we have a text Data whose True Value is that it's not a
17:34
Spam which means that it's one hot encoding is one and zero so let's say this is not a Spam but our predicted
17:41
values our predicted values after or predicted values here are 08 and 02 then
17:48
the cross entropy loss is negative of we'll need to sum over all the classes Yi into log of Pi Yi is the true value
17:56
Yi is the true value and Pi is the predicted value
18:02
right so let's multiply so we'll multiply one which is the true value multiplied by log of8 so 1 will be
18:10
multiplied by log of8 and 0 will be multiplied by log of0 2 and we'll take the negative sign of this so 0 * log
18:18
point2 is 0 and then 1 * log point8 if you take the negative that's 2 2231 why
18:25
is this a good measure of loss because if our predicted value was 1 and zero which is exactly equal to true this
18:32
second will anyway be zero but the first but the first entry will be 1 into log 1
18:39
which will be equal to zero so if the predicted value equals to the True Value then our loss will be
18:45
zero which is exactly what we want so this negative of Y log Pi is a very good
18:51
loss function to be to calculate the loss in the case of this categorical
18:56
predictions in the in classification tasks uh so to give you just a
19:03
brief visual flavor negative of log negative of log of x looks something
19:08
like this so this is X and this is negative of log of x and we want X to be
19:14
as close to one as possible which is this probability of the correct class So eventually we'll start out from some
19:20
high loss and our goal is to make the loss as close to zero as possible
19:25
another advantage of the Cross entropy loss is that it's differentiable so it's very useful for us in the case of back
19:31
propagation right um okay so let's actually Define the cross entropy loss now along with this uh calculation of
19:39
accuracy loader what we are also going to do is that we are going to define a loss function which is the cross entropy
19:45
loss why can't we use just classification accuracy and take the inverse of that accuracy maybe to get
19:51
the loss it's because classification accuracy is not a differentiable function so we will use the cross
19:57
entropy loss as a proxy to maximize the accuracy this is the same as the cross entropy loss which we use to pre-train
20:03
the large language model so uh okay so what we are going to
20:08
do now is that let's say we get an input batch and a Target batch always when an input and Target batch is given your
20:15
visual mind should take you to this figure where we have an input batch and we have a Target batch so what what has
20:21
to be done here is that once you get the input batch you pass in through the model and then you only look at the
20:27
Logics of the last output token because that contains the most information and then you find the categorical cross
20:33
entropy loss between this logic sensor which is the output of the last token and the target batch so the logic sensor
20:42
output can be something like uh8 and 02 and the target out is one Z so when you
20:48
calculate the cross entropy loss you'll get some value of the loss function so I'll just show you here torch.nn do
20:56
functional cross entropy this is the P torch functionality which we are using over here to find the cross entropy
21:03
loss awesome and this is differentiable so it will be very useful for us when we do the back
21:09
propagation okay so we will use this calculate loss batch function to compute the loss for a single batch and we can
21:16
also use it to calculate the loss for a multiple set of batches so for to calculate the LW for multiple batches we
21:24
have to use similar code lines as we used over here so so if number of batches is not specified then we take
21:32
the number of batches to be equal to the length of the data loader if number of batches is specified then it's equal to
21:37
the minimum of the number of batches specified and what is the length of the data loader very similar to the accuracy
21:43
classification code which we saw then what we are going to do is that we are going to take one input batch one target
21:49
batch calculate the loss between all the samples of the input batch and the target batch using this calculate loss
21:54
batch which will implement the categorical cross entropy we are going to add the loss every time we get a loss
22:01
we are going to add the loss and then that is in the total loss um awesome and then what we are
22:08
ultimately going to do is that we'll divide the total loss with the number of batches so that will kind of give us an average loss per batch and this is the
22:16
loss which we will eventually try to minimize using back propagation that is the whole workflow which we are going to
22:22
follow so now what we can do is that we can implement this loss function on our data set again we have not implemented
22:29
back propagation so the loss will be very high but I just want to show you the initial values of the training loss
22:35
the validation loss and the test loss so here again I'm setting the number of batches equal to five U because actually
22:41
the train data loader has 130 batches I think so that will take a long time to calculate and anyway we have not done
22:48
the training here so I just want to illustrate that the loss can be found on five batches like this so you you
22:55
implement the Cal Closs loader function and you pass in train loader then validation loader and the test loader
23:02
and then you also pass in the number of batches so then you can print out the training loss you can print out the
23:07
validation loss and you can print out the test loss and you can see these are the high these are the values which are
23:12
pretty high it's again if you in the accuracy we saw that the accuracy was very bad and that is reflected in the
23:18
loss values as well now we will Implement a training function to fine tune the model which means that we'll
23:24
adjust the parameters to minimize the training loss and and then we'll also print out the validation loss and we'll
23:31
print out the test loss so let's start looking at that part of the code right now so until now we
23:38
have finished a number of steps here we have finished uh let's see we have
23:43
finished downloading the data set pre-processing the data set create data loaders initialize model load pre-train
23:49
weights modify model for fine tuning Implement evaluation utilities which is the loss and the accuracy
23:56
basically and now we we are at this stage where we will actually fine tune the model which means that we'll Define
24:02
the training Loop and we'll Implement back propagation so this is the training Loop
Fine-tuning training loop implementation
24:07
which we are going to Define first we'll have the EPO which means one Epoch is going through the entire data set once
24:14
right so let's say if you if you are running in one particular Epoch the
24:19
second Loop is that you have to go within each batch so each batch has eight samples at least that's how we
24:25
Define the training data loader to be so then we'll look at each particular sample and then we'll calculate the loss
24:32
on the current batch uh and uh we'll Implement a backward pass to calculate
24:38
the loss gradients and then we'll update the model weights using the loss gradient so here what we are doing is
24:43
that W new is equal to W old
24:49
minus Alpha * the partial derivatives this is exactly what we written over we
24:54
wrote over here also uh and then once the weights are
25:00
updated we print the training and the validation loss and then we keep on doing the same thing for multiple number
25:06
of epox so that the parameters are getting updated so the simplest way to think about this is that the most
25:12
important step is this backward pass once we do the backward pass we get the loss gradients that's why we needed the
25:18
loss function to be differentiable once we get the loss gradients with respect to the parameters we can actually update
25:24
the parameters and once we do this enough number of times the par parameters will get updated and
25:29
hopefully we'll reach a value of the loss where the loss function is minimized this is the exact same
25:36
training function which we had implemented to pre-train the llm and
25:41
here's what I'm what I want to show you is that when we finetune the model on supervised data which means data set
25:48
such as the spam no spam label I showed you we need to again train the model so there is training process involved in
25:55
pre-training and there is training process involved in fin tuning that's why it's called pre-training actually
26:00
because it's before this second training process which needs to be implemented so let's see how the
26:06
training process is implemented in code right now so this section I have named as finetuning the model on supervised
26:13
data so until now we have actually not trained the model on the data set at all
26:18
which means that that's why the parameters are not optimized so in this section we'll Define and use the
26:23
training function to fine tune the pre-trained llm and improve its spam classification accur
26:28
accuracy uh a note here is that if you have followed these lectures you'll see that the training function is very close
26:34
to the train model simple function which we used for pre-training earlier the only distinction is that we are tracking
26:40
the number of examples here the number of text samples instead of tracking the number of tokens which we had calculated
26:47
earlier so in the code what we are going to do is that there are seven steps the first step is that we have to set the
26:53
model to training mode uh so here you see we set the model to train training mode that's the first
26:59
step the second step is reset the loss gradients from previous batch so when we look at each every batch we have to
27:06
reset the loss gradients again so let's say we are looking at one batch right now uh we reset the loss gradients from
27:13
the previous batch iteration then the third step is calculating the loss gradients and updating model weights
27:19
these are the most important step so then what you do is you find the loss in that batch and then you calculate the
27:26
loss gradients through a backward propagation then you do Optimizer do step this is where the optimizer comes
27:31
into the picture in on the Whiteboard I showed you simple vanilla gradient desent over here but in practice we'll
27:38
use a a more complicated optimization algorithm which keeps track of the previous gradient which keeps track of
27:45
the previous gradient Square Etc so that the optimization is done in a in a
27:51
better Manner and so that the model does not get stuck in local Minima then the next step is that we
27:58
keeping track of the number of examples so we just keep track of the number of examples which we are seeing so input
28:04
batch. shape zero is that let's say if each batch has eight samples when you
28:09
look at the first shape uh first value of the batch shape it will give us the number of samples in the batch so for
28:16
example if the batch has eight uh eight samples and the number of tokens is
28:22
120 so then we'll get eight here input badge. shape zero which will give us the number of samples over here
28:30
so then we keep track of the number of example scen so you can just think of this example scene as when you look at
28:35
one text message that's one example seen when you look at second text message you increment the number of example seen by
28:41
one whenever you go through a full batch you increase the global step by one
28:46
right awesome now here we have that if Global step percentage of evaluation
28:53
frequency equal to zero so we have to specify an evaluation frequency now if the training batch has 130 if the
29:00
training uh data loader has 130 batches in training and if the evaluation frequency
29:08
is 50 it means that for after 50 batches
29:13
are processed after 50 batches are
29:19
processed for each Epoch after 50 batches are processed in each Epoch we
29:25
print and what are we going to print we are going to print the training loss and we are going to print the validation
29:32
loss so this evaluation frequency just specifies how after how many batches are
29:38
completed we print the training and the validation loss so here later we are
29:44
going to set the evaluation frequency to 50 which means that after 50 batches are processed in each Epoch we are going to
29:50
print so in every Epoch we are going to print on an average of two times because 130 divided 50 is around 2.6 so we are
29:57
going to print 2 two times in every Epoch okay awesome so now to print the
30:03
training loss and the validation loss we are going to calculate the evaluate model so evaluate model gives you an
30:09
option to specify the evaluation iteration which means that the number of batches you want to use for evaluation
30:16
sometimes if you want to show quick evaluation on a sample data set you don't want to use all the batches so
30:21
here you can just set the number of evaluation iterations to be five or 10 since the number of batches is 130 this
30:28
this will really save us time when we print out the train loss and the validation loss so this actually
30:33
evaluation step is optional but when we do the training you'll see that the train loss and validation loss are
30:38
printed after every 50 batches due to this evaluation step then what we are going to do is
30:44
that after every Epoch we are going to calculate the training accuracy and validation accuracy and we are going to
30:50
print it out so after every Epoch what we are going to do is that we are going to print the training and the validation
30:55
accuracy and after every batches we are going to print the training loss and the validation loss so let's do the training
31:03
process now for me this training process took uh around 8.8 minutes and I have a
31:08
MacBook Air 2020 um it does not have very high end configurations but it's a good laptop if you have an i5 or i7
31:16
laptop or a Macbook this training should take only 7 to 10 minutes for you so here you can see that this is the main
31:23
code where we write about the training so we are going to use adamw optimal
31:28
izer let me show you a bit about this t. optim adamw it's a modification of the
31:33
Adam Optimizer with weight DEC so it's very good to avoid local Minima this
31:40
algorithm converges in a smooth Manner and it also leads to faster convergence you can try various things here you can
31:46
try Adam you can try to change the learning rate weight Decay so this is why this kind of code opens the door for
31:53
research if you just use chat GPT you will never get to change all of these things which are happening
31:59
under the hood but once I share this code with you you can try playing around with various parameters and try seeing
32:05
the effect on the loss function on the accuracy Etc so this is the optimizer which we
32:11
have defined right now and then what we are going to do is that we're going to call this train classifier simple so I'm
32:18
calling this train classifier simple function and I have to I have to pass the model so the model which I'm passing
32:24
in is the GPT model class which we have created with the modified architecture
32:31
so the modified architecture is this where the architecture has a classification head on top of
32:37
it let me show you yeah this is the modified architecture which has this classification head on top of it this is
32:44
the model which we are passing in and then we pass the train loader the
32:50
validation loader the optimizer which is the admw uh number of epo evaluation
32:55
frequency so this evaluation frequency as I mentioned here is after 50 batches we print the train loss and validation
33:01
loss and evaluation iteration is basically when you print this train loss and validation loss how many batches you
33:08
want to evaluate so I'm just doing five batches here so that the calculations would be quick if you do evaluation
33:14
iteration equal to 50 batches or 100 batches it will just take more time to do the evaluation of course this is not
33:20
the best way to evaluate evaluate because we are only evaluating on five later in I have a code where we actually
33:27
evalate on the entire data set for now this gives us a good sense at every iteration how the training loss and
33:33
validation loss is progressing awesome so after I run this code you can see that I've already run it and it's 8.83
Analysing training results
33:40
minutes so if you look at the training loss the training loss goes down to 0.083 and the validation loss goes down
33:47
to 0.074 training accuracy improves to around 100% And validation accuracy is
33:55
97.5% you can even print the training loss and validation loss and
34:00
along with it you can also print the example scene because then you can see the more examples the model sees the
34:07
more text messages you can see that the training loss goes down as indicated by the blue line and the validation loss
34:14
also goes down as indicated by the Orange Line This is actually perfect training because training loss is very
34:19
low validation loss is also very low that's awesome that indicates that there is not too much overfitting here so as
34:26
we can see based on the sharp downward slope the model is learning well from the training data and there is little to
34:32
no indication of overfitting that is there is no noticeable gap between the training and the validation set
34:38
losses that is exactly what we wanted if the validation loss is much higher than training loss let's say if the
34:44
validation loss is somewhere here that is a sign of overfitting Now using the same plot we
34:50
can also plot the classification accuracies so as the loss is decreasing the training and the validation loss you
34:55
can also see that the training accuracy has shown by the blue line is increasing and then it reaches one the validation
35:01
accuracy also increases and it reaches around 97 and plate uh one thing to note is that it's
35:08
important to note that we have set evaluation iteration to be equal to five as I mentioned over here we have set the
35:14
evaluation iteration to be equal to five so that's not so the values which we are seeing here of the accuracy are not
35:21
representative of the accuracy on the entire data set since we only evaluate on five batches so this this means that
35:28
our training and validation performance were based on only five batches for efficiency during training to calculate
35:34
the performance matrics for the training validation and entire testing set for the full data
35:40
set uh we can also do that so all we need to do is that then we have to run the calculation accuracy loader and then
35:47
we have to pass in the train loader we have to pass in the model and we have to pass in the device either it's a CPU or
35:53
a GPU so what this calculation accuracy loader will do as we have already defined earlier uh this calcul calculate
36:01
uh this calculate accuracy loader will take in our model and then it will do the prediction it will compare it with
36:08
the actual value then it will print out the accuracy and it will do this for all the batches in the training set so it's
36:14
not only five batches so this um this accuracy measure for the training
36:20
testing and validation data set is a much better representative than these plots because these plots are only for
36:26
evaluation iteration which was set to be equal to five so let's print out these train
36:33
accuracy validation accuracy and test accuracy on the entire data set so when you print out these you will see that
36:38
the training accuracy is 97% the validation accuracy is also 97%
36:44
and the test accuracy is 95% so the training and the test set performances are almost identical a
36:51
slight discrepancy between the training and the test set accuracy so the test set accuracy slightly less right
36:57
compared to the training it suggests that there is small amount of overfitting although there's small only
37:02
2% difference is there but it still indicates that slight amount of overfitting is there on the training data typically the validation set
37:10
accuracy is somewhat higher than the test set accuracy because the model development often involves fine-tuning
37:15
parameters on the validation set this situation is common but the Gap
37:21
could potentially be minimized by adjusting the model settings such as increasing the dropout rate or the
37:26
weight Decap parameter in the optimizer configuration as I mentioned before once I share this notebook with you you will
37:32
have a lot of scope to experiment so you can experiment with dropout rate in the model architecture you can even
37:39
experiment with learning rate parameter weight DK parameter in the optimizer um you can also experiment
37:46
with things like unfreezing certain parameters so if if you remember from our previous lecture the only parameters
37:52
which are being trained here is of course the output classification head and along with that we are also training
37:57
the last Transformer block the 12th Transformer block and the final normalization layer you can do some
38:04
changes here so you can make sure that the last three Transformer blocks are trained Etc you can make sure that maybe
38:11
this is false and that leads to better answers who knows so this kind of experimentation is open and I'll be very
38:19
happy if you experiment with various options that will even improve your understanding further and try to see if
38:25
you can increase the test accur further to match that of the training
38:31
accuracy awesome so until now what we have done is that we have uh um let's see what all we have done we
Testing model on new data
38:39
have fine tuned on the supervised data and we have even plotted the training
38:44
and the validation loss now the last step is remaining which is using model on new data so whatever is shown in the
38:51
tick mark here downloading the data set pre-processing the data set creating data loaders initializing the model load
38:57
pre-train weights modify model for fine tuning Implement loss and accuracy functions then actually doing the backo
39:05
pass and fine tuning the model and training and validating the model these nine steps we have done now what we have
39:11
to do is that we have to use the model on new data which the model has not seen before so that is the real test whether
39:17
our model our large language model how its performance is as a Spam classifier so let's go to the last
39:24
section of this project right now and uh let's see whether our model is actually performing well on data which it has not
39:31
seen so after fine-tuning and evaluating the model in the previous sections we are now in the final stage of this
39:37
chapter where we will use the model to classify spam messages right so finally
39:43
let's use the fine tuned GPT based spam spam classification model we'll need to
39:48
define a function first we'll need to define a function called classify review which will take in any text and it will
39:54
predict whether it's a Spam or not and what this function will do is that it will do a number of things first it will
40:00
uh and let me actually write this down in description so let's say a text is given such as
40:07
you let's say a text is given such as you on a lottery right if a text is
40:13
given the first thing which we will do is that we'll convert this text into token IDs we'll convert this text into token
40:20
IDs actually there is a nice representation of the data pre-processing which we had looked at before I'm just I'll just take you to
40:26
that part so that you can see how this yeah so if a new text is given we'll
40:31
first convert the text into token IDs something like this and that's the first thing which we have written in the code
40:37
we'll first use tokenizer in code so this is the tick token this is the tick token tokenizer
40:44
which we are going to use it's a bite pair encoder it takes in any sentence and converts it into a bunch of tokens
40:50
right then we will uh we'll look at the supported context length and that's equal to uh 1024 in this this case
40:58
because the uh so model. positional embedding weight shape that is a shape
41:04
of the embedding weight Matrix and to give you an idea of what the shape size is it has the number of rows equal to
41:11
the context length and it has number of columns equal to the embedding Dimension so the number of rows will
41:18
give us the context length and that's why we are using the embedding shape zero to find the context
41:24
length so the reason we find this context length is that we are going to compare it with the maximum length so
41:30
what we did here is that we have we have found the maximum token length token ID length from the training set which means
41:36
which is the text message which is the longest and we have got that length let's say that length is equal to 120 so
41:43
uh if that length is equal to if that length is actually higher than the context length then we have to trunet
41:51
everything down to the context length so sequences which are way higher than the
41:56
maximum length we have to find the minimum of the maximum length and the supported context length so if the
42:01
maximum length is actually higher than 1024 then we are going to take the
42:06
context length and truncate all the sequences to be equal to the context length in the cases where this does not
42:13
happen our maximum length will be used and then all the input text will will
42:18
have those many token IDs so let's say if the uh maximum length is 120 and you
42:24
have received a text message such as uh you have won a
42:31
lottery let's say you have received this text message and when you convert it into token IDs you you have seen that
42:37
the length is only 50 so what you will do is that you have to extend this to 120 by adding some end of text
42:45
tokens so you add 70 end of text tokens here which are this 50256 and you make sure that the length
42:52
of the uh text is equal to the maximum length this is very important because when you batch it every sentence needs
42:59
to have the same number of token IDs so you have to pad this you have to pad every input sequence to match the
43:06
maximum length so the maximum length ideally is the length which we have got
43:11
from our training data set so what's the maximum email length in the training data set but if it's higher than the
43:16
context length the maximum length will be set equal to the context length so whenever you give a new test input it's
43:22
first converted into token IDs and then it's padded with this end of text token which is 50256 so that the length is
43:29
equal to the uh length is equal to the maximum length then we convert it into a
43:35
tensor to add the batch Dimension uh and then we perform the model inference so we first calculate the prediction so we
43:42
get the logit tensor which is the logits of the last output token and then we apply torch. arac so we have seen this
43:50
implementation um let me recap your understanding we have seen this implementation in this part of the code
43:56
right where we take the AR Max and this gives us the prediction whether it's spam or not a Spam and then that is our
44:02
final answer so this model this model here is our train model which we are using now for inference for inference on
44:09
any new text message so the main magic happens in this line where our input
44:15
tensor is passed through this model and then we predict the label but before that we have to make sure that the token
44:21
IDs are equal to the maximum length now what now let us actually take two sentences and let us pass them through
44:28
our classify review function and let's see whether our model predicts them as as spam or no spam so the first sentence
44:35
I'm taking is you are a winner you have been specially selected to receive ,000 cash or $2,000 reward clearly it looks
44:42
like a Spam rate and this is from a testing set my model has not seen it in the training data I'm going to pass it
44:48
through the classifier review function and let me print out the output and our model is clearly recognizing the output
44:54
to be that this is a spam then let's take a second sentence hey just wanted to check if we are still on for dinner
45:01
tonight let me know I'll again pass it through the model and I'll check whether it's spam or no spam this looks like a
45:07
very legitimate message right and it's clearly not a Spam and model makes a correct prediction that it's not a
45:13
Spam so this seems that our model is doing an amazing job it's actually recognizing spam as spam and not a Spam
45:21
as not a Spam when I share this code with you I actually encourage you to play around with several different text
Next steps for exploration and research
45:27
messages and check how the large language based model is doing but this is an awesome example which we have
45:33
finished I never thought an llm could be used for classification task but this kind of an architecture when I saw
45:40
attaching a classification head on top of the GPT architecture it really blew my mind it's awesome and it really works
45:46
we have brought down the loss we have increased the accuracy and we have tested this model on new text samples
45:52
and it seems to be performing well um this is pretty awesome right and through this I hope you also understood the
45:59
concept of fine tuning remember we have used pre-trained weights from gpt2 but
46:04
we needed to do the training procedure once more so that is one disadvantage
46:09
you might say of fine tuning that you need to spend more time on doing additional training on specific data set
46:15
what is the specific data set which we are using here it's the spam collection uh but this additional tuning
46:22
also gives us an advantage that now our model is specifically working very well to this data set and it can act as a
46:28
Spam classifier we can even go ahead and save the model in case we want to reuse the
46:34
model later and please keep this trick in mind because if you do not save the model you'll need to train it again so
46:41
just tor. save it's an awesome functionality implemented by toor P torch and I I'll share the link to this
46:48
also tor. save allows you to save the model parameter so that you can just use them
46:53
later uh and then you can load the same model parameters using tor. load and you
46:59
specify the path where you saved the model parameters and then you can directly use the loaded parameters to do
47:04
inference or to do further fine tuning Etc that will save a lot of time and effort for you this brings us to the end
47:11
of this lecture where we have successfully implemented uh llm spam classifier project and uh this project
47:19
showed you how to combine fine tuning with pre-training on a very specific
47:25
data set I I hope you understood why it is called pre-training and fine tuning and why we need fine tuning if we did
47:32
not do fine tuning Our model was having a very bad prediction so if you see above we had a special section where we
47:39
had displayed the model prediction yeah so if we did not fine tune and if you give something in the prompt itself like
47:45
is the following text spam answer with a yes or no the model could not answer correctly that's why you need to fine
47:52
tune you need to change the GPT model architecture so that the model starts answering better and its accuracy is
47:58
improved the same thing what you learned right now the same code can be applied to a wide range of classification tasks
48:04
with different range of different data sets and I encourage you to explore with different data sets that will not only
48:10
improve your understanding but it will make you much more confident as an llm engineer now I have taught you the nuts
48:16
and bolts of how to do fine tuning so you should not be scared of when people say the word fine tuning it just
48:22
changing the model parameters training it again on specific data so that it performs well on that data set in the
48:30
next set of lectures we are going to look at instruction fine tuning so until now we have looked at classification
48:36
fine tuning right which is just one one category of fine tuning but another major category is instruction fine
48:42
tuning so we'll actually be building our own chat bot which can answer specific which can answer or reply to Specific
48:49
Instructions so we'll cover that in the subsequent set of lectures thanks everyone I'm I hope you are enjoying
48:54
this whiteboard approach Plus this coding approach as you are following please keep a track of the notes please
49:01
make your own notes and run your own code ask questions uh discuss with each other so that your understanding is
49:07
improved maybe change the data set instead of spam collection maybe use a heart disease data set and run the same
49:13
code who knows you'll develop an awesome model this opens a lot of research opportunities not only with respect to
49:19
llm architecture changing and testing various llm architecture but also with respect to applying this architecture on
49:26
various CL classification projects thanks so much everyone I look forward to seeing you in the next lecture

***





