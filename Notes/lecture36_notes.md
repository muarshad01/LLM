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

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

***

* 20:00

***

* 25:00

***

* 30:00

***

* 35:00


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









