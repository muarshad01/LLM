
* 20,000 characters
* 5,000 tokens

* [OpenAI - tiktoken](https://github.com/openai/tiktoken)


#### Divide the dataset into training and validation
```python
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
```

#### Use DataLoader to chunk training and validation data into input and output datasets

***

* 10:00

* stride = context_size

1. Input
2. LLM Output
3. Target

* Loss is computed b/2 LLM output and target

***

* 15:00

#### Training
* For Training -  (Input, Target) pairs give training loss
#### Validation
* For Validation - (Input, Target) paris give validation loss

***

* 15:00

***

* 20:00

***

* 25:00

***

* 30:00

```python
import os
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
```

***

* 35:00

* stride = context_size = 4

```python
from previous_chapters import create_dataloader_v1
# Alternatively:
# from llms_from_scratch.ch02 import create_dataloader_v1

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
```

***

* 40:00


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




***


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









