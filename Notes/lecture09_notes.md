#### Create Input-Target Pairs
* Creating input-target pairs essentially Or input-output pairs 
* Auto Regressive (AR) model also called self-supervised learning or unsupervised learning

#### Creating Input-Target Pairs
* DataLoader fetches the input-target pairs using a sliding-window approach. 


#### 2.6 Data sampling with a sliding window

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
```

```python
enc_sample = enc_text[50:]
```

```python
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)
```

```python
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
```

***

#### Implementing a Data Loader

* PyTorch works with Tensors
* https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
* We'll implement a data loader that fetches input-out target paris using sliding-window approach.

* To impliment efficient data dataloaders, we collect inputs in a tensor x, where each row represents one input context. The second tensor y contains the corresponding prediction targets (next words), which are created by shifting the input by one posotion. 

* In the case of LLM one input-output pair corresponds to the number of prediction tasks as set by the context size that is very important.

***

40:06
from the data set which we defined earlier awesome so let's see the arguments which this function takes it
40:13
of course takes the text file which is the data set which we have then it takes the batch size which is how many batches
40:21
how many CPU processes we want to run parallell if you don't specify anything this will be by default four so if the
40:28
number of threads on your CPU are four or eight you can run those many processes parallely max length is basically equal
40:36
to the context length so we here I showed you a context length of four uh but when gpt2 or gpt3 those high level
40:44
llm models are designed they usually Implement a context length of 256 which
40:49
means they are so strong that the model can look at 256 words and predict the next word during
40:55
training then stride is one 28 so stride as I mentioned is when we create input output batches how much we need to skip
41:02
before we create the next batch a number of workers is also the number of CPU threads which we which we can run
41:11
simultaneously awesome so the first thing which we do is Define the tokenizer and here we are using the tick
41:16
token which is the bite pair encoder used by GPT and then we create the data
41:21
set so here see we are creating an instance of this GPT data set V1 class which we defined over here uh and here
41:29
we provide the input text the tokenizer is this tick token which is the bite pair encoding the max length is 256 and
41:36
the stride is 128 awesome so an instance of the GPT data set one is created and
41:41
we are calling it data set this data set is then feeded or loaded into the data
41:46
loader look at this data loader method it's it's it takes the data set as an
41:52
attribute basically what is happening in this step is that this data loader will just check this get item method in this
41:59
in this class and then it will return the input output pairs basically based on what is mentioned in the get item
42:07
that is exactly how it's going to work and then we just return the data loader which are the input output pairs that's
42:13
it which is happening over here so uh essentially in this part what we did was
42:19
we implemented a a data set which is a method or rather I would say it's a
42:25
library python and then what we are going to do is that we also implemented the data loader so the data set was
42:32
initially implemented and we created a class called GPT data set V1 an instance
42:38
of this class was created and then it was fed to the data loader method what this method essentially did was it
42:45
accessed the get item and then essentially what it's going to do it's just going to create those input output
42:50
tensors which we have defined in the GPT data set V1 class why did we do this
42:55
data loader between because it will really help us to do parallel processing and it can also uh analyze multiple
43:03
batches at one time so here I want to explain the difference between batch size and number of workers so there is a
Batch size and parallel computing
43:09
difference between batch size and the number of workers batch size is basically the number of uh batches the
43:17
model processes at once before updating its parameters so to make sure that the
43:22
model updates its parameters quickly the data is usually chunked into batches so that after analyzing four batches in
43:29
this case the model will update its parameters rather than going through the entire data set num workers is different
43:35
it is basically for parallel processing on different threads of your CPU uh and data loader enables us to do
43:42
all this if we did not do this then defining the batch size num workers would be very
43:48
challenging now what we are going to do we are going to test the data loader with a batch size of one and a context
43:54
size of four this will develop an intu of how the data set V1 class and the
43:59
create data loader function work together so if you found the previous part a bit challenging to understand
44:06
please try to focus on this part where we are going to show the Hands-On implementation of the data set class and
44:12
the create data loader function so first what we do here we just read the
44:18
text uh and then we are going to create a data loader and convert the data
44:23
loader into python iterator to fetch the next entry in the data let me show you what this means so here
44:30
what we are doing is we are going to create the data loader but with a batch size of one and a context size of
44:36
four then what we are going to do we are going to iterate through the uh data loader and we are going to print the
44:43
first batch and I I want to show you what this batch looks like so I'm just going to print this uh and here you see
44:49
this is the printed answer so what the first batch basically gave me is the input tensor and the output tensor
44:56
that's it so this is the input tensor and when you shift it by one you get the output tensor that's it essentially we
45:03
did all this to get this input output pair which we had already seen before so if you look at the input output pair
45:09
it's very similar to um the input output pair which we had looked at the
45:16
beginning of this lecture so if you remember at the beginning of this lecture we had looked at this input
45:22
output pair 1 2 3 4 uh and then 2 3 4 5 right what we have essentially obtained
45:28
right now is exactly similar but through the data loader so it's much more structured and here we can specify a lot
45:35
more parameters like batch size maximum length stride so here the stride is
What is stride?
45:40
equal to one we uh we can even show you the effect of stride so let me actually
45:46
show you the figure to explain uh what actually changing the stride really looks like so um here is
45:55
the figure for explaining The Stride in some detail
46:01
yeah okay so if you look at this
46:09
figure okay so let's look at this figure the first part of this figure shows the
46:14
input stride of uh one and the second part of this figure shows the input
46:19
stride of four so if you look at the first figure the first input of batch
46:24
the input of batch one is in the heart of and the input of batch two is the heart of the so see there is just a
46:31
stride of one year between batches between inputs of different batches but
46:37
if you look at the second figure the input of batch one is in the heart of which is this and then you stride by
46:43
four 1 2 3 and four so the next input will be the city stood the
46:49
see so typically a more stride means that you you move over the data in a
46:55
faster Manner and so less computations will be there that's what you specify
47:00
When you mention the stride so here stride is equal to one which which means that we'll be doing something very
47:06
similar to uh what is mentioned in this uh example which I'm highlighting with a
47:13
star right now awesome so here is how we create the input and output
47:19
batches so what I want to mention here is that the first batch variable contains two tensors the first tensor
47:25
stores the input token ID these and the second St tenser stores the target token
47:30
IDs which are these and since the max length is set to four each of the two tensors contain
47:37
four token IDs that is also very very important to remember note that an input
47:43
size of four is relatively small uh or the context size it is common to train
47:48
llms with input sizes or context sizes of at least 256 now you can also see the second
47:54
batch see the second batch is 367 2885 and this is the output of the
48:00
second batch so the second batch input is this the second batch output is this so if you compare the First with
48:07
the second batch we can see that the second batch token IDs are just shifted with one position see this is the first
48:13
batch input token IDs this is the second batch input token IDs now can you think what would be the stride in this case
48:19
based on the explanation which I provided to you the stride will be equal to one
48:26
because because if you look at the first input batch and the second input batch it just shifted by
48:33
one um so the stride setting dictates the number of position the input shift
48:41
across batches emulating a sliding window approach that's why it's called as the sliding window approach batch
48:47
sizes of one such as we have sampled from the data loader so far are used for illustration purposes if you have
48:54
previous experience with deep learning you may know that small batch sizes require less memory but lead to more
49:00
noisy model updates just like in regular deep learning the batch size is a
49:05
trade-off and hyperparameter to experiment when training llms so as I mentioned the batch size is number of
49:12
data the model has to process before updating its parameters so if the batch size is very small the parameter updates
49:18
will be very quick but the updates will be noisy if the batch size is very large the model will be going through the
49:24
entire data set before or the model will be going through the large batch before making the update so
49:31
the update will not be as noisy but it will take a lot of time so you need to make sure this hyper parameter is set
49:38
correctly before we move on uh and end this lecture the last thing which I want to show you is the effect of batch
Effect of larger batch size
49:47
size uh so let's actually take a brief look at that essentially uh what happens
49:54
when the batch size is more than one so the batch size so here you see I'm creating a data loader with a batch size

***


50:00
equal to 8 and then I'm running this so see the input tensor and the output tensor and here I'm also incre
50:07
increasing the stride to four so if you look at the first input and the second input you will see that there is not an
50:14
overlap because the side stride is equal to four and here the batch size is equal to
50:19
eight so see the input has essentially eight uh input tensor has eight inputs
50:25
which means the batch size equal to 8 and the output tensor will also have eight rows because the batch size is
50:31
equal to eight so the model will essentially process this batch before making the parameter
50:38
updates uh okay so now here you can see as I mentioned we have increased the
50:44
stride to four this is to utilize the data set fully um we don't skip a single word but
50:51
also avoid overlap between batches since more overlap could lead to increased overfitting so that is one advantage of
50:58
having a more stride so here you see in these two examples if you look at the second if you look at the second example
51:04
over here the stride is equal to four so there is no overlap between the input batch one and the input batch two which
51:11
might be good for overfitting per uh which might be good to prevent overfitting whereas if you look look
51:16
over here the stride is just equal to one so the input batch one and two have a lot
51:22
of overlap that might lead to overfitting which is not great for
Lecture recap
51:28
training this actually brings us to the end of today's lecture where we covered number of things first we covered what
51:35
does it mean to uh break down the text into input Target pairs input Target
51:42
pairs are definitely needed for training of llms and we started with this example
51:48
uh this example right here on the screen let me just take you to that yeah
51:54
so we started with yeah so we started with this example to
52:01
illustratively explain what the input Target pair looks like so the target pair is just the input which is shifted
52:08
by one then we saw how to convert this into X and Y pairs so if the input is 1
52:15
2 3 4 the output is 2 3 4 5 here the context size is four which means that if
52:20
one is the input the output should be two if one and two is the input the output should be three if 1 2 3 is the
52:27
input the output should be four if 1 2 3 4 is the input the output should be five
52:33
so in one input Target pair there are four computations or four predictions because the context size is four the
52:40
rest of the lecture was devoted to creating input Target pairs like this but in a structured manner for that what
52:47
we actually did was we used something which is called as data loader and then we paired the data loader with the data
52:54
set so first we created this GPT data set V1 class uh we created a data set
53:00
and then we fed that data set into the data loader what this data loer essentially did is is just it created
53:06
these input output pairs but in a very structured manner why structured because we can Define many things we can Define
53:13
things like stride batch size number of workers
53:18
Etc uh so there are essentially three things which are very important for you to remember and let me take this example
53:25
by rubbing some things of the screen so the first thing to remember is the uh
53:31
stride so stride basically dictates here yeah so this this
53:38
schematic here shows the difference between stride so if you see the top top level schematic here um this shows a
53:46
stride of one which means that the input of batch one and batch two will just be
53:51
differing by one word so there is a lot of overlap which might lead to overfitting what people people usually
53:56
do is that they keep the stride length equal to the context length so that you don't even miss any word but there is
54:03
not an overlap between different input batches that's the meaning of stride the second thing is the uh batch size so
54:11
what the batch size lets you do is it tells the model how many batches of data
54:17
you want to process at once before making the parameter updates so towards the end of this lecture we showed an
54:23
example where the batch size was equal to 8 which means that all these eight input
54:28
output pairs will be processed before making the parameter updates and the last parameter which we did not explore
54:34
too much is actually number of workers what this means is that you can exploit the parallel Computing facilities in
54:42
your own computer because all of us have multiple threads on our computer so you
54:47
can make sure the Computing happens on Parallel threads by specifying number of
54:52
workers so what the data loader essentially does is that ultimately we can extract input output pairs through
54:58
the first batch second batch Etc in the data loader and then this now will be converted into Vector embeddings which
55:05
then we will feed into our model training the next section or the next
55:10
lecture is going to be a token embedding or vector embedding I know everyone that today's lecture was a bit more complex
55:17
especially because of the data set and data loader but this part is very rarely covered by anyone and I really wanted to
55:24
make sure I cover it in a lot of detail thank you so much everyone I hope you are enjoying and understanding these
55:29
lectures I'm deliberately making them a bit long so that everything is covered from scratch thanks everyone and I look
55:36
forward to seeing you in the next lecture
