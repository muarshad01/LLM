
* [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* (Input, Target) pairs

***

* 5:00

* [tiktoken - OpenAI](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

***

* 10:00

* `<|endoftext|>`  is token ID 50256 for GPT-2

***

* 20:00

***

* 25:00


that it just has eight rows over here and it can either have zeros or ones so that's why the size here is 8 and this
26:07
what I'm showing here is just one batch remember that there are 747 uh examples corresponding to spam
26:16
and 747 examples corresponding to no spam so if you add these both together
26:23
you'll get that the total number of data which we have is 1494 right out of that
26:28
the training data is 70% right so we will have batches corresponding to that so 70% of 1494
26:36
let's see so 7 into
26:41
1494 uh that's 1045 and I think we have printed this above
26:47
uh yeah 1045 so the training data set overall has 1045 samples right so the
26:55
training data set overall has uh training data set overall has
27:01
1045 samples in the training data set and now if each batch has eight such samples
27:10
the batch size is eight right which means each batch will have eight samples if each batch has eight samples eight
27:17
samples in each batch how many batches do you think will be in the training data
27:24
set so then the number of batches will be
27:30
number of batches will be 1045 divided by 8 which is approximately 130
27:36
batches and we can actually test this what I've done at the end of this code is that I have printed the length of the
27:42
training loader which will give me the number of training batches and here we can see that we have 130 training batches and that exactly fits our
27:49
intuition which you have written on the Whiteboard similarly what you can do is that you can print out the length of the
27:56
validation loader the length of the test loader as well and you'll see that there are 19 validation batches and 38 test
28:02
batches remember that we have 20% of the data as test and 10% of the data as
28:08
validation that's why the number of test batches are exactly two times that of the number of validation
28:14
batches and why does the length of the training loader give you the number of batches because if you print out one
28:21
batch that has eight eight samples right so actually the training loader this is
28:27
just one batch so the the training loader Dimensions will be so one batch of the training loader
28:33
has the dimension of uh 88 rows and 120 column right and there are 130 such
28:39
batches so if you print out the dimension of the entire training loader it will be 130 by 8X 120 it will be a
28:46
threedimensional tensor similarly for the testing loader
28:51
the dimensions would be uh 38 38 8 120
28:58
and for the validation the dimensions would be 19820 the dimensions of the validation
29:05
loader right so now until this part this concludes the data preparation steps
29:11
which means that now we have got these data loaders we have got the training loader validation loader and test loader
29:16
through these data loaders we can easily extract the batches the input batch and
29:21
the label batch which we need at any time and that just better so now we have created input and Target pairs remember
29:29
when we trained the llm the target pair here was also a token ID
29:34
representation it was just shifted from the input to the right by one so the difference here is that the target pair
29:41
here are zero and ones they are not token IDs awesome so until now we have looked
Recap and next steps
29:47
at this entire First Column is now over in the previous lecture we saw downloading the data set and
29:53
pre-processing the data set in today's lecture we saw how to create data loaders for the training data set the
29:59
validation data set and the testing data set in the next lectures we are going to start looking at the stage two which is
30:06
we are going to initialize the llm model we are going to load the pre-train weights we are going to modify the model
30:11
for fine tuning and then finally in the subsequent lectures we'll evaluate the F tune model so we have done the hard work
30:19
of data pre-processing data cleaning and using data sets and data loaders which is usually so important before we
30:26
directly jump to the model training itself so I hope you all are liking these
30:31
lectures I usually try to maintain an approach of whiteboard plus coding since
30:37
my aim is to train ml Engineers who are not just good at using chat GPD uh but I
30:43
want every one of you who are following these lectures to have very strong fundamental and theoretical understanding because that's what's
30:50
lacking in today's Engineers thanks a lot everyone and I look forward to seeing you in the next
30:56
lecture

***


Coding th





