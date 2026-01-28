
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

gpt2 tokenizer from tick token and then allows us to pad or truncate the
20:14
sequences to uniform length defined by either the longest sequence or predefined maximum length if the user
20:21
defines their own maximum length what we can now do is we can create an instance of the spam data set class using the
20:27
train. CSV file uh which we obtained in the previous lecture and here you can see I do not
20:34
set the maximum length so the maximum length is computed from the data set itself and when you print the maximum
20:39
length you can see that it's 120 that makes sense uh since the longest sequence in our data set contains no
20:46
more than 120 tokens it seems like a common length for text messages so one
20:51
sentence is around 15 to 20 tokens let's say so six sentences uh then that's the maximum
20:57
length so it's worth noting that the model can handle sequences of up to 1024
21:02
tokens because the context length of gpt2 which we have defined as the model here is
21:08
1024 right so you can pass max length Max up to a maximum value of 1024 over
21:15
here when you call this function um now what we are going to do
21:20
is that we are also going to uh pad the validation and test data sets to match the length of the longest training
21:27
sequence it it is important to note that any validation and test samples exceeding the length of the longest
21:33
training examples are truncated so now what we are going to do is that when we create an instance of the spam data set
21:39
class for the validation and the test data set we pass in the maximum length and that maximum length will be equal to
21:46
120 which is the maximum length in the training data set so we will encounter
21:51
this Loop where maximum length has been defined so there might be some sequences
21:56
in the tra testing and validation set which are last lger than the maximum length so we will need to truncate those
22:01
sequences that's what I've written over here however one thing to not is that
22:07
you can even set the maximum length equal to none here there is no such requirement that the maximum length
22:12
which you set here has to be equal to the maximum length in the training data set uh you can try out by setting this
22:19
to none as well okay now uh so here you can see
22:24
that you can create instances of the validation data set and the testing data set as well and then you can print out
22:31
the maximum length but right now if you print out the maximum length it will just be equal to 120 because we have
22:37
passed in that so if you print out the maximum length you can see that it's 120
22:43
because we passed in this parameter as a user defined and this was already
22:49
120 all right now the data set has been defined right now the data set will
Coding the Data Loaders
22:55
serve as an input to the data loader so so remember there are two things here first uh we have to implement a data set
23:03
and the data set will then be served as an input to the data loader so now what
23:08
we are going to do next is that we are going to use the data set as the input and then we will instantiate data
23:14
loaders right uh okay so in when we create or when we pass the data set as
23:21
an input to the data loader remember that we can set the batch size and we can also set the number of workers
23:27
that's for parallel processing so here what we are doing is that we are setting the batch size equal to 8 and we
23:32
are setting the number of workers equal to zero setting number of workers equal to zero is just for the sake of
23:38
Simplicity we don't want any parallel processing here drop last equal to True
23:43
means that if the last batch has a smaller data we just drop it uh great so
23:51
now here you can see train loader you create an instance of the data loader and then you pass in the train data set
23:58
as the input you similarly you can create the validation loader and the test loader as well so now this test
24:05
loader validation loader and trade loader when you run this part they'll be initialized and then you can use these
24:11
loaders to essentially create entire data sets in this format what I'm showing to you on the screen right now
24:17
so for example when you run the training data loader and you can extract batches from it now you can extract the first
24:24
batch and then it will give you the uh encoded and the you can extract the
24:29
second batch you can extract the eighth batch in a very easy manner so after you run this what we can now do is that we
24:36
can run a test to make sure that the data loaders are working and indeed returning batches of the expected size
24:43
so here what I'm doing is that I'm iterating through the training data loader uh right till the end and then
24:49
then I'm going to print the input badge Dimension and the label batch Dimension so if you uh iterate through the
24:56
training loader till the end you'll see that the input badge Dimension has the size of 8 by 120 can you think what this
25:02
means and the label badge Dimensions has torch dot size 8 can you try to think what this means you can pause the video
25:09
here for a moment so these input B Dimension means that since the batch size was eight
25:16
every input batch has eight rows and 120 columns because the maximum token ID
25:22
length was 120 so this is exactly what I've shown you over here right on the screen what you're seeing right now
25:29
uh on the screen what you're seeing is if you look at yeah if you look at this first answer
25:38
this right here what I'm showing with the arrow right now that that's the input badge and if you look at this input
25:45
batch you'll see that it has eight rows and it has 120 columns so that's why the size here is 8 by
25:52
120 okay similarly you can look at the labels label sensor and here you can see


***


25:59
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




