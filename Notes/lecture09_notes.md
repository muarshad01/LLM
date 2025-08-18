
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from
0:10
scratch Series in the previous lecture we took a look at bite pair encoding and uh we saw
0:19
that how bite pair encoding algorithm can be used for something which is called as subword tokenization so we saw
0:27
the difference between word based subword based and character based tokenization and we looked in detail how
0:33
GPT models such as GPT 2 3 and 4 use the bite pair encoding algorithm for
0:40
tokenization if you have not seen the video for the previous lecture again I would highly ENC encourage you to go
0:47
through this so that you will follow along pretty well in this lecture if you are coming to this playlist for the
0:52
first time welcome and uh we follow a very specific style in this playlist where we do a mix of writing on the
0:59
White board plus showing you everything from scratch in the jupyter notebook
1:04
code editor so that the theoretical understanding is also strong and the coding background is also
1:11
strong up till now we have looked at tokenization which is needed for large
1:17
language models so if you think of the whole process we are currently at the data pre-processing stage before the
1:23
data is given for the llm training in the pre-processing the first step is tokenization then we come to something
1:31
called Vector embeddings we have not seen Vector embeddings yet uh and then after that we feed these Vector
1:37
embeddings to the uh training or for the training process before we come to Vector
1:44
embeddings there is one very important lecture which we need to cover and that is the topic of today's lecture creating
1:51
input Target pairs essentially input output pairs if you look at other
1:57
machine learning tasks such as classification ation it's usually usually very clear right what is the
2:03
input and what is the output if you want to distinguish between cats and dogs from images the images of cats and the
2:10
images of dogs will be input and whether it's a cat or whether it's a dog will be
2:15
the output if you consider a regression problem on the other hand let's say if you want to predict the price of a house
2:22
B based on its area the area of the houses is the input and the price is the
2:28
output so creating the input output pairs or the input Target pairs is pretty easy for large language models we
2:35
use a specific technique for creating these pairs and it's very important to devote a separate lecture for you to
2:42
understand this so let's get started with today's lecture as I mentioned before now only
2:48
one last step is remaining before we move to creating Vector embeddings which will then be fed to training the large
2:55
language model and then last and then that last step is essentially create getting the input Target pairs so first
What are input-target pairs in LLMs?
3:02
when I say input Target pairs what do I mean and what do input Target pairs looks like so let's say uh this is my uh
3:13
sentence right which is the text sample llms learn to predict one word at a time
3:21
so the blocks which are marked in blue will be the input to the llm and the
3:29
block which are marked in red will be the Target or the output which the llms
3:35
have to learn and why are there these different rows so these are different iterations
3:42
let's look at the first iteration in the first iteration the input is llm and the
3:48
based on this input the out uh llm has to learn the output which is the learn so the next word is always the output
3:55
whatever comes after the prediction is masked or it's not shown to the llm
4:00
this is what happens in iteration number one now let's look at iteration number two so learn which was the output or the
4:09
Target in the first iteration now is a part of the input so in the second iteration llms learn that is the input
4:17
and two is the target that's the target pair that's the second
4:23
iteration in the third iteration two which was the output of the previous iteration now becomes the input so so
4:29
llms learn to is the input in the third iteration and predict is the output I
4:35
hope you have started understanding the pattern now in every iteration there is only the next word which is the output
4:42
and whatever comes before that is the input these are the input Target
4:48
pairs that's very important to remember so uh here also you'll see that llms
4:54
learn to predict is the input and one is the output so at every stage of the iteration process uh llms have input
5:02
which is the part of the sentence up till the word which needs to be predicted and the word which needs to be
5:09
predicted that is essentially the output this this figure which I'm saying
5:15
is just for illustration purposes in today's lecture we'll learn something about context length which means how
5:22
many words are given as the input the output length is always one one word will be predicted but we can essentially
5:28
choose the input context length now uh in every
5:34
iteration The Words which are after the target are essentially masked so the
5:39
llms cannot access The Words which are past the target so there are two things to remember here the first thing to
5:46
remember is that within the sentence itself we break down the sentence into input and a Target which is the next
5:52
word uh then in the second thing to remember is that in subsequent iterations whatever was the output in
5:58
the previous iteration then becomes the input so this is a auto regressive model
6:04
why Auto regressive because the output of the first iteration becomes an input of the next iteration like let's look at
6:10
these two iterations in in this iteration let me show it with a different color so that it becomes easy
6:16
in this iteration one the result one was an output right but see in this
6:22
iteration one is now a part of the input and then the next word is the output so it's called an auto regressive and it's
6:28
also called a self-supervised learning or you can think of it as unsupervised
6:34
learning itself because we are not labeling the input and the output the sentence structure
6:40
itself uh is used to predict or is used to determine what is the input and the
6:45
output we do not have to do any special labeling so in cats and dogs we have to manually label this is a cat this is a
6:51
dog right for the image classification but here to create the input Target pairs we don't have to say that look
6:58
this label this as the input label this as the output we'll just write a simple code which utilizes the sentence
7:04
structure itself and breaks down the sentence into input and the output so this is also an example of unsupervised
7:11
learning and it's also called Auto regressive I hope you have understood these two
7:16
concepts so in pre-training we always do unsupervised learning because the sentence structure is exploited to
7:23
create input output pairs or input Target pairs so I hope you have
7:28
understood how the the input Target pairs look like and we are going to create this in today's lecture in Python
7:36
uh if you understand up till this it's actually pretty easy to code it out in Python but I have I feel that students
7:42
don't really understand this part intuitively and and hence they find the coding part of it a bit
7:48
difficult okay now I want to mention a few things uh which I've just which just
7:53
serve as a summary of what all I explained up till now the first thing is what we are essentially doing here is
8:00
that we are given a text sample and uh based on the text sample we are
8:05
extracting input blocks that serve as the input to the llm correct and the llm
8:12
prediction task during the training is to predict the next word that follows the input block so for example if you're
8:19
looking at this input block the llm task is to predict the output or the next
8:24
word based on this input uh and that's what the llm is trained for
8:30
and the last point to remember is that during the training process we will mask out all the words that are past the
8:37
target so in every iteration the target is the target word right like in this iteration time or let me take an earlier
8:43
iteration in this iteration two is the target so when we are doing this iteration the llm does not see anything
8:49
which comes after two so this part is essentially
8:55
masked and we'll see how to implement all of these features in code
9:00
Okay so until now I just wanted to explain what is the purpose and what is the aim of today's lecture and now we
9:07
are going to code the input Target pairs in Python so I hope you are ready for this coding so let's get started with
9:15
coding great so this coding section I've have titled creating input Target pairs
9:21
as always I'll be sharing this Jupiter notebook code also with you along with the video so that you can run the code
9:29
and check check whether you have understood the concept or not yourself so in this section we are going
9:35
to implement a data loader that fetches the input Target pairs using a sliding
9:41
window approach so there are two parts of this sentence which might be confusing to you what is data loader
9:48
that's part number one and what is the sliding window approach that's part number two don't worry I'll explain to
9:54
you both of these in a lot of detail uh to get started what we will initi do is that we'll take the whole
10:01
the verdict short story so remember our data set for this entire coding Journey
10:07
for this entire playlist is this short story The Verdict let me show you uh how
10:12
it actually looks like so this is the short story called The Verdict this is the data set which
10:19
we have been using I think this was published in uh so let me check the verdict edit won it was published in
10:27
1908 and we are using using this as the data set it's a toy data set but it's
10:32
important because whatever we learn right now it scales exactly the same way for larger data sets as well so we are
10:39
going to use this data set and remember in the last lecture we looked at the bite pair encoding tokenizer we are
10:45
going to encode this entire text using the bite pair encoding tokenizer it's a subword tokenizer so the tokens Can Be
10:53
characters the token can be words the tokens can be subwords as well so if you if you are not familiar with the bite
10:59
pair encoder please look at the previous lecture which we have covered great so we have uh defined the
11:06
tokenizer already which is the bite pair encoder tokenizer and what we'll be first doing is we will read the entire
11:14
data set and store it in a variable called raw text and then we will encode the entire raw text remember what an
11:21
encoder does is takes this text and converts it into token IDs and let us
11:27
actually run this right now so I ran this right now and you will see that uh I've have printed out the length of the
11:33
encoded text which means it's 5145 that means the vocabulary size which we have is 5145 what does a
11:40
vocabulary mean well we covered this in the previous lecture but let me show it to you again a vocabulary essentially
11:47
looks something like this uh yeah so this is
11:55
how let me go to the yeah this is how a vocabulary looks like like essentially
12:00
we'll have different tokens and to every token a token ID will be uh attached so
12:06
vocabulary is essentially dictionary which maps The Tokens into token IDs remember since we are using the bite
12:12
pair encoder the tokens won't be words but they can be subwords or characters also so essentially what this size 5145
12:20
conveys is that our vocabulary for the for this text which we have as the data
12:26
set has the length of 514 5 which means we have 5145 tokens and corresponding
12:32
token IDs great so uh I have just written this in
12:37
blue executing the code above will return 5145 that is the total number of
12:42
tokens in the training set after applying the bite PA encoding tokenizer great so what I'm going to demonstrate
12:50
right now to you is just I'm going to look at the first uh so what I'm going
12:56
to do as I'm going to remove the first 50 tokens for the from the data set just so that the demonstration becomes a bit
13:02
better uh after you remove the initial 50 tokens it results in a slightly more
13:08
interesting text passage you can keep the entire tokens as well just to make the lecture more interesting what I'm
13:14
going to do here is that the encoded tokens were in ENC or encoded unders
13:20
scroll text so I'm going to Define one more variable called called encoded undor sample which just removes the
13:26
first 50 tokens from the data set great now uh first I what I want you all
Coding input-output pairs in Python
13:33
to do is I want you to pause here for a moment and think about this question
13:39
yourself uh think about the question that let's say you are given this data set right now and you you'll I hope you
13:47
understood this input output Target pairs which I mentioned what's the simplest thing which comes to your mind
13:53
how can you convert this data set into such kind of uh input output Target
13:59
pairs what will you need to make this conversion can you think about it a bit
14:05
think about the simplest way don't think about complex algorithms anything like that what is the simplest thing which
14:11
comes to your mind you can pause the video here for some time because if you answer it it will really improve your
14:20
understanding so let me reveal the answer now one of the easiest and most
14:26
intuitive ways to create the input Ty Target pairs for the next word
14:31
prediction task is to create two variables X and Y where X contains the
14:38
input tokens and Y contains the targets which are essentially the input shifted
14:43
by one so let me explain to you how this logic works okay so what do we exactly need
14:50
here let's say if uh let's say if my input is 1 2 3 and 4 let's say say if my
14:59
input is this I want my output array let's say if this is my input array X I
15:05
want my output array to be looking something like 2 3 4 and
15:13
5 so what I have done here exactly is that if one is the
15:19
input two should be the output it's very similar here if llm is the input learn
15:25
should be the output correct if one and two is the input then three should be
15:31
the output which means that if llm learn is the input then two should be the
15:36
output if 1 2 3 are the input which means that if llms learn two is the
15:44
input then the output should be four which means that predict should be the output and then finally if 1 2 3 4 if
15:53
all of these four words are the input then five should be the output which means that if llm learn to predict is
16:00
the input then the output should be equal to one this is what I actually want to
16:06
create and how do I determine the size of this uh why do why did I take the
16:11
input size X to be four and the output size to be four so basically that is called as the context size the context
16:18
size is how many words do you want to give as input for the model to be making its prediction so here the context size
16:25
is equal to four right so if we give up to four four words the model will be able to predict the next
16:32
word that is what the context size actually means so we want to create input output arrays like this so let me
16:39
show you how those can be created for this data set of the verdict which we
16:44
have seen okay so first we have to determine the context size as I told you the context size determines how many
What is context size?
16:51
tokens are included in the input so let me explain the context size a bit more here currently we are choosing context
16:58
size of four you can choose anything which you want and you can play around with this code when I share it with you
17:04
so the context size of four means that the model is trained to look at a sequence of four words or
17:11
tokens to predict the next word in the sequence so the input X is the first
17:16
four tokens let's say 1 2 3 4 and the target Y is the next four tokens which is 2 3 4 5 so that is meant by context
17:25
size so the input if the input is 1 2 3 4 the output is 2 3 4 5 what does it mean if the input is one the output will
17:33
be two if the input is 1 two the output will be three if the input is 1 2 3 the
17:38
output will be four if the input is 1 2 3 4 the output will be five but the
17:43
input cannot be one 2 3 4 5 because then the context size would be exceeded to
17:49
think of it intuitively the context size is basically how many words the model should pay attention at one time to
17:56
predict the next word so let's now take a simple thing so we have this encoded
18:01
sample which contains the token IDs of the encoded data set what I will first
18:06
do is that I will first take the four elements which are the first four elements that is my X which is the input
18:13
and then I'll just shift this x Matrix X array by one and then that will be my Y
18:19
which is the output so let's print out X so the IDS which are associated with the
18:24
first four encoded samples are 290 4920 2241 and
18:30
287 and then the IDS which are associated with Y which is the output is
18:36
4920 2241 287 and 257 what does this mean if the input ID is 290 then the
18:44
output will be 4920 if the input is 290 and 4920 the output is
18:51
2241 if the input is 290 4920 and 2241 the output will be 287 and if the input
18:59
is 290 4920 2241 and 287 the output will be 257 this is how the input output pairs
19:07
are actually constructed so uh what we can do now is
19:12
that processing the inputs along with the targets and remember the targets are just the input shifted by one position
19:20
we can then create the next word prediction tasks as follows so what I just explained to you I've written this
19:26
in code so I have created two uh variables here called context and
19:32
desired and I'm looping in so the context size is four so this Loop will
19:38
go from 1 to five so in the when I is equal to 1 which is the first iteration
19:43
the context will be just the first token ID which is 290 and the desired will be
19:49
the next token ID which is 4920 awesome when I is equal to 2 then
19:55
the context will be the first two tokens which is 290 and 4920 and the desired
20:01
will be the next token which is 2241 if I is equal to 3 uh then the
20:08
context would be the first three token IDs which are 290 4920 and 2241 and the
20:14
output will be or the desired will be the next token which is 287 and if I is
20:20
equal to 4 then the context will be the first four tokens IDs and the desired
20:26
will be the next which is 257 so everything on the left of the arrow here refers to the input the large
20:33
language model would receive uh and the token ID on the right hand side of the
20:39
arrow represents the target token ID which the llm is supposed to
20:44
predict so when we constructed these input output pairs this is what it actually means there are four prediction
20:51
tasks here it's not one prediction task so when I created this input output pair
20:56
of X and Y and even here here when I showed you the input output pair of X and Y it's not just one prediction task
21:04
but there are four prediction tasks which are happening here and these are the four prediction tasks because the
21:10
context size was four if the context size was eight there would have been eight prediction tasks in each input
21:15
output pair so when you look at input output pairs usually regression and
21:20
classification problem one input output pair corresponds to one prediction task image of a dog needs to be classified as
21:27
whether it's a cat or a dog but in the case of llms one input output pair corresponds to the number of
21:34
prediction tasks as set by the context size that is very important now what I'm
21:40
going to do is that I'm going to take this simple the same code but I'm going to decode it into text so that you can
21:47
get a feel of what is exactly happening here so here you can see I've taken the same code but I'm printing the decoded
21:54
context and I'm printing the decoded desired value so if and is the input
21:59
established is the output if and established is the input himself is the output if and established himself is the
22:06
output the next word which is in is the output and if and established himself in is the input then the next word uh that
22:13
is the output now this is exactly what we started the lecture with right you remember we started the lecture with
22:19
this and uh now what we have done is that through code we have just created
22:25
very simple input output pairs X and Y and we have seen how these pairs can be used to create the input and the output
22:33
awesome right so this is just the first step of what we need to be doing what we
22:39
have done so far is we have created the input output pairs that we can turn into
22:44
use for the llm training so later we are going to do llm training and so we have created input output pairs now but we
Creating a DataLoader
22:51
need to create them in a much more structured manner we need to create them for the entire data set not just that
22:57
later we will parallel processing so if we have multiple CPUs and we need to do
23:02
parallel Computing we need to do Computing in batches so we are going to do this in a
23:09
very structured Manner and for that what we are going to be doing is we are going to use something called as uh data
23:17
loader so now there is only one more task which is remaining before we can look at the vector embeddings in the
23:23
next lecture and that is implementing an efficient data loader
23:29
uh that iterates over the input data set and Returns the inputs and targets as P
23:34
torch tensors So currently we have got the input output arrays right but they are not tensors why do we need tensors
23:42
because all the optimization procedures which come later we we are going to use py torch and py torch works with tensors
23:50
so we need input tensors and we need output tensors no need to worry if you don't know what a tensor is you can just
23:56
think of it as a two-dimensional array for now or multi-dimensional array no need to worry about this this will not
24:02
stop you from understanding this lecture so our goal is this we want to
24:07
implement a data loader which creates two tensors an input tensor which contains the text that the llm sees and
24:15
the target tensor that includes the targets for the llms to predict that's it basically we have to create something
24:21
exactly like what I showed you in the code before but we need to create it in a tensor format and we need to do it in
24:28
a structured manner so that's why we are going to use something called as data
24:33
set and data loader so the these are data sets and data loaders which are in Python and here you can just see some
24:40
examples which have been done for some classification data sets but essentially data sets and data loaders enable you to
24:47
load or process the data in a much more efficient and compact manner as we'll see right
24:53
now awesome so now what we are going to do in the next step is we are going to Implement a data loader and uh for the
25:02
efficient data loader implementation we will use the pytorch inbuilt data set
25:07
and data loader classes so these are the data set and data loader classes link which I've shown right now I'll also
25:14
attach the link in the video description before going into the code
25:19
further I just want to show you what we expect the data loader to do so that you have a visual understanding I have seen
25:26
that until you get a visual understanding sending the code becomes very difficult to really Master but if
25:31
you know what you want to implement it's really very easy so in this section what are we doing we are implementing a data
25:38
loader that fetches the input output Target pairs using a sliding window approach let's see what this means so uh
25:47
here so what we are going to do is that let's look at this sample text in the
25:52
Heart of the City stood the old library A Relic from a Bagon era let's say this
25:58
is the kind of sentence and we want to create input output Pairs and we are going to create input output pairs with
26:05
a context size of four okay so let's say if the input is in the heart of let's
26:13
say the input is in the heart of the output tensor will be shifted by one
26:18
right as we already saw so the output will be the heart of the correct so the
26:26
first input pair is in the heart of and the first output is the heart of the now
26:31
in this input output pair there will be four prediction tasks the first is that
26:36
if the input is in the prediction should be the if the input is in the the
26:42
prediction should be heart uh let me switch to a different color if the input is in the heart the
26:50
prediction should be off and if the input is in the heart of the prediction should be the so uh first of all we have
26:59
an X which is the input tensor and Y which is the output tensor now let's see
27:05
what the row of every tensor are representing uh so we collect the inputs
27:11
in a tensor X so we collect the inputs in a tensor X
27:16
where each row represents one input context so let's look at the tensor X if you see each row each row of this is one
27:24
input context in the Heart of the City stood the so each row represents one uh
27:31
input context so the first input output pair will be in the heart of and the
27:36
first output will be the heart of Thee the second input will be the CT stood the and the second output will be CT
27:44
stood the old so basically what I earlier in this lecture I showed you one
27:49
input output pair when we look at tensors if you look at each row each row
27:54
of the X tensor is an input each row of the Y tensor is the Corr oning output so
27:59
the 50th row of the X X tensor and the 50th row of the Y tensor will be the 50th input output
28:06
pair and in each input output pair there are four prediction tasks here as I as I
28:12
told you because the context size is equal to four so essentially we are doing the same thing what we did in the
28:17
earlier part of the lecture but we just take the entire text uh we put it in tensors in the rows of the tensor so we
28:24
split the text into four words so the first four words are the first row the second four words are the second row and
28:31
if you look at the output tensor it just the input tensor shifted by one that's it if you look at each row of the output
28:37
tensor it's just the input tensor row shifted by one so the second tensor y uh the second
28:44
tensor y contains the corresponding prediction targets next Words which are created by Shifting the input by one
28:51
position this is very important for everyone who is watching this lecture to understand uh it all we are doing is
29:00
next word prediction task so let me again explain this so that I I really
29:05
want this concept to be understood because it's the heart of everything which we are going to do but let's look
29:11
at the second row in the second row there will be four prediction tasks the first prediction task is when the input
29:17
is the the output is City when the input is the city the output is stood when the
29:23
input is the city stood the output is the and when the input is the city is stood the the output is the old the
29:31
output is old so basically each input output pair
29:36
corresponds to one prediction task and corresponds to four prediction tasks and we are just predicting the next word
29:43
that's it it's as simple as that and that is exactly all we are actually doing in the code also so now I'm
29:50
returning back to the code and uh This what I showed you here right now on the
29:56
Whiteboard uh creating such kind of input tensor and output tensor is exactly what we are
30:02
going to do in the code so if you if you have understood this the code will be much easier to understand so we are
30:08
going to implement a data loader which creates these input and the output tensors and this will be done in four
30:14
steps the first is tokenizing the entire text because uh we are going to deal
30:20
with token IDs right now I'm showing you words over here but actually we'll deal with token IDs so we need the encoded
30:26
text uh then we'll use a sliding window and now do you understand why is it called
30:32
sliding window because uh look at this blue look at the blue window and the red window
30:40
here so the blue is the input and then you slide it by one row slide it by one
30:46
world and that's the output so each input and output pair is just sliding so you slide the input and then you get the
30:53
output pair uh and then finally what we'll be doing is that we'll return the total
31:00
number of rows in the data set and we'll return a single Row from the data set I'll show what this means so here we are
Implementing DataLoader in Python
31:07
going to define a class um and this class is going to take a data set now uh
31:15
for this what we are doing is from tor. utils we are importing data set and data loader the data loader will come a later
31:22
right now we have our goal is to define the data set and the data set cannot just be a bunch of tokens we need to
31:31
make sure the data set is in input output pairs so what do we do here first
31:36
we see what are the arguments which are taken when we create an instance of this class so when we create an instance of
31:41
this class we basically need to specify four things we need to specify all the text file which we have so this is the
31:48
data set the txt file then we need to specify the tokenizer so here we are
31:53
using the bite pair tokenizer then we need to specify the max length the max length is the context size here we are
32:00
going to use a context size or context length of four and then there is something called as stride so what
32:07
stride means we'll come come to that in a moment but right now just know that there are four arguments which this
32:12
class actually takes now what we'll be doing is that we'll be creating two arrays the first is called input IDs the
32:19
second is called Target IDs what we'll be doing is that in input IDs uh we are
32:26
so here you see we are going to uh loop over the entire data set so
32:32
when I is equal to 1 the input ID the input chunk will have token IDs which
32:38
are from 0 to 4 and the target will be token IDs from 1 to five which is just
32:44
shifted by one and then you append this to the input ID's tensor and the target
32:49
ID's tensor so when I is equal to one what will be appended is the first row of the input and the first row of the
32:57
output so let me actually rub rub erase this a
33:03
bit okay so in the code we saw the iterations right so when I is equal to 1 the first row will be appended to the
33:09
input tensor and the first row will be appended to the output tensor so if you
33:15
see the input ID is is the input tensor and this input chunk is being appended
33:21
that's the first row now when I is equal to 1 we slide over to the next so if I
33:27
is equal to one we uh we we look at the next chunk so the first chunk will
33:33
be in the heart of in the heart of the second chunk here is the city stood the
33:39
that's the third chunk the fourth chunk is old library a so we are going to
33:45
divide the entire data set into chunks like this and then we are going to append that to the input tensor and just
33:51
we are going to slide the chunk by one and then that we are going to up to the output tensor this is how we are
33:58
creating the input and the output tensors and then we are going to Loop
34:03
till we reach the end of the data set so length of data set minus the max length
34:10
why minus max length because the last thing will include the context size and
34:15
so we don't want to spill over the data set essentially if you are finding this section a bit hard to understand just
34:22
remember that all we are doing here is that we have a data set we are chunking the data set so we have in first we have
34:29
the input chunk and then we have the output chunk actually I think let me explain this using a Hands-On example
34:37
over here so let's take this this example itself which I have written here right now uh what we are essentially
34:44
doing here is that we are first going to have an input chunk so two Implement efficient data loaders that's my first
34:50
input chunk then what I'm going to do then I'm going to shift it by right
34:56
shift it to the right by one and then I'll have my output chunk so the output chunk is Implement efficient data
35:02
loaders V so that's my first input output chunk so the input chunk will be
35:07
the first row of the tensor the output chunk will be the second row of my tensor and then I'll slide so the way
35:13
I'll slide will depend on my stride so if my stride is equal to let's say one
35:19
it means that my next input will be Implement efficient data loaders and my next output will be sorry my next input
35:27
will be Implement efficient data loaders V and my next output would be efficient data loaders V collect so at each time
35:35
the output is just the input shifted by one and stride determines how much we slide so in the example which we took
35:42
here if you see the in the first input is in the heart of but the second input
35:47
is the city stood the so the first input is in the heart of the second is the
35:52
city stood the which means that the stride is also equal to four so the first first input was this and then we
35:59
moved and then the because the stride was four the second was the second input
36:04
was the it stood the if the stride was equal to one the the next input would have been the heart of the
36:11
city but that is not the second input here which means we have used a stride of four in this
36:18
example that's why we also need the stride because we need to know how much to slide when to create the next input
36:24
output batch I hope you have understood this part this is a bit of a tricky portion so please ask in the comment
36:30
section if something is unclear over here so until this part we have created
36:35
the input output tensor but now we have to define a method which is called as get item what this method does is that
36:43
based on the index which we provide it just Returns the that particular row of the input and that particular row of the
36:49
output so if the index is equal to zero this will return the first row of the input tensor and the first row of the
36:55
output tensor why is this method get item needed because when we create a
37:00
data loader the data loader will look at this method and then only it will create
37:06
the input output pairs so if you look at the data loader what uh it needs the
37:12
data set in this map style or the iterable style right now we are using a map style data set which means we need
37:18
to have this get item what the get item will do is that it will actually tell it
37:23
will tell the data loader uh what what kind of input and Target should we have
37:30
so here we are clearly saying that if it's based on the index if the index is 50 the input will be the 50th row of the
37:38
input tensor the target will be the 50th row of the output tensor so this G item
37:44
is finally what the data loader will be using now let me recap what all we have learned so far so this GPT data set
37:52
origin class we have just implemented is based on the pytor data set class it
37:57
defines how individual rows are fetched from the data set each row consists of
38:03
number of token IDs assigned to an input chunk tensor this basically means that
38:08
each row the length of or the number of tokens in each row is equal to the context length so here if you see the
38:16
number of tokens in each row in the heart of four tokens because the context size is four the target chunk tensor
38:23
contains the corresponding targets so uh as I've mentioned over
38:29
here I'll actually upload the links to this data set data loader and also maybe
38:34
uh a small tutorial so that uh you know this thing becomes a bit more clearer
38:40
and I think you should have a stronger understanding of uh this data set and data loader portion because many people
38:47
just skip over this completely now what we'll be doing since our data set is ready and this format is ready we will
38:53
feed the data set into the data loader so this is where the data loader comes into the picture and what we'll be doing
39:01
is we'll be doing the four things we'll initialize the tokenizer create the data set why because this data set needs this
39:10
class needs four attributes the the text the tokenizer the max length and the stride then what we'll be doing is that
39:17
we'll put drop last equal to true because if the last batch it's shorter
39:22
than the batch size uh it's dropped to prevent loss spikes during training you
39:27
don't need to know about this too much but just remember that when we do batch processing if the last batch is shorter
39:34
than the specified batch size uh it is important to prevent loss spikes during
39:39
the training process awesome so now we Define this function called create data
39:46
loader this is very important because this is the one this function will implement the batch processing the
39:52
parallel processing which we will need uh and that is governed by the batch size but more than that what this
39:58
function will do is that it will actually create uh it will help us create the input output uh data pairs
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
