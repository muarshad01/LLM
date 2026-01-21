text?
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from
0:10
scratch Series today is the last lecture in the GPT architecture module which we
0:16
have been covering and in today's lecture what we are going to do is that we are going to generate text from the
0:23
output tensor which we have reached at in the last lecture so first let me give you a
0:30
review of how an llm generates text because this figure right here is what will guide us throughout this entire
0:37
lecture and then I'll quickly take you through the recap of what all we have completed so far and what's the main
0:44
objective of today's lecture so let's get started everyone what a large language model
0:51
does is essentially the following it generates tokens given a certain
0:57
sequence of input tokens it generates the output token and uh to generate the output
1:04
token the model is given a context size that's the maximum number of tokens which the model looks at before
1:11
predicting the next token so uh let's say when we are
1:16
looking at the first iteration this is the input tokens hello I am and the
1:22
model has to predict the next token and then the next token is
1:28
actually appended to the input tokens so in the second iterations now the inputs are hello I am
1:36
a uh so the token which has been generated in the previous round is appended to the input for the next
1:44
iteration and in the next iteration the input tokens are hello I am a and then
1:49
the output is model great then in the third iteration
1:55
we have hello I am a model which are the input tokens and then the output tokens
2:00
are ready similarly we continue doing iterations until there is a specific
2:06
number of Max maximum number of new tokens which can be generated so let's say we have reached the six iteration
2:13
and hello I am ready to help so the new tokens which have been generated are model ready to and help and Dot so five
2:21
new tokens so if the maximum number of new tokens is five then the output from
2:27
the model is hello I am a model ready to help this is the main uh mechanism through
2:33
which the next token is generated from a given list of tokens what's very important in this process is also the
2:40
context size that's the maximum number of previous tokens the llm is going to look at before predicting the next token
2:47
in gpt2 architecture the context size was 1,24 so just keep this in mind keep this
2:54
iterative procedure in mind where the output from the earlier iteration is appended to the next iteration and the
3:00
next word is predicted and this keeps on happening until we have reached the maximum number of new tokens this is the
3:07
exact mechanism which we are going to implement in today's lecture first let's recap what all we
GPT Model recap
3:13
have learned so far in this series we started out with getting an input such as every effort moves you that's the
3:19
input sentence we converted it into token IDs we converted it into input embeddings sorry we converted it into
3:26
token embeddings we added positional embeddings to this token embeddings which led to input embeddings then we
3:32
added a Dropout layer the output from this Dropout layer was passed into the Transformer block the Transformer block
3:39
which is shown in blue color is where all the magic really happens and the core engine of the Transformer block is
3:44
multi-head attention before the multi-ad attention there is a normalization layer after the multi-ad attention we have a
3:51
Dropout and a shortcut connection then after the shortcut connection we have another layer of normalization we have a
3:57
feed forward neural network we have another Dropout layer then we have one more shortcut connection over here and
4:03
then we get the output from the Transformer after the output is obtained
4:09
from the Transformer we have one more layer normalization layer and then we have a final neural network which is
4:16
also called as the output head which gives us this output tensor all of this has been covered in
4:21
the previous two lectures where we coded out the entire Transformer block and in the last lecture we coded out the entire
4:27
GPT model so if you have not seen seen the previous lectures I highly recommend you to do that so that you can follow
4:33
along in this lecture as well so uh let us see the exact flow which we have
4:39
implemented so far and what we need to do next so the output or the main goal of the today's lecture is that look at
4:46
this output tensor which we have obtained this output tensor looks pretty big right so your question would be how
4:52
do I go from this output tensor to prediction of the next word which I have shown in this visual representation in
4:58
today's lecture first we are going to understand that on a whiteboard and then we are going to go to code to predict
5:06
the next word so at the end of today's lecture we will actually take an input and get words as the next
GPT Model visual flow
5:12
predictions okay so here's the entire pipeline which we saw remember the format in which we have received the
5:17
input batches uh here there were two batches and each batch has certain number of tokens so in this case each
5:24
batch had four tokens we focused on the first batch which has four tokens and that's every moves you we converted
5:32
every token ID there are four token IDs here 6 6109 3626 610 and 345 we converted every
5:42
token ID into a token embedding so the embedding size was 768 over here and
5:48
then what we did was we added positional embedding to these token embeddings since the context size is four over here
5:54
we have four positions and there is a 768 dimensional embedding for the positional
6:00
uh embedding so we add the token embedding to the positional embedding and that
6:05
gives us the input embedding for every token for every effort moves and U so we
6:10
have these four input embeddings for these four tokens then what we do is we pass these input embeddings through a
6:17
Dropout layer which randomly turns off certain elements to zero this output which I'm highlighting in the yellow
6:23
color right now that is passed as an input to the Transformer this is where all the magic happens in the trans
6:29
former we first start with a normalization layer which makes sure that the mean and variance across so
6:35
mean is zero across every row and the variance is one across every single row then we apply the multi-head attention
6:42
which converts these embedding vectors into context vectors which carry richer meaning they also carry meaning about
6:48
how the particular token attends to all the other tokens uh or how one token
6:54
relates to all the other tokens in the sentence this mask multi-ad attention is
6:59
the key evolution in the llm architecture and that's why modern GPT
7:04
architecture such as the GPT 4 which we all have been using perform so brilliantly if multi-ad attention was
7:10
not there the llm outputs would not be so coherent and so meaningful after this
7:15
we have a Dropout layer then we have a shortcut connection followed by a layer
7:21
normalization followed by a feed forward neural network followed by another Dropout
7:28
layer followed by a short shortcut connection and then we get the Transformer block output remember here
7:34
the dimensions up till this stage are exactly preserved so we still have four tokens here and every token is still an
7:40
embedding size of 768 but now the embeddings are much more richer because they also contain uh
7:47
context about how one token relates to other tokens then we have after coming out of the Transformer we have a layer
7:55
normalization and then we have the final output layer which is this fin neural network now after this stage is where
Converting output logins into next word prediction
8:03
today's lecture begins so first I want you to look at the output which we have received when we come out of the entire
8:10
GPT model so the number of rows are still the same we have four rows every
8:15
effort is the second row moves is the third row and U is the fourth row but
8:20
the number of columns are now 50257 which is the vocabulary size and
8:26
uh why do we have those many number of columns the the reason is because every
8:31
effort moves you these are four tokens right which is also the context size when the context size is equal to four
8:37
there are actually Four input output prediction tasks which are happening there is not just one input output
8:44
prediction task so you might be thinking how come there are four input output prediction task well the first
8:49
prediction task is when every is an input you have to predict what's the output and that should be effort right
8:55
so you look at these output Logics so these are called logics you look at the 50257 output logits for every and then
9:04
you find that index which has the highest value so let's say this index has the highest value right remember
9:11
this index also correspond to probabilities so this index will give us
9:17
which word in the entire vocabulary should come after every and then
9:22
hopefully this token ID here or this index corresponds to effort so then if when every is the input effort is the
9:29
output similarly when every effort is the input we look at the row for effort
9:36
and we try to look at that index which has the maximum value let's say this is that Index this is that token ID then we
9:43
go to our vocabulary and look for the word which corresponds to this token ID and that will be moves after all the
9:49
weights and parameters have been optimized when every effort moves is the input then we look at again the row
9:56
corresponding to moves and we try to look at that token ID or that index
10:02
which gives us the maximum value and then we look at the vocabulary and we
10:07
try to find out what that ID corresponds to and that will be you only after these three prediction tasks are done then we
10:14
come to the fourth prediction task which is the main prediction task when the input is every effort moves you you have
10:21
to predict the output right so you look at the fourth row which is the final row and you try to find that ID which gives
10:28
you the maximum value and then you find the word corresponding to that and that
10:33
word will hopefully be forward so then the next word which will be predicted by this LM is every effort knows you
10:39
forward and remember the size of the output over here since we have only one
10:45
batch over here the number of rows in the output are equal to four and the number of columns are equal to the
10:50
vocabulary size which is 50257 uh when the number of batches are equal to two the output tensor size will
10:57
be 2 into 4 into 50257 what we have to do is that from this tensor we have to extract that word
11:04
which comes after every effort moves you so let's see how the first step is to
11:10
take a look at this output tensor and make sense of it that we have four tokens and the number of columns is
11:16
equal to the vocabulary size awesome as I told you every row corresponds to something specific so the first row
11:22
corresponds to what token should come after every the second row corresponds to what token should come after every
11:28
effort the third row corresponds to what token should come after every effort moves you and only the fourth row
11:34
corresponds to what token should come after every effort moves you so we are going to extract the last Vector from
11:41
this tensor that's step number two so we will look at the vector which is corresponding to um U which I'm marking
11:50
right now and we'll extract this Vector after this Vector is extracted this is
11:55
called as the logits vector we look at the logits which are present and you'll see that these logits don't add up to
12:01
one which means that the logits currently don't represent the probabilities then what we do from Step
12:07
number two to step number three is that we are going to apply soft Max so we are going to apply soft Max function here so
12:14
that we are going to convert these logits into a set of probabilities so now when you look at the last Vector for
12:20
you you'll see that all of the values add up to one so this gives us probabilities that the probability of
12:25
the next token being the first element in the vocabulary is let say 0.1% the
12:30
probability of the uh second token being the uh the probability of the next word
12:37
being the second token is let's say uh 02 Etc so you can do this for all the
12:42
tokens and then you in the next step which is Step number four you identify the index position or token ID of the
12:49
largest value so you find the index which corresponds to the largest probability so here clearly it looks
12:56
like 02 uh is the probability which seems to be the largest and then I find the index
13:02
for this particular element and it turns out that in this case let's say the token ID is equal to 57 which means that
13:09
the highest probability for the next word after every effort moves you is the word or the token with the token ID
13:16
equal to 57 and then all we do is that we go to we go to our vocabulary and we
13:22
decode the word which corresponds to the Token ID of 57 and we hope that it's equal to forward up till now we have not
13:28
trained the llm architecture so the next word will not be what we expect because the training has not been done at all it
13:34
will be a random word but after all the training is done which will be the subject of our next module the next word
13:41
should be what we actually expect so this is the exact procedure which we are going to follow and remember there's one
13:48
last step after this token ID is obtained we'll have to append this token ID to the previous inputs for the next
13:54
round why this step number five exist is I hope you remember this diagram which we saw at the start of this lecture the
14:01
step number five exists because after you predict the next token after you predict the Next Generation token the
14:08
task does not stop here we have to append this token to the input in the second iteration and then we have to
14:13
repeat this process until we reach the last iteration which corresponds to the maximum number of new tokens okay so I
Next word prediction visualised
14:20
hope you have understood the process I've just uh written a section where I explain this process again so that you
14:27
can revise and reinforce your Concepts so in the previous section we have seen that the GPT model out outputs the
14:34
tensors with this shape batch size number of tokens and the vocabulary size
14:40
remember this is exactly what we saw over here uh the shape is equal to the
14:45
batch size multiplied by number of tokens multiplied by the vocabulary size okay so uh this is the output tensor and
14:55
now the question is that how do we go from this output tensor to the generated text and as I explained to you in the
15:02
visual map there will be different steps so first what we'll do is that we'll get this output tensor and then what we'll
15:08
do is that we'll extract the last row from this output tensor after extracting
15:14
the last row these are the Logics we'll convert it into a set of probabilities by applying the soft Max and we'll
15:20
extract that token ID with the highest probability and then we'll find the word which cor or find the token which
15:26
corresponds to that token ID and then we'll append that token to the previous
15:31
inputs and then we'll do the next round of iterations this is exactly what we'll be
15:37
doing so basically we'll take the index of the highest value we'll get the token ID we'll decode it back to text that
15:44
produces the next token and will append to the previous input so I've written the same thing here so that you can you
15:50
know reinforce and uh master your understanding of these Concepts so this
15:55
stepbystep process enables the model to generate text sequence ially building coherent phrases from the initial input
16:02
context so as I showed to you over here we are going to over here we are going
16:08
to repeat this process over multiple iterations right and that's also what I've written over here yeah in practice
16:16
what we do is that we repeat this process or multiple iterations until we reach a user specified number of
16:22
generated tokens so we have to specify how many new tokens you need and only when we reach that reach that me many
16:29
number of new tokens we stop so the figure below illustrates the process of generating one token ID at a time so
16:35
let's zoom into this figure further so let's say the initial input tokens which are provided as the input to the llm are
16:41
hello I am with these token IDs what the llm will do in iteration number one is
16:47
that it will predict the next token ID using the procedure we saw before and let's say the token ID is 257 which
16:53
corresponds to the Token o this token is then appended in the second iteration so
16:58
now we are we have come to the second iteration the inputs are hello I am a and then we make the output which is the
17:05
next uh given given these inputs what is the next token which comes out and then
17:11
the token word which or the actual token is model and then this token is now appended to the input of the previous
17:17
iteration now we are at iteration number three similarly we get an output at iteration number three and this proceed
17:24
proceeds till the end why do we do only six tokens because there is a provision
17:29
for Max new tokens and that has been set to six that's why we only do six iterations remember the number of
17:36
iterations which we do will be determined by the maximum number of new tokens which have been
17:43
specified so then when the input is hello I am since the maximum number of
17:48
new tokens is six the output will be hello I am a model ready to help dot
17:55
this this is the output which the GPT has has generated that's why generative AI we have generated something like this
18:03
completely from scratch now that was not present as an input or a training data this is generated as a
18:10
new text um and this is the process underneath all of it so now when you use
18:15
GPT today or tomorrow or any time uh hopefully this lecture Series has kind of shined a torch to this black box for
18:23
all the other students who don't know how this next work next word prediction task Works GPT operates like a black box
18:29
but not for all of you who have been watching this video series because I'm trying trying to deconstruct how the
18:35
next word is actually predicted given the input tokens awesome so uh in iteration number
18:44
one the model is provided with tokens corresponding to hello I am and then it
18:49
predicts the next token with ID 257 great and then that is again appended to
18:55
the input and this process is repeated till the model produ produces the complete sentence hello I am a model
19:02
ready to help after six iterations why only six iterations because the maximum number of new tokens was set to six okay
19:10
so I hope everyone has followed with me until this part and I want you to recap
19:15
these steps the step number one is to look at the output tensor step number
19:21
two is to extract the last Vector step number three is to convert logits into probabilities step number four is to
19:27
identify the index position position of the largest value and step number five is to append token ID to the previous
19:33
inputs and we are going to keep on doing this until maximum number of new tokens has been reached this is the exact same
19:40
thing which we are going to implement in code right now so the next part of this lecture is diving into code so uh let's
19:48
jump into code right now okay so here what we are going to do
Coding the next token generator function
19:53
is we are going to generate text from output tokens and uh we are going to implement the same process what we had
19:59
seen in the uh on the Whiteboard okay great so the first thing what we
20:05
need is that uh we need the inputs uh we need the inputs to be
20:12
provided and the inputs are usually provided in the format which looks like
20:17
this um yeah this is the input batch so in the input what we'll be doing is that
20:22
let's say this is a batch with two inputs the first uh batch has four tokens and the second batch has four
20:29
tokens so the inputs will be provided like this and that's called as the idx
20:34
so the shape of idx is batch which are the number of uh batches which we have
20:40
uh and that's the number of rows and the number of columns is equal to uh end tokens so so that you visualize what the
20:47
inputs will look like I'll just copy paste the inputs which I showed to you before so I'll just copy paste this over
20:54
here so that this is a visual reference for you so this is the
21:00
this is the input this is the format of the input batch which is passed into this generate
21:06
text simple so we are defining a function which is generate text simple
21:11
and what it will do is that it will take model and what is model model has
21:18
been defined before model is actually let me go back yeah model is an instance
21:25
of the GPT model class so see this is the GP model class which we had coded in
21:30
the last lecture which takes the inputs and then outputs the logits
21:35
tensor um so that's the second input to the function which we are going to Define today and that function is
21:42
generate text simple so model sorry model is the first input so we have to pass in an instance of the GPT model
21:47
class we have to pass in the inputs and remember I told you about the maximum number of new tokens so we have to pass
21:53
in that as an argument and we also have to pass in the context size because the
21:59
context size specifies how many words we have to look at before predicting the next World great now let's start the
22:05
coding the first thing what we have to do is that we have to make sure that the number of tokens which the llm is
22:12
looking at at any given point is determined by the context size so let me show you what that actually means um
22:19
here I'm just taking a random example to demonstrate this to you so take a look at this example over here let's say
22:26
these are the inputs uh let's say these are the um this is the input tensor which is
22:31
given to the llm and here you will see that this input tensor actually has two rows which means there are two batches
22:37
but I want you to look at so two rows because there are two batches but look at the number of tokens so number of
22:43
tokens here are actually equal to eight so there are eight tokens here now what
22:49
if the context size is equal to five so if the context size is equal to five we
22:54
cannot look at eight tokens before predicting the next word we only can look at five tokens so this command which is it takes in the
23:01
input and then it only looks at the number of elements specific to the context size so in the code we are going
23:08
to write this Command right now so what this command will do is that it will look at the input if the input token
23:14
size which is the number of columns is equal to the context size then it's fine but if it's not it will take the last
23:20
Elements which are equal to the context size so now the context size is equal to five so it will look at the last five
23:26
tokens as an input from the first batch and it will look at the last five tokens as an input from the second badge this
23:32
is what this idx colon minus context size colon is going to do it will restrict the input so that we only look
23:39
at the number of tokens equal to context size so that's the first uh command which we have written that's idx c n d
23:47
which is condition idx condition great now what we are going to do is that we are going to pass this
23:54
input to the model so the model is the GPT class this is where all all the main functions are happening what this model
24:01
will do is that it will take the input and then it will pass the input through token embeddings positional embeddings
24:07
Dropout layer Transformer blocks another normalization layer and the final output layer and it will return this logic
24:14
tensor and remember that this logic sensor which is which is received what are the dimensions for it the dimensions
24:21
are batch comma number of tokens comma the vocabulary size this is the
24:27
dimensions of this logic sensor and then what we have to now do is that we have to extract the last row from this logic
24:33
tensor remember what we did um on the white Bard when we looked at the logic
24:38
tensor which was this tensor that is in Step One in Step number two what we have to do is that we have to extract the
24:44
last Vector from this logit tensor so now what we are going to do in the second step is that let's say if the
24:50
logit tensor looks something like this so if I have two batches this is my first batch and this is my second batch
24:58
and then then in each batch I have four tokens and then here I'm taking a vector
25:03
embedding Dimension equal to five so what we have to do is that from each of these batches we have to take the last
25:10
we have to take the last row so from the first batch we have to take the last row from the second batch we have to take
25:15
the last row and the way this will happen is through this command Logics colon minus one and colon so what this
25:22
first colon is that which means you do nothing to the batch argument but the second is minus one which means that you
25:30
look at the first batch and then you look at the last you take the last row you look at the second batch and you
25:36
just take the last row this is what we are going to do so now we are going to apply this function which will just take
25:41
the last row out of every batch in the logic stenor so logits colon minus one
25:46
colon is going to result in this where we just take the we just take this
25:53
Row from the first batch and we take the last row from the second batch and we just uh stack them
25:59
together so this is the second command here where we only focused on the last time step or the last row okay and now
26:07
when we execute this command the dimensions become the batch which are the number of rows and the vocabulary
26:13
size so one thing which I would like to clarify here is that here I mentioned this five as the
26:20
embedding Dimension right but this is actually the vocabulary size the number of columns here are equal to the
26:26
vocabulary size and what this what this function this is
26:32
equal to the vocabulary size and what this function does is that it just takes the last row so when we get this final
26:39
output the number of rows are still equal to the number of batches and the number of columns are equal to the
26:44
vocabulary size we get rid of the second dimension which was equal to the number of tokens all right so now we have this uh
26:52
these two we have the rows which correspond to the last row in every batch and the next
26:59
step is applying soft Max and converting these Logics into a set of probabilities
27:04
this is exactly what we are going to do we are going to apply soft Max and dimension equal to minus one uh which
27:11
will ensure that for every row which we have extracted soft Max will be applied Along The Columns of those rows so when
27:17
we look at each of the tokens so let's say you when we look at this batch so when you look at each batch um when when
27:25
you sum up the probabilities for each batch they will sum up to one so remember that now that the size here
27:31
is just batch size number of rows and number of columns equal to the vocabulary size so now all of these will
27:38
be transformed into values such as these and if you add up these values so for
27:44
the first batch you'll just add up these values and that will sum up to one for the second batch you'll add up these
27:49
values and that will sum up to one remember the goal is to predict a new token for every batch we have
27:55
inputed so this is the next step and then the final step is we are going to look at that index with the highest
28:01
probability value and uh this is exactly the step which we had mentioned here
28:07
also yeah so after converting the Logics into probabilities we look at that index
28:13
which has the highest value so this is exactly what we are doing in this step we look at that index with the highest
28:18
value uh that token ID and then in the last step we append that token ID idx
28:24
next to the initial uh token IDs which were stored in idx so in this last step we do the appending
28:31
part which has been mentioned over here so look at step number five over here you have to append the token ID
28:38
generated to the previous inputs for the next round and this is what is shown in this torch. cat which is concatenation
28:45
and idx next is the input is the ID which corresponds to the highest probability and that is appended to the
28:52
current Uh current indices and we you see we are in a Loop
28:58
here so the number of times we are going to do this appending operation is by the time we reach the maximum number of new
29:05
tokens these are the number of iterations remember on the Whiteboard what we saw uh on the Whiteboard we had clearly
29:12
seen that uh the number of iterations over here the number of iterations which were six iterations over here are equal
29:19
to the number of Maximum maximum number of new tokens so that's what's been written in
29:24
the code we are doing the number of iterations equal to the maximum number of new tokens and then we are going to
29:30
keep on adding these new tokens to the input tokens and that's it this is how we are
29:36
going to predict the new tokens corresponding to the next words and that's exactly what's happening in the
29:41
GPT model this is how you go from all of the complicated GPT model architecture
29:47
to predicting the next World I have just written some text over here in the
Role of softmax in next token prediction
29:52
preceding code the generate Tex simple function we use a soft Max to convert the Logics into probability distrib R
29:58
bution from which we identify the position with the highest value uh now
30:03
you might think that since we are only looking at the index with the highest value and soft Max is monotonic why do
30:09
we need the soft Max why can't we just find the index from the logits that index which gives the highest value soft
30:16
Max is monotonic so that index is going to remain same whether we apply soft Max or
30:21
not um so in practice the soft Max step is redundant which means it's not really needed to find the position with the
30:28
highest uh highest score because the position with the highest score in the softmax output is the same position in
30:35
the logic sensor so in other words we could apply the torch. AR Max here we have applied
30:41
it to the soft Max torch. AR Max we have applied to soft Max generated output
30:46
right we could have applied this to the logic sensor directly and get identical results so here I could have just
30:52
replace this with logits and the results would have been the same so then your question would be
30:58
then why are we using the softmax in the first place uh what the importance of softmax
31:05
is that we wanted to show you the full process of transforming Logics to
31:10
probabilities which can give additional intuition the probabilities give us some intuition of how much percentage
31:15
contribution does each token have in the next word prediction task and this will
31:21
also help us because uh in the next module where we'll do the GP training we
31:27
will introduce additional sampling techniques where we will modify the softmax output such that the model does
31:32
not always select the most likely token and will introduce some variability and creativity in the generated text this is
31:39
the important part the model does not always take the U output with the maximum
31:46
probability to make sure that the generated text has some variability some creativity we will explore some other
31:53
options where the softmax select some other tokens and for that definitely we need to apply the soft Max because we
31:58
need the outputs to be in some format of probabilities so although soft Max was not needed in the current code it will
32:04
be useful later when we look at things such as temperature variability in selecting the outputs Etc don't worry
32:11
about these terminologies right now I'll cover that in detail when we come to the next module now what we can do is that we
Testing the next token generator function
32:18
have written this whole function right why don't we test it on some sample text so let me take the model input as hello
32:25
I am um these this is is my model input and the reason I'm taking this input is
32:30
that this is the same input which I have used on the mirror whiteboard over here so I take this model input and uh then
32:38
what I'll first do is that I'll first encode it into token IDs so I first use my encoder to encode this model input
32:44
and convert it into a tensor so remember the shape of the input should be batch size here I have only one batch and the
32:51
number of tokens so it should be a tensor and it should be a tensor of token IDs so I have my encoder which has
32:57
been defined through tick token so if you have been following these lectures you will know that we have been
33:03
generating our encodings through tick token which is the tokenizer used for open AI models it's a bite pair encoder
33:10
so we are using that to encode this sentence and now I have generated my input sample now what we'll do is that
33:16
before passing in the model we'll first put the model in evaluation mode this bypasses some layers such as
33:22
normalization layer Dropout layer because we are not training the model here we are just evaluating so it just
33:27
just makes the model a bit more efficient and then we will just call this generate text sample
33:33
function we'll call this generate text sample function and we'll put model equal to model we have already defined
33:40
the model before and let me again take this over here so that you you have full
33:45
grasp and the GPT model configuration which we are using has been defined over here this is the configuration we are
33:51
using a vocabulary size of 50257 context length of 1024 768
33:57
embedding dimension 12 attention heads 12 Transformer blocks and dropout rate of
34:02
0.1 so this is the model which has been defined um because that's needed to be
34:08
passed into our function so I'm just writing it over here for your reference I'll code it out I'll comment it out I
34:15
will share the code so that you can run it on your own laptop okay so then we'll
34:20
run this generate text simple function we'll pass in the model we'll pass in the inputs this these are my inputs
34:26
right now remember the input is the second argument over here then we have to pass Max new tokens and context size
34:32
so let me pass in that so my Max new tokens is six and the context size which I'm I'm passing is GPT configuration
34:38
context length which is 1024 and then I'll just print out the output so I have six new tokens right
34:46
and the input was these tokens 1 5496 11 314 and 716 so now these are the inputs
34:54
and if you look at the output tensor you'll see that six new tokens have been appended over here 27018 2486 474 843
35:04
30961 42 3 48 7267 these are the six new tokens which
35:09
have been appended uh through because of our generate text simple function so these are the next Words which our GPT
35:16
model has predicted and here you can see that the output length is 10 why 10 because four were the number of input
35:23
tokens and Max new tokens were six so six additional words or tokens have been generated now we can use the decode
Analysing the next token predictions
35:30
method and based on our vocabulary and the bite pair encoder which we have used we can convert these new tokens back
35:36
into text so it seems that the next text is now some random text and the next
35:42
text is not as great as what I had written on the Whiteboard over here uh
35:48
on the Whiteboard the next text was hello I am model ready to help but here the next text is something completely
35:54
random right now why is this completely random the reason this text is completely random is that because we
36:00
have not trained the model yet the model has 124 million parameters and all of those parameters are completely random
36:06
right now those are not trained now it's just a matter of training the whole GPT architecture has been set up completely
36:13
we have implemented the full GPT architecture and initialized a GPT model instance but with random weights we need
36:19
to train these 124 million parameters and for that we have the whole NY module dedicated which are a uh next maybe six
36:27
seven number of lectures or even more but for now if you have reached this stage just be happy and proud that you
36:34
have run a 124 million gpt2 architecture model completely on your laptop you have
36:39
taken an input and you have predicted the outputs and uh this is the first step towards understanding how GPT
36:46
really works when you go to chat GPT and when you type hello I am let me let's actually do that so I'm going to chat
36:53
GPT right now and let me type hello I am and let
37:01
me say complete this sentence I'm providing no context here and does not
37:06
make too much too much sense but here you can see that based on the past interactions which I've had with chat
37:12
GPT it it results in some output which is at least quite coherent it's much
37:17
better than the output which we have received over here right but that's fine we have not implemented the training but
Recap and summary
37:23
we have essentially implemented all the nuts the bolts and the building block for building out this entire GPT
37:30
architecture on our own completely from scratch we have not used any library from Lang chain or anything we have
37:36
defined this GPT architecture fully from scratch we have learned about all the sub modules involved in the GPT
37:42
architecture we have coded all these sub modules and ultimately now we have reached a stage where given an input we
37:48
can predict an output so let us go back to the schematic which we started this lecture
37:53
with and let's see whether we have implemented everything which we we actually wanted to implement uh yeah
38:00
this is that schematic I was talking about so we have I think had six to seven lectures in this GPT architecture
38:06
module and in these lectures we have implemented every single thing which has been mentioned in this schematic all the
38:12
things we have been mentioned all the things have been covered so when you are given an input which is a text such as
38:17
every effort moves you now we are we are ready to predict the next words we have
38:23
reached the stage where we had an input and we have implemented the whole GPT architect piure to produce the output
38:29
it's just that the training has not been done yet but it's fine we'll come sequentially to that part in fact we
38:35
have learned this whole thing in U this module we first started with the GPT
38:41
backbone where the code was not implemented but we had a dummy GPT class then we implemented layer normalization
38:47
J activation feed forward Network shortcut connection we coded out the entire Transformer block after that and
38:54
in today's lecture we coded out the entire GPT architecture and we also got the final the next words given a set of
39:01
input tokens so these were a comprehensive set of lectures but if you have reached the end you should be proud
39:06
of yourself and I just want to write that
39:13
you um you did it that is what I I want to
39:19
write just to keep you motivated uh to keep on following the next lectures which are coming because
39:25
if you have reached this stage you have already reached much farther than 95% of students so it's amazing many other
39:32
students might just be using GPT but you are one of the few students who have now coded out a gpt2 architecture on your
39:39
own on your local machine which which is predicting the next word which is predicting the next token and I find
39:45
that incredibly satisfying and motivating in the next set of lectures what we are going to do is that we are
39:51
going to do the training for the 124 million parameters in the gpt2 model and
39:56
then the output which is generated will start getting much better and it will be
40:01
better and better and better so the whole goal of the next set of lectures is to make this set of outputs better
40:08
but now since the architecture is in place uh doing the next part um will be
40:14
a bit easier because we can directly work from the architecture which we have built thank you so much everyone I hope
40:21
you are enjoying these lectures I say this at the end of every lecture but I deliberately try to keep a mix of uh
40:28
very detailed whiteboard notes such as this plus the coding because I feel that students to really Master large language
40:35
models you need an understanding of theory intuition as well as detailed code I'll be sharing the entire code
40:41
file with you and I encourage you to play with this code ask doubts on YouTube um and we'll try to clarify as
40:48
much as possible thanks a lot everyone I look forward to seeing you in the next video

***
