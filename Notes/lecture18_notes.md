ulti-head attention recap
0:00
[Music]
0:05
hello everyone welcome to this lecture in the build large language models from
0:11
scratch Series this is the second part of the multihead attention lectures in
0:17
the previous part we looked at implementing multi-head attention in the following
0:22
way what we did is that we had the input tokens so let me show you this figure
0:30
which summarizes everything yeah so this was the input
0:37
Matrix which we had essentially the number of rows here represent the number of tokens which we have and each token
0:44
was encoded as a three-dimensional input embedding Vector right in the first part of the
0:51
multi-ad attention what we essentially did was we created multiple weight matrices for the query key and the value
0:59
so if we have two attention heads we'll create two weight matrices for the query
1:04
two weight matrices for the keys and two weight matrices for the values and then
1:10
we will multiply the inputs with these weight matrices to get two queries two
1:16
keys and two values so the main problem with this
1:21
approach is that here you can see that there are two Matrix multiplications which are needed as we saw earlier gpt3
1:28
used 996 attention attention heads so if you have 96 attention heads you'll need
1:33
96 multiplications to get the queries Matrix 96 multiplications to get the
1:39
keys Matrix and 96 multiplications to get the values Matrix that's not very
1:44
efficient right so today what we are going to see is that how can we make
1:49
sure that the number of multiplications are reduced in particular what if we just need to do one multiplication for
1:57
quick Keys one one multiplication for queries and one multiplication for values and then once we do one
2:04
multiplication then we can split the queries again into two parts we can split the keys into two parts and we can
2:10
split the values also into two parts for two heads and then we can perform the rest so once we get the copies or the
2:17
multiple matrices for queries keys and values what we can do is multiply queries with keys transposed to get the
2:24
attention scores then we can get the attention weights and we can multiply them with the values to get the context
2:31
vectors so the rest of the procedure can be bit similar but what if we can reduce
2:37
the number of Matrix multiplications at the start this is what we are going to look at in today's lecture so let's get
2:43
started this procedure is called implementing multi-head attention with weight splits and it's definitely much
2:49
more efficient than the multi-head attention which we saw previously so in the previous lecture the way we
2:56
implemented multi-head attention was something like this we had the the causal attention module and we
3:02
calculated the causal attention module result which is the context Vector the context Vector for every single
3:07
attention head and then we concatenated the results from the different uh
3:13
context vectors together and that led to a large Matrix this was the process now what we are going to do is we are going
Multi-head attention with weight splits introduction
3:20
to follow a slightly different procedure and that's called multihead attention with weight splits and it's much more
3:26
computationally efficient so the main idea is that in the previous code we had maintained two separate classes we had
3:33
maintained a class for the multi-head attention rapper and we had maintained a class for the causal attention and then
3:39
we combine both of them into a single multi-head attention class so here in
3:44
the in the top what you can see over here is what we did previously We performed two Matrix multiplications to
3:52
obtain the two query matrices q1 and Q2 Q2 what we are going to do right now is
3:57
what if the weight Matrix which we start start out initially itself was a larger weight
4:03
Matrix and then uh we multiply the inputs x with the query Matrix to get
4:09
the queries and then after that we split the queries into two components so here you see the
4:16
difference in the previous case we multiplied x with wq1 and we multiplied x with
4:22
wq2 but what if we multiply x with a WQ which is already a large Matrix which
4:27
consists of D out so here you see the dimensions of this initial weight Matrix are larger and that these Dimensions
4:36
already include the number of heads so this Dimension 4 is the D out which is
4:41
two multiplied by the number of heads so this D out is already specified before
4:47
so the weight Matrix for the queries keys and the values which we specify already will kind of include the head
4:53
Dimension and then we when we get the queries keys and values we'll split them based on the number of heads so here we


***

5:00
can see where we split the queries Matrix into two q1 and Q2 because there are two heads so ultimately the rest of
5:07
the procedure will remain the same but we are just reducing the number of Matrix multiplications so if you look at this
5:14
weight Matrix right now the number of attention head is specified right uh so how is it specified so D out
5:21
is equal to 4 and D out is equal to head Dimension multiplied by the number of heads so the head Dimension is equal to
5:29
2 because each head has a dimension of two this is what we had done here each head
5:34
had a dimension of two each attention head here you see D out equal to two which was implemented in the previous
5:40
case so each head had a dimension of Two And there are two heads so the D out in this larger trainable Q Matrix already
5:48
includes number of heads I'll explain to you in detail what this means right now if you just get an
5:55
intuitive idea of what we are trying to do that will be very helpful
6:00
okay so now uh let us get started with the code so we are going to implement multi-head attention with weight splits
6:07
right so instead of maintaining two separate classes so here you can see earlier in our code we had the we had a
6:15
causal attention class which did all the computations of attention scores attention weights Etc and then we
6:21
integrated this causal attention class with the multi-head attention rapper so we had a multi-head attention rapper and
6:27
we created multiple instances or multiple causal attention objects within this
6:33
rapper now the idea is instead of maintaining two separate classes why don't we combine both of these Concepts
6:39
into a single multi-head attention class also in addition to just merging
6:45
the multi-head attention rapper with the causal attention code let's make some other modifications to implement
6:51
multi-head attention more effectively so as I told you earlier in the multi-ad attention rapper which we
6:57
had earlier multiple heads are implemented by creating causal attention objects and uh the causal attention
7:04
class independently performed the attention mechanism earlier and then the results from each attention head were
7:10
effectively concatenated now we are going to implement a class which is called as multi-head attention class and
7:16
we are going to integrate the multi-head functionality as well as the causal attention functionality everything
7:22
within a single class the way we are going to do do this is that the this class splits the input
7:29
into multiple heads by reshaping the query key and value tensors let's see what this means don't worry about this
7:36
sentence in this lecture I have constructed a Hands-On example so that you understand the code which we are
7:42
about to write so first let's look at the multi-ad attention class and how we are going to Define it this code right
7:48
here which I'm showing on the screen is at the heart of the Transformer mechanism so you see we have the in init
7:54
Constructor which is invoked by default and then there is the forward method at the end of the forward Method All We are
8:01
going to do is calc calculate the context Vector for each of the input embedding vectors but what happens in
8:07
the middle that is the main key which you really need to understand okay so I
8:13
could have just taken you through this code but I have seen that if I take students through this code it becomes
8:19
very difficult for them to wrap their heads around what exactly is going on especially because if you see the
8:24
dimensions there are four dimensional tensors which are involved in this code and there is a very good reason for why
8:30
we need four dimensional tensors so if you if you just go through the code you will you will think that you have
8:36
understood it but you would not have because there are lot of subtleties with respect to the dimensions so what we are
8:42
going to do is that we are going to go to the Whiteboard and I constructed this example completely from
8:48
scratch so we are going to take a simple example we are directly going to start from the input and we are going to do
8:54
all the steps on the Whiteboard which are implemented in this code then you
8:59
will find that understanding the code is extremely easy at every step of the code I'm going to take you to the Whiteboard
9:06
and I'm going to explain to you what exactly is going on okay so let's get started I've tried to distill this down
9:12
to 11 steps and uh I I will explain everything related to matrices
9:18
Dimensions extremely clearly I will not assume anything everything is written down on the Whiteboard so that you will
9:24
not be afraid of this code I have seen several other YouTube videos and even lectures where uh people just
9:33
explain this as if it's very easy to understand but you need to decompose it into individual layers and explain every
9:39
single one of them okay so I'm going to start with the forward method and I will
Defining inputs
9:44
explain every single line here step by step so first the forward method takes the input X right let's see what that
9:51
input looks like and what it means so the first step to the attention make the
9:57
multi-ad attention with weight splits that code or the multihead attention class is that we have to start with the


***

10:04
input the way we will specify the input is that the input has three dimensions the First Dimension is the batch the
10:11
second is the number of tokens and the third is the input Dimension right what is this D in the D in is basically every
10:19
token is represented by a vector embedding so this D in is the dimension
10:25
of that Vector embedding so 1A 3 comma 6 means that I have three tokens you can
10:31
think of one token as one word for Simplicity so let's say the three tokens are
10:38
the cat sleeps let's say these are my let's say
10:45
these are my three words then what we are essentially doing here is that we
10:51
are converting each of these words we are converting each of these words into six dimensional vectors so the will be a
10:57
six dimensional Vector which is the first row so which is the first row over
11:02
here so look at the first row over here this is the six dimensional Vector for the this is the six dimensional Vector
11:09
for cat and this is the six dimensional Vector for sleep so you'll see that this is a 1x 3x 6 for Simplicity I have taken
11:17
the batch size equal to 1 okay so this is a 1x 3x 6 tensor why
11:23
3x 6 because we have three rows here and we have six columns uh each row consists of six six
11:30
dimensional vectors so I hope you have understood how the input has been defined so here you can see the input
11:36
shape is B comma number of tokens comma input Dimensions so I hope you you have understood this okay now let's come to
11:43
the next step the next step is what we have to do is we have to essentially decide two things we have to decide what
Decide output dimension, number of heads
11:49
our output Dimension is going to be and we have to determine the number of
11:55
Heads This output Dimension is basically we have the input input embedding Vector
12:01
for each token right ultimately we will get a context Vector for every token so
12:06
ultimately similar to this input embedding Matrix which is 3x which is 3x 6 we will have a context embedding
12:13
Vector which is three because we have three tokens multiplied by D
12:21
out so now we have to also decide what is D out which is the so each each token
12:28
will have a context vector what is the dimension of that Vector we have to decide so now I am deciding that D out
12:34
will be equal to 6 which is same as D in this is typically done in GPT based models the D in and the D out are the
12:40
same second thing we also have to decide is how many attention heads do we want to have so I have decided that we are
12:48
having two attention heads right now in GPT the number of attention heads are 96 and the D out is also pretty large but
12:55
exactly what we are doing right now can be scaled to a larger D out and larger number of heads okay so I'm using D out
13:03
is equal to 6 and number of heads equal to three so then each head will have a dimension which is called as head dim
13:10
and we'll look at that also later each each head will have a dimension of head dim which is equal
13:19
to head dim which is equal to essentially the D
13:25
out divided by the number of heads which is equal to 6 / 2 and that will be equal
13:33
to three so then the dimension of each head is equal to three and since there
13:38
are two heads the total D out will be equal to six okay so this is the second decision point the third decision point
Initialize trainable key, query, value weight matrices
13:45
which I have to make is that I have to initialize or it's not a decision Point rather but it's the third step so the
13:51
third step which we have to do is initialize trainable weight matrices for the key query and the value so we have
13:58
to initi w k WQ and WV okay so remember that the
14:06
input which we have which for now can be thought to be six rows and three columns has to be multiplied sorry three rows
14:13
and six columns so the input is three rows and six columns so when you construct these trainable weight
14:18
matrices for the keys query and value their first Dimension has to be equal to D in because if you look at the input
14:25
Dimension the input the number of the last dimension of of the input is D in
14:31
and for these for d for this input to be compatible with this WK WQ and WV in
14:37
multiplication you need the first argument here to be equal to D in so actually the dimensions of the
14:44
trainable key query and value matrices are D in multiplied by D out which is 6
14:50
by 6 because D in is equal to 6 and D out is equal to 6 so we have to initialize the these three vectors these
14:57
three matrices rather and I have shown these random initializations here so you can see that WQ is a six diens or a 6x6
15:05
tensor WK is a 6x6 tensor and WV is a 6x6 tensor let us let me show you in
15:13
code where these matrices are actually initialized so if you look at the code these matrices are actually initialized
15:19
in the init Constructor so w query W Key and W value these are trainable weight
15:26
matrices as you can see the dimensions are D in D out and we are using the
15:31
linear layer of neural networks with the bias equal to zero to set the uh initial
15:37
values for these why do we use a neural network linear layer because it's optimized for initializing the weights
15:43
so it's much better when we do the back propagation later so this is where the
15:49
the trainable weight matrices for query key and value are initialized in the init Constructor so it's called by
15:55
default when we create or it's these Matrix are created by default when we create an instance of the multihead
16:02
attention class all right so up till now we have essentially initialize these trainable
16:08
weight matrices w k WQ and WV Now we move to step number four step number
Calculate the key, query and value matrices
16:14
four is the step from which computations actually start so we have the input now right and uh we have the these matrices
16:23
we have trainable Keys the trainable queries and the trainable values what
16:28
we'll now be doing doing is that we will multiply the input with these matrices to ultimately get the keys the queries
16:35
and the values so what are the dimensions of the input the dimensions of the input are
16:42
one one M uh me just write this again the
16:48
dimensions of the input are 1 multiplied by 3 because we have three rows
16:53
multiplied by six columns correct and the dimensions of each of
17:00
these qu key query and value trainable weight matrices are six which is D in
17:06
multiplied by 6 which is D out so when you multiply the input with
17:12
these weight matrices the result which you'll get is 1x 3x 6 so you'll get the
17:18
keys you'll get the keys Matrix which is 1x 3x 6 you'll get the queries Matrix
17:23
which is 1x 3x 6 and you will also get the values Matrix here which is 1x 3x
17:28
six let's try to understand what this 1 3 and six is so I as I've written over
17:34
here one is the batch size which we are taken to be one three is the number of
17:40
tokens because we have three tokens and D out is basically the output Dimension
17:46
so the way to interpret these keys saries and value Matrix is that each row
17:51
basically corresponds to one token so the first row corresponds to the first token the second row corresponds to the
17:57
second token and the third row corresponds to the third token and there are six dimensions in each row because
18:04
each token is a six dimensional representation because D out is equal to
18:09
6 now let me show you in the code where this is calculated so if you go down below here
18:16
you see the keys queries and values what we have done is that we have passed in the input to this neural network linear
18:23
layer so what this does is that the trainable weight mates for the key query and value are applied on this input and
18:29
we get the keys queries and the value Matrix as we saw on the Whiteboard the shape of this is B which is the batch
18:35
size the number of rows is equal to the number of tokens and the number of columns is equal to the D out which is
18:42
equal to six okay now we move to the next step
18:47
and this is the step where four dimensional tensor start to come into the picture right so until now we have
18:53
three dimensional tensors for the keys queries and values right why the fourth dimension needs to
19:00
come into the picture is that until now these three dimensions are for batch size number of tokens and D out but
19:06
there is no dimension for the number of heads uh or the head Dimension rather so
19:12
this is where we come to next so what we are now going to do is that we are going to unroll the last dimension of the keys
Unroll key, query, value dimensions to include num_heads
19:20
queries and values to include the number of heads and the head Dimension what
19:25
this means is that if you look at this last dimension of the keys here in fact even for queries and values this is D
19:31
out right and D out is essentially number of heads into head Dimension as we have seen earlier so let me take you
19:38
yeah so here remember what we saw head Dimension is equal to D out divided by
19:44
number of heads so D out is equal to head Dimension multiplied by the number of
19:50
heads so that's what we are actually going to do we are going to unroll the last dimension of the keys saries and
19:56
values to include number of heads and head Dimension right so we have D out which is equal to 6 which is a decision
20:02
which we have made and we have also made a decision with respect to the number of attention heads which is equal to two So
20:08
based on these two decision points the head Dimension is fixed and the head Dimension will be
20:14
equal to 6 / 2 which is equal to 3 so what we are going to do next is that we
20:19
had this 1X 3x 6 matrices right for the key SAR and value now we are going to
20:24
roll them into 1 by 3 by 2 by 3 so now instead of D out we will have
20:31
number of heads which is equal to two and head Dimension which is equal to
20:40
three so let's see what the reshaped keys queries and values Matrix actually look like uh so when you reshape the
20:48
keys queries and value Matrix they start looking like this and I'll tell you how to interpret four dimensional tensors
20:54
also so for the sake of Simplicity uh let's first analyze the queries Matrix
20:59
so this Matrix is 1x 3x 2x 3 how do you Analyze This four dimensional tensor for
21:06
now forget about the first which is the number of batches okay so next look at
21:11
three so this three is the number of rows so this is my first row and uh this is my first token also
21:18
right this is my second token and this is my third token correct that's why
21:25
there there is this three now let's look at this two what is this two this two is the number
21:31
of heads so if I go in each token right now let's see if I go in first if I go in the first token the first row
21:37
corresponds to the first head and the second row corresponds to the second head that's why there is this two and if
21:43
I go within each head I'll see that there is there are three dimension the First Dimension the second dimension and
21:50
the third dimension so remember the head Dimension is equal to three so the way to interpret this Matrix is start from
21:57
the outermost value so three why three because there are three tokens then go to each token why two because there are
22:04
two heads in each token then go within each head why three because the dimension of each head is three each
22:10
head is a three dimensional Vector so in this same way we can analyze the queries the keys and the
22:16
value Matrix also so the keys Matrix will also be uh 1X 3x 2x3 and the values
22:24
Matrix will also be 1X 3x 2x 3 as I mentioned before let me repeat it again
22:29
each row over here is a token so if you look at the value Matrix let's look at the second row the second row
22:35
corresponds to the second token if you now look at the first row of the second
22:41
row this that is the threedimensional head Vector for the first head if you
22:48
look at the second row that's the three-dimensional head Vector for the second head remember every token has two
22:54
attention heads so you can think of as two people paying attention to each token token because we have two
22:59
attention heads that's why there are two rows corresponding to every token now if we come to the code this
23:06
line has been mentioned over here so see we have to unroll the last Dimension so now the D out will be replaced with the
23:14
number of heads and head Dimension so this is exactly what has been done over here keys. view so now keys will be
23:21
replaced with keys do view B common number of tokens common number of heads and head Dimension this is exactly
23:29
uh what we we just saw on the Whiteboard so in this step the three dimensional tensors have been converted into four
23:35
dimensional tensors to include the number of heads and the head Dimension great now we move to the next step so if
23:43
you see uh if you see this let's look at this argument which is
23:48
three um so the shape of this is 1A 3A 2 comma 3 right now let's look at this
23:55
first this this entry this is three now this three is the number of tokens
24:02
which means that currently these matrices are grouped according to number of tokens right so I'm saying that this
24:08
is the first token this is the second token and this is the third token and then I further dive into
24:15
number of heads and the dimensions in each head but it turns out that later when we want to compute the attention
24:21
scores the only way the computation can proceed ahead is if we Group by the number of heads so instead of grouping
24:27
by the number number of tokens I actually want to group by the number of heads and we have two heads here right
24:34
so I want to flip these Dimensions I want to flip these Dimensions here so that the first row
24:41
represents the first head the second will represent the second head and each will have a 3X3 let me show you what I
24:47
mean so now what we are going to do is we are going to group The matrices by the number of heads okay so currently
Group matrices by number of heads
24:54
the keys queries and the values Matrix have the dimensions of one which is the batch three which is the number of
25:00
tokens two which is the number of heads and three which is the head Dimension so
25:06
we are grouping with respect to the number of tokens but now I want to group with respect to number of heads so again
25:13
I want to switch the dimensions to be I want to switch this this
25:18
to uh let me write it again yeah I want to switch this two
25:24
with three and this three should come over here so I want the the matri to have the dimensions of B comma number of
25:31
heads comma number of tokens and head Dimension so I want the dimensions to be 1 comma 2 comma 3 comma 3 so what I'm
25:38
going to do in the code also you'll see we are going to transpose keys quaries and value and we are going to transpose
25:44
one comma 2 now why do we do one comma 2 over here because python has zero indexing so index zero is this since we
25:51
want to Interchange the number of tokens and the number of heads the indexes which we need to transpose are index
25:57
number one and index number two that's why we are doing Keys queries and value and transpose 1A 2 so let's see what the
26:05
result actually looks like so when you when you make the 1A 3A 2A 3 to 1A 2
26:11
comma 3A 3 now the transposed queries keys and Valu start looking like this
26:16
and now you will see that they are grouped by head so the first thing what we can do is that let's look at this
26:21
block in the queries so we are analyzing the queries Matrix now the shape of the queries Matrix is what
26:29
1 comma let me write it here again 1 comma
26:35
2 comma 3 comma 3 right that is the essentially
26:45
the shape of the queries Matrix and we are going to analyze this so let's start with this two which is the number of
26:52
heads so if you look at the first block here uh let me erase this right now now
26:58
and then draw it again yeah so if you look at the first block over here which are marking with these curly braces
27:04
that's the first head if you look at the second block here that's the second head so see now this two because this two
27:11
comes over here now we can group with respect to number of heads so the first block shows everything with respect to
27:17
head one and the first row over here is the first token the second row over here is the second token and the third row
27:23
over here is the third token similarly if you look at head number two the first row is the first token the second row is
27:30
the second token and the third row is the third token so now we have the dimensions as
27:37
number of tokens and head Dimensions come last so each token if you see each
27:42
token has a three dimensional Vector because the head Dimension is equal to three so the the reason this helps is
27:49
because since we can now group with respect to heads we can compute the attention score for each head separately
27:56
so remember there is one attention score there are there is an attention score Matrix for the head one and there is an
28:02
attention score Matrix for the head two and then we we are going to U concatenate them together right so it
28:08
makes sense to group with respect to the head and that's why this step exist this keys. transpose it's very difficult for
28:16
students to understand this unless you see this visual example of why we are essentially doing this transpose the
28:22
main reason we do do this transpose is that here you see we are grouping with respect to uh
28:29
we are grouping with respect to number of tokens here but that's not good if you want to compute the attention scores
28:35
for each head parall so we group with respect to number of heads so that's why it's important to flip number of tokens
28:41
and number of head Dimension and that's exactly what we have done okay now let's
Finding attention scores
28:46
go to the next step the next step is to find the attention scores so remember now we have the uh we have the queries
28:54
Matrix we have the keys Matrix and we have the values Matrix in exactly the shape which we want so now we can do uh
29:01
we can go ahead and find the key queries and the keys transpose to get the
29:06
attention score so let me show you how this is done first let me rub all of
29:13
this
29:20
okay okay so now I have rubbed all of this so what we are now essentially going to do is that um this is the
29:28
head number one right this is the head number one of the queries and this is the head number one of the keys so what
29:35
this this shape will help us do is that when we do queries multiplied by Keys transpose it will
29:41
directly uh take the equivalent product of head one of the queries with head one
29:46
of the keys and then head two of the queries and head two of the keys but remember when we take the keys
29:53
transpose what's really important to us is that now the shape of the keys is B B common number of head is common number
29:59
of tokens and head Dimension so what it's really important to us is number of tokens and head
30:05
Dimension so remember the formula for calculating the attention score is
30:11
queries multiplied by Keys transpose right so here also we are going to do queries with respect to Keys transpose
30:17
but what exactly do we have to transpose we have to transpose uh we have to transpose this
30:24
so we have to transpose the last two dimensions and let me show you what that transposed key Matrix looks like yeah so
30:31
this is the transposed key Matrix now here you can see the key Matrix uh if you see the first row it's
30:39
4143 -1. 423 and - 2.71 31 right so when we do
30:47
Keys transpose Keys transpose 2 comma 3 it will transpose along the last two Dimensions so now that that row which we
30:53
saw has now become a column over here so this is the keys transposed
30:59
and here is the queries Matrix and I've shown the keys transpose over here so the queries matrix dimensions is 1A 2A 3
31:06
comma 3 the keys transpose Dimension is 1A 2 comma 3 comma 3 so they they they
31:11
are compatible for multiplication and the way the multiplication will now proceed is that the head one will only
31:18
be multiplied by the head one of the keys transposed the head two here will only be multiplied with the head two of
31:24
the keys transposed and ultimately when we do this multiplication we we will get the attention scores Matrix so this is
31:31
the attention score Matrix which we have and the dimensions of this are B number
31:37
of heads number of tokens and number of tokens let me show you why
31:43
um okay so if you look at what we are multiplying here the query's dimensions
31:48
are B comma number of heads comma number of tokens comma head Dimension right and when we do Keys
31:54
transpose uh 2 comma 3 the dimensions here are B number of heads head
32:00
Dimensions comma number of tokens so essentially you can think about it like we are multiplying two matrices with the
32:05
dimensions number of tokens comma head Dimension multiplied by head Dimension number of tokens so what will the
32:12
resulted Matrix will have number of tokens rows and number of tokens columns and the first two Dimensions here will
32:18
stay the same because they are the same in both of these matrices we are multiplying so the resultant attention
32:24
scores will have the dimensions of B number of heads number of tokens and number of tokens it's fine if you forget
32:31
these Dimensions but you should be able to interpret what is going on here so let's see what is going on here remember
32:38
we have we are grouping with respect to head so that stays the same this first uh this first block which I've
32:44
highlighted right now that is head number one and the second block which which I've highlighted right now that is
32:50
essentially head number two this is the first thing to understand okay then what we are doing
32:56
when you look at the let's look at head number one for now if you look at the first row the first row essentially
33:03
consists of the attention score between of the first word with all the other words right so remember our sentence was
33:12
the actually let me write it over here that will be much better so our sentence
33:19
was the the cat
33:28
the cat and here it was sleeps
33:35
right the cat let me just write it over here yeah
33:43
the cat sleeps and the same words I'm also going to write over here so the first row
33:49
is let me write it over here actually the first row is
33:56
the the second row is
34:02
cat and the third row is
34:08
sleeps so that's why the final two dimensions are number of tokens comma number of tokens because if you look at
34:15
the second row now if you look at the second row now the first element of the second row tells us information about
34:22
the attention between cat and the the second element of the the second
34:28
row tells us the information between cat and cat so if the query is cat how much
34:33
attention should you pay to cat the third element here tells us the information between cat and sleep which
34:39
means if the qu if the query is cat how much attention should you pay to sleep so that's why the shape of the attention
34:45
Matrix for every head is number of token rows and the number of token columns because an attention score exists
34:51
between each token for every other token so whenever you see these Dimensions right don't get confused by it try to
34:57
always understand the meaning behind it that's why we had so many lectures on the attention mechanism before just so
35:03
that when we reach this stage understanding all of this becomes easy so remember until this stage we have
35:09
computed the attention score so this is exactly what is done here remember what we saw on the Whiteboard to compute the
35:16
attention scores we'll take the queries and we'll multiply with the keys. transpose 2 comma 3 because uh in
35:24
transposing 2 comma 3 we'll make sure that the correct queries and the attent
35:29
and the KE transpose product is taken to calculate the attention
35:35
scores and uh this is also implemented in the code so if you see in the code the attention score is the product is
35:41
the scaled product between queries and the keys great so here it shown dot product
35:48
for each head now you'll understand why I'm saying each head because as I showed you before each head has number of
35:54
tokens comma number of tokens attention scores and for for one head it's here and for the second head it's below okay
Finding attention weights
36:01
now we come to the next step the next step is to essentially find the attention attention weights okay so uh
36:09
if you look at this attention score over here right now you'll see that for every token there is an attention score with
36:15
respect to every other token right but that's not what's the mechanism in causal attention what causal attention
36:21
says is that when you look at the you should only look at the attention score between the and what comes before it so
36:28
the and the all the other elements here so let me show them with a different let
36:33
me first rub this uh so
36:39
that yeah so what causal attention mechanism dictates is that when you look
36:44
at the first word which is the only the attention score between the and what
36:49
comes before it should survive so all of this should go to zero if you look at
36:55
cat only the attention score of the Words which come before so the and Cat should survive this should go to zero
37:01
and when you look at sleeps attention scores of all Will Survive because all the words come before it this is what we
37:07
are actually going to implement next so to do that first what we are going to do is we are going to take the
37:13
attention scores which we have and replace all of the elements above the diagonal with negative
37:19
Infinity the reason we uh replace this with negative Infinity is because after
37:24
this point we are going to implement the soft Max function so that each row sums up to one and when we Implement soft Max
37:31
whatever is there in the infinity will automatically go to zero so it will kill it will kill two birds in the same Stone
37:37
we will implement the causal attention mechanism and we'll also make sure that all the rows sum up to one but before we
37:44
Implement soft Max we do one more thing we divide every single element here with the square root of the head Dimension
37:51
and this when we looked at the lecture for uh self attention we saw why this is done this is essenti to make sure that
37:59
the variance between the when we take the dot product between the queries and the keys the variance scales up with the
38:06
number of dimensions and to prevent the variance from blowing up we have to divide by the square root of the head
38:12
Dimension this also makes sure that the values in
38:17
the values before we compute the soft Max are not very high and that's generally useful for back propagation
38:24
and leads to stable gradients so what we'll be doing is that the head Dimension as we saw is three right
38:31
because the D out is equal to 6 and the number of heads is equal to 2 so each head Dimension is equal to three so
38:38
we'll divide this after replacing the elements above the diagonal with
38:43
negative Infinity we'll divide this with square root of 3 which is square root of head Dimension and that leads to this
38:50
Matrix over here or this tensor I should say and then we apply soft Max to this
38:55
tensor so we make sure that every row here sums up to essentially so if you
39:00
look at each row in this you'll see that it it's summing up to one and the reason it sums up to one is we are applying
39:07
soft Max so now I can make claims interpretable claims so when I say that when I look at the second token cat I
39:13
should pay 96% attention to the and I should pay 4% attention to cat when I
39:19
look at sleeps I should pay 4% attention to the I should pay 26% attention to cat
39:26
and I should pay pay 69% attention to sleeps remember these values are not optimized but when they are optimized uh
39:34
when we look at back propagation later uh the fact that these values sum up to one will carry meaning because we
39:41
can make interpretable statements such as what I was making right now remember that after we are going to apply soft
39:47
Max uh the attention weights have exactly the same dimensions as the attention scores which is going to be
39:54
the batch size number of heads number of token tokens and number of
39:59
tokens so this is the dimension of the attention weights which is 1A 2A 3A
40:06
3 uh so if you look closely to go from attention scores to attention weights we
40:11
actually M we actually have very we have a rich number of steps and it's
40:17
important for you to understand all of these first what we did is we applied a mask so that all elements above the
40:23
diagonal are negative Infinity then we divided by the square root of the the head Dimension then we applied soft Max
40:30
this is how we got the attention weights now let's see how that is done in the code H before that one thing usually we
40:37
can also Implement Dropout after this so you can mention a dropout rate which is actually one of the arguments in the
40:44
multi-ad attention class but here I'm not implementing Dropout for the sake of simplicity so if you look at the code
40:51
here we have got the attention scores the first step as I said is to create this mask and and then apply this mask
40:58
to the attention score so that all the elements above the diagonal are negative Infinity that's what this step is doing
41:05
uh here the mask actually has also been defined over here see this is the upper
41:10
triangular mask which is all the elements about the diagonal one then they are replaced with negative infinity
41:16
and that's applied to the attention scores so this will ensure that all the elements above the diagonal of the
41:21
attention scores are equal to negative infinity and we are only considering context length here why context length
41:29
because let's say context length is three it means that maximum if three words are given we can make prediction
41:36
of the next word so when we implement this Dropout mask we only Implement a mask of context length comma context
41:43
length there is no point in implementing a bigger mask because anyway we are not going to look at more tokens than the
41:50
context length at a time and if it happens that we are looking at a batch where the number of tokens are less than
41:56
the context size this statement makes sure that then the mask stops at number of tokens but this is a detail which
42:04
probably uh you can Overlook right now if you're understanding all the other things that's what the most important if
42:10
you understand this Minor Detail it's awesome then the next step is to apply soft Max but as I told you before
42:17
applying soft Max we defi we divide every element with the square root of the head Dimension if you look at the
42:23
keys. shape let's look at keys. shape uh this is going to be the keys do shape so
42:29
keys do shape of minus one which means that we are going to look at the last Dimension which is the head Dimension so
42:36
we are essentially dividing by square root of head Dimension here and then we apply the soft Max y Dimension equal to
42:43
minus one because we need to make sure that all The Columns of a row sum up to one and then as I said we can even
42:50
Implement Dropout if needed towards the end so up till now we have reached a stage where we have obtained the
42:58
attention weights is basically and I hope you understand the meaning behind this final attention weight matrix it's
43:05
not just important to understand how the dimensions work so to make sure you understand the meaning let me go through
43:11
the meaning of this attention weight Matrix once more um this what I'm
43:16
highlighting right now is the attention weights for the first head this second block is the attention weights for the
43:21
second head in each attention uh head block you will see that the size is
43:28
number of tokens rows and the number of tokens columns so each value is essentially the
43:35
attention weight between let's say this is the attention weight between this is the attention weight between the second
43:40
row which is the second token and the second token this is the attention weight between uh the third token as the query
43:49
and the first token as the key so basically you'll see that every single
43:55
element here has some meaning it essentially encodes the attention weight between the query and the particular key
44:03
okay now let's go ahead the last step which we are going to implement is that
Finding multi-head context vectors
44:08
we have to calculate the context Vector remember the aim of all the attention mechanisms is to ultimately compute the
44:14
context Vector Matrix and this is exactly what we are going to do and to compute the context Vector Matrix we
44:20
take the attention weights and we multiply them with values remember the value Matrix was The
44:27
Matrix which we had computed earlier let me show you where the value Matrix was in case you have forgotten it because we
44:33
have done so many things uh yeah so here was the value
44:38
Matrix which we had computed we have not used it until now it will only be used in this last
44:44
step okay so the attention weights will be multiplied by the values Matrix to get the context Vector Matrix so let's
44:50
see how the dimensions work out here okay the attention weights as we looked earlier over here
44:57
the attention weights have the dimensions of B comma number of heads
45:03
comma number of tokens comma number of tokens and as we saw earlier the values
45:08
Matrix has the dimensions B comma number of heads uh comma number of tokens and head
45:15
Dimension so effectively let's see whether these matrices can be multiplied so this is number of tokens and number
45:22
of tokens and that will be multiplied by number of tokens and head dim so the
45:27
number of columns here is number of tokens and the number of rows here is number of tokens so the number of
45:34
columns in the first Matrix are matching the number of rows in the values Matrix
45:39
so we can see that these two matrices can essentially be multiplied so multiplication is possible and now let
45:46
us see how the multiplication will actually work in practice this is the final attention weights Matrix here it's
45:53
mentioned attention scores but I should have called it attention weights remember there is a difference between
45:58
scores and weights attention weights in in the attention weights each row sums up to one that's not the case with
46:04
attention scores okay so this is the attention weights and this is my values
46:11
this my values Matrix so this 1A 2A 3A 3 and this 1A 2A 3A 3 and when we multiply
46:19
the resultant output will be B comma number of heads comma number of
46:24
tokens uh comma the head Dimension so here you can see that the
46:31
context Vector output is B comma number of heads comma number of tokens comma
46:36
head Dimension which is 1 comma 2 comma 3 comma 3 Let's interpret this again uh
46:42
so here you can see that there are two heads so this is head number one and this is head number two and in each head
46:48
there are number of tokens so if you look at each head there are three rows so each row corresponds to one token but
46:55
now if you look at what what each row represents each row represents the context Vector for that particular token
47:02
and it has the dimensions equal to head dim because head dim is equal to three
47:08
so that's the meaning of this uh context Vector Matrix which we have reached but
47:14
now remember there is a problem here right or not a problem uh but we have to
47:19
somehow merge this number of heads and head Dimension back together because the
47:25
resultant context Vector Matrix remember what we saw earlier the let me scroll up
47:30
a bit so if you if you if you looked at the goal which we had when we started
47:36
this lecture the goal was that the resultant context Vector Matrix should
47:42
have the dimensions of uh yeah as I mentioned to you the
47:48
goal was that the resultant context meor Matrix should have D out right as the
47:54
dimension so we should again pull back the head Dimension and the
47:59
number of heads together so that we can get the resultant Matrix which has the D out Dimension
48:05
preserved uh whereas let's see what we have obtained until
48:10
now well until now the context Vector Matrix which you have
48:20
obtained yeah the context Vector Matrix which we have obtained has number of heads and head dimensions in separate
48:26
positions so first what we'll do is that we'll bring them closer together so that we can then merge them to get the D out
48:33
so what we are going to do is now we are going to swap this this number of tokens
48:38
index with the number of heads index so that the dimension of the context Vector Matrix is so that the
48:46
shape of the context Vector Matrix is changed so the next step is basically
48:51
step number 10 and that is to reformat the context vectors So currently the
48:57
context Vector shape is B comma number of heads comma number of tokens comma head Dimension right and I want the
49:02
number of heads to come here so that they're closer to the Head dim and I want the number of Tok number of tokens
49:08
to go here so I want the resultant Matrix to be B comma number of tokens
49:15
comma number of heads comma head Dimension so essentially what I will do
49:22
is after I compute the context Vector Matrix I'll do a transpose of the first
49:27
index and the second index and so the resulting context Vector Matrix now which has the
49:33
dimensions of B comma number of tokens comma number of heads comma head Dimension looks like this so here you
49:40
see now the interpretation is different now this is my first token now the grouping is with respect to tokens this
49:46
is my second token and this is my third token and in each token there are two heads so if you look at the first token
49:52
there are two heads and if you look at the first head this is the vector with respect to the first head the context
49:58
Vector context vector and the second row is the context Vector with respect to the second head for the first
50:05
token now let's see how all of this is implemented in code actually all of what we saw right now is just implemented in
50:11
one line of code but to understand this we really have to understand first of all how the attention weights are
50:16
multiplied with values the multiplication really makes sense and why do we do this transpose 1A 2 the
50:23
reason we do this transpose 1 comma 2 is to get the context Vector mat in this shape the reason we get it in this shape
50:29
is now you can see the number of heads and head Dimension are closer together so we can merge them um into the D out
50:37
more easily so here you can see this is what we have reached until now where the
50:42
context Vector is obtained and it's in the correct format now the last step what we have to do is that we have
50:50
to um let me show you the last step what we have to do is
50:56
essentially we have to combine the results from multiple heads so see this is the context Vector Matrix which we
51:02
have obtained right now right so if you look at the first token which I've highlighted over here this is the first
51:08
head and this is the second head now what I will do is that when I look at the first token I will combine these two
51:14
together into one row so that it will be uh six the
51:19
dimension will be six so so here these are three and these are three right so
51:25
I'll combine the outputs from both of these heads into one output so let's see how this looks like so then the first
51:31
row will so then we'll flatten this is called flattening will flatten each token output into each row so the head
51:39
one and head two outputs are combined together so if you look at the final output the first row consists of merging
51:46
of the two heads for the first token the second row consist of the merging of the
51:52
two heads for the second token so for the second row we merge these two out outputs into one single row and for the
51:58
third token we merge these two outputs into a single row so you'll see that the
52:04
F the this is the third row so this now what I what I'm showing on the screen here is my final context Vector Matrix
52:12
and how to interpret this if you look at the first row the first row is the context Vector
52:19
context Vector for the first token first row is the context Vector for the first token why does it have six elements
52:25
because d out is equal to 6 the second row is the context Vector for the second token and the third row is the context
52:32
Vector for the third token so overall you see we first split the D out into number of heads and head Dimension and
52:39
now we brought it back together to get the D out so in the final shape you will not see the number of heads it's all
52:45
merged into this D out so this is my final answer right now and the shape of
52:51
this is 1A 3 comma 6 which is B comma number of tokens comma D out
52:57
so this is exactly what is done here what we do is that we uh we take this
53:02
context vector and we reshape it into B comma tokens comma D out why this
53:07
continuous is needed is because we want to make sure that when we reshape matrices they are in the same blocks of
53:13
memory so when we reshape uh tensors let's say and if they're in different memory blocks it becomes difficult so
53:19
first we make sure that using this continuous they in the same memory block then we reshape them so that the final
53:26
output is B which is the batch size number of tokens which is equal to three in the example we saw and D out which is
53:33
the output Dimension which is equal to six and then there is an optional projection layer towards the end so if
53:39
you look at the out out output projection it's again a linear layer and whose parameters can be learned this is
53:46
not really necessary but sometimes it is implemented in practice now this is exactly the entire procedure for how the
53:53
multi-ad attention is implemented from scratch and here we saw the multi-head attention for the example which we have
54:01
so the first token is again the the the second token is
54:11
cat and the third token is sleeps me write this
54:18
again yeah the third token is sleeps so you see through this entire procedure we obtained the enriched context Vector
54:25
representation for these tokens similarly when you deal with large volumes of text you take sentences you
54:32
break them down into tokens then into token IDs then into input embeddings and similar to this procedure you get
54:39
context vectors for each token which you have ideally when we run the actual code
54:44
we will have multiple batches but I showed only one batch right now for Simplicity so uh to whoever who have
Hands on example testing
54:52
reached until this stage I want to say that thank you for following with me for so to many lectures I know these
54:57
lectures are becoming very long but unless I explain every single thing in detail it's very difficult for you to
55:04
understand all the details so congratulations if you have reached this lecture you have successfully understood
55:11
how the multi-head attention works and I think there are very few people who really understand this entire piece of
55:16
code block by block okay so I'll share this notebook
55:22
with you and whatever I explain to you on the Whiteboard all the steps which which I laid out in front of you on the
55:29
Whiteboard uh all of those have been explained here as step one to Step 11
55:34
and uh I have explained added a detailed explanation of the multi-head attention class in today's lecture I did not just
55:40
want to read this but I wanted to construct a practical example to show you how the dimensions actually work and
55:47
uh it took me a long time to make this example but now I think it's worth it because it really helped me explain it
55:53
and I hope you understood it better so now we can actually test out the multi-head attention class so here are
55:59
my inputs uh as I showed you on the Whiteboard we have three tokens and we
56:04
have six the embedding Dimension is six the only change here what I'm going to
56:09
do I'm going to create a batch so I'm going to going to create a batch of two such inputs and I'm going to stack this
56:15
batch on top of each other so I'm going to assume a d out equal to six exactly
56:20
what we saw on the Whiteboard and context length equal to six and then we are going to implement the multihead
56:26
attention class so D in is equal to 6 uh D out is equal so D in is equal to six
56:34
right because each um each token has the input embedding dimension of six D out
56:40
equal to 6 the context length which we are using is equal to uh six then
56:48
uh yeah the dropout rate which we are considering is zero we can even include
56:53
the dropout rate so the dropout rate will change this this last layer of Dropout and randomly block out some
57:00
attention weights this is good for generalization and the number of heads equal to two so we create an instance of
57:06
this class and then create the context Vector Matrix and you'll see for the first batch the context Vector Matrix
57:12
has three rows and six columns let's see if the shape matches what we had seen on the
57:18
Whiteboard uh okay so let me scroll down
57:24
below yeah this is the final context Vector Matrix which we had
57:31
obtained yeah so this also had three rows okay I think I need to scroll up a
57:37
bit yeah this is the final context Vector Matrix which we obtained and this also had three rows and six columns the
57:44
values might be different because we have done the initializations differently here I have taken random initializations in the python code there
57:51
are some other initializations every time you initialize we take from a goian distribution so the values might be
57:57
different but let's check the shape so this is 3 comma 6 three rows and six columns and here also we can see that
58:03
three rows and six columns awesome so the shape matches but you'll see that since there are two batches here's the
58:09
context Vector Matrix for the first batch and here's the context Vector Matrix for the second batch so the
58:16
multi-head attention class which we defined is extremely powerful because it can also handle multiple batches at once
58:22
we can even do 50 data batches and then it will just have one 1 two it will have
58:27
50 such context Vector matrices okay that's it that brings me
58:33
to the end of this section or the end of this lecture so in this lecture we implemented the multi-head attention
58:39
class that we'll be using in the upcoming lectures to implement and train the llm this code is fully functional but we
58:47
we used relatively small embedding sizes and number of attention heads to keep the outputs readable so as I showed you
58:53
we only use two attention heads but gpt3 actually was 96 attention heads so the
58:59
smallest gpt2 model had 12 attention heads and a context Vector embedding size of 768 the largest gpt2 model had
59:07
25 attention heads and a context Vector embedding size of 1600 and gpt3 has even higher so the
59:14
gpt3 largest model has 96 attention heads I think and generally in GPT
59:19
models the D in is equal to D out in the example which we saw D in was equal to D out equal to 6 but here the D in and D
59:27
out are much larger around 768 Etc again thank you so much everyone for reaching
Conclusion
59:34
the end of this attention series it's been one of the longest and most comprehensive series which I have
59:40
covered and uh I really enjoyed learning about all of these things I can see that
59:46
many llm practitioners cannot understand these Dimensions or they do not take time to go through understanding the
59:53
theory the building blocks behind how the attention mechanism Works they just implement the code bases which are
1:00:00
available which completely abstract away all of these things so I don't think
1:00:06
that's the good way or the correct way to learn about large language models if you want to be a true llm engineer or a
1:00:11
machine learning engineer you have to understand how nuts and bols work otherwise you might be able to deploy
1:00:18
applications but to make real inventions you will have to go into the code base change a few things understand
1:00:24
Dimensions as you might have seen and I have stressed this many times dimensions and linear algebra are at the heart of
1:00:30
becoming a very strong ml engineer it all comes down to Dimensions other students might be scared of a
1:00:36
four-dimensional tensor right but if you understand how it works based on what I showed to you on the Whiteboard my aim
1:00:42
is that you should not be scared of these higher dimensional matrices once you write it down and once you
1:00:48
understand what's going on it really becomes easy that's why I really recommend writing things down you can
1:00:54
write on a whiteboard you can even write on a piece of paper but make sure you write things down then you'll remember
1:00:59
them for a longer period of time I hope you all are enjoying these lectures thank you so much everyone and look
1:01:05
forward to seeing you in the next next lecture where we'll actually start building the llm model thanks a lot

***

