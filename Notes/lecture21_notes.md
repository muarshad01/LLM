#### GELU Activation Function
* Feed Forward Neural Network
* We'll implement a small NN sub-module that is a part of LLM transformer block.
* Two activation functions commonly implemented in LLMs
1. GELU -> $$x^{*}\phi(x)$$ --> CDF of standard Gussian distribution
2. Swi GLU


#### Dead Neuron Problem
* dead neuron problem which means that if the output from one layer is negative
and if RuLU activation function is applied to it the output becomes zero and then
it stays zero because uh because we cannot do any learning of after that so the neurons which are associated with that particular output they don't contribute anything to the learning process once the output of the neuron becomes negative and that's called as
the dead neuron problem so learning essentially stagnates of course. RuLU has a huge number of other advantages this nonlinearity which is introduced over here makes neural networks expressive it gives the power to neural networks but the reason we we are looking at the
disadvantages of ReLU is that understanding the disadvantages of RuLU will open an opportunity for us to learn about the GELU activation function and why it is used in LLMs?

* 5:00 


so first let's start understanding about
6:09
the mathematical representation of the
6:12
uh JLo activation function so
6:14
mathematically the J activation function
6:17
is the product of X which is essentially
6:20
just the identity variable so J of x
6:23
equal to X into this 5 of X and
6:26
essentially f of x is the cumulative
6:29
distribution function of the standard
6:30
goian
6:32
distribution so I just have this opened
6:34
door here so a standard goian uh
6:38
cumulative distribution function looks
6:40
like this as a function of X so the G is
6:43
a product of X multiplied by this and
6:46
then uh if you
6:49
actually uh try to understand what is
6:51
happening so let's look at this five of
6:55
X for X greater than zero so if x is
6:58
very high you see that it's almost equal
7:00
to one so which means that for positive
7:02
values of X we are slowly tending to one
7:05
which means that the for very positive
7:08
values of X the G of X will tend to X
7:10
into one which is X so for very high
7:13
values of X this will almost tend to the
7:15
linear function which is quite similar
7:18
to the positive branch of Ru but what
7:20
happens to the negative values of this
7:23
is pretty interesting the negative
7:24
values here you can see that they are
7:26
not equal to zero so X will be
7:28
multiplied with these negative values
7:30
and that's why for negative values of x
7:32
g will not be zero like it is in The Rao
7:35
activation
7:36
function so now what we can do is that
7:39
instead of using this complicated
7:41
cumulative distribution function what uh
7:44
people generally do is that they use an
7:47
approximation for the JLo activation and
7:50
the approximation which was actually
7:51
used for training gpt2 looks something
7:54
like this so here you can see that guu
7:56
of X is equal to5 * x * 1 + tan hunk of
8:02
2 byk into x + this cubic term no need
8:07
to worry about this term but just know
8:09
that instead of worrying about this 5 of
8:11
X which is the cumulative distribution
8:13
function and there is also goian
8:17
involved and it's a bit difficult to
8:18
compute f of x it's better to use
8:21
numerical approximations right so when
8:23
gpt2 was trained the J function which
8:25
they actually used was was this
8:27
approximation which is very close to the
8:29
actual J function right now if you want
8:32
to compare this with the ru function
8:34
here I have shown the plots of the J
8:37
activation function along with the Rao
8:39
activation function I want you to pause
8:41
the video for a while here and try to
8:43
look at the similarities and differences
8:45
between these
8:48
two okay so the first thing which should
Why do we use GELU?
8:51
immediately be clear to all of you is
8:53
there are lot of differences for
8:54
negative values of X so for X less than
8:57
Z you can see that the Galu activ
8:59
function is not really zero so if you
9:01
zoom into this further you'll see that
9:03
for most of the values of X it's not
9:05
zero it tends to zero but it is not zero
9:08
whereas The Rao activation function for
9:10
X less than 0 was Zero
9:13
throughout also if you look at the
9:15
positive values of X you'll see that
9:17
this is kind of not exactly linear here
9:20
for short values of X but for large
9:22
values of X it's fully
9:24
linear uh which is exactly what's
9:27
happening for The Rao activation
9:28
function so although this positive side
9:30
of the jilu looks like yal to X it's not
9:33
exactly y equal to X there are some
9:36
minor
9:37
differences but you can say that for X
9:39
greater than Z it's almost similar to
9:41
the ru for X greater than 0 but for X
9:44
less than 0 there are big differences
9:46
which start to emerge which actually
9:47
make Jou much better than the
9:50
ru uh so what are the advantages of the
9:53
Jou or the ru activation function well
9:55
the first Advantage which you
9:57
immediately see from this graph over
9:58
here is you can see that J activation is
10:01
smooth throughout right here there is a
10:03
discontinuity in Ru there is a


***

10:05
discontinuity at x equal to zero which
10:06
makes it not differentiable J activation
10:09
on the other hand is smooth throughout
10:11
so it's differentiable across all X
10:13
that's the first Advantage the second
10:15
Advantage is that it's not zero for
10:17
Negative X so that solves the dead
10:20
neuron problem even if the output of a
10:22
neuron after is negative even if it goes
10:25
through J it will not become zero so the
10:28
neuron won't become dead it will still
10:30
keep on contributing to the learning
10:31
process that's the second reason so
10:33
first reason is differentiability second
10:36
reason is it prevents the dead neuron
10:38
problem and third reason is that it just
10:40
seems to work better than Ru when we do
10:43
experiments with
10:44
llms so as always activation functions
10:47
are hyperparameters right so we need to
10:49
test out multiple activation functions
10:51
to see which one performs better and we
10:53
have generally seen that JLo performs
10:55
much better in the context of large
10:57
language models compared to Ray so now
Coding the GELU activation class
10:59
what I want to do is that first I want
11:01
to go to code uh and I want to uh write
11:05
a class for the JLo activation function
11:07
write a create a class for the J which
11:09
implements the uh forward pass which is
11:12
that it essentially implements the this
11:15
function which I showed over here so if
11:17
you look at this
11:18
function whenever J receives an input X
11:22
it it transforms it into this through
11:24
this function and then we get this
11:26
output as shown in this JLo activation
11:28
function so what I'm doing here is that
11:30
I'm creating a class called J Loop and
11:33
what I'm doing here is that I'm defining
11:35
a forward method which takes in an input
11:37
X and it returns this value it returns.
11:40
5 into X into 1 + tan H square root of 2
11:44
by pi into x +
11:46
0.044 into x 3 this is exactly what has
11:50
been um written over here in this black
11:54
box over here the reason we are using
11:56
this is because the same activation
11:57
function was used for training G
11:59
gpt2 and remember when we are
12:01
constructing the llm architecture here
12:03
we are mimicking the parameters used in
12:06
the smallest model of
12:08
gpt2 so now we have created a class for
12:10
the Jou activation awesome so here I've
12:13
written a simple plot function we just
12:15
plots the JLo and the railu activation
12:17
function and it Compares them side by
12:20
side we already looked at these two
12:21
plots on the Whiteboard and we saw the
12:23
similarities and differences between
12:25
them so here I have just summed up the
12:27
points because of which the J activation
12:29
function is used in llms so as we saw
12:32
the smoothness of the JLo can lead to
12:34
better optimization properties during
12:36
training as it allows for more nuanced
12:39
adjustments to the model parameters so
12:42
it's fully differentiable railu has a
12:44
sharp corner at zero which can sometimes
12:47
make optimization harder especially in
12:49
networks that are very
12:51
deep um especially in networks that are
12:54
very deep or have complex
12:57
architectures unlike ra which output
12:59
zero for any negative input J allows for
13:02
small nonzero output values so it
13:05
prevents the dead neuron problem so this
13:07
means that during the training process
13:09
neurons that receive negative input can
13:11
still contribute to the learning process
13:14
in reu neurons which receive negative
13:16
input just get an output of zero so they
13:18
become dead they don't contribute to the
13:20
learning process this problem is avoided
13:22
in Rao in Jou sorry Jou avoids this dead
13:26
neuron problem and that's why it's used
13:28
in the case of of large language models
13:30
now what we are going to see next is
13:32
that okay now that we understand about
13:34
the Jou activation function we are going
13:36
to actually look at the architecture of
13:39
this uh feed forward neural network so
13:42
you see when you zoom into the neural
13:44
network you'll see that there is a
13:45
linear layer here there is a JLo
13:47
activation here and there is another
13:49
linear layer here so up till now we
13:51
understood about the J activation right
Feed forward neural network architecture
13:54
but now I want to tell you a bit about
13:55
what the linear layer actually looks
13:57
like
13:59
so let let's go to that part of the
14:01
Whiteboard and let me show you how the
14:03
linear layer looks like okay so this is
14:06
how the feed forward neural network
14:08
actually looks like uh don't worry if it
14:10
looks a bit complicated it's actually
14:12
quite simple so let's say we receive a
14:14
token uh so let's say the feed forward
14:17
neural network receives a token and the
14:20
number of uh the dimensions of the token
14:23
is equal to the embedding Dimension and
14:25
for gpt2 the smallest size that is equal
14:27
to 768 so let's say this is the
14:30
embedding dimension of the token which
14:31
means that every token is projected into
14:33
a 7 768 dimensional space so as this
14:37
token passes through the different
14:39
layers of the Transformer block which we
14:41
saw over here the good thing about this
14:44
Transformer block is that the
14:45
dimensionality of the the token is
14:47
preserved so even if we pass from here
14:50
to here to here to here and finally we
14:52
go to the input of the feed forward the
14:55
dimensionality of the token remains 768
14:58
throughout this entire procedure and
15:00
that's one of the big advantages of the
15:02
way the Transformer block is constructed
15:05
so keep this embedding dimension in mind
15:07
as you try to understand the neural



***



15:09
network architecture right so here we
15:12
can see that these are the inputs to the
15:13
neural network it's a 768 dimensional
15:16
input vector and this is the first
15:18
linear layer over here all of these
15:22
weights which you can see connected to
15:23
the neurons and then here you can see
15:26
here is the second linear layer so you
15:28
might be thinking what is the number of
15:30
neurons which is used so the number of
15:32
neurons which is used over here is four
15:35
* the number of uh inputs here so the
15:38
number of neurons here will be four
15:40
multiplied
15:44
by so let me write this so the number of
15:47
neurons here will be 4 multiplied by
15:49
768 so this will be um close to 3,00
15:54
3,200 neurons over here so in the first
15:57
layer what happens is that the inputs
15:59
are projected into a larger dimensional
16:01
space just so that we make the neural
16:03
network more expressive and capture the
16:05
properties between the inputs and in the
16:07
second layer the inputs are compressed
16:09
back to the original embedding size so
16:11
the output which is received from this
16:13
neural network has the same dimensions
16:16
as the input it matches the original
16:18
input Dimensions so the output
16:20
Dimensions will also be equal to
16:22
768 so the dimensionality of the input
16:25
is preserved through this neural network
16:27
as well the expansion so you can think
16:30
of this neural network as an expansion
16:32
contraction neural network and remember
16:35
that expansion contraction neural
16:36
networks are very powerful because they
16:38
preserve the size of the input but at
16:40
the same time uh they allow to explore a
16:43
re they allow for a richer exploration
16:46
space so what happens is that when we
16:48
expand this when we uh in the first
16:51
linear layer we do an expansion right
16:53
projecting into a dimension which is
16:55
four times larger we can capture more
16:57
properties between the inputs and that's
17:00
what essentially makes Transformers so
17:01
powerful due to layers like these if
17:04
this layer was not there probably we
17:06
would have missed out the capturing the
17:08
meaning between some sentences when we
17:09
predict the next word so that's why this
17:12
layer is very important so you can think
17:14
of the neural network essentially as uh
17:18
taking one token and then modifying each
17:20
dimension of this token place by place
17:24
because the input is 768 Dimension the
17:26
output is also 768 Dimension and we are
17:28
looking at one token at a time so this
17:30
is very different than the attention
17:32
mechanism right in the attention
17:33
mechanism we look at one token and we
17:35
look at the relationship of that token
17:37
with other tokens in the neural network
17:40
that's not in this feed forward neural
17:41
network we don't consider other tokens
17:44
at all we just look at one token and
17:46
then we pass the
17:48
input and then each dimension of the
17:52
input is modified and then we get the
17:55
output so that's the difference between
17:57
the feed forward neural network and the
18:00
essentially this multi-ad attention
18:02
module which we saw let me yeah so let
18:05
me zoom in
18:08
here yeah that's the difference between
18:10
the feed forward module this feed
18:12
forward module it only focuses on the
18:15
specific token and the multi-ad
18:17
attention which we saw earlier and
18:18
because that looks at the relationship
18:20
of one token with other tokens as
18:24
well uh awesome now what we can actually
18:27
do is that let us go to python code and
18:29
implement this speed forward neural
18:32
network U with the expansion and
18:34
contraction it's again shown over here
18:37
what we are going to consider in Python
18:39
so what we are going to do is that we
18:40
are going to look at an input which
18:43
essentially has three tokens and each
18:46
token has the size of
18:49
768 and that to we are going to look at
18:51
two such batches so in batch number one
18:54
we'll have three tokens in batch number
18:56
two we'll have three tokens and each
18:58
token we have a size of 768 now remember
19:01
what happens in the first linear layer
19:03
just look at one token at once uh the
19:05
768 is projected into a 3072 dimensional
19:08
space then we have the J activation
19:11
function after this linear layer so
19:13
after this first layer there is a JLo
19:15
activation function which we learned
19:17
about earlier remember the JLo
19:19
activation preserves the dimension so
19:22
the input to the JLo is 3072 Dimension
19:24
the output is 3072 now the final layer
19:28
compress so the input Dimension to the
19:30
final layer is 3072 and the output
19:33
Dimension is 768 so if you see the
19:35
output tensor Dimension it's exactly the
19:37
same as the input tensor two batches
19:40
three tokens in each batch and 768 the
19:43
embedding dimension of each token so I
19:46
hope you have understood the visual
19:48
nature of the neural network which we
19:50
are about to construct because if that's
19:52
the case you'll really understand what's
19:53
going on in the code very deeply so we
Coding the feedforward neural network class
19:56
are going to construct this class which
19:58
is called feed forward and uh when an




***



20:01
instance of this class is created this
20:02
init Constructor is called by default
20:05
and what it does is that it creates this
20:07
self. layers which is basically nn.
20:10
sequential so if you are not aware of NN
20:12
do sequential uh you can it's a p torch
20:16
module basically for constructing or
20:18
chaining a neural network
20:21
together um so essentially you can
20:23
Define multiple layers and create a
20:25
neural network by adding these different
20:27
layers together that's why is called
20:29
sequential so here what we are doing is
20:31
that um first if you see we have a GPT
20:35
configuration and let me actually pull
20:36
that configuration here once more so
20:38
that you are aware of the configuration
20:41
which we are using yeah so this is the
20:43
GPT configuration which we are using and
20:45
I'm going to paste it over here so that
20:47
it's a
20:49
reference okay so yeah so here I'm going
20:53
to paste the GPT architect GPT
20:55
configuration which we are using now
20:58
let's look look at how the sequential uh
21:00
layer is constructed first we have a
21:02
linear layer which we saw on the white
21:04
board and the input dimension of this
21:06
linear layer is the number of embedding
21:08
Dimension which is 768 for GPT and the
21:11
output of this linear layer dimensions
21:13
are 4 into 768 because see the first
21:16
layer is the projection layer so it
21:18
takes in an input of the embedding
21:19
Dimension and the IT outputs 4 into 768
21:23
then we have a j activation function and
21:26
then we have the second layer the input
21:28
to the second layer is 4 into 768 and
21:31
the output of the second layer is
21:33
768 so the CFG embedding Dimension is
21:37
that this we are taking this
21:38
configuration and we are looking at the
21:40
embedding Dimension which is
21:42
768 um and if you print this out if you
21:45
print GPT config 124 million embedding
21:48
Dimension you'll see that it's 768 right
21:50
so this is the feed forward neural
21:52
network which is constructed it has
21:54
expansion so you can think of this as
21:56
expansion let me write a comment here
21:58
actually
21:59
this is
22:00
expansion uh then this is the
22:03
activation the J activation and then
22:06
finally we have the
22:09
contraction so it's a three-step process
22:11
expansion activation contraction and
22:13
this feed forward neural network is
22:15
constructed like this right and then we
22:17
have the forward method which just
22:19
Returns the
22:20
output uh from this layer so it will do
22:22
the expansion it will do the J it will
22:24
do the contraction and it will return
22:27
the output and remember that the output
22:29
has the same dimensional size as the
22:31
input
22:33
right so I have just written some text
22:35
over here as we can see in the preceding
22:37
code the feed forward module is a small
22:39
neural network consisting of two linear
22:41
layers and a JLo activation function
22:44
right uh in the 124 million parameter
22:48
gpt2 model GPT model it receives the
22:51
input batches with tokens that have an
22:53
embedding size of 768 as we saw earlier
22:56
now we can actually uh now we have
22:59
everything in shape now we can
23:01
actually create an instance of this
23:03
field forward class and remember we have
23:05
to pass in the configuration which we
23:07
have earlier constructed so that it can
23:09
extract the embedding Dimension right so
23:11
we create an instance of this feed
23:13
forward class and pass in the GPT
23:15
configuration which we are using one
23:17
more thing to mention is that this Jou
23:19
is basically the j class which we had
23:22
defined earlier at the start of this
23:24
lecture so that's available to the feed
23:27
forward uh feed forward class okay so
23:30
now I will define an input X so as we
23:32
saw on the Whiteboard X will have two
23:34
batches each batch will have three
23:36
tokens and the embedding dimension of
23:38
each token is going to be
23:39
768 that's the input now when I uh I
23:44
create an instance of the feed forward
23:46
class FFN and then pass the input to
23:48
this instance so what will happen when
23:50
the input goes through this instance it
23:53
will um the this init Constructor will
23:56
be called so self. layers will be
23:58
defined and the neural network will be
24:01
constructed with this architecture and
24:03
then the forward method will be called
24:05
when the forward method will be called
24:07
first the expansion will be applied on
24:08
the input then the activation will be
24:10
applied and then the contraction will be
24:12
applied and all along the size of the
24:14
input will be preserved so then the
24:16
output will have the same size as the
24:19
input and you can print out the output
24:20
shape which is 2x 3x 768 again it's the
24:24
same size of the input as we had seen on
24:25
the white board so the speed forward
Feedforward neural network advantages
24:28
module which we implemented in this in
24:31
this lecture plays a crucial role in
24:33
enhancing the model's ability to learn
24:35
from and generalize the
24:37
data why can it do that because although
24:40
the input and the output dimensions are
24:41
same it internally expands the embedding
24:44
Dimension into a higher dimensional
24:46
space through the first linear layer
24:48
this expansion is followed by a
24:51
nonlinear jalu activation and then a
24:53
contraction block back to the original
24:55
Dimension with the second linear
24:57
transformation
24:59
so such an expansion contraction design
25:01
allows for the exploration of a richer
25:03
representation space and thus it
25:05
enhances the models ability to learn
25:07
from and generalize the data always
25:09
remember that when you learn about these
25:11
neural network architectures first ask
25:13
the question why is it even there what
25:15
if I remove this um you'll see that if
25:19
you remove it the model's ability to
25:20
learn from data is hampered
25:22
significantly and remember that in gpt2
25:25
we have 12 Transformer blocks and each
25:27
Transformer block we have a feed forward
25:29
neural network like this so we'll have
25:31
12 neural network like this so 12
25:33
expansion contraction blocks now imagine
25:35
the exploration power which our model
25:39
has okay the second thing which I really
25:42
want to highlight here which I also
25:44
highlighted at the start is that there
25:46
is a uniformity in the input and the
25:48
output Dimensions when we look at the



***



25:50
gpt2 architecture this simplifies the
25:52
architecture by enabling the stacking of
25:54
multiple layers and this makes the model
25:57
much more
25:59
scalable so um let me explain this once
26:03
more yeah when we looked at this
26:05
Transformer block at every single layer
26:08
as I mentioned at every single layer
26:09
normalization multi-ad Dropout fade
26:12
forward neural network the dimension is
26:14
preserved throughout so that way we can
26:16
stack multiple layers together because
26:18
we don't have to worry about dimensional
26:20
mismatch that's one of the biggest
26:22
advantages of this Transformer Block
26:24
it's very flexible that way and we can
26:26
stack multiple layers on top of each
26:28
other and that makes the model much more
Summary
26:32
scalable okay so this actually brings us
26:34
to the end of this lecture where we
26:36
covered about the J activation function
26:38
and the feed forward neural network to
26:40
which the J activation function is
26:42
linked so now in this entire GPT
26:45
architecture we have now covered four
26:47
things we have covered the GPT backbone
26:50
we have covered layer normalization in
26:52
the previous lecture and in today's
26:54
lecture we covered the jalu activation
26:55
along with the feed forward neural
26:57
network in the next lecture we are going
26:59
to look at shortcut connections so
27:01
shortcut connections are these basically
27:03
these plus signs if you would have seen
27:06
if you zoom into this Transformer block
27:08
there are this plus signs here right
27:10
this plus this plus signs with an
27:14
arrow this plus sign here with an arrow
27:17
these are shortcut connections and you
27:18
might be wondering what they are what
27:20
they do we'll learn all about that in
27:22
the next lecture I hope you are
27:25
understanding everyone from this
27:26
lectures and please try to execute the
27:29
code which I'm sharing after every
27:31
single lecture that way the conceptual
27:33
understanding and the code understanding
27:35
will also develop much further I'm
27:37
deliberately splitting these lectures
27:39
into separate so that you understand and
27:42
discover about each model of the GPT
27:44
architecture on your own without
27:46
confusing you too much thank you so much
27:48
everyone and I look forward to seeing
27:50
you in the next lecture

***

