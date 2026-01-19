#### Transformer
* Transformer block is the fundamental building block of GPT and other llm architectures.
* The transformer block is repeated 12 times in GPT-2 small (124M parameters)

#### Trransformer Block
1. Multi-ad attention
2. Layer normalization
3. Dropout
4. Feed forward layers
5. GELU activation.

***

* 5:00


inputs are multiplied by these weight matrices and then we get the queries Matrix we get the keys Matrix and we get
5:14
the values Matrix right and again in the case of multi-head attention we get
5:19
multiple copies of this so here I'm seeing I'm seeing I'm showing two attention heads so there are two queries
5:25
Matrix there are two keys Matrix and two values Matrix what happens next is
5:30
something very interesting we will take the dot product between the queries and the keys that
5:36
gives us our attention scores the attention scores are normalized to give us the attention weights which have been
5:43
shown over here and then the attention weights are multiplied with the values uh Matrix and then we get a set
5:50
of context vectors the whole aim of multi-ad attention or any attention mechanism is to convert embedding
5:56
vectors into context vectors what are context X vectors the simplest way to think about them is that they are a
6:02
richer representation than the embedding Vector embedding Vector consists of semantic meaning of a particular word
6:09
right it contains no information about how that word relates to other words in a sentence context Vector goes beyond
6:16
embedding embedding it not only captures the semantic meaning of the particular
6:22
word but it also captures the relationship of how that word relates to
6:27
the other tokens in the sentence or how that word attends to other tokens in the
6:32
sentence that's why it's called attention mechanism and the whole goal of this multi-ad attention is to get
6:38
context vectors so for every attention head there is a SE separate context Vector Matrix which is generated and
6:45
ultimately you merge the context Vector matrices from different attention heads and you get this
6:51
combined uh context Vector Matrix and then this is the context
6:56
Vector Matrix which is the output of the multihead attention so whenever you see the multi-head attention layer over here
7:03
this mask multi-ad attention you have Vector embeddings as inputs and then you have context uh you have context vectors
7:11
which are the outputs coming from multi-ad attention if you want to revise some of these Concepts please go to
7:17
those lectures which we covered on the attention mechanism and uh then so the multi head
7:23
attention was a part of our attention mechanism series but if you look at the other four components we have have been
7:29
covering this in the last three lectures itself so let us look at each of these components separately and quickly revise
7:36
what they do these components also form part of the Transformer block the first
7:42
major component is layer normalization so this component does the function of
7:48
normalizing the layer outputs so let's say these are the outputs uh from any layer it can be a Dropout layer it can
7:54
be the multi-ad attention layer let's say these are the outputs from any layer without normalization they will let's
8:01
say have some random mean and some random variance after applying layer normalization the values of these
8:07
outputs will be changed so that the mean will be zero and so that the variance will be
8:13
one why is this needed because it solves two issues for us first it leads to a
8:18
bit of a stability in the during the back propagation it
8:24
ensures that the values are not too large so that the gradient does not explode or the gradient does not vanish
8:30
during back propagation the second is that it solves the problem of internal coari shift which means that during the
8:37
training process the inputs which are received to a certain layer may have different distributions or different
8:43
iteration that's a big issue because that holds the training it becomes very
8:48
difficult to update the weights and then the training takes a long time for convergence layer normalization solves
8:54
this issue and if you look closely at the Transformer block you'll see multiple places where layer
9:00
normalization happens or is implemented it is implemented before the multi-ad
9:06
tension and it is also implemented before the feed forward neural network so it's actually implemented two times
9:13
within the Transformer block uh just as a note the layer
9:18
normalization is also actually implemented after the Transformer block um but we are not going to look at that
9:24
right now the second component are Dropout again a dropout layer can be
9:29
applied after any layer and what Dropout does is that it looks at the layer outputs and then it randomly turns off
9:38
some of the outputs so if you see on the right hand side on the left hand side we
9:43
are showing a neural network before Dropout so these are the units or the layer outputs of the preceding layer
9:49
after passing through Dropout some neurons here or some inputs are randomly
9:54
turned off why are they randomly turned off because it improves generalization
10:00
during training some neurons get lazy they get so lazy that they don't update
10:05
themselves at all they depend on other neurons uh and during testing that's a
10:10
big problem because these lazy neurons are not learning anything once we Implement Dropout a lazy neuron will see
10:16
that the other neurons which are doing all the work they're not there in that iteration so the lazy neuron has no
10:22
choice but to learn and update its weights so that's the reason Dropout helps generalization it prevents
10:29
overfitting that's the second component the third component which we looked at is this feed forward neural network and
10:36
that had the J activation function the construction of this speed forward neural network is pretty interesting we
10:43
preserve the dimension of the inputs so let's say the input to the neural network is like this and when you think
10:50
of an input always think of a token with an embedding Dimension so in gpt2 the
10:57
embedding Dimension was 760 so if you have a word let's say the word is forward or step that word will be
11:04
converted into an embedding of 768 Dimension let's say that is the input which is passed to this neural network
11:11
first we have a layer of expansion which means that there is a hidden layer of neurons and uh the number of neurons
11:18
here is four times larger than the embedding Dimension so the number of neurons will be 4 * 768 so the first
11:25
layer expands which means that we are going from 7 68 Dimension to 4 into 768
11:32
Dimension and then there is a second layer which is the compression layer where we again come back to the exact
11:37
same Dimension which we started with so the input dimension of this neural network and the output dimension of this
11:44
neural network is exactly the same uh then you might be thinking why is this
11:49
expansion and contraction done in this in the first place the expansion and contraction is done so that we can
11:55
explore a richer space of parameters what this expansion does is that that it takes the inputs into a much higher four
12:02
times higher Dimension space and there we can uncover much better relationships between parameters and it generally
12:08
helps llms learn better and why do we compress it back to the same output same
12:13
Dimension because we want the input and the output Dimensions to be same that helps scalability that way you can stack
12:20
multiple neural networks like these together without worrying about Dimension changing now after this layer of neurons
12:28
every neuron has to have an activation function right uh and that activation
12:33
function which is used in this case is the JLo activation function generally everyone is familiar with Ru but J is a
12:40
slight variation J is not equal to zero for X is less than zero that is one
12:46
change and the second change is that Jou is differentiable at x equal to Z it's fully smooth and likee Ru Ru is not
12:53
differentiable at x equal to0 so for X greater than 0
12:59
J generally approximates Ru which means y equal to X it's not exactly equal to X
13:04
but it reaches that so jalu generally solves the dead neuron problem which
13:10
means that if the output of a neuron is negative and it passes through a Ru uh
13:15
it will just be zero but when it passes through Jou it won't be zero and that's
13:21
when the neuron will continue learning in Ru what happens is that if the input to a neuron is if the output of a neuron
13:28
is negative and if if it passes through reu the output is zero and then it the neuron becomes dead it cannot learn
13:34
anything after that point and learning stagnates that issue is solved by J and it generally turns out in experiments
13:41
with llms that the Jou activation does much much better than Ru so that's why
13:46
the activation function which is used after this layer of neurons is the Jou activation
13:52
function and the last component which we looked at um when we look at the
13:59
Transformer block components is shortcut connections so what happens in shortcut connections is that the output of one
14:06
layer so here if you see the output of one layer is added with the output of the previous layer see here there is one
14:13
more path which has been created similarly here you will see the output of this layer the output of this layer
14:20
is added with the output of the previous layer and then we create this path the reason shortcut connections are
14:26
implemented is that it solves the vanishing gradient problem so on the left hand side if you see the gradients
14:33
the outermost layer has a gradient of 0.5 but as you back propagate and you
14:38
reach layer four layer three layer 2 and layer 1 you'll see that the gradient has reduced to a very small value as we back
14:45
propagate to the first layer this is the vanishing gradient problem and then if the gradients become very small the
14:51
learning stops and that's not good for training the llm whereas if we implement
14:57
this shortcut connection it gives another route for the gradient to flow it makes gradient flow much more stable
15:04
and that's why the vanishing gradient problem is solved so if you see layer five gradient magnitud magnitude is
15:11
1.32 and layer three layer 2 and layer 1 all have magnitudes around 0 2 and3 so
15:18
the gradient has not become vanishingly small we have solved the vanishing gradient problem in fact the gradient
15:24
magnitude looks to be pretty stable over here that's why shortcut connections are such an important part of the
15:29
Transformer block so now let's zoom out and take a look at these five components
15:34
together we learned about layer normalization we learned about Dropout we learned about feed forward neural
15:40
network we learned about how it is linked with Jou and finally we learned about shortcut connections now all of
15:47
this have to be stacked together when we create the Transformer block and we'll follow the specific order as which is
15:54
mentioned in the schematic what is this order exactly first we'll start with the layer
16:00
normalization then we will stack the multi-ad attention on top of it let me show with the different
16:06
color uh then we will add the Dropout layer this plus with this Arrow this
16:12
thing here this is the shortcut connection right then we'll add the layer normalization two then we'll add
16:17
the feed forward neural network with J activation then we'll add another Dropout and then we'll add another
16:23
shortcut connection this is exactly what we'll be doing in code now uh before going to code I just want
Transformer block shape preservation
16:30
to explain some conceptual details so when a Transformer block processes an
16:35
input sequence each element is represented by a fixed size Vector let's
16:40
say the size is the embedding dimension for each element one point which is extremely important to note is that the
16:47
operations within the Transformer block such as the multi-head attention and the feed forward layers you remember the
16:54
expansion contraction layer are designed to transform the input vectors such that
17:00
the dimensionality is preserved that's extremely important to note so when you
17:05
look at this Transformer block and when you look at an input which is coming into the Transformer and if you look at
17:11
the output which is going out of the Transformer the outputs have the same
17:17
exact same form and the dimension as the input this is an extremely important
17:23
point which I want to bring to your attention that's why it becomes so easy to stack multiple Transformer blocks
17:28
together we saw that gpt2 has 12 Transformer blocks right the reason it becomes so easy to stack them together
17:34
is that Transformer blocks preserve the dimensionality the dimensionality of the input the dimensionality of the input to
17:42
the Transformer is the same as that of the output from the Transformer so if you look at the input every input is
17:49
basically tokens and the token is converted into these embedding uh
17:54
embedding vectors right that's the input let's say to the Transformer
18:00
um now if you see the output the output has exactly the same size so every token
18:05
will have a corresponding output and it will have exactly the same Dimension as what was there in the
18:12
input uh many students don't U register this importance of the Transformer that
18:18
the dimensionality is preserved but it's one of the most important features of the way the Transformer block has been
18:24
created we could have easily created the Transformer block so that the output Dimension is different
18:29
but that would not help us scale the Transformer blocks now we can just tack different Transformer blocks together
18:34
without worrying about Dimensions just uh for revision the self
18:40
attention block is different than the feed forward block the self attention block analyzes the relationship between
18:46
input elements so it analyzes the relationship between how one input element is related to other input
18:52
elements and it assigns an attention score right but the feed forward neural network just just looks at each
18:59
element separately so when you looked at the neural network here let me take you to the yeah so this
19:06
is the neural network component right and we are looking at one input token at a time one input token with 768
19:12
dimensions and the output is also 768 Dimensions which means we are only looking at one input at a time and not
19:18
its relation with the other inputs that's one difference between the multi-ad attention mechanism and the
19:24
feed forward neural network okay so now uh if you all have
19:30
understood the theory and the intuition behind the Transformer block now it's time to jump into code so I'll be taking
Let us jump into code!
19:37
you to python code right now and let's code out the different aspects of the Transformer block together so here as
19:45
you can see yeah so GPT architecture part five coding attention and linear
19:51
layers in a Transformer block before we proceed I just want to discuss a bit about the configuration which we are
19:57
going to use here we are going to use the configuration which was used in gpt2 the smallest size where they had 124
20:04
million parameters so here the vocabulary size was 50257 the context length was 1024
20:11
remember the context length is the maximum number of input tokens which are allowed to predict the next
20:17
token this is needed when we uh represent the positional embeddings then we have the embedding Dimension so
20:23
remember every token is converted into uh Vector embedding the dimension is 68
20:29
here n heads is the number of attention heads that's 12 n layers is 12 that is
20:35
actually the number of Transformers so remember the number of attention heads and the number of Transformers are
20:40
different within each Transformer there is a multi-ad attention block that can have multiple attention
20:47
heads and then when we have the Dropout layer we just have the dropout rate so this specifies that on an average 10% of
20:54
the neurons or the elements of the layer will be set to to zero that's why it's 0
21:00
one right now and then the query key value bias is set to false because we don't need this bias term right now we
21:07
are going to initialize the weights of the query key and value Matrix randomly without the bias
Coding LayerNorm and FeedForward Neural Network class
21:14
C okay before going to the Transformer block we need to revise what all we have
21:19
coded for the other blocks before so we saw the layer normalization right and we
21:25
had defined a class for the layer normalization before what this class does is that it simply takes an input it
21:31
subtracts the mean from the input and it divides by the square root of variance that make sure that the elements are
21:38
normalized to keep their mean equal to zero and standard deviation or variance equal to 1 remember in the denominator
21:45
we also add a very small value to prevent division by zero you may be wondering what the scale and shift is
21:52
these are trainable parameters which are added so you can think of them as uh parameters which are learned
21:59
during the training process so this is the layer normalization here the embedding Dimension is the input and the
22:06
normalization is performed along the embedding Dimension so let's say every step moves you forward right every is a
22:14
token so actually let me go to the Whiteboard to show you how the normalization is actually
22:20
done just so that you get a visual representation yeah so now uh if you
22:26
see um let's look at the these tokens and every token let's say has an embedding Dimension here of
22:32
768 in normalization what is done is that we look at individual rows and then
22:38
we normalize across the columns so we make sure that let's say the mean is zero and the standard deviation is one
22:45
now these values can be the output from any layers so let's say there is a layer normalization here right so what this
22:52
does is that it receives inputs from here and the input will have the dimension of 768
22:59
because the dimension is preserved so we'll take the mean along the 768
23:04
Dimension we'll make sure the normalization is such that the mean along the columns which is the embedding
23:09
Dimension is zero and the standard deviation is one that that will be the output from the layer normalization so
23:16
the size will be the same as the input but just what will change is that every row will have mean of zero and standard
23:23
deviation of one then we have the J activation function and as we saw it's defined by
23:30
this approximation which is used in gpt2 once we use this approximation the JLo
23:35
starts looking like what we had seen um in the building block so if I
23:42
just go to that particular graph so when I zoom in over here see this is the JLo
23:48
activation function and how it looks like the approximation used when gpt2 model was developed was this kind of an
23:56
approximation and uh actually in one of the previous lectures we saw the
24:02
function which was actually used so let me scroll up towards that to show you
24:07
that function yeah I think it is over here so if you see this this function
24:15
this is the function approximation which was actually used by researchers for training uh gpt2 this is the J
24:22
activation function approximation this is exactly what we have written over here and then we have the forward neural
24:29
network which takes in the input um which has the dimensions embedding
24:34
Dimension it expands it to four times the embedding Dimension then we have the
24:39
J activation and then we have the contraction layer so it takes the input which is four times the embedding
24:45
Dimension and brings it back to the original embedding Dimension so here you can see that the feed forward neural
24:51
network consists of an expansion and activation and a
24:56
contraction uh this is exactly what we have seen on the white board over here so let me just go to that portion of the
25:03
Whiteboard where we saw the feed forward neural network just to REM just to give you a
25:10
refresher um so here is yeah so this is that expansion
25:16
contraction neural network which we have just coded out in Python so the input is expanded to four times the dimension
25:22
size then we have the J activation function and then we have the contraction to the original embedding
25:30
size okay so now we have coded out different classes so we have a layer normalization class we have a j
25:36
activation class and we have a feed forward neural network class now we are ready to code the entire Transformer
Coding the transformer block class in Python
25:43
block using these building blocks which we learned about before okay so this is the class for the
25:50
Transformer block first let me go to the forward method and tell you a bit about
25:56
what we are doing here to understand this this sequence you just need to keep
26:02
in mind this sequence which we have in this figure so let me take you to that figure
26:10
right now yeah this sequence right over
26:15
here I'll just rub everything which is there on the screen so that you get a better look at this sequence there are
26:22
two many colors and too many symbols on the screen right now so I'm just getting
26:27
rid of I'm just getting rid of all of
26:34
these okay so I hope you are able to see the sequence now so this is the exact
26:40
same sequence which we are going to follow uh and keep this in mind right now it's fine you can even take a look
26:46
at this whiteboard later and revise when the video is when you look at the video so first we have the layer normalization
26:52
followed by attention followed by Dropout followed by shortcut layer so let's see these four steps initially so
26:59
see we have a layer normalization followed by the attention followed by the uh Dropout followed by the shortcut
27:06
Okay so until this point we have reached this stage this stage and then we have the
27:13
next steps which is another layer normalization uh so another layer
27:20
normalization yeah in the next four steps we have another layer normalization then feed forward Network
27:27
then Dropout and short cut so four steps and here you can see another layer normalization feed forward uh feed
27:35
forward neural network then drop out and then shortcut so this is what is happening in
27:40
the forward method if you understand the diagram which was shown on the Whiteboard this is pretty simple but
27:45
let's see when we create an instance of this Transformer block class what are the different objects which are created
27:51
so at is the multi-head attention object this is an instance of the multi-head attention class which we had defined in
27:58
one of the previous lectures what this class does is that it takes the embedding vectors and converts
28:04
them into context vectors and uh the input Dimension is the embedding Dimension the output is the same as the
28:10
embedding Dimension we have to specify context length over here which is 1024 let's revise this again context
28:18
length is 1024 and uh yeah then number of heads is the
28:25
number of attention heads which is I think 12 over here Dropout is the dropout rate 10% which we have used
28:30
before and this query key value bias is set to false if you want to revise multi-ad attention please go to one of
28:37
the previous lectures where we have covered this so whenever this uh at
28:43
atten uh an instance of this multi-ad attention class is created it takes in the input embedding and converts it into
28:50
context vectors the size here is batch size number of tokens and embedding size so
28:57
if the number of tokens are let's say four or five here there will be four and each will have the embedding size which
29:03
is let's say 768 in our case that's the embedding Dimension and the batch size can be uh any batch size which we have
29:11
defined so this is the at object which is the multi an instance of the multi-ad attention class then we have another
29:18
object called FF which is an instance of the feed forward class and we saw that feed forward class over here we just
29:24
have to specify the configuration here so that from the configur we can get the embedding dimmension and this class what
29:31
it does is that it creates uh these layers um and with the J activation
29:37
function and initialize the weights randomly so this is the um FF object
29:45
which is an instance of the feed forward class so wherever FF is used here we the input is passed in and it goes through
29:52
the expansion the J and the contraction and the output is the same dimensions as the input
29:58
then we have Norm one and Norm two so Norm one is the first normalization
30:03
layer and Norm two is the second normalization layer so you see the first normalization layer we are using before
30:10
the multi-ad attention the second normalization layer we are using before the feed forward neural network that's
30:16
why sometimes these are also called pre-normalization layers why pre because they are used before the multi-ad
30:22
attention and before the feed forward neural network and the last object is the drop shortcut which is basically a
30:29
Dropout uh which is basically an a Dropout layer so nn. Dropout is already
30:36
predefined from pytorch so you can actually search Dropout pytorch and it
30:42
will take you to this documentation I'll also share the link to this in the YouTube description
30:49
section okay so now when you look at the forward method I have explained to you the norm one uh which is the object Norm
30:58
normalization first normalization layer object then at drop shortcut shortcut
31:03
which is the Dropout layer we do not need to create a separate class for the shortcut connection because what we do
31:09
is that we just add the output of this back to the original input so when you
31:14
look at the shortcut mechanism look at where the arrows are there so if you look at this
31:21
Arrow if you look at this Arrow over here what we are doing is we are adding the this input over here
31:28
uh this input over here to the output from the Dropout right so the output
31:34
from the Dropout is added to this input which is over here let me actually mark
31:39
this with yellow so that you can get a clear understanding so the input is being marked with an yellow star here
31:45
and the output from the Dropout is marked with another yellow star and we are adding these two yellow stars
31:50
together that's what this first shortcut connections ISS so what we are doing is
31:56
that um shortcut is initially in initialized to X which is the input and
32:02
then X gets modified to the output of the Dropout so we are essentially adding the Dropout output to the input which is
32:09
exactly what we saw on the white board in the second shortcut layer let's see what happens in the second shortcut
32:16
Connection in the second shortcut connection what is actually happening is that uh here you see there is an input
32:23
to the second normalization layer and there's an output from the second dropout so we are adding the input to
32:29
the second normalization layer with the output from the second Dropout and you will see in the code what we are doing
32:36
so shortcut equal to X where X right now is the input to the second normalization layer and uh when we reach this step X
32:44
is equal to the output so here we are actually adding the output to the input of the second normalization layer which
32:51
is exactly what we saw on the Whiteboard so after all these operations are performed in the forward method the
32:58
Transformer block Returns the modified input which is the same dimensions as the input Vector remember that if
33:05
you keep in mind or visualize this Blue Block which I'm showing on the screen
33:10
right now you will easily understand what is happening in the code because in the code we have just followed a
33:16
sequential workflow of all of these different modules together okay so the Transformer block
33:23
is as simple as this once we have understood about the previous modules we just stack them together to build the
33:29
Transformer block now I think hopefully you would have understood why I have spent so many lectures on the feed
33:35
forward neural network we had one full lecture on the feed forward neural network and and Jou we had one full
33:41
lecture on layer normalization and we had one full lecture on shortcut connections as well the reason I spent
33:47
so much time separately on those lectures is that all of those different aspects come together beautifully when
33:53
you try to code out the Transformer block from scratch okay so here I have just written
Transformer block code summary
33:59
an explanation of what we have done in the code so the given code defines a Transformer block class in py torch that
34:06
includes a multi-head attention mechanism and a feed forward neural network so this is the multi-head
34:11
attention mechanism and this is the feed forward neural network layer normalization is applied before each of
34:17
these two components so before the multi-ad attention mechanism and once which is before the feed forward neural
34:24
network that's why it's called as pre-layer Norm older architecture such as the original
34:30
Transformer model applied layer normalization after the self attention and feed forward neural network that was
34:36
called as post layer Norm post layer Norm so researchers later discovered
34:43
that post layer layer Norm sometimes leads to worse or many times leads to worse training Dynamics that's why
34:50
nowadays pre-layer Norm is used so we also implement the forward
34:55
path where each component is followed by shortcut connection that adds the input of the block to its output so here you
35:01
see this shortcut connection adds the input of this whole block to the output this shortcut connection over here adds
35:08
the input of this whole block to the output now what we can do is that we can
Testing the transformer class using simple example
35:14
initiate a Transformer block object and let's feed it some data and let's see what's the output so here what I'm
35:21
defining is that I'm defining X which is my input it has two batches each batch has four tokens and the embedding
35:28
dimension of each token is 768 now I'm passing this this through the Transformer model let's first visualize
35:35
what will happen once this x passes through this Transformer model or Transformer class rather when X first
35:41
passes through the Transformer class uh normalization layer is applied so now try to visualize this try to visualize
35:48
every token as a row of 768 columns so every row is normalized which
35:55
means that the mean of every row and the standard deviation of every row will be one that will be done for all the four
36:01
tokens of one batch and then the same thing will be done for all the four tokens of the second batch so then the
36:07
normalization layer is applied to X and then all of the four tokens of one batch will be transformed so that the mean of
36:15
every row and the standard deviation or the variance of every row will be equal to one mean will be zero sorry the mean
36:22
of every row will be zero and the standard deviation will be equal to one after that that we pass every token of
36:29
both the batches to the self attention mechanism or the multi-head attention rather and the output of this is that
36:36
every token embedding Dimension is converted into a context Vector of the same size so if you look at the first
36:43
token of the first batch that has 768 dimensions that's a embedding Vector which does not encode the attention of
36:49
how that should relate to the other input vectors when we implement this the
36:55
resultant is the embedding Vector which essentially has four tokens and each
37:01
token has 768 Dimensions but now the resultant will be context vectors main
37:06
aim after this attention mechanism or after this attention block is to convert the embedding vectors into context
37:12
vectors of the same size then we apply a Dropout layer which randomly drops off
37:17
some U uh some parameter values to zero and then we add a shortcut layer this is
37:24
the first block you can see in the second block the output of the previous block passes through a second
37:30
normalization layer then through a feed forward neural network where the dimensions are preserved so after coming
37:36
out from the feed forward neural network the dimensions would again be uh two batches multiplied by four
37:44
tokens multiplied by 768 which is the dimension of each token and then we again have a Dropout
37:51
layer and then we again add the shortcut mechanism to prevent the vanishing gradient so when we return the X we
37:57
expect the output to be 2x 4X 768 which is the same size as the input now let's
38:04
check whether that's the case so this is my X and now I'm creating an instance of the Transformer block but remember I
38:10
need to pass in this configuration so when I create when you create an instance of the Transformer block you
38:15
have to pass in the configuration and remember again this is the configuration which I'm using over here which defines
38:21
the context length embedding Dimension number of attention heads number of Transformer blocks and the dropout rate
38:29
okay so now we pass in this configuration and then we just print out the output um and the input shape is 2x 4X
38:37
768 and you'll see that the output shape is exactly the same 2x 4X 768 what I
38:43
really encourage all of you to do is uh when you watch this lecture try to understand the dimensions try to write
38:50
down the 2x 4X 768 on the Whiteboard apply the layer normalization try to see
38:55
how the dimensions work out through all of these different building blocks and try to see that when you
39:01
reach the end the dimension is exactly preserved which is 2x 4X
39:06
768 okay so I have just added some notes here so that we can conclude this
39:11
lecture so as we can see from the code output the Transformer block maintains the input
39:17
Dimensions indicating that the Transformer architecture processes sequences of data without altering their
39:23
shape throughout the network this is very important the Transformer block processes the data without altering the
39:30
shape of the data the preservation of shape throughout the Transformer block
39:35
architecture is not incidental but it is a crucial aspect of the design of Transformer block itself this design
39:42
enables its effective application across a wide range of sequence to sequence tasks where each output Vector directly
39:49
corresponds to an input Vector maintaining a on toone relationship however the output is a
39:55
context Vector that encapsulates in information from the entire input sequence remember that the output
40:01
contains so much information it's very rich output Vector because it contains information about how the uh in the
40:08
input sequence how every token relates to the other tokens of the input sequence that's the whole idea of the
40:14
attention mechanism which we looked about before or which you understood before so the GPT architecture is the
40:20
broad level within that there is a Transformer model within that there is a attention mechanism so these three
40:26
things power each other other it starts from the attention mechanism which is a key component of the Transformer block
40:33
the and the Transformer block is the key component of the whole GPT architecture so finally this means that
40:40
while the physical dimensions of the sequence length and feature size remain unchanged as it passes through the Transformer block the content of each
40:47
output Vector is reint is re-encoded to integrate contextual information from
40:52
across the entire input sequence so this is just saying that although the input and the output dimensions are the same
40:59
the output contains a lot more information since it also contains information about how each token relates
41:05
to the other tokens in the input that actually brings us to the end
Lecture summary and next steps
41:10
of this lecture I just want to show you one thing as we are about to conclude
41:16
um I want to show you um how what all we have learned so far relates to each
41:23
other so when we looked at the GPT architecture we saw that there are four
41:29
things which are important the layer normalization the J
41:36
activation uh let me draw that again the layer normalization which is here the J
41:44
activation which is here the feed forward neural network the feed forward neural network
41:50
component and finally the shortcut connections and we also saw today how all of these four come together together
41:57
to build the entire Transformer block and we coded out this Transformer block together now in the next lecture what we
42:04
are going to see is that how the Transformer block uh leads to the entire GPT GPT
42:11
architecture so remember what I said earlier it all starts from this
42:16
attention which is the mass multi-ad attention that forms the core of the Transformer block now that we have coded
42:23
the Transformer block our task is not over because remember that after the Transformer block there are lot of
42:29
pre-processing steps and then finally we have to use the output Vector in order to predict the next next token and we
42:37
still have to see how that is done so the last step is still remaining and the last step is
42:42
the final GPT architecture but try to understand this sequence here it all
42:49
starts from the attention block then so it all starts here the magic starts here
42:55
at the attention block then that forms the core of the Transformer
43:01
block which I'm marking over here and the Transformer block essentially forms
43:06
the core of the entire GPT
43:12
architecture I hope you have got this sequence in mind so that's why for us it was very important to first spend a lot
43:18
of time understanding attention we have covered five to six lectures on that then it was very important to spend a
43:24
lot of time on every single component of this Transformer Block in the next lecture we are finally going to see uh
43:31
how the Transformer block leads or fits in the entire GPT architecture or rather
43:37
put in other words how can we pre-process the or postprocess the output from the Transformer block so
43:43
that we can predict the next word remember the whole goal of GPT style models is that given an input given an
43:50
input let's say the input is every effort moves you how to predict the next word up till now we have just seen that
43:56
the Transformer block retains the input shape right the Transformer block output but how is this converted to next word
44:03
prediction that's what we are going to see in the next lecture okay so now with the Transformer
44:10
block implemented we have all the ammunition or building blocks needed to implement the entire GPT architecture
44:17
and here you can see the next lecture which I've already planned is coding the entire GPT
44:24
model so I hope everyone you are liking this style of lectures where I first cover intuition plus theory on the
44:30
Whiteboard and then I take you to the code please follow with me and then try
44:35
to implement the code on your own try to write things on the Whiteboard Because
44:40
unless you understand the nuts and bolts of how large language model works it will be very difficult to invent
44:46
something new in this field it will be very difficult to truly be a researcher or engineer and such kind of fundamental
44:52
understanding will help you as you transition in your career as well thanks thanks everyone and I look forward to
44:58
seeing you in the next lecture

***

