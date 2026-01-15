#### Multi-head Attention

* We are going to study two types of multi-head attention mechanisms the first type is basically by just concatenating the context Vector matrices, which is obtained from different query key and value matrices and the second approach is a more unified approach, which is more commonly implemented in modern LLMs and modern code bases. 

***

* 5:00

***

* 10:00

* What is multi-head attention?

* The term multi-head essentially refers to dividing the attention mechanism itself into multiple heads. Each head will be operating independently to give you a understanding related to code. When we get the attention scores here right now or the attention weights from one set of query key and value, we say that this is one head this is one attention head right. We are not decomposing into multiple query keys and values.
* There is only one query Matrix.
* There is only one Keys Matrix and one values Matrix.
* What multi-head attention does is that it extends the causal attention mechanism so that we have multiple heads and each of these heads will be operating independently.
* Then what we do in multi-head attention is we basically stack multiple single attention head layers together so what we'll simply do is that we will create multiple instances of the causal self attention mechanism each with its own weights and then combine their outputs.
*  It's actually very simple what the output which we had obtained before for one attention head we'll just combine them together and I'll show you how it can be done so as you might have expected this can be a bit computationally intensive but it makes llms powerful at complex pattern recognition tasks so researchers have found out that although when you stack multiple heads the computations which need to be made increase it really helps

***

* 15:00

llm perform much better and when you look at Chad GPT right now it does have multiple attention heads it none of the
15:12
modern llms work with a single attention head so let me show you what it means by stacking multiple attention heads
15:18
together this diagram actually encapsulates it all earlier when the when I showed you the flowchart we just
15:25
had one trainable Matrix for the query one trainable Matrix for the keys and one trainable Matrix for the values
15:31
right so that was one attention head now when we look at multi-ad attention in
15:37
this figure we have shown an example of having two attention heads so what we do here is that instead of one trainable
15:44
weight Matrix for the query key and value we have two so here you see we have two trainable weight matrices for
15:50
the query we have two trainable weight matrixes for the key and we have two trainable weight matrices for the values
15:57
these will be multiplied by the input Vector input X and then we'll get two queries Matrix we'll get two keys Matrix
16:05
and we'll get two values Matrix previously we just had one query key and value Matrix right then what we'll do is
16:12
that we'll multiply the queries with the keys transpose and get the attention
16:18
weights then we'll get the attention scores we'll multiply with the value Matrix and we'll get two sets of context
16:23
vectors so this is the first set of context vector and this is the second set of context vector
16:29
earlier we just had one context Vector right we had one context Vector Matrix
16:35
now we get two context Vector matrices so if you look at this input here which has been
16:40
highlighted this input 7.2 and 1 let me highlight it once
16:46
more yeah this input 7.2 and one earlier we converted every threedimensional input input into a two dimensional
16:53
context Vector right but now this 300 threedimensional input will be converted into two context vectors each of the
17:00
output dimmension which in this case is two so then what we will do is that we'll get these two context vectors and
17:06
we'll concatenate them together to give the ultimate context Vector Matrix so earlier if you see the context
17:14
Vector Matrix was 6x2 so in this case the 6x2 and 6x2 will be aggregated
17:20
together and the final context Vector Matrix size will actually be
17:26
6x4 it will be 6x4
17:32
okay so it actually looks something like this earlier when we applied the single
17:37
head attention we just got one context Vector Matrix right but now since we have two attention heads we have two
17:42
context Vector matrices and each context Vector Matrix has a dimension of
17:48
6x2 so when you stack these two and you stack them along their last Dimension
17:53
which is two so the number of rows actually remain the same but the number of column colums increase so when you
18:00
stack these two along the column the final con concatenated context weight Vector weight context Vector Matrix will
18:07
be 6x4 that's all which is actually happening in multihead attention so
18:13
let's go to our flowchart and see really what is going on okay so as we saw earlier we had this
18:23
trainable query key and value matrices right now in multi-ad attention we have
18:29
multiple of these matrices if we have five attention heads we have five trainable query matrices five trainable
18:35
key matrices and five trainable value matri even here we had one one queries
18:41
Matrix one Keys Matrix and one values Matrix right in multi-ad attention we have multiple of these similarly we have
18:49
multiple sets of attention scores we have multiple sets of attention weights and ultimately when we multiply these
18:55
with the values uh we have two set of context Vector matrices so if we have
19:01
two attention heads if you look at the first context Vector Matrix this is from attention head number one which
19:07
basically means it's from the first set of queries keys and values if you look at the
19:13
second uh context Vector Matrix you'll see that this is from attention head two
19:19
so the dimensions of the first context Vector Matrix is 6x2 and the dimensions
19:24
of the second context Vector Matrix is 6x2 now these are concatenated along the
19:30
columns and so we get a final context Vector Matrix which has the dimensions of
19:35
6x4 that is all what is being implemented in the multi-head attention that's why it's called multi- head
19:40
because we are aggregating the output of multiple attention heads so if you find yourself getting
19:47
confused just remember the main idea what we are doing is that we are just running the attention mechanism multiple
19:53
times that's it that's the only thing to remember and this one figure which I've shown over here really summarizes it all


***

20:00
we have multiple copies of the trainable query key and the value Matrix yeah here
20:06
we have multiple copies of the trainable query key and value Matrix we have multiple copies of the queries keys and
20:13
the value Matrix and we have multiple context Vector matrices which are stacked together now let us Implement multi-head
Coding multi-head attention in Python
20:20
attention in code for that you need to remember that we have already implemented the causal attention class
20:27
and uh the output of this class is that it returns a context Vector correct so just remember this and let's move
20:33
forward to Extended s extending single head attention to multihead attention so
20:38
in Practical terms implementing multi-head attention involves creating multiple instances of the self attention
20:44
mechanism each with its own weights and then combining their outputs so if you create one instance of the causal
20:51
attention you get one context Vector so then you'll create another instance of the causal attention you'll get another
20:57
context vector and you will merge them to that's exactly what we are going to implement in the code so to do that we
21:05
we will actually Implement a multi-head attention rapper class that Stacks multiple instances of our previously uh
21:12
implemented causal attention module so here's the multi-head attention rapper class now what is happening in this
21:19
multi-ad attention rapper is actually pretty simple okay so we get the output from the causal attention mechanism
21:26
which is written over here and we first Define number of attention heads so if
21:31
the number of attention heads is five uh what we'll do is that we'll get outputs from uh we'll get five outputs
21:39
and then we'll concatenate them together so first let's look at the forward method it's torch do cat which is which
21:46
is concatenation and dimension equal to minus one why minus one because we are concatenating along the columns as we
21:52
saw before and then here you can see we are looping over all the attention head in self. heads and what is self. heads
22:00
so self. heads is essentially it will create an instance of the causal attention class for how many number of
22:06
heads so here if you see we are looping over the number of heads so if we specify the number of heads equal to two
22:13
we will create two instances of the causal attention class and then the results of the two instances will be
22:19
essentially uh stored in this function head of X okay and head is essentially in self.
22:27
heads so what will will be doing is that if number of heads is five we'll create five instances of the causal attention
22:33
class and the outputs what we get the outputs we'll just concatenate them together along the columns that's it
22:40
it's as simple as that that's why we learned about the causal attention mechanism earlier because if you
22:46
directly come to multi head attention you'll find it difficult to understand all of these so remember D in is the
22:53
vector dimension of our tokens the input embedding Vector Dimension D out is the
22:58
output Dimension which we want context length is in this case the number of tokens but it can be anything what you
23:04
said Dropout is the dropout rate okay so for example if we use this
23:10
multi-head attention rapper class with two attention heads and causal attention output Dimension equal to two this
23:16
results in a four dimensional context vectors because D out into number of heads is equal to four so remember what
23:23
is happening here and that is Illustrated in this diagram below
23:29
uh take a look at this diagram yeah yeah take a look at this diagram D
23:36
out is specified earlier so if D out is equal to two and we have two attention
23:41
heads then the final output will be 2 into 2 which is equal to 4 if D out
23:46
equal to two and we have three attention heads we'll get three context Vector matrices right so then there will be one more which is added here so then the
23:53
output of the final concatenated context Vector Matrix will be uh basically Bally six rows into this will be six columns
24:01
so D out will be six in this case that's just what is written here so the final Dimension the column dimension
24:08
of the context Vector is D out into the number of heads now let us actually create an instance of this uh multi-head
24:15
attention wrapper and see how it's working so uh first we have to specify the batch okay and I think we have
24:22
defined the batch earlier so let me actually copy paste that uh let's see where the batch has been
24:34
defined I remember we had defined the batch somewhere earlier and that's why I am implementing it below yeah here so
24:41
this is where we have defined the batch and the inputs so let me actually bring the inputs here itself so that you have
24:48
an understanding of uh what the inputs were so the inputs were six tokens your
24:54
journey begins with one step yeah these are the inputs and uh this was the batch
24:59
which we had so let me put all of this together and bring all of this to our


***


25:05
current code right so before this part let me just type in the inputs and the
25:12
batch right so these are my inputs my inputs are six tokens okay and uh with three
25:20
dimensional Vector embedding for each now I'm going to create a batch so which has two inputs which are stacked on top
25:26
of each other so this is the inputs and I have a batch of two inputs so let me print this so you'll see the batch has a
25:33
shape of 2A 6 comma 3 correct so now my context length is going to be badge do
25:39
shape of one because here I'm assuming that the number of tokens is equal to the context length which is equal to six
25:45
so this is the number of tokens which is equal to six in this case D in which is the vector embedding Dimension is 3 in
25:52
my case and I'm going to use a d out equal to 2 that's all we need to create an of the
25:58
multi-ad attention wrapper we need to specify D in which is three D out which is equal to 2 context length which is
26:05
equal to 6 Dropout will will choose to be number of equal to zero and number of heads we are going to choose right now
26:12
to be equal to two I think yeah number of heads equal to two and this bias term is just false this is for initialization
26:19
in the trainable weight trainable weight metries for queries the keys and the
26:24
values awesome so let me create an instance of this multi head attention wrapper and then it's it's defined in
26:31
mha multi-head attention and then after I create an instance I'll pass in the batch which has been defined earlier and
26:39
then the result will be the context vector and let's print out the context Vector shape so if you print out the
26:44
context Vector shape you'll get it as 2 comma 6A 4 can you try to think why
26:50
there is why the shape is 2A 6A 4 so let's look at each Dimension individually the First Dimension is two
26:57
because we have two two batches so this is the output for the first batch this is the output for the second batch now
27:02
let's look at each output so the second dimension is six so there are six rows why are there six rows because there are
27:09
six tokens and we have a context Vector for each token that's why there are six rows but if you look at the columns for
27:15
each there are four columns so you might be thinking but D out is equal to two so there should have only been two columns
27:21
right so the thing is that there were only two columns for one attention head but now we have two attention heads so
27:28
the two columns and two columns will be aggregated together and that's why we have four columns in the final context
27:34
Vector Matrix so this is the result of the multi-head attention wrapper these are the this is the final context Vector
27:41
Matrix which we have after we implement the multihead attention and this is that Matrix which is further passed to the
27:47
large language model for the training procedure all of these lectures which we had ultimately resulted into this this
27:55
context Vector Matrix which we are seeing right now to get to this Matrix we have to first learn about U we have
28:01
to learn about causal attention before that we have to learn about query key and values right to learn about causal
28:07
attention we had to learn about query key and values attention scores attention weights Etc then context
28:12
Vector after learning all this right now we have seen that we can concatenate
28:17
different attention heads together to get the final context Vector Matrix and
28:23
the values which you see here are not trained so these are random values right now but later you'll see that when the
28:29
large language model is trained these values itself get updated and they get better and better and better and they
28:35
encode information about how every word relates to other words attends to other words in a
28:42
sequence so the first dimension of the resulting context vectors is two since we have two input text as I already
28:48
mentioned the second dimension refers to six tokens in each input six and the third dimension refers to the four
28:55
dimensional embedding of each token why four dimensional because two dimensions for D output multiplied by the number of
29:01
heads which is equal to 2 so 2 into 2 equal to 4 okay so in this section we basically
Making multi-head more efficient
29:08
implemented a multi-ad attention wrapper that combines multiple single attention head modules right it's actually pretty
29:15
simple if you understand causal attention we are just stacking the outputs of the causal attention along the
29:20
columns note that these are processed sequentially why the head X for each
29:26
head right so each output is processed sequentially we first find the output for the first attention head then we
29:32
find the output for the second attention head Etc the output is the context vector and then we stack them together
29:38
that actually leads to a lot of inefficiencies this sequential Computing we can improve this implementation by
29:44
processing the different attention heads in parallel uh one way to achieve this is
29:50
by Computing the outputs for all attention heads simultaneously via matrix multiplication currently what we
29:57
are doing is that we are we are looking at one attention head finding finding the output we are looking at second

***

30:02
attention head finding the output that's extremely inefficient why don't we essentially combine the output for all
30:08
the attention heads simultaneously and how can we do that let's explore in the next section so this will be the next
30:14
video where we'll be looking at implementing multi-head attention with weight splits now this is the actual
30:20
multi-head attention mechanism which is implemented in GPT uh the multier attenion with weight
30:28
splits because it's much more efficient so the next lecture is going to be awesome let me see if there is something
30:34
else to be covered in the current lecture so we looked at this figure yeah I think we have covered everything for
30:40
today's lecture one last thing which I want to see is that I want to ask chat GPT itself how
30:47
many how many attention heads were present for
30:53
training GPT 3 and GPT 4 let's
31:01
see so here you can see that uh gpt3 there were 96 attention heads uh in the
31:08
largest version but for GPT 4 the exact number of attention heads have not been
31:13
publicly disclosed by open AI but I assume that it would be much higher than 96 presuma presumably more than 150 even
31:22
but these are the number of attention heads which need to be concatenated together right now in the demonstration
31:27
which just saw two attention heads but think about the amount of computations if you need to accumulate the results
31:34
from 96 attention heads together that's why training a large language model requires huge amount of computational
31:41
resources okay I hope all of you have understood multi-ad attention this is
31:47
the culmination of all the previous lectures which you which you diligently I hope have gone through and uh in the
31:54
next lecture I'll teach you about multiple multi-head attention with weight splits this is a very interesting
32:00
lecture which we have planned because I I take a Hands-On example um with actual
32:05
values in matrices and I show you how to actually compute exactly the multi-ad attention from start to finish thank you
32:12
so much everyone I hope you enjoying these lectures I look forward to seeing you in the next lecture


