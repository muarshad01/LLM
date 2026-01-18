#### GELU Activation Function
* Feed Forward Neural Network
* We'll implement a small NN sub-module that is a part of LLM transformer block.
* Two activation functions commonly implemented in LLMs
1. GELU -> $$x^{*}\phi(x)$$ --> CDF of standard Gussian distribution
2. Swi GLU


#### Dead Neuron Problem
* __Dead neuron problem__, which means that if the output from one layer is negative and if RuLU activation function is applied to it the output becomes zero and then it stays zero because uh because we cannot do any learning of after that so the neurons which are associated with that particular output they don't contribute anything to the learning process once the output of the neuron becomes negative and that's called asthe dead neuron problem so learning essentially stagnates of course. RuLU has a huge number of other advantages this nonlinearity which is introduced over here makes neural networks expressive it gives the power to neural networks but the reason we we are looking at the disadvantages of ReLU is that understanding the disadvantages of RuLU will open an opportunity for us to learn about the GELU activation function and why it is used in LLMs?

***

* 5:00 

#### Approximation for the GELU activation 
* The approximation which was actually used for training GPT-2 looks something like this"

* $$\text{GELU(x)}\approx 0.5.x.(1+tanh[\sqrt{\frac{2}{\pi}}.(x+.055715.x^3)])$$

***

* 10:00

* First Advantage: ReLU discontinuity at x=0, which makes it not differentiable. JELU activation on the other hand is smooth throughout so it's differentiable across all X.
* Second Advantage is that it's not zero for Negative X so that solves the dead neuron problem even if the output of a neuron after is negative even if it goes through JELU it will not become zero so the neuron won't become dead it will still keep on contributing to the learning process that's the second reason.


1. first reason is differentiability 
2. second reason is it prevents the dead neuron problem
3. third reason is that it just seems to work better than ReLU


* when we do experiments with LLMs so as always activation functions are hyperparameters right so we need to test out multiple activation functions to see which one performs better and we have generally seen that JELU performs much better in the context of LLM compared to ReLU.


```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

```python
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
```

***

* 15:00 

* __Expansion and contraction allows__ for a rich exploration space.

***

* 20:00

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





