## Shortcut Connections
* Also know as skip connections or residual connections
* They were first proposed in the field of computer vision to solve the __problem of vanishing gradients.__
* gradients become progressively smaller as we propagate backwards through a neural network and when gradients become very small -> making it difficutl to train earlier laryers.
* The weight updates are not made learning becomes stagnant and so convergence is delayed.

***

* 5:00

* becomes extremely small because a product of small quantities becomes even more smaller so by the time the training is finished and by the time we get the gradients of the loss with respect to all the weights of this layer we find that this gradient becomes very small and then it even starts approaching zero do you see the problem which happens when the gradient start approaching zero so let's say if you have a particular weight uh the gradient update or the weight update rule looks something like the the weight in the new iteration is the weight in the old iteration minus Alpha which is the step size multiplied by partial derivative of loss with respect to the weights right um so partial derivative of L with respect to

* Shortcut connections create alternative an path for gradient to flow, by skipping one or more layers.
* This is achieved by adding output of one layer to output of a later layer.
* They are also called skip connections.
* Play a crucial role in preserving flow of gradients during training backwards pass
* [Visualizing the Loss Landscape of Neural Nets - 2028](https://arxiv.org/abs/1712.09913)

***

* 10:00

***

* 15:00

neural Nets so just look at this left figure which is without using skip connections the Lost landscape look like
15:08
this we have several local Minima and there are several Peaks and valleys whereas if you include skip connections
15:15
you will see that the number of local Minima are not that much there is just seems to be a very smooth landscape with
15:22
one single local Minima that's what skip connection does since the Lost function landscape becomes so smooth the grade
15:28
Med flow also become smooth and that's the advantage of having skipped connection so if you forget the
15:34
mathematical deriv derivations and if you are a person who is good at visual learning just keep this loss function
15:40
landscape in mind and remember that adding skip connections will help us go from the left which has a number of
15:46
oscillations the loss function is not smooth to the right where the loss function as you can see is pretty
15:52
smooth now that we have learned about the mathematical intuition and visual
15:57
understanding of the skip connections or shortcut connections let's go to code right now and uh let's actually
16:04
Implement shortcut connection so what we are going to do in code is that we are going to look at a neural network like
16:09
this which will take in the inputs of a certain size let me Mark it with
16:15
different color the neural network will take in the inputs of a certain size and we will stack multiple layers
16:21
together and we will give a provision to add the shortcut connection so the output of any layer we can add to the
16:27
input of the previous layer this is what we are going to implement right now and we are also going to check
16:33
the similarities or we are going to compare the gradient flow magnitudes
16:38
without shortcut connections and with shortcut connections so let's get to code right
Coding shortcut connections in Python
16:44
now so here we are going to implement this class example deep neural network
16:49
and uh when an instance of this class is created by default this init Constructor is called and this takes in some
16:56
arguments first it takes in the layer sizes which are basically uh how many neurons you want in each layer so the
17:03
layer sizes can be three three 3 3 and one which means that there are five
17:09
layers with three neurons and the final layer has one neuron so if the layer size is 3 3 3 3 one it looks something
17:16
like this so here you can see that there are five layers with three neurons and there is one layer with one neuron
17:23
that's the first argument which is the layer size which this function will take the second argument is use shortcut so
17:29
if this can be true or false if it's false we'll not use the shortcut connections if it's true we'll use the
17:35
shortcut connections so let's see how this is constructed first we construct a neural network by using this nn.
17:41
sequential as I also showed you in the last lecture nn. sequential is a very
17:46
important module provided by pytorch where we can connect different neural network layers together I'll also share
17:53
this link when I upload the YouTube video so you can find this in the description okay so here as you can see
18:00
we are we are chaining different layers together so if you look at the first layer the input is layer size is zero
18:07
the output Dimension is layer size is one and then we have a j activation function similarly for every layer the
18:14
input Dimension is described by the layer size the output Dimension is described by the layer size and we have
18:19
a j activation function for all of these layers um which have been constructed
18:26
right uh so here if you can can see maximum provision which I have allocated
18:31
over here is for layer sizes bracket five which means the layer sizes can at
18:37
Max uh have let's say uh 0 to five so if you see over here this is 0o this is one
18:44
this is two this is three this is four and with this five so there are six uh
18:49
there are six elements here so uh this is how the different
18:54
neural this is how the neural network is connect is created by chaining the different layers together now what does
19:00
this layer sizes zero and one mean it means that let's say for the first layer the it takes this as the input so the
19:06
input Dimension is three the output Dimension is three for the second layer the input Dimension is three the output
19:12
Dimension is three so just just like what has been shown over here the layer sizes is constructed so
19:19
let me explain this to you intuitively just so that you get a sense of Dimension right so for this layer over
19:25
here for the first layer over here uh let me zoom into this further to just
19:30
show you the first layer and how it essentially the input and the output dimensions of that layer are
19:42
constructed yeah so if you zoom into this first layer here it has three inputs so the input Dimension will be
19:48
three and it has three outputs because it's going to the next layer which has three outputs now if you look at this
19:54
next layer here um its inputs are equal to three because it it takes in the
20:00
output of the previous layer which has three neurons so it has three inputs and the outputs are also equal to three
20:06
because the next layer size is equal to three so to get the input size of a of a current layer we look at the input
20:12
dimensions of that layer or the previous layer previous layer output and then to
20:17
get the output size of the current um layer we look at the next layer Dimension size so that's how these layer
20:26
sizes are constructed and we access the particular index to get the input
20:31
Dimension and we access another index to get the output dimension of a layer also keep in mind that we are using the JLo
20:38
activation function which we learned about in the previous lecture so this is where we create uh we initialize the
20:43
layers now this forward method is where all the magic actually happens so let's try to understand this forward method in
20:50
detail if shortcut is not applied then what we'll do is that we'll just take
20:55
the output of every layer and then uh we will we will first take the output of
21:01
the first layer then we will again go to this for Loop the output of the first layer will then serve as the input to
21:07
the second layer and this process will continue until we get the output of the final layer but now let's see what what
21:14
happens when we use the shortcut so at the first layer let's say what will happen at the first layer is that we'll
21:20
take the input and then we'll add with the first layer output so this is exactly what has been shown over here at
21:26
the first layer We'll add the in put with the first layer output correct and
21:32
then what we'll do is that so then X will be X Plus layer output and then we'll go to the loop again in the second
21:38
pass of the loop what will happen is that um the output which we had computed
21:44
previously which was this X Plus input that will be added with the second layer
21:50
output so we will perform this operation then when we go to the next iteration of
21:57
the loop this out output which was computed previously will be added to the next layer's output so then this
22:03
operation will be performed similarly when we reach the end we'll apply all of the shortcut connections and then we'll
22:09
get the final output variable so at every step of the process we take the output of that layer and we take and we
22:15
add the output from the previous layers so it's an accumulation of all shortcut connection so when we reach this final
22:21
shortcut connection it's the accumulation of all the four shortcut connections which have come before
22:26
it so that's how the short shortcut connection is applied it just one simple line of code xal to X Plus layer output
22:34
so here you can see this code implements a deep neural network with five layers and each consisting of a linear
22:40
layer and JLo activation in the forward pass we iteratively pass the input through the layers and optionally add
22:46
the shortcut connections uh if self. use shortcut attribute is set to
22:52
true so now what we can do is let us use this code to First initialize a neural network without the shortcut connections
Gradient flow without shortcut connections
23:00
and later we'll initialize the neural network with shortcut connection so the neural network is
23:05
initialized like this so three 3 3 3 and one so we have five layers of three neurons and one output neuron and the
23:13
input is a three-dimensional input which is if you look at this figure here so we
23:19
are now looking at this neural network this neural network which does not have
23:24
let me show the arrow here this neural network which does not have shortcut connection and you'll see that this neural network takes in three inputs it
23:32
goes through this five layers and then we compute the output right so let if you get the layer sizes as 3 3 3 3 3 1
23:41
this is the sample input which has three input values one 0 and minus one and then we are going to set use shortcut
23:48
equal to false and create an instance of the example deep neural network and then we get this output which is model
23:54
without shortcut now what I also want to do which is the main thing here is that I want to print out the gradients at every
24:01
single layer so let me first show you how that is going to work so if you look
24:06
at each layer so if you look at the first layer uh or rather this is the first
24:12
layer right if you let me show this with a different color just for Simplicity so if you look at the first layer here
24:18
there are three neurons and each neuron will have three weights so the weight Matrix will be a 3X3 Matrix in fact for
24:27
every every layer since the input is three and the output Dimension is three for every layer we'll have a 3X3 weight
24:34
Matrix for this layer we'll have a 3X3 for this layer we'll have a 3X3 weight Matrix for this layer we'll have 3x3 for
24:40
the final layer we'll have a 3x1 weight Matrix right and when you do the backward pass you first find partial
24:47
derivative of loss with respect to all of the values in this weight Matrix and you update them so when I what I'm going
24:54
to do is that I'm going to do an iteration of the backward pass in which all these weight values will be updated
25:00
and then I'm going to find the mean of these nine values and that I'm going to call as the
25:06
mean gradient in that particular layer so if I look at this layer I have this 3x3 Matrix I'll find the mean of the
25:13
mean of all the gradients to get the mean gradient value of that layer if you are unfamiliar of the concept of
25:19
backward pass we have a neural networks from scratch Series so I really encourage you to go through that to
25:25
understand this but the main idea is that we have this output Y and then we
25:30
Define a loss which is y minus the ground truth so if the ground truth is zero then the loss will be y - 0 squ and
25:38
then we have to find the partial derivative of the loss with respect to all the weights in the first layer we
25:44
have to find the partial derivative of the loss with respect to all the weights in the first layer then we have to find the partial derivative of the loss with
25:51
sorry partial derivative of the loss with the weights in the output layer first then we have to go backwards then find the partial derivative of the loss
25:57
with the fourth layer then with the third layer and then finally we'll find the partial derivative of the loss with
26:02
the weights in the first layer that's how the backward pass will be implemented so here you can see the
26:08
target is zero which we want to match the output is the model of X and then what we are going to do is we are going
26:14
to just use a squared loss and then do loss do backward what this will do is that this will calculate gradients in
26:20
every single layer for that 9 by9 Matrix and then what I will do is that I will just take a mean of the Matrix in every
26:27
layer that will print out the gradients which I've calculated for one backward pass in every layer so I just written
26:34
what I'm doing here in the preceding code we specify a loss function that computes how close the model output and
26:40
a user specified Target is this is exactly what happens in actual code we have a predefined output and we have the
26:47
output predicted by our model we Define a loss function based on our output and the True Value then we find the
26:54
gradients of that loss with all the parameters and then we update the gradients and that's how our model gets
27:00
better and better and better as it tries to reach um lower values of
27:06
loss okay so I have here I have said that um we have 3x3 gradient values at
27:13
every layer and we print the mean absolute gradient of these 3x3 values to obtain a single gradient value per
27:20
layer uh so backward method is a convenient method in pytorch that computes the loss gradients during the
27:26
backward pass right so here you can see what we are doing is loss. backward in this one single command uh we just
27:33
compute the gradients in the backward pass for all the layers so now we can print these mean gradients for every
27:39
layer and you'll see that layer 4 has the gradient mean of 0.05
27:45
0.005 but you can see what happens as we move backward to layer three to Layer Two to layer 1 and finally to the first
27:52
layer the first layer has the gradient value of 0.002 so so this is a clear illustration
27:59
of the vanishing gradient problem right the gradient has reached so low that it has almost become equal to zero when we
28:05
reach the first layer now what we are going to do is that we are going to instantiate a model with skip connection
Gradient flow with shortcut connections
28:12
set to true so we are going to use shortcut equal to true and then we are going to print out these gradients so
28:18
we'll follow the exact same procedure we'll do this loss dot backward but now what we'll do is that we'll add another
28:25
path for the gradients to flow and as shown sh in the Whiteboard what that that will do is that that will add a
28:31
skip connection between every layer so see these green colored skip


*
*     and now let's see what the mean gradient value is in every layer so now I have put use
28:44
shortcut equal to true and I going to print the gradients at every layer so you'll see that layer four which is the
28:50
last layer as a mean gradient of 1.32 and as we do the backward backward pass
28:56
and move to layer three layer two layer one and layer zero the layer three gr layer zero gradient mean is 0.22 which
29:03
is not negligible at all in fact it's not close to zero and this value is much higher than without using the shortcut
29:11
connections this clearly shows that using the shortcut connection solves the vanishing gradient
29:17
problem in fact you also see that the gradient value stabilizes as we progress towards the first layer and does not
29:23
shrink to a vanishingly small value this is an impractical proof and demonstration that shortcut connections
29:31
actually uh help us prevent the vanishing radiant problem so in conclusion shortcut connections are very
29:37
important to overcome the limitations posed by The Vanishing gradient problem in deep neural networks uh and they are
29:44
a very core building block of large language models such as llm they facilitate more effective training by
29:51
ensuring consistent gradient flow across layers when we train the GPT model so as
29:56
we can see in this lecture we learned about the conceptual understanding of um
30:02
shortcut connections on the Whiteboard and we also looked at the from scratch coding implementation
30:09
of shortcut connection so we connected two we constructed two deep neural networks one without shortcut
30:15
connections where we saw that the gradient flow had the vanishing gradient problem as we reached the first layer
30:21
the gradients vanished to almost zero then we used shortcut Connection in the
30:26
Deep neural network and then we saw that the gradient flow has stabilized and we solve the vanishing gradient problem so
30:33
now hopefully you'll be able to better appreciate why we are actually using the shortcut Connections in the first place
30:39
it really helps to stabilize training and that helps us when we uh when we are
30:45
constructing the architecture for large language models now what I want to do is
Summary
30:50
that just want to quickly um want to quickly go to the Transformer block architecture and show
30:56
you where we use the shortcut connection so here you can see now hopefully you'll
31:01
start to better appreciate these arrows so let me rub whatever is mentioned here so that you can clearly see these
31:09
arrows yeah so now hopefully you'll start to understand and appreciate the
31:15
significance of these arrows over here these arrows are the shortcut connections between different layers
31:21
these arrows provide an alternative path for the gradient to flow so here you can see we can we can actually add a
31:26
shortcut connection between with any layers of the Transformer block and that helps the solve the vanishing gradient
31:33
problem because ultimately we'll do the backward propagation through this entire Transformer block from the output to the
31:40
input so we have to make sure that gradients don't vanish and learning continues uh in a stable way that brings
31:48
us to the end of this lecture and now we have covered several things for building the Transformer Block in the previous
31:55
lectures we covered the layer normalization J activation and feed forward neural network in today's
32:00
lecture we have covered the shortcut Connections in a lot of detail and now we are fully ready with all the building
32:06
blocks to understand the Transformer block which will be the main part which we'll cover in the next lecture so thank
32:13
you so much everyone this brings us to the end of this lecture I hope you are learning a lot and in detail my whole
32:20
goal is that every lecture should provide you intuitive understanding theoretical understanding as well as
32:25
coding knowledge and I try to cover all of these three aspects in my lectures I hope you are liking these lectures if
32:32
you have some doubts or questions please put it in the YouTube comments and I'll try to address it in the next video
32:37
thanks a lot everyone and I look forward to seeing you in the next video

***



