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

***

* 20:00

***

* 25:00

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


***



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
