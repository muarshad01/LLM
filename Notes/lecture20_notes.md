## Layer Normalization 

#### Vanishing gradient problem
* Training DNN with many layers can be challenging due to two things as it can either lead to:
1. Vanishing gradient problem or it can lead to vanishing/exploding gradient problem or
2. unstable training dynamics
3. Layer normalization improves the stability and efficiency of NN training.
4. __Main idea__: Adjuct outputs of NN to have mean=0 and variance=1. This speeds up convergence.
***

* 5:00 

#### Gradient depends on layer output
1. If the layer output is too large-or-small, gradient magnitudes can become too large-or-small. This affects training. __Layer normalization__ keeps gradients stable.
2. As training proceeds, the inputs to each layer can change __(internal covariate change)__. This delays convergence. __Layer normalization__ prevents this.

***

* 10:00

* $$\\{x_1, x_2, x_3, x_4\\}$$

* $$\text{Var} = \frac{1}{4}\[(x_1-\mu)^2 + (x_2-\mu)^2 + (x_3-\mu)^2 + (x_4-\mu)^2\]$$

* $$Normalized = \[\frac{(x_1-\mu)}{\sqrt{var}}, \frac{(x_2-\mu)}{\sqrt{var}}, \frac{(x_3-\mu)}{\sqrt{var}}, \frac{(x_4-\mu)}{\sqrt{var}}\]$$


normalization really helps to prevent this layer normalization make sure that um since we
10:17
are normalizing which means that as we'll see the variance of the standard deviation is kept to one we'll make sure
10:23
that the mean and standard deviation of the output from every layer is fixed and
10:29
this reduces the problem of internal coari shift which accelerates convergence so there are two main
10:35
reasons why layer normalization is employed the first reason is that it keeps the training procedure stable by
10:42
preventing the vanishing gradient or the exploding gradient problem and the second uh the second major reason for
10:49
using layer normalization is that it prevents or reduces the problem of internal covariate shift and that
10:56
accelerates convergence and we get to a result for faster so that's why layer normalization is employed not just in
11:03
the GPT or the llm architecture which we are going to see right now but in fact in many deep learning architectures
11:09
layer normalization is very frequently used okay uh so what exactly is the main
What is layer normalization ?
11:16
idea of layer normalization it's very simple so we look at a specific layer
11:23
and we'll look at the outputs of that specific layer and what we'll do is that we'll adjust those outputs so that they
11:29
have a mean of zero and they have a variance of one let me illustrate this through a simple example let's say you
11:35
are looking at a neural network and these four are the outputs from one specific layer of the neural network
11:41
right uh so the four outputs are X1 X2 X3 and X4 and X1 is equal to 1.1 X2 is8
11:51
X3 is 2.3 and X4 is 4.4 in layer normalization what you do is you find
11:57
two quantities first you find find mean of this so the mean will just be X1 + X2
12:02
+ X3 + X4 / 4 so in this case it will be 2.15 and the second thing which you do
12:08
is you find the variance so the variance will be 1X 4 because there are four um
12:14
quantities here and then we'll sum up X1 minus the mean whole square + X2 - the
12:20
mean square + x3 - the mean whole square + X4 minus the mean whole Square so that
12:26
gives us the variance value now what we do is we perform the normalization procedure which means that for every
12:33
variable we subtract the mean and we divide by the square root of the variance which is the stand standard
12:38
deviation so X1 will be replaced by X1 minus mu / square root of variance X2
12:45
will be replaced by X2 - mu / square root of variance X3 will be replaced by
12:50
x3 - mu / square root of variance and X4 will be replaced by X4 minus mu / square
12:57
root of variance so when you do this normalization it leads to these four values do you notice something about
13:04
these four normalized values if you add these together in the numerator you will
13:09
have X1 + X2 + X3 + X4 - 4 * mu so that will be zero which means that the mean
13:17
of this these normalized values is equal to zero and if you compute the variance of these normalized values through this
13:23
formula you'll see that the variance of these normalized values is is
13:29
actually equal to 1 that's the most important thing which uh which is which
13:35
you should realize or you you should understand is that after performing the normalization procedure uh the values
13:42
which you get these four values their mean is equal to zero and their variance is equal to one that's the whole idea
13:49
behind normalization the in normalization we adjust the outputs of every layer of
13:54
neural network to have mean of zero and variance of one and it turns out that this simple procedure helps us in the
14:02
stability neural network training and it also helps us reduce the problem of the internal coari
14:08
shift um so let us actually uh see this in code but before that I want to tell
14:14
you where the layer normalization is used and uh we discussed this at the start of the lecture but um the input is
14:22
converted into an input embedding then we add the token embeddings and then here right we feed it before going into
14:28
the multi head attention we have a layer normalization layer so that the inputs to the multi-ad attention are normalized
14:35
even after the multi attention there is a layer normalization layer before feeding into the neural network module
14:42
within the Transformer block remember this Blue Block here is the Transformer block so the layer normalization layer
14:48
appears two times here and then it appears once more again outside the
14:54
Transformer uh so in GPT and modern Transformer architecture layer normalization is typically applied
15:00
before and after the multi-head attention module like what we have seen over here and it also appears once

***

15:07
before the final output layer and we saw this when we coded in the last lecture here if you see within the Transformer
15:14
block layer normalization appears two times before and after the multi-ad attention but even before the output it
15:20
we employ it once so overall it appears three times that's why we need to define a separate class of the layer
15:27
normalization this one figure which I'm which I've shown over here actually illustrates the procedure of layer
15:34
normalization um let's say we have a neural network layer these are the five inputs uh to the neural network right
15:41
and these are the six outputs of the neural network without the normalization you'll see that their mean is not equal
15:47
to zero and their variance is not equal to one but after we perform normalization on these layer outputs
15:54
which means that for from every output here we are going to subtract the mean and divide by the square root of the
16:00
variance so you'll get these as the resultant values after applying layer normalization and if you take a mean of



***


16:07
these values you'll get that mean is equal to zero and the variance of these values is equal to one this is the simple
16:14
illustration which describes uh what happens underneath the hood for layer
16:21
normalization now what we are going to do is that I'm going to take you through code and we are going to implement layer
16:26
normalization first on a neural network which looks like this and then we are going to create a separate class of
16:32
layer normalization which we can integrate within our GPT architecture so let's jump into code right now all right
Layer normalization basics in Python
16:40
so here's the code file for layer normalization the first thing which we
16:46
are going to do is start out with a simple example to illustrate how layer
16:51
normalization is implemented in practice and then we will actually fill out this
16:56
layer normalization class which we had created in the previous lecture so let's
17:02
get started what we are doing here is that we are essentially let me take you to the white board to demonstrate what
17:08
we are doing here we'll have a simple neural network layer and the neural network is
17:15
constructed such that we have two batches of inputs so here is batch
17:20
number one and here is batch number two and each batch has five inputs so batch
17:26
number one has X1 X2 X3 X for X5 and batch number two has X1 X2 X3 X4 H5
17:34
X5 now here we are looking at one layer of neurons and there are six neurons
17:39
here so when these inputs essentially pass through this first layer of neurons
17:45
we have the output which is produced and uh there will be six outputs for batch
17:50
number one which is y1 Y2 Y3 y4 y5 Y6 and there will be six outputs for batch
17:56
number two y1 Y2 Y3 y4 y5 Y6 so if you look at the
18:02
inputs the shape of the inputs will be two rows and five columns because we have two batches and each batch will
18:09
have five inputs right then we have a sequential layer which essentially takes in five inputs and it has six outputs
18:17
this sequential layer is this uh the second layer which I've shown you over here this layer of six neurons and after
18:25
every neuron here we essentially have The Rao activation function so which has been mentioned over here if
18:31
you don't know what Rao activation function is it's fine for this lecture we don't need to understand
18:38
this uh the output of the layer is that the layer is then applied on this input
18:43
batch and we get the output can you try to understand why the shape of the output is like this so here you can see
18:50
that we have two rows and we have six columns if you look at the first row
18:56
this represents the six outputs from the first um batch and if you look at the second
19:02
row this represents the six outputs from the uh second batch and that's what
19:09
exactly being shown here y1 Y2 Y6 and batch two has y1 Y2 up till Y6 so this
19:15
can be y1 this is Y2 this is Y3 this is Y6 for batch one and this is y1 this is
19:22
Y2 and this is Y6 for batch number two okay so this is the layer which we have
19:28
and now what we are we are going to do is that after this layer we are going to uh apply the batch normalization uh
19:34
sorry we are going to apply the layer normalization so uh here I have simply
19:42
explained that we have a neural network which consists of a linear layer followed by The Rao activation layer to
19:48
quickly illustrate what The Rao activation function actually looks like take a look at this this image over here
19:56
so if x is positive The Rao is just y = to X but if x is equal to negative The
20:01
Rao is zero so there is a nonlinearity here that's The Rao activation
20:07
function now what we are going to do as I mentioned is we are going to apply layer normalization so the way it is
20:13
applied is very similar to the Hands-On example which we saw on the white board over here this
20:18
example uh the same thing will be applying to the first batch and the same thing will be
20:25
applying to the second batch so what we'll be doing is that when you look at the first batch we will do y1

***

20:33
minus so y1 will be replaced by y1 - mu divided by uh square root of we write
20:41
this again divided by square root of variance Y2 will be replaced by Y2 minus
20:50
mu uh divided square root of variance
21:00
and like this similarly the last output which is Y6 here Y6 will also be
21:06
replaced with Y6
21:12
minus mu divided by square root of
21:19
variance and first we process the first batch and then we process the second batch in a very similar manner so now
21:26
let's go to code to see how this is done so now what we are going to do is that
21:32
we we have the output which is this tensor right and then we are doing output do mean Dimension equal to minus1
21:39
why Dimension equal to minus1 because we have to take the mean along the columns so first we look at the first batch
21:46
outputs and we want to take the mean of this so we do output do mean Dimension equal to minus1 and this keep Dimension
21:53
equal to true that is very important because if we don't include keep Dimension equal to true The Returned
21:59
mean would be a two dimensional Vector instead of a two into one dimensional
22:04
Matrix so essentially uh if you use keep dim equal to true the output which you
22:11
get for the mean is this so the first value here corresponds to the mean of
22:17
the first batch the second value here corresponds to the mean of the second batch since we used keyd equal to true
22:23
the shape of this output is that it's a matrix uh or rather uh yeah it's a a two
22:28
into one dimensional Matrix over here right now if we did not use keep dim equal to true this would not be a matrix
22:35
in fact it would just be a two- dimensional vector and that's generally not good because it's good for the
22:42
dimensions to be preserved as we are doing all of these calculations similarly for the variance
22:48
what we are doing is that we are taking the variance across the column for both the batches and we use keep them equal
22:54
to true and then you print out the mean and then you print out the variance for every batch
22:59
so for the first batch of data the mean is. 1324 for the second for the first
23:04
batch of data the variance is 0.02 31 for the second batch of data the mean is
23:10
216 2170 and for the second batch of data the variance is
23:15
0.398 so remember the two uh uh two commands which we have used
23:21
here dim equal to minus1 because we have to perform that operation along the columns and keep dim equal to true
23:28
because we have to retain um the dimension of the final mean and the
23:34
variance Matrix which we have if we did not use keep D equal to True later when we subtract this mean from every
23:40
individual element it will lead to some problems so we want to avoid that so in
23:45
this text over here I have just explained why we used keep dim equal to true and why we used Dimension equal to
23:51
minus1 so if you have some confusion along those lines please read this text when I share this Google or when I share
23:58
this Jupiter notebook with you great and now what we are going to do is that we are going to uh
24:05
subtract the mean so like over here we are going to subtract the mean and divide by the square root of variance so
24:13
we have the output Matrix which is there we are going to subtract the mean which is now again you can see the mean is
24:19
also a tensor which has two rows and one column and we are going to subtract the
24:25
mean from the output and we are going to divide by the square root of variable this is the main normalization
24:31
step so this is my output now and the normalized layer outputs are given like
24:37
this uh the first row again corresponds to the normalized outputs of batch one the second row corresponds to the
24:44
normalized outputs of batch number two so here I'm just printing out the mean and variance of the batch one and batch
24:51
two so if you look batch one and batch two so if you look at the mean you'll see that the mean of the first batch is
24:56
almost close to zero this is is 10us 8 which is really very close to zero we
25:02
can approximate it to zero for the second batch the mean is again very close to 10us 8 again that's almost
25:09
equal to zero and if you look at the variance for both the batches you'll see that the variance is equal to one
25:15
awesome this is exactly what we wanted right which means that the layers have been normalized now so note that the


***

25:21
value 2.9 into 10- 8 is the scientific notation for 2.9 * 10us 8 this value is
25:28
very close to zero but not exactly zero due to small numerical errors in Python
25:34
what we have is this uh we can turn on the turn off the scientific mode So
25:39
currently the scientific notation is on that's why we are getting these uh um
25:45
values which have been represented in the scientific notation we can turn off the scientific notation and then let's
25:51
print out the mean and the variance so you'll see that the mean for both the batches is equal to zero and the
25:57
variance for both the batches is equal to one great and now we'll achieve the
Coding the Layer normalization class in Python
26:02
goal which we started out this lecture with we want to create a class for layer normalization um what would be the
26:11
output of this class basically this class will take in the um the output of
26:16
a layer and it will apply the um normalization to that so let's look at
26:23
where this layer normalization step is implemented so the layer normalization step is is implemented here the layer
26:30
normalization step is implemented here so at both of these places when we get
26:36
uh when the inputs are received to this block and when the input is received to this block um we have certain number of
26:44
tokens uh which is let's say let's say we are looking at the uh Contex size for the number of
26:53
embedding vectors which we have but the main thing which I want to point out is that the the number of columns which we
26:59
have is equal to the embedding size and this is the embedding Vector Dimension which we are using so for gpt2
27:09
this embedding size is equal to
27:15
768 so when we look at the so when we look at let's say the first row over
27:21
here the first row corresponds to the embedding for the first token which is an input to this layer normalization
27:27
right so so we will take the mean and we'll take the variance along the column
27:32
Dimension which is 768 so we'll take the mean of all of this and take the variance and then do the normalization
27:39
similarly we'll do this for every single row um for the input to this layer
27:44
normalization as well as the input to this layer normalization so if you see the dimension when an instance of this
27:51
class is created we have to pass in the embedding Dimension why do we have to pass in the embedding Dimension because
27:57
we'll see that we are going to implement something like scale and shift these are trainable parameters but the size of the
28:04
scale and shift will be governed by the embedding Dimension and that's the same as the input Vector to the layer Norm
28:11
module and that will be the same as the output Vector so the input to the layer Norm
28:16
module will have certain number of rows but it will have embedding columns the output of the layer normalization which
28:22
is Norm X will have the same dimensions as the input because normalization does not change dimensions and we are going
28:28
to scale the output with the scale and shift I'll come to that in a moment so the main part of this layer
28:35
normalization class is the forward method which takes in the input which I described so when you think of the input
28:40
think of the input as having certain number of rows but mostly focus on the number of columns which will be the
28:46
embedding dimensions in each row let's say we have 768 embedding Dimension so in the first step what we do is that
28:53
along the column we take the mean ex exactly similar to what we
28:58
actually did over here just keep this example in mind um then what we do is along the
29:04
column we take the variance and then we subtract the mean and then we divide by
29:09
the square root of variance note that we have added uh this small variable Epsilon so this is a small constant
29:16
which is added to the variance to prevent division by zero during normalization so we don't want to divide
29:21
by the square root of zero right so we add a small uh variable here which is called self. Epsilon or Epsilon so this
29:30
is the output of the normaliz layer normalization but we do one more step here which is we multiply by the scale
29:36
and we multiply by the shift so let me explain what the scale and shift are uh
29:42
just like we are training we are going to train the embedding parameters the positional embedding parameters the
29:47
neural network parameters in the GPT architecture the scale and shelf uh the
29:52
scale and shift are two trainable parameters which have the same Dimension as the input
29:59
that the llm automatically adjusts during training if it is determined that doing so would improve the model's
30:05
performance on the training task so this allows the model to learn appropriate scaling and shifting that best suits the


***


30:12
data uh it is processing so you can think of it as knobs turnable knobs or
30:17
fine-tuning parameters which we have added just to make sure that
30:22
uh the layer normalization proceeds smoothly or if we want to tweak the layer normaliz ation of bit we can
30:28
always do it with the scale and with the shift these are trainable parameters which means we don't specify the values
30:34
of these parameters at the start and remember that the these scale and shift
30:41
have the embedding Dimension so that we can multiply the U the scale and the
30:47
output of the normalization layer together so remember when you look at each row each row has 768 Dimensions
30:55
right um or the embedding dimensions and this self do scale is again a vector
31:01
which has embedding Dimensions so you can do element wise multiplication here and then you can do an element twise
31:07
addition here and the Epsilon parameter is set to a small value such as 10us 5
31:13
but the main step which is being performed in this normalization layer is that we take the input and then every
31:19
row of the input we subtract with the mean subtract the mean and divide by the square root of variance so that the mean
31:25
of the resultant row is zero and the standard deviation of the variance is
31:30
equal to 1 that's it this is the layer normalization class and uh when we bring
31:38
all the elements of the GPT architecture together we'll see that we'll replace this dummy class now with the layer
31:44
normalization class which we just defined I just have one small note
31:49
towards the end which is regarding biased variance so uh in our variance
Bessel’s correction
31:55
calculation method we have opted for an implement mation detail by setting unbiased equal to false so here so you
32:02
might be thinking what happens if unbiased equal to True right if unbiased equal to True we'll apply something
32:08
which is known as besels correction and in besels correction when we do the variance calculation we divide by n
32:15
minus one instead of dividing by n like what we did right now so if you look at this uh
32:22
this implementation which I did on the Whiteboard to calculate the variance I'm dividing by n right which is the number
32:28
of um number of inputs if we do the besels correction we divide by nus one
32:36
so this four is replaced by three this does not matter too much in the case of large language models where the
32:42
embedding Dimension n is so large that the difference between n and n minus one is practically negligible if you are
32:48
interested to learn about bessels correction um there are many good
32:54
articles on this and you will see that it's the use of uh n minus one instead of n in the formula for sample variance
33:01
and that leads to an unbiased uh variance but for now it totally works
33:07
for us if we set the unbiased equal to false it does not lead to too much of a
33:12
difference so the reason we choose this approach is to ensure compatibility with gp2 Mod gpt2 models normalization layers
33:20
and uh it reflects tensor flows default Behavior which is to set unbiased equal to false this was actually what was used
33:27
to implement the original gpt2 model and that's why we are also using unbiased equal to false over here so now that we
Testing the Layer normalization class
33:34
have defined a class for layer normalization let's try the layer normalization in practice and apply it
33:40
to the batch input right so we have defined this batch over here this batch over here let's try to
33:48
uh pass in this batch through the layer normalization layer so I'm defining a layer normalization class I'm creating
33:54
an instance of it and as I told you over here we need to pass in the embedding Dimension right so I'm saying that the
34:01
EM embedding Dimension is equal to five so I'm taking in batch example so let's
34:06
see where batch example has been defined yeah so this is the batch
34:13
example I'm taking in this example and then what I'm doing is that I am
34:19
applying the layer normalization layer to this batch and then it will do all of the
34:24
steps which I have mentioned over here first it will will find the mean over all the dimensions and then it will
34:31
essentially find the variance it will subtract the mean and then divide by square root of variance so let me go to
34:37
the batch example again and tell you what it's going to do um let me print
34:43
out the batch example actually right over here so if you print
34:52
out print out the batch example you'll see that it's this right so so what this
34:59
layer normalization will do is that it will first take the mean of this subtract mean from all of these elements
35:05
and divide all of these elements by square root of the variance of this first batch then it will do the same for


***


35:12
the second batch and then when you get the mean and the variance of the resulting
35:18
output so output Ln is my resulting output and if you get the mean and the
35:23
variance you'll see that the mean for both the batches is zero and the standard deviation or the variance for
35:29
both the batches is equal to one so as we can clearly see based on the results the layer normalization code works as
35:36
expected and normalizes the value of each of the two inputs such that they have a mean of zero and variance of
35:43
one uh so I hope you have understood this concept I tried to explain this through through whiteboard as well as
35:50
through code and I hope that you have understood the basics as well as the coding aspect of it now let's see what
35:56
all we have learned in this lecture so far so in this lecture what we have learned is that we learned about layer
36:04
normalization and let's see how this fits into the context of the entire GPT architecture so to master the GPT
36:10
architecture we need to learn all of these building blocks in the previous lecture we learned about the GPT
36:16
Backbone in this lecture we looked at layer normalization in the next lecture we are going to look at this thing
36:22
called G activation then we'll look at feed forward Network and then we'll look at shortcut connections so in separate
36:29
lectures we're going to cover all of this and then you will see that all of this essentially comes together to teach
36:34
us about the Transformer block and only then we'll be fully equipped to understand the GPT
36:41
architecture I don't think there are any other YouTube videos out there which cover the GPT architecture from scratch
36:48
in so much depth but I believe this much understanding is necessary for you to truly understand how the llm
36:54
architecture works one last point which I need to mention is is that in this lecture U sometimes I said batch
Layer vs Batch normalization
37:01
normalization and then I corrected to layer normalization right if you also have this confusion remember that layer
37:08
and batch normalization are very different from each other in layer normalization we usually normalize along
37:14
the feature Dimension which is the columns and layer normalization does not depend on the batch size at all no
37:21
matter what the batch size is we just take the output of every layer and normalize it based on the mean and the
37:27
standard standard deviation batch normalization on the other hand we do the normalization for an entire batch so
37:33
it definitely depends on the batch size now U the main issue is that the
37:38
available Hardware typically dictates the batch size if you if you don't have too powerful of a hardware you might
37:45
need to use a lower batch size so batch normalization is not that flexible because it depends on the batch
37:51
size which depends which depends on the available Hardware on the other hand layer normalization is pretty flexible
37:58
it leads to more flexibility and training for it leads to more flexibility and stability for
38:04
distributed training so if you have if you are in environments which lack resources and Hardware capabilities are
38:11
not there which leads to low batches Etc or if you don't want to care about batch size you want the normalization to be
38:18
independent of the batch size layer normalization is much better in terms of flexibility so don't make the error or
38:25
don't make the confusion between layer normalization versus batch normalization this brings us to the end
38:32
of today's lecture as I mentioned in the subsequent lectures we'll talk about Jou we'll talk about feed forward Network
38:38
and we'll also talk about shortcut connections and then later we'll see how all of it comes together to make the GPT
38:45
architecture thank you so much everyone and I look forward to seeing you in the next lecture

***















