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

* $$\sigma^2 = \frac{1}{4}\[(x_1-\mu)^2 + (x_2-\mu)^2 + (x_3-\mu)^2 + (x_4-\mu)^2\]$$

* $$Normalized = \[\frac{(x_1-\mu)}{\sqrt{\sigma}}, \frac{(x_2-\mu)}{\sqrt{\sigma}}, \frac{(x_3-\mu)}{\sqrt{\sigma}}, \frac{(x_4-\mu)}{\sqrt{\sigma}}\]$$

***

* 15:00

5. In GPT-2 and modern Transformer architectures, layer normalization is typically applied before and after the multi-head attention module and before the final output layer.

***

* 20:00

***

* 25:00


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



















