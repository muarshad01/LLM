## Model initialization with pre-trained weights

***

* 5:00

* Step-1: Load pre-trained GPT-2 weights
* Step-2: Modify the architecture by adding a classification head

* [OpenAI GPT-2 Weights](https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights)

***

* 10:00

***

* 15:00

***

* 20:00

* Step-3: Select which layers you want to finetune

* Since we already start with a pretrained model, it is not necessary to finetune all layers.
* This is because the lower layers capture basic language structures and sementics, which are applicable across a wide range of tasks and datasets.
* Finetuning last layers is enough.

* We will finetune:
  * Final output head
  * Final transformer block
  * Final LayerNorm module
  * We'll freeze all other parameters

* Step-4: Extract last output token

***

* 25:00

hardcoding the output nodes but we are getting the number of classes and we are setting the final number of output nodes
26:04
equal to that so for example if you have three classes such as technology sports or medicine our same code is going to
26:11
work for this modification because then the final number of output nodes will be equal to three that just a small detail
26:17
which I wanted to mention so before we construct the model architecture we can print the original model architecture so
26:24
we print out the original model architecture and you can see that there are 12 Transformer blocks so here you
26:31
see so the number of Transformer block goes from 0 to 11 that's why there are 12 Transformer blocks and there is an
26:37
output projection layer at the end um awesome so this is the output
26:42
head so here you see this has input dimension of 768 and output feature dimension of 50257 this is the one which
26:49
we plan to change to two instead of 0257 we just want two as the output
26:55
features uh so as discussed earlier the GPT model consists of embedding layers token embedding and positional embedding
27:02
followed by 12 identical Transformer block followed by a final layer normalization and the output layer
27:09
output head so what we are going to do is that we are going to replace the output head with a new output layer as
27:15
we have Illustrated in this figure over here we want to replace this original output head with this new output head
27:22
right to do that to get the model ready for classification fine tuning we first freeze the model which means that we are
27:29
going to first make all the layers non-trainable so the way to freeze the
27:34
entire model is that we just do for all the parameters in model. parameters we
27:40
say that requires grad is equal to True equal to false which means that we are not going to update this parameter at
27:46
all which means that we are going to freeze all the model parameters then we are going to we are going to tell this
27:53
model which are the parameters which we are going to find tune so as I mentioned
27:58
there are three sets of parameters which we are going to fine tune there is the final output head there is the final
28:04
Transformer block and there's the final layer Norm module so first let's modify the final output head architecture so we
28:11
are going to say that model. output head is now the size is input features are
28:17
the embedding Dimension the input feature Remains the Same that's equal to 768 the output features now is equal to
28:24
the number of classes so if number of classes equal to two the output features will will be two which is in this case
28:29
spam versus no spam if number of classes is three the output features will be equal to three so that's one simple
28:35
change which you make this indicates to pytorch that these parameters need to be updated these parameters need to
28:42
change um okay so this new model model. output output layer has its requires
28:50
grad attributes set to True by default which means that it's the only layer in the model that will be updated during
28:56
training so here we have freezed all the parameters but when we change the model output head structure the new model
29:02
output head output layer has requires grad attribute set to True by default which means we are going to update those
29:09
parameters uh so additionally as I mentioned we are going to update two more sets of parameters we are going to
29:15
look at the last Transformer block the 12th Transformer block and we are going to modify its parameters as well and we
29:22
are going to look at the final layer normalization uh which connects the output of the Transformer block to the
29:29
final output head these we are going to make as trainable so I mentioned this to you over here right the final
29:34
Transformer block and the final layer normalization those we are going to make trainable so here you see what I'm doing
29:41
I'm saying that you look at the Transformer blocks and this minus one indicates that you look at the last
29:47
Transformer block you look at the parameters in the last Transformer block and then you set all of those parameters
29:53
to be equal to trainable by setting requires grad equal to true and then you look at look at the model final
29:59
normalization parameters which will be shift and scale and those you change to params do required grad equal to true
30:05
now you see there is a lot of scope for experimentation here you can even switch this off and try to see the result you
30:11
can make the parameters of last two Transformer blocks to be trainable you can switch this off you can see the
30:17
results you can maybe make this as false and check the results so whatever code which I'm showing to you right now
30:23
there's a lot of room for exploration over here awesome so now you can see that we
30:29
have added a new output layer and Mark certain layers as trainable or non-trainable great so let us just take
30:36
one sample input so the sample input corresponds to do you have time uh so


***



30:42
this has four token IDs and what we are going to do is that we are going to pass the inputs through our model now and
30:47
let's see the output so as expected do has two tokens you has two tokens have
30:53
has two tokens and time has two tokens corresponding to spam or no spam and as I mentioned we are going to look
30:59
at the last we are going to look at the last row and we'll extract only the last row to predict whether spam or no spam
31:07
and the reason we saw was that the last row the last token is the only one with an attention score to all other tokens
31:13
so it contains maximum information until now we have not done the training the models the parameters in this output
31:21
head um the parameters in this output head and the parameters in the final
31:27
Transformer block final layer normalization are still random they have not been trained on our spam and no spam
31:33
data set but uh that's fine currently I'm I just want to show you the output
31:40
uh the output will be random for now but I just want to show you the dimensions so when the input is do you have time
Extracting last token output
31:46
you will see that for four tokens for each of the token there are two outputs here and we are going to look at the two
31:52
outputs of the last token and we are going to see whether for spam the Valu is higher or no spam higher and then we
31:58
are going to choose the one with the higher value and make our prediction like that okay uh so remember that we are
32:07
interested in fine tuning this model so that it returns a class label that indicates whether the model input is
32:12
Spam or not a Spam to achieve this we don't need to F tune all the four output rows as I mentioned we don't need to F
32:18
tune all these four output rows but we can focus on a single output token we
32:24
will focus on the last row corresponding to the last output token since it contains all the information so to
32:29
extract the last output token we are simply going to use this command outputs colon minus one and colon which will
32:35
extract the last output Row from this tensor uh so the reason why minus one is
32:43
comes in the middle here is that look at the tensor Dimension the number of rows is in the second second position right
32:50
so that's why the second position we have to specify minus one since we are looking at the last row and when you
32:56
specify this you'll see that out of this 4 the last output is extracted which is minus 3.58 983 and 3.99 02 so until now
33:05
what we have done is that we have just modified the model architecture right we have not trained we have not trained the
33:10
model on our data set and to train the model on our data set what we need to do is that we need to define the loss we
Next steps
33:17
need to define a loss function and then we need to implement back propagation that's when the model will be
33:22
trained so what we'll be doing in the next section is that we'll be detailing the process of transform forming the
33:28
last token into class label predictions and then we'll calculate the model accuracy and then we'll calculate the
33:34
loss function once we have the loss function based on our underlying training data which has been collected
33:40
from this machine learning ucne repository once we have the loss function then we are ready to do back
33:45
propagation and then we are ready to fine tune the parameters so then we'll do the training and testing after that
33:50
in the subsequent lectures so next lecture we'll focus on calculating the classification loss and accuracy
33:58
uh thank you so much everyone this brings us to the end of the lecture we are now quite close to performing this
34:03
Hands-On project and taking it to completion because until now what we have done is that we have reached these
34:08
steps we have reached step number here where we now the model is ready to be fine tuned now in the next step we have
34:14
to just implement the loss and the accuracy evaluation utilities and then we'll finetune the model test the
34:20
finetune model on new data as well so there are lot of fun lectures coming up and at the end of this set of lectures
34:27
you will will have build your own classification fine tuning completely from scratch uh thank you so much
34:33
everyone I hope you learned a lot and I look forward to seeing you in the next lecture


***





