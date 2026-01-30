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

***

* 30:00

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






