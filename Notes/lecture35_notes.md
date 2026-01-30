## Model initialization with pre-trained weights


***

* 5:00

gpt2 and then we will modify the model architecture bit for fine tuning uh and then finally we will
5:33
Implement evaluation Utilities in today's lecture we will do step number four step number five and step number
5:40
six uh so it will be a comprehensive lecture and let's get started now we
5:46
have these data loaders training testing and validation so now we come to the GPT architecture so here you see in this
5:52
lecture Series so far we have constructed this architecture which is uh which I zoomed in on the screen right
5:58
now uh don't focus on these two images on the right just look at this Gray colored architecture this is the llm
6:04
architecture which we have focused up till now what we are going to do first is that we are going to first load the
Loading OpenAI GPT-2 pretrained weights
6:10
pre-trained gpt2 weights into this architecture and if you have not seen
6:15
the previous lectures let me just give you that recap open a has basically made the gp2 gpt2 weights free freely
6:22
available to the public and they have made weights available for multiple parameters 107 million 124 million 7 74
6:30
million Etc opena even had a public announcement for uh gpt2 and that they release these
6:37
weights what we are going to do is that instead of pre-training ourselves which would involve a huge amount of cost and
6:44
computational resources we are just going to load the pre-trained GPT 28s into this GPT model and we have done
6:51
this before when we uh when we trained our large language model so let's get
6:57
into code right now to see how this part is done and then we'll move to step number two okay so I'm going to take you
7:04
to code right now most of this lecture which we are going to do today will involve going through the code so I'll
7:10
explain each part of the code to you step by step okay so now our first task is to
7:16
prepare the model which we will use for classification fine tuning to identify spam messages and what we are going to
7:22
do is that we are going to use the same architecture which we have used and then load the pre-trained weights later later
7:29
we'll do a slight modification at this final layer but for now let's just see how to load the pre-train weights so you
7:36
can see that uh GPT when you download the weights from gpt2 you'll have models
7:41
small model medium large and the extra large we are going to choose the GPT small with gpt2 small which has 124
7:49
million parameters right so we have a base configuration which means that the vocabulary size is 50257 the context
7:57
length is 1024 the dropout rate is zero and the query key value bias term is set
8:03
to True these are the same values which were used when gpt2 was trained and since we are recycling those weights we
8:09
are using the same weights the pre-trend gpt2 weights we are retaining this configuration we are going to upload
8:16
this base configuration with the model which we are going to choose and here you see we are choosing from this model
8:22
configs dictionary we are choosing this choose model equal to GPT small so we have updated this base configuration
8:28
with our model configuration is GPT to small in this last code what we are doing is that if our training data set
8:36
has some text messages whose maximum length is greater than the context length which is
8:41
1024 uh what we are going to do is that we are going to set the maximum length
8:46
equal to the context length in short we are going to remove all of the uh tokens
8:53
which have higher length than the context length this is because our llm
8:58
can only process tokens with the maximum length equal to the context length and that is equal to 1,24 in this particular
9:06
case awesome now that we have the configuration ready what we are going to do is that we are going to uh we are
9:15
going to download the gpt2 parameters and there is a specific way to download
9:21
the gpt2 parameters and for that I'm going to I'm going to take you through
9:27
the code file right now so let me take you through vs code yeah so as I mentioned there is a
9:34
specific way to download the GPT parameters and we have written this code called download and load gpt2 what this
9:41
code does is that it basically downloads the weights and the entire model details
9:46
which have been prescribed by open when they made the gpd2 weights public so we
9:52
are going to download all these files and then we are going to convert or we
9:58
are going to return two things we going to return settings and we are going to return params what settings is basically
10:04
is just these configurations the vocabulary size the context length the embedding Dimension number of attention
10:11
heads and number of Transformer layers what this params is basically is that the params is a dictionary uh and
10:18
this dictionary has has been constructed in a very specific format so here is how the params dictionary looks like when


***


10:25
the params dictionary is returned we get five we get a dictionary with five Keys we get token embeddings we get
10:32
positional embeddings we get all the parameters which are present in this Transformer block which I'm marking in
10:38
blue right now then we get the final normalization layer scale and shift parameters there's a separate lecture
10:45
where we actually explain all of these parameters and how these keys are imported from gpt2 but for now all you
10:52
need to know is that when you run this function when you run this function load
10:58
uh download and load gpt2 it will give you the settings dictionary which is a list of the uh gpt2 configuration and it
11:06
will give you the params dictionary which consists of all the parameters organized in a specific format you get
11:13
the token embeddings positional embeddings the Transformer layer parameters and the output final
11:18
normalization layer parameters as well essentially the params dictionary contains all the parameters you will
11:24
need now let's go back to code again uh right so what we are going to
11:30
do is that from this file called GPT download 3 we are going to import download and load gpt2 that's the
11:36
function which I just showed you right now and when you run this function you have to pass in two arguments the first
11:41
is the model size and the model size we are going to get from this uh choose model and that's going to be 124 million
11:48
that's the model size and then you have to pass in the directory where you want to store the parameters so the directory
11:54
I'm passing it as gpt2 when you run this code you will get two dictionaries settings and the params
12:01
settings will contain the configuration file params will essentially contain all the different um parameters of gpt2 then
12:11
what we do is that we initialize a instance of the GPT model class which we have defined earlier uh and then we call
12:18
this function load weights into GPT what this function does is that it takes the params dictionary and then it loads all
12:25
the weights from params into our model so let me show you what this load dictionary does it takes this
12:33
model uh it takes this model which we have created and at all the different
12:38
places of this model where there are trainable parameters such as multi-head attention layer normalization feed
12:44
forward neural network token embedding positional embedding whatever I have marked with an arrow right now has pable
12:50
parameters right so this load this load weights into GPT function
12:55
what this does is that it takes the GPT to parameters and it loads all of those
13:01
parameters into this model basically you can think of our model being fully
13:06
equipped with the best parameters which have directly directly been loaded from gpt2 in one of the previous lectures
13:13
which is called pre-training uh we have explained this entire code what is this function load
13:18
weights into GPT the GPT model class etc for now just you can just follow along
13:24
by understanding that we are loading the parameters of gpt2 into our model so
13:30
that our model is already pre-trained and if you have already loaded this before this should run very
13:36
fast because the file already exists if you are running this for the first time please keep in mind that the total U
13:44
parameter file size which is provided by gpt2 if you see these seven files if you
13:50
add it up it comes to be around 500 megabytes so it may take time depending on the internet speed once this is
13:56
downloaded you'll see that your model is now updated with all the weights from gpt2 you can even test whether the model
14:03
was loaded correctly so you can pass in the input text every effort moves you and then you have the output function
14:10
generate teex simple which we had defined earlier what this function does is that it takes in the input text it
14:16
passes the input text through our model and then it generates an output so it generate 15 new
14:22
tokens so here you can see that every effort moves you is the input and then the 15 new tokens are forward dot the
14:29
first step is to understand the importance of your work awesome right which means that the pre-train parameters are working because this is
14:35
reasonable this text makes sense it is proper English now until now we are at a
14:42
point where the GPT model parameters are loaded into our architecture now we come
14:47
to the next stage where we have to start fine tuning the model right but before we start fine tuning the model as a Spam
14:54
classifier let's see if our model can already classify spam messages by prompting it with instructions so note
15:01
that until now the model just predicts next tokens we have not yet trained it to predict whether spam or no spam but
15:08
let's see if our model has inherently learned these capabilities so what I'm going to do is that instead of providing
15:14
text such as every effort moves you in the text prompt itself I'm going to say is the following text spam answer with a
15:21
yes or no and then I'm going to give the text you are a winner you have specifically selected you have been
15:27
specifically selected to receive ,000 cash or $2,000 reward note that we have


***



15:32
not given the model any data set about our spam or no spam so far we are just checking whether based on the gpt2
15:39
training itself can it answer this so when you pass it through the generate teex simple function let's see the
15:45
answer so this is the question and the answer with gpt2 generates is that the following teex spam answer with yes or
15:53
no you are a winner so it clearly fails the model struggles with following
15:58
instructions and this is because the model has only undergone gone pre-training right it
16:03
lacks any fine tuning so without any classification fine tuning as we saw the model is not being able to uh perform
16:11
correctly and that is expected so now let us go to step
Adding classification head to model architecture
16:16
number let us go to step number two so step number one was loading prer and
16:21
gpt2 weights into the model and that we have finished right now and now we are moving to step number two step number
16:28
two is modif ifying the architecture by adding a classification head so let me explain this to you in detail actually
16:35
so you might be thinking that our model has been trained to predict the next token right how are we doing
16:41
classification so here's the part where this magic happens so if you remember the output layer so let's look at this
16:47
linear output layer in the text classification or in the text generation task for which this
16:54
llm is typically trained on this output layer looks like this where you have input which is 768 of the embedding
17:01
Dimension size and the output is equal to 50257 because that's the vocabulary
17:06
size so when you have every effort
17:13
moves you if this is the input the output for every for every Row the
17:19
output will have 50257 columns because that's equal to vocabulary size so there will be 50257
17:27
entries for every 50 257 entries for effort 50257 entries for moves and 50257
17:34
entries for youu so if you want to predict the next token after every effort moves you you look at the final
17:40
row and then you choose that token ID with the maximum probability that gives you the next token this is how you
17:45
predict the next tokens but now we don't need the next token prediction right now
17:51
our job is to Simply classify whether it's a yes or no so what we are going to
17:56
now do is that uh we are going to do the same thing but the output Dimension will
18:01
change every effort moves you this is my input right now for every token we want
18:07
two outputs either it's a yes or it's a no so two outputs for every two outputs for
18:14
effort two outputs for moves and two outputs for U so to get the final answer we are going to look at the final row
18:21
which is U since it contains all of the previous information and then we are going to see the yes value and then we
18:27
are going to see the no value these values will be indicative of probabilities so then we are going to
18:33
based on Which is higher we'll classify whether it's spam or no spam so instead of having this final neural network
18:40
output layer size is 50257 we are going to replace replace
18:47
the original linear output layer with a layer that maps from 768 hidden units
18:52
into only two units and what are these two units corresponding to the two units are corresponding to Simply span
18:59
uh versus no spam this is the only change which we
19:05
are going to do in the llm architecture when I saw this for the first time I was pretty Amazed by it because I had never
19:11
seen a classification head so this can be thought of as the classification head right now so let me just write the name
19:18
this can be thought of as a classification head I was pretty Amazed by this because
19:25
I had only done classification using neural networks and decision Tre before I never thought you can add this
19:31
classification head on top of a GPT architecture and use that itself as the classifier it might be overkilling it
19:39
because even a decision tree or a neural network might work but this is just a fun application to consider that llms
19:44
can actually be used to perform classification tasks whether llms perform better than neural networks or
19:50
decision trees that's a question of open research and that needs to be figured out
19:56
still okay so this is the classific head now which is added on top of the GPT model architecture and that is used to
20:04
classify whether the answer is yes or no okay one more thing which I would like
Select layers which want to fine-tune
20:09
to mention before we dive into the code is that we can actually select which layers we want to find tune so of course
20:15
when you add this classification head this was not present in the original gpt2 architecture so these parameters we
20:21
will need to find tune but we have an option to choose among all of these parameters uh gpt2 has already given me
20:29
many parameters so how much do I need to find tune so that's a call which you need to make right so one thing which I
20:35
mentioned here is that since we already start with a pre-train model as we have loaded the gpt2 weights it is really not
20:42
necessary for us to F tune all the layers this is because the lower layers
20:47
such as the token embedding layer the positional embedding layer the layer normalization here Etc these lower
20:54
layers really capture the basic language structures and semantics which are applicable across a wide range of tasks
21:00
and data set this is very important the lower layers if you see the lower layers
21:05
have token embedding which captur captures the semantic meaning it has the positional embedding and then we have
21:12
some amount of multi-ad attention Etc where token embeddings are converted into context vectors or input embeddings
21:18
are converted into context vectors which contain information about how much attention one token pays to all other
21:24
tokens and gpt2 has been trained on huge amounts of tech so it already contains some information
21:31
about meaning Etc so if you give an email let's say there is already some intution baked in about whether this
21:38
email is kind of a Spam or what kind of information is it representing that information is already captured in these
21:45
lower layer somehow because gpt2 is a very smart and intelligent model it does
21:50
capture this information so we have a choice as to which layers we want to find tune so the choice which we are
21:56
making here is that we are only going to find the last layers so we are definitely going to fine tune this


***

22:01
classification head we will definitely F tune the final linear normalization layer which has the scale and the shift
22:08
parameters and we are going to fine tune the final Transformer block so remember
22:13
the gpt2 architecture had 12 Transformer blocks like this right instead of fine-tuning all of the
22:21
12 Transformer blocks we are only going to f tune the final Transformer block we are going to assume that all the other
22:27
Transformer blocks inherently contains some information about what the text in the data represents so this is a good
22:34
mix between achieving good accuracy as well as reducing the computational cost
22:39
remember if you are to F tune everything again then what's the purpose of loading pre-trained weights right the reason we
22:45
loaded pre-train weights from gpt2 is because it would hopefully capture some semantic meaning uh as to what the text
22:53
represents so what we are going to do is that we are only going to f tune three things we we are going to fine tune the
22:59
final output head which is this classification head over here because of course that was not present in gpt2 then
23:06
we are going to find tune the final Transformer block the 12th Transformer block and we are going to f tune the
23:11
final layer normalization module that is what we are going to do these three things we are going to fine tune rest
23:18
all the other parameters we are going to freeze which means we are not going to train the remaining
23:24
parameters uh and one more thing which I wanted to explain before we go to the the code is that let's see here right so
23:31
every token will produce output two tokens right so let's say every effort
23:36
moves you is my sentence so what is the output whether it's spam or not spam which of these four tokens should I look
23:43
at should I look at the first token the second token third token or fourth token because all of them will have this yes
23:48
no values which token should I look at so here there's a nice schematic to mention that why should we always
23:55
extract the last output token we should always EXT ract the last output token because the last token is the only one
24:03
which with an attention score to all the other tokens if you look at the second token let's say it will only contain the
24:09
attention with respect to First token if you look at the third token it will only contain attention with respect to
24:15
previous two tokens whereas the last token has all the information it contains attention with respect to all
24:20
the previous tokens so if you want to predict whether it's a Spam or not a spam you want to predict from that row
24:28
which contains maximum amount of information present in the sentence so that is equal to the last token right
24:33
here so we are going to extract the last token output row and we are going to use
24:39
that to predict whether the email is Spam or whether the email is not a Spam so this is exactly all of this is what
24:45
I'm going to show you in the code right now so let's go to code uh the first thing which we are going to do is add a
24:52
classification head at the top which is what I showed you over here in the figure we are going to replace the this
24:58
original output head with a classification head so in this section we modify the
Coding the finetuning architecture
25:04
pre-trained large language model to prepare it for classification finetuning to do this we replace the original
25:11
output layer which maps The Hidden representation to a vocabulary size of 50257 with a smaller output layer that
25:18
maps to only two classes zero and one so look at this we want this kind of an output layer which only has two outputs
25:25
either zero or one so two neurons at the end so one thing which I would like to
25:30
mention is that mention is that we could technically use a single output node since we are dealing with a binary
25:36
classification task right uh however this is not a generic approach if we use
25:41
a single output head that's not generic if we have more number of classes so here what we have what we are doing is
25:47
that we are doing a more General approach what we are going to say when we code the model architecture is that
25:52
we are going to say that the final number of output nodes should be equal to the number of classes so we are not
25:58
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

