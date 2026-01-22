#### Stage_2: Foundational Model
1. Training Loop
2. Model Evaluation
3. Load Pretrained Weights

* Gradient descent back-propogation algorithms

* $$L_{BCE}=\frac{1}{N}\sum_{i=1}^{N}y_i.\log(p(x_i))))+(1-y_i).\log(1-p(x_i))$$

***

* 5:00

#### Using GPT to generate text

* Input - predicted - trarget values

***

* 10:00

#### Outputs -> Predicted (Logits Tensors)
* Inputs - GPT model - Output token IDs

1. Tokenizer convers tokens into token IDs
2. GPT model converts token IDs into logits
3. Logits are converted back to into token IDs

***

* 20:00

#### Finding the loss between targets and outputs

```python
import torch
from previous_chapters import GPTModel
# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference
```



```python
import tiktoken
from previous_chapters import generate_text_simple

# Alternatively:
# from llms_from_scratch.ch04 import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```
***

* 25:00

```python
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]
```

```python
with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
```


```python
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
```

***

* 30:00

lecture which is actually defining the loss between the targets and the output so now let let's see what's going on
30:29
here so we have the inputs and this is the probability tensor which indicates
30:35
what's the probability of the next token for every input token what I've marked in blue over here
30:41
is the target indexes so if we want the actual answer to be that for every it
30:47
should be effort for every effort it should be moves and for every effort moves it should be U the the correct
30:53
answer of the indices which we want is 1 four and five these are the true values uh so the token IDs here are 3626
31:04
610 and 345 but uh in the sample code which I've shown over here we just has
31:11
seven uh we just has seven vocabulary size the token IDs which are 1 four and five those are the correct ones so now
31:18
one thing which I would like to mention is that let's say these indexes which are marked with the star right now these
31:24
are the actual indexes which we want right but if you see the values corresponding
31:30
to those those values are not the highest because our llm is not trained yet so what I'm going to do now is that
31:37
I'm going to collect the probabilities at these indexes so I'm going to take this index I'm going to take this index
31:43
and I'm going to take this index and I'm going to find the probabilities at these these indexes or these indices rather
31:50
and I'm going to collect these together so for example for the first batch what I collect can be 144
31:58
and3 these are the set of probabilities and for the second batch I'll collect another set of three probabilities like
32:04
this this is exactly what we are doing in the actual problem in the actual problem we know that the target indexes
32:11
are these so what I'll do is that I look at my probability tensor and I'll get
32:16
the probability which is corresponding to these indexes I know it will not be maximum because my llm is not optimized
32:23
but I'll just write down these probabilities for now so uh I I have these Target indices and let me call
32:29
them i11 i12 I13 for the first batch and i21 i22 and i23 for the second
32:37
batch so what I'm uh so what I'll be doing now is that I'll be uh looking at
32:43
batch one and I'll be looking at badge two and I'll find the value corresponding to these indices in the
32:48
probabilities tensor so in this in in the probability tensor which look which
32:54
will look something like this for the 50257 vocabulary size also I'll find the
32:59
probabilities corresponding to i11 i12 and I13 for batch number one and I'll find the probabilities corresponding to
33:07
i21 i22 and i23 for batch number two so for batch number one I'll aggregate
33:12
these probabilities together and they look like this P11 p12 p13 for batch two I'll aggregate these probabilities and
33:19
they look like p21 P22 and p23 remember these are the probabilities which are
33:24
not maximum right now the goal of training is that to get all of these
33:29
values as close to one as possible and why do we want all these values close to one as possible because then we'll make
33:36
sure that the output indexes which have the maximum probabilities will be closer
33:41
to i11 i12 I13 i21 i22 and i23 which are
33:46
my targets so the goal of the llm Performing better now is reduced to this
33:53
problem that I want these probabilities to be as close to one as possible I want all six probabilities to be as close to
33:59
one why are there six probabilities because there are two batches each batch has three tokens so there are three
34:05
prediction tasks so that's why there are three probabilities for the first batch three probabilities for the second
34:11
batch so let me merge these probabilities together and then this will be P11 p12
34:18
p13 and this will be p21 P22 and p23 now what I want is I want all of
34:25
these P11 p12 p13 p21 P22 and p23 I want all of these to be as close to one as
34:32
possible how do I enforce this mathematically first let's see the workflow so we had the logit tensor we
34:39
converted it into probabilities through soft Max and then we had the target probabilities tensor what this target
34:47
probabilities means is that we have the IND indices corresponding to the Target values so we have these indexes i11 i12
34:55
I13 i21 i22 i23 these Target probabilities are just the merge merge
35:01
Target probabilities which is P11 p12 dot dot dot up to p23 these are
35:07
the six probabilities and we want all of these to be as close to one as possible
35:13
to all of you students who have studied uh classification and the loss this problem would be familiar since we are


***


* 35:00


dealing with probabilities it's natural that logarithms and cross entropy will come into the picture so instead of
35:26
directly dealing with this number numbers it's much better mathematically and from an optimization P perspective
35:31
to just take the logarithm of these values and it comes out to be this and then we take the average of all of these
35:38
logarithm values and it comes out to be this and then we take the negative of this average so that's 10.77%
35:58
Target probabilities and we find the mean of so we take the then the log of this so we take log P11 log p12 and the
36:07
last is Log p23 and then what we do is that we find the mean of this
36:14
mean and uh so then that will be Sigma which will be the summation of log 1 one
36:21
summation of log P11 DOT log p23 divided 6 and then I'll just
36:30
take the negative of this and that's called the negative log
36:38
likelihood so then that will be negative of summation of log of P11 dot dot dot up till log of
36:47
p23 divided 6 so now see we want uh we want P11 to be close to uh one we
36:56
want p all the probabilities to be close to one so we essentially want this negative log likelihood to be as less as
37:03
possible so this cross entropy loss we want to be as low as possible and as close to zero as possible so uh why are
37:12
we taking the negative log likelihood the reason we take the negative log likelihood is
37:19
because it just makes more sense if you if you don't take the negative value then uh the loss function would look
37:27
reverse of this the loss function would look reverse of this and then we'll have to reframe the problem as trying to
37:33
maximize the loss instead it just makes more intuitive sense if the negative
37:40
looks like this and now the whole
37:45
goal instead it makes more physical and intuitive sense if the log likelihood
37:50
looks like this and now our whole goal is to bring down this loss as low as
37:56
possible what will happen if this loss is brought down to let's say zero if the loss is brought down to zero which means
38:02
that the mean which means this value is Clos to one which means that there is a higher chance that P11 p12 p13 up till
38:10
p23 are equal to one that is exactly what we want the goal of training is to get these values as close to one as
38:18
possible so that is the main uh main hope now so initially let's say when we
38:24
start the training procedure our value uh the x value whose log we are going to
38:30
take that will not be close to one it will be somewhere maybe here it will be somewhere 0 to one and somewhere here
38:36
the goal is to go here slowly so that the mean of the probabilities will be equal to
38:44
1 okay so this is the negative log likelihood and it's also called as
38:51
the uh it's also called as the cross entropy loss and uh we want to minimize
38:57
this cross cross entropy loss as much as possible that will ensure that the indexes at which our
39:03
tokens uh indexes at which we predict the next token that matches with the target indexes so then the output and
39:10
the target indexes will match closely to each other and then we know that the
39:16
large language model is actually doing a very good job so this is exactly what we are going to do right now in the code
39:22
one thing which I would like to mention about the cross entropy loss is that the cross entropy loss essentially measures
39:29
the difference between two probability distributions so here what we are actually doing is that we are just
39:34
adding discrete probabilities but in a sense this can also be called as the categorical or I should call it the
39:40
cross entropy loss so now let's see the sequence of steps which we are going to implement
39:46
the logit tensor which is a tensor of probabilities uh has the shape of 2 into 3x
39:53
50257 why because it looks something like this so every effort moves and then
39:58
we have 50257 uh so this is the first batch and
40:03
this is the second batch what we are going to do is that we are going to flatten this so that these two first
40:10
batch and second batch are merged together right so these are so this is



***

* 40:00



my output tensor right now every effort moves you I really like it's a merging of these two batches and my target so my
40:24
target is 2x3 why is it 2 by three because for the first batch this is the
40:30
target index of the first input every this is the target index of every effort and this is the target IND index of
40:37
every effort moves 107 is the target index of I so it will be really 588 is
40:43
the target index of I really and 11311 is the target index of I really like so
40:49
then chocolate then what we are going to do is we are going to flatten this out so then this will be 3626
40:55
610 uh 345 107 588 and
41:00
11311 now ideally what we need is that we are going to look at every year and
41:06
we are going to look at the index corresponding to 3626 so let's say if this is the index corresponding to 3626
41:13
this will be P11 for effort we are going to look at the index corresponding to 610 let's say it's this one so that will
41:20
be p12 similarly for like which is the last we going to look at the index
41:26
corresponding to 1 311 so let's say this is this so this is p23 so what we'll get is that we we'll
41:32
get P11 p12 p13 p21 P22 p23 we'll take the log of them we'll add take the mean
41:40
and then do the negative this whole process is encapsulated in this just one line of code tor. nn. functional cross
41:48
entropy logits so this is the logits flat I'm calling this logic flat tensor
41:54
and the second argument here is the target flat what this uh this code will
42:00
do is that tor. nl. function cross entropy logit flat comma targets flat
42:05
first it will convert this logits tensor into a tensor of probabilities until now the logits is is not does not represent
42:12
probabilities so it this function will apply soft Max to the Lo logits and then
42:17
what it will do is that as I told you before it will find the negative log likelihood so first it
42:24
will uh find the probabilities P11 p12 p13 p21 P22 and p23 in this logic flat
42:31
tensor corresponding to the index indices in the Target flat tensor it will take the negative log and then it
42:38
will find the mean this one line of code is going to do all of the steps for us that's why python is so powerful in one
42:45
line of code we'll not only convert this Logics into uh a tensor of probabilities
42:51
through soft Max but we will also get the negative log likelihood and that's the final loss function which we are
42:57
looking for so this one line of code is going to give us the loss between the output and the Target and that's why
43:04
it's a very important piece of code uh to do all of these steps in just one
43:10
line of code I think it's pretty amazing but what I want you all to focus on is what we really are doing over here what
43:17
we are essentially doing is that we have inputs and those inputs are every effort moves and I really like we
43:24
are passing them through the GPT architecture and then we get this logits we get this logit tensor what we are
43:30
going to do with this logit tensor is that we'll first merge them merge this tensor merge the batches and call it
43:37
logits flat then what we'll do is that corresponding to every token we are going to look at the indexes which
43:44
corresponds to the Target and then we are going to find the probabilities now in an Ideal World these probabilities
43:51
will be close to one because that would mean that the output and the in uh Target will match
43:58
but since the llm is not optimized these probabilities which correspond to the Target tensor indices will be probably
44:05
be very small and our goal is to bring all of these probabilities close to one as possible so that's why we'll uh take
44:12
the negative log of these probabilities and take the mean that's the cross entropy loss and through one line of
44:18
code cross entropy tor. nn. functional doc cross entropy between these two
44:24
tensors we are going to implement this exact same operation which I just described in words right now so you can
44:30
even check this and let me type it torch.nn do functional cross
44:37
entropy and you can check it finds the cross entropy loss between the input
44:43
Logics and the target this is exactly um what we did right
44:49
now okay now I'm going to implement this same thing in code we are going to find the cross entropy loss in code okay so
Coding the LLM cross entropy loss
44:56
for first what we are going to do is that uh we are going to find the P1 P2
45:01
P3 so we are going to find P11 p12 p13 and p21 P22 and p23 that's the first
45:10
step so remember here in the sequence of steps over here first we find these Target probabilities right uh and to
45:17
find these Target probabilities what we'll do is that we will uh we'll take the target indexes which
45:25
are uh the indexes corresponding to the True Values and what we are going to do
45:30
is that we are going to uh take the probas so let me see what the probab probas is so this probas is the
45:38
output uh this probas is the output probability tensor and what we are going to do is that we are going to find the
45:45
token probabilities corresponding to the Target indexes and to do that what we are going to do is that we are going to
45:51
use this line of code and what this will just do is that it will take the target indices and it will index or it will
45:59
look at this probas which is the tensor of output probabilities and then it will look for those particular indexes so
46:05
basically what it will be doing is that for zero it that's P11 that will be this
46:11
for one that will be P1 two and for two that will be p13 similarly when we look
46:17
at the second batch over here um what we are doing over here is
46:22
that uh that this is one which means that it's going to look at
46:28
the second batch so one way to understand what's going on here is to look at the probas and try to see the
46:34
dimensions so uh let me write it here again for reference just so that your understanding is clear so the probas is
46:41
actually it will look something like this first let me write it for the first
46:49
every effort moves right and then
46:55
uh this is 50257 the number of
47:02
columns and then second is I really
47:09
like and the size of this so this whole thing is the probas tensor which is the
47:15
probability tensor and the size of this is so we have two batches we have three tokens and
47:22
50257 so what we are essentially doing here is that
47:27
uh let me scroll down below yeah what we are doing here is that in this line text idx
47:34
is equal to zero which means that we are first looking at the first batch so we are looking at this batch and then what
47:42
we are doing is that we are looking at row 0o row one and row two and from row
47:48
0o we'll get the um we'll get the so let's look at the Target answer we'll
47:54
get the index corresponding we'll get the value corresponding to 3626 index from Row one we'll get the value
48:00
corresponding to 61 0 and from row two we'll get the value corresponding to
48:05
345 now similarly here what we are doing here is that we are looking at so text
48:10
idx equal to 1 which means we are looking at the second batch we are looking at the second batch and then uh
48:17
row zero Row one and row two so then we'll look at the targets tensor again
48:22
and for row zero we'll take the value corresponding to index 1107 for Row one we'll take the value corresponding to
48:28
index 588 and for row two we'll take the value corresponding to index
48:33
11311 so that is essentially uh p21 P22 and p23 so these
48:40
three values are the P11 p12 P2 p13 and these three values are p21 P22 and p23
48:48
the whole goal is to get these values as close to one as possible right then what we'll do in the Second Step as I told
48:54
you uh over here once we get these P11 p12 p13 p21 P22 p23 we are going to
49:01
merge these together right so that's what's written here we are going to concatenate these six values together
49:07
and you can see that they appear like this the next step is that we are going to take the
49:13
log uh actually after concatenation yeah we take the log over here and then we print these log values then we'll take
49:20
the mean of these log values and then we are going to do the negative of the log likelihood
49:27
uh so as I told you we can also not do the log likelihood and just do positive log likelihood but then that would mean
49:35
maximizing the loss that does not make sense so in deep learning it's conventional to use negative log lik Le
49:40
so that we minimize the loss right so this is now the negative log likelihood loss value which we ultimately need to
49:47
minimize now as I told you there's a much simpler way of doing this using the
49:53
torch.nn do functional cross entropy the first step of this is to flatten the logits and the targets and to use this
49:59
one line of code so that's what I'm going to do now I'm going to take the logits which was earlier 2A 3A 5257 and
50:08
I'm going to flatten it I'm going to flatten the these first two Dimensions together so that it's 6A
50:14
5257 this is exactly what we saw over here look at this it's 6A
50:20
5257 batch one and batch two are merged together then what I'm going to do is
50:25
that I'm going to look at the Target stenor and I'm going to flatten it so if the target stenor is this two rows and
50:31
three columns I'm going to merge these two rows into six values and then I'm going to just write
50:38
one line of code this one line of code will first convert these Logics into a soft Max and then it will find the
50:45
values corresponding to the Target indexes then it will take the log of these the mean and then the negative log
50:51
likelihood all of this will be done and I will get my categorical cross entropy loss of 10 remember I want to take this
50:57
loss as close to zero as possible that's the goal so when we train the large language models later the goal is to
51:03
bring this categorical cross entropy loss to as low as possible before we end the lecture I
Perplexity loss measure
51:09
want to show you one last thing there is another major of loss and uh that's like
51:15
cross entropy itself but a minor modification and that's called as perplexity so perplexity actually
51:22
measures how well the probability distribution predicted by by the model matches the actual distribution of words
51:29
in the data set and so in that sense it is more interpretable and it's more
51:34
interpretable way of understanding model uncertainty in predicting the next token and I'll show you why in a minute
51:41
remember that lower per perplexity score also means better predictions so the formula for
51:47
perplexity is just to find the exponent of the loss and surprisingly this simple uh
51:54
modification leads to a lot more intuition so if the exponent of the loss
51:59
in our case is 48725 it means that the model is roughly
52:04
as uncertain as if it had to choose the next token randomly from about 48725
52:10
tokens in the vocabulary so the number of tokens in the vocabulary is 50 uh 257
52:16
I think now what this means is that if the input is uh
52:22
every effort moves right every effort moves is the input currently our llm is
52:29
at a stage that to predict the next token that's as as if we have to choose
52:35
between 48725 tokens that's pretty pretty bad right it means that there is
52:40
a lot of uncertainty in predicting the next token if the perplexity score was equal to two that's pretty good which
52:46
means that we just need to predict between two tokens so that means our llm
52:51
is very accurate but in this case the perplexity score is 48725 48725 which means that 4 48,000 tokens
53:00
are kind of equally likely to become the next token which means that our model is not good at all uh you can see how it is
53:07
more interpretable than getting a categorical cross entropy loss of equal to 10 when I get a categorical cross
53:13
entropy loss of 10 I don't really know how it relates to my vocabulary size also but if you get a uh entropy or a
53:21
perplexity score of 48725 you can kind of relate it to your vocabulary size and make interpretable
53:28
predictions like these that the 48,000 next token 48,000 tokens in our
53:33
vocabulary of 50,000 are equally likely to be the next token and that's pretty
53:38
bad because our llm is not yet trained so this brings us to the end of today's lecture where basically what we
Recap and next steps
53:46
did is we took at we uh we first started with the inputs then we started with the True
53:53
Values which were known as the targets and then what what we did was we found the output values this was kind of a
53:59
revision we got the Logics tensor we converted it into a tensor of probabilities and then we got these
54:05
output tokens so then we Tred to find the loss between the targets and the
54:11
output using the cross entropy loss and we ultimately
54:16
saw that using one single line of code torge nn. functional. cross entropy we can find the loss between the loged
54:23
sensor and the target tensor as the lecture concluded we even looked
54:28
at another way of measuring loss which was called as perplexity which is much more intuitive and the way perplexity is
54:35
calculated it just e rais to loss which is exponent of loss and I also mentioned
54:40
the perplexity concept over here in our case the loss was 10.79 and the
54:45
perplexity value was 48725 and this is usually more
54:50
interpretable awesome in the next lecture what we are going to do is that we are going to take an actual
54:56
data set from a book which is called the verdict and first we'll tokenize this data set we'll divide it into input
55:04
output input output Pairs and then we are going to get the llm output and we are going to find the loss function or
55:11
the loss value for this entire uh data set until now in this lecture we just
55:17
looked at uh in the code we just looked at two sample inputs right we looked at
55:23
um let me yeah we looked at these two inputs in the next lecture we'll be scaling this up and look at an entire
55:29
text data set so next lecture will be a lot of fun and uh we'll run the entire
55:34
architecture on this um Hands-On example you can replace this example with any
55:40
book which you like Harry Potter book any other book so next lecture is going to be a lot of fun I hope you all are
55:46
liking these lectures we have now started a new module which is on U the
55:52
training large language model and we are slowly making a lot of progress on in building the large language models we
55:59
have already finished stage one now we are on stage two and rapidly moving towards completion thanks a lot everyone
56:06
and I look forward to seeing you in the next lecture

***











