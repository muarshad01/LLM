#### Generating Text from output Tensors

***

* 5:00


#### Output logits 
* Number of Tokens (4) X Vocubalry Size (50,257)
* 1-batch = 1 X 4 x 50,257
* 2-batch = 2 X 4 x 50,257

* __Step-1__:
  * Number of Tokens (4) X Vocubalry Size (50,257)
  * 1-batch = 1 X 4 x 50,257
    * Every [...|...|...]
    * Effort [...|...|...]
    * Moves [...|...|...]
    * You [...|...|...]
* __Step-2__:
  * Extract the last Tensor
    * You [...|...|...] -- Logits Vector
* __Step-3__:
  * Convert logits into probabilities by applying softmax
* __Step-4__:
  * Identity the index position (token ID of the largest value)
* __Step-5__:
  * Apply token ID to previous inputs for the next round

***

* 10:00

***

* 15:00

***

* 20:00

#### 4.7 Generating text

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
```

***

* 25:00

the last row and the way this will happen is through this command Logics colon minus one and colon so what this
25:22
first colon is that which means you do nothing to the batch argument but the second is minus one which means that you
25:30
look at the first batch and then you look at the last you take the last row you look at the second batch and you
25:36
just take the last row this is what we are going to do so now we are going to apply this function which will just take
25:41
the last row out of every batch in the logic stenor so logits colon minus one
25:46
colon is going to result in this where we just take the we just take this
25:53
Row from the first batch and we take the last row from the second batch and we just uh stack them
25:59
together so this is the second command here where we only focused on the last time step or the last row okay and now
26:07
when we execute this command the dimensions become the batch which are the number of rows and the vocabulary
26:13
size so one thing which I would like to clarify here is that here I mentioned this five as the
26:20
embedding Dimension right but this is actually the vocabulary size the number of columns here are equal to the
26:26
vocabulary size and what this what this function this is
26:32
equal to the vocabulary size and what this function does is that it just takes the last row so when we get this final
26:39
output the number of rows are still equal to the number of batches and the number of columns are equal to the
26:44
vocabulary size we get rid of the second dimension which was equal to the number of tokens all right so now we have this uh
26:52
these two we have the rows which correspond to the last row in every batch and the next
26:59
step is applying soft Max and converting these Logics into a set of probabilities
27:04
this is exactly what we are going to do we are going to apply soft Max and dimension equal to minus one uh which
27:11
will ensure that for every row which we have extracted soft Max will be applied Along The Columns of those rows so when
27:17
we look at each of the tokens so let's say you when we look at this batch so when you look at each batch um when when
27:25
you sum up the probabilities for each batch they will sum up to one so remember that now that the size here
27:31
is just batch size number of rows and number of columns equal to the vocabulary size so now all of these will
27:38
be transformed into values such as these and if you add up these values so for
27:44
the first batch you'll just add up these values and that will sum up to one for the second batch you'll add up these
27:49
values and that will sum up to one remember the goal is to predict a new token for every batch we have
27:55
inputed so this is the next step and then the final step is we are going to look at that index with the highest
28:01
probability value and uh this is exactly the step which we had mentioned here
28:07
also yeah so after converting the Logics into probabilities we look at that index
28:13
which has the highest value so this is exactly what we are doing in this step we look at that index with the highest
28:18
value uh that token ID and then in the last step we append that token ID idx
28:24
next to the initial uh token IDs which were stored in idx so in this last step we do the appending
28:31
part which has been mentioned over here so look at step number five over here you have to append the token ID
28:38
generated to the previous inputs for the next round and this is what is shown in this torch. cat which is concatenation
28:45
and idx next is the input is the ID which corresponds to the highest probability and that is appended to the
28:52
current Uh current indices and we you see we are in a Loop
28:58
here so the number of times we are going to do this appending operation is by the time we reach the maximum number of new
29:05
tokens these are the number of iterations remember on the Whiteboard what we saw uh on the Whiteboard we had clearly
29:12
seen that uh the number of iterations over here the number of iterations which were six iterations over here are equal
29:19
to the number of Maximum maximum number of new tokens so that's what's been written in
29:24
the code we are doing the number of iterations equal to the maximum number of new tokens and then we are going to
29:30
keep on adding these new tokens to the input tokens and that's it this is how we are
29:36
going to predict the new tokens corresponding to the next words and that's exactly what's happening in the
29:41
GPT model this is how you go from all of the complicated GPT model architecture
29:47
to predicting the next World I have just written some text over here in the
Role of softmax in next token prediction
29:52
preceding code the generate Tex simple function we use a soft Max to convert the Logics into probability distrib R
29:58
bution from which we identify the position with the highest value uh now
30:03
you might think that since we are only looking at the index with the highest value and soft Max is monotonic why do
30:09
we need the soft Max why can't we just find the index from the logits that index which gives the highest value soft
30:16
Max is monotonic so that index is going to remain same whether we apply soft Max or
30:21
not um so in practice the soft Max step is redundant which means it's not really needed to find the position with the
30:28
highest uh highest score because the position with the highest score in the softmax output is the same position in
30:35
the logic sensor so in other words we could apply the torch. AR Max here we have applied
30:41
it to the soft Max torch. AR Max we have applied to soft Max generated output
30:46
right we could have applied this to the logic sensor directly and get identical results so here I could have just
30:52
replace this with logits and the results would have been the same so then your question would be
30:58
then why are we using the softmax in the first place uh what the importance of softmax
31:05
is that we wanted to show you the full process of transforming Logics to
31:10
probabilities which can give additional intuition the probabilities give us some intuition of how much percentage
31:15
contribution does each token have in the next word prediction task and this will
31:21
also help us because uh in the next module where we'll do the GP training we
31:27
will introduce additional sampling techniques where we will modify the softmax output such that the model does
31:32
not always select the most likely token and will introduce some variability and creativity in the generated text this is
31:39
the important part the model does not always take the U output with the maximum
31:46
probability to make sure that the generated text has some variability some creativity we will explore some other
31:53
options where the softmax select some other tokens and for that definitely we need to apply the soft Max because we
31:58
need the outputs to be in some format of probabilities so although soft Max was not needed in the current code it will
32:04
be useful later when we look at things such as temperature variability in selecting the outputs Etc don't worry
32:11
about these terminologies right now I'll cover that in detail when we come to the next module now what we can do is that we
Testing the next token generator function
32:18
have written this whole function right why don't we test it on some sample text so let me take the model input as hello
32:25
I am um these this is is my model input and the reason I'm taking this input is
32:30
that this is the same input which I have used on the mirror whiteboard over here so I take this model input and uh then
32:38
what I'll first do is that I'll first encode it into token IDs so I first use my encoder to encode this model input
32:44
and convert it into a tensor so remember the shape of the input should be batch size here I have only one batch and the
32:51
number of tokens so it should be a tensor and it should be a tensor of token IDs so I have my encoder which has
32:57
been defined through tick token so if you have been following these lectures you will know that we have been
33:03
generating our encodings through tick token which is the tokenizer used for open AI models it's a bite pair encoder
33:10
so we are using that to encode this sentence and now I have generated my input sample now what we'll do is that
33:16
before passing in the model we'll first put the model in evaluation mode this bypasses some layers such as
33:22
normalization layer Dropout layer because we are not training the model here we are just evaluating so it just
33:27
just makes the model a bit more efficient and then we will just call this generate text sample
33:33
function we'll call this generate text sample function and we'll put model equal to model we have already defined
33:40
the model before and let me again take this over here so that you you have full
33:45
grasp and the GPT model configuration which we are using has been defined over here this is the configuration we are
33:51
using a vocabulary size of 50257 context length of 1024 768
33:57
embedding dimension 12 attention heads 12 Transformer blocks and dropout rate of
34:02
0.1 so this is the model which has been defined um because that's needed to be
34:08
passed into our function so I'm just writing it over here for your reference I'll code it out I'll comment it out I
34:15
will share the code so that you can run it on your own laptop okay so then we'll
34:20
run this generate text simple function we'll pass in the model we'll pass in the inputs this these are my inputs
34:26
right now remember the input is the second argument over here then we have to pass Max new tokens and context size
34:32
so let me pass in that so my Max new tokens is six and the context size which I'm I'm passing is GPT configuration
34:38
context length which is 1024 and then I'll just print out the output so I have six new tokens right
34:46
and the input was these tokens 1 5496 11 314 and 716 so now these are the inputs
34:54
and if you look at the output tensor you'll see that six new tokens have been appended over here 27018 2486 474 843
35:04
30961 42 3 48 7267 these are the six new tokens which
35:09
have been appended uh through because of our generate text simple function so these are the next Words which our GPT
35:16
model has predicted and here you can see that the output length is 10 why 10 because four were the number of input
35:23
tokens and Max new tokens were six so six additional words or tokens have been generated now we can use the decode
Analysing the next token predictions
35:30
method and based on our vocabulary and the bite pair encoder which we have used we can convert these new tokens back
35:36
into text so it seems that the next text is now some random text and the next
35:42
text is not as great as what I had written on the Whiteboard over here uh
35:48
on the Whiteboard the next text was hello I am model ready to help but here the next text is something completely
35:54
random right now why is this completely random the reason this text is completely random is that because we
36:00
have not trained the model yet the model has 124 million parameters and all of those parameters are completely random
36:06
right now those are not trained now it's just a matter of training the whole GPT architecture has been set up completely
36:13
we have implemented the full GPT architecture and initialized a GPT model instance but with random weights we need
36:19
to train these 124 million parameters and for that we have the whole NY module dedicated which are a uh next maybe six
36:27
seven number of lectures or even more but for now if you have reached this stage just be happy and proud that you
36:34
have run a 124 million gpt2 architecture model completely on your laptop you have
36:39
taken an input and you have predicted the outputs and uh this is the first step towards understanding how GPT
36:46
really works when you go to chat GPT and when you type hello I am let me let's actually do that so I'm going to chat
36:53
GPT right now and let me type hello I am and let
37:01
me say complete this sentence I'm providing no context here and does not
37:06
make too much too much sense but here you can see that based on the past interactions which I've had with chat
37:12
GPT it it results in some output which is at least quite coherent it's much
37:17
better than the output which we have received over here right but that's fine we have not implemented the training but
Recap and summary
37:23
we have essentially implemented all the nuts the bolts and the building block for building out this entire GPT
37:30
architecture on our own completely from scratch we have not used any library from Lang chain or anything we have
37:36
defined this GPT architecture fully from scratch we have learned about all the sub modules involved in the GPT
37:42
architecture we have coded all these sub modules and ultimately now we have reached a stage where given an input we
37:48
can predict an output so let us go back to the schematic which we started this lecture
37:53
with and let's see whether we have implemented everything which we we actually wanted to implement uh yeah
38:00
this is that schematic I was talking about so we have I think had six to seven lectures in this GPT architecture
38:06
module and in these lectures we have implemented every single thing which has been mentioned in this schematic all the
38:12
things we have been mentioned all the things have been covered so when you are given an input which is a text such as
38:17
every effort moves you now we are we are ready to predict the next words we have
38:23
reached the stage where we had an input and we have implemented the whole GPT architect piure to produce the output
38:29
it's just that the training has not been done yet but it's fine we'll come sequentially to that part in fact we
38:35
have learned this whole thing in U this module we first started with the GPT
38:41
backbone where the code was not implemented but we had a dummy GPT class then we implemented layer normalization
38:47
J activation feed forward Network shortcut connection we coded out the entire Transformer block after that and
38:54
in today's lecture we coded out the entire GPT architecture and we also got the final the next words given a set of
39:01
input tokens so these were a comprehensive set of lectures but if you have reached the end you should be proud
39:06
of yourself and I just want to write that
39:13
you um you did it that is what I I want to
39:19
write just to keep you motivated uh to keep on following the next lectures which are coming because
39:25
if you have reached this stage you have already reached much farther than 95% of students so it's amazing many other
39:32
students might just be using GPT but you are one of the few students who have now coded out a gpt2 architecture on your
39:39
own on your local machine which which is predicting the next word which is predicting the next token and I find
39:45
that incredibly satisfying and motivating in the next set of lectures what we are going to do is that we are
39:51
going to do the training for the 124 million parameters in the gpt2 model and
39:56
then the output which is generated will start getting much better and it will be
40:01
better and better and better so the whole goal of the next set of lectures is to make this set of outputs better
40:08
but now since the architecture is in place uh doing the next part um will be
40:14
a bit easier because we can directly work from the architecture which we have built thank you so much everyone I hope
40:21
you are enjoying these lectures I say this at the end of every lecture but I deliberately try to keep a mix of uh
40:28
very detailed whiteboard notes such as this plus the coding because I feel that students to really Master large language
40:35
models you need an understanding of theory intuition as well as detailed code I'll be sharing the entire code
40:41
file with you and I encourage you to play with this code ask doubts on YouTube um and we'll try to clarify as
40:48
much as possible thanks a lot everyone I look forward to seeing you in the next video

***





