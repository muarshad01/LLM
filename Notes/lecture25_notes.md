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

***

* 30:00

***

* 36:00

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


