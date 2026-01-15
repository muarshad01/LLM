#### Implementing Multi-head Attention With Splits

* $$\text{head-dim} = \frac{d_{out}}{n_{heads}}$$

***

* 5:00

#### Example
* __Step-1__: b, num_tokens, d_in = (1, 3, 6)
  * context_vector = 3 x d_out

* __Step-2__: Decide d_out, num_heads = (6, 2)
  * Usually d_out = d_in
  * $$\text{head-dim} = \frac{d_{out}}{num_{heads}}$$ = 6/2 = 3

* __Step-3__: Initialize trainable weight matrices for key, query, and value $$(W_k, W_q, W_v)$$ = (d_in, d_out) = 6 x 6


***

* 15:00

* __Step-4__: Calculate Keys, Queries, Values Matrices $$(W_k, W_q, W_v)$$ = (d_in, d_out) = 6 x 6
  * Input X $$W_k$$, Input X $$W_q$$, Input X $$W_v$$
  * Input = (1, 3, 6) = (b, num_tokens, d_out)
  * $$W_k = W_q = W_v = (6, 6)$$

* __Step-5__: Unroll last dimension of Keys, Queries, and Values to include num_heads and head_dim
  * $$\text{head-dim} = \frac{d_{out}}{num_{heads}} =\frac{6}/{2} = 3$$
  * (b, num_tokens, d_out) = (b, num_tokens, num_head, head_dim)
  * (1, 3, 6) = (1, 3, 2, 3)

***

* 30:00

* __Step-6__: Group matrices by number of heads
  * (b, num_tokens, num_head, head_dim) -> (b, num_head, num_tokens, head_dim)
  * (1,3,2,3) = (1,2,3,3)


* __Step-7__: Find Attention Scores
  * Queries X Keys.Transpose (2, 3)

***

* 35:00

* __Step-8__: Find Attention Weights
* Mask attention scores to implement Causal Attention
* Divide by $$\sqrt{\text{head-dim}} = \sqrt{\frac{d_{out}}{n_{heads}}} = \sqrt{\frac{6}{2}} = \sqrt{3}$$
* Attention weights (1, 2, 3, 3) = (b, num_head, num_tokens, num_tokens)


***

* 40:00

* __Step-9__: Context Vector
  * Attention Weights X Values
  * (b, num_heads, num_tokens, num_tokens) = (b, num_heads, num_tokens, head_dim)

***

* 45:00

* __Step-10__: Reformat the Context Vector
  * (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)

***

* 50:00

* __Step-11__: Combine Heads
* (1, 3, 6) = (b, num_tokens, d_out)

***

* 55:00

| LLM | Parameters | Attention heads | Embedding Size| 
| ---| --- | --  | ---|
| GPT-2| 117 million parameters | 12  | 768|


understand all the details so congratulations if you have reached this lecture you have successfully understood
55:11
how the multi-head attention works and I think there are very few people who really understand this entire piece of
55:16
code block by block okay so I'll share this notebook
55:22
with you and whatever I explain to you on the Whiteboard all the steps which which I laid out in front of you on the
55:29
Whiteboard uh all of those have been explained here as step one to Step 11
55:34
and uh I have explained added a detailed explanation of the multi-head attention class in today's lecture I did not just
55:40
want to read this but I wanted to construct a practical example to show you how the dimensions actually work and
55:47
uh it took me a long time to make this example but now I think it's worth it because it really helped me explain it
55:53
and I hope you understood it better so now we can actually test out the multi-head attention class so here are
55:59
my inputs uh as I showed you on the Whiteboard we have three tokens and we
56:04
have six the embedding Dimension is six the only change here what I'm going to
56:09
do I'm going to create a batch so I'm going to going to create a batch of two such inputs and I'm going to stack this
56:15
batch on top of each other so I'm going to assume a d out equal to six exactly
56:20
what we saw on the Whiteboard and context length equal to six and then we are going to implement the multihead
56:26
attention class so D in is equal to 6 uh D out is equal so D in is equal to six
56:34
right because each um each token has the input embedding dimension of six D out
56:40
equal to 6 the context length which we are using is equal to uh six then
56:48
uh yeah the dropout rate which we are considering is zero we can even include
56:53
the dropout rate so the dropout rate will change this this last layer of Dropout and randomly block out some
57:00
attention weights this is good for generalization and the number of heads equal to two so we create an instance of
57:06
this class and then create the context Vector Matrix and you'll see for the first batch the context Vector Matrix
57:12
has three rows and six columns let's see if the shape matches what we had seen on the
57:18
Whiteboard uh okay so let me scroll down
57:24
below yeah this is the final context Vector Matrix which we had
57:31
obtained yeah so this also had three rows okay I think I need to scroll up a
57:37
bit yeah this is the final context Vector Matrix which we obtained and this also had three rows and six columns the
57:44
values might be different because we have done the initializations differently here I have taken random initializations in the python code there
57:51
are some other initializations every time you initialize we take from a goian distribution so the values might be
57:57
different but let's check the shape so this is 3 comma 6 three rows and six columns and here also we can see that
58:03
three rows and six columns awesome so the shape matches but you'll see that since there are two batches here's the
58:09
context Vector Matrix for the first batch and here's the context Vector Matrix for the second batch so the
58:16
multi-head attention class which we defined is extremely powerful because it can also handle multiple batches at once
58:22
we can even do 50 data batches and then it will just have one 1 two it will have
58:27
50 such context Vector matrices okay that's it that brings me
58:33
to the end of this section or the end of this lecture so in this lecture we implemented the multi-head attention
58:39
class that we'll be using in the upcoming lectures to implement and train the llm this code is fully functional but we
58:47
we used relatively small embedding sizes and number of attention heads to keep the outputs readable so as I showed you
58:53
we only use two attention heads but gpt3 actually was 96 attention heads so the
58:59
smallest gpt2 model had 12 attention heads and a context Vector embedding size of 768 the largest gpt2 model had
59:07
25 attention heads and a context Vector embedding size of 1600 and gpt3 has even higher so the
59:14
gpt3 largest model has 96 attention heads I think and generally in GPT
59:19
models the D in is equal to D out in the example which we saw D in was equal to D out equal to 6 but here the D in and D
59:27
out are much larger around 768 Etc again thank you so much everyone for reaching
Conclusion
59:34
the end of this attention series it's been one of the longest and most comprehensive series which I have
59:40
covered and uh I really enjoyed learning about all of these things I can see that
59:46
many llm practitioners cannot understand these Dimensions or they do not take time to go through understanding the
59:53
theory the building blocks behind how the attention mechanism Works they just implement the code bases which are
1:00:00
available which completely abstract away all of these things so I don't think
1:00:06
that's the good way or the correct way to learn about large language models if you want to be a true llm engineer or a
1:00:11
machine learning engineer you have to understand how nuts and bols work otherwise you might be able to deploy
1:00:18
applications but to make real inventions you will have to go into the code base change a few things understand
1:00:24
Dimensions as you might have seen and I have stressed this many times dimensions and linear algebra are at the heart of
1:00:30
becoming a very strong ml engineer it all comes down to Dimensions other students might be scared of a
1:00:36
four-dimensional tensor right but if you understand how it works based on what I showed to you on the Whiteboard my aim
1:00:42
is that you should not be scared of these higher dimensional matrices once you write it down and once you
1:00:48
understand what's going on it really becomes easy that's why I really recommend writing things down you can
1:00:54
write on a whiteboard you can even write on a piece of paper but make sure you write things down then you'll remember
1:00:59
them for a longer period of time I hope you all are enjoying these lectures thank you so much everyone and look
1:01:05
forward to seeing you in the next next lecture where we'll actually start building the llm model thanks a lot

***
