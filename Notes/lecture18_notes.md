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


to the attention score so that all the elements above the diagonal are negative Infinity that's what this step is doing
41:05
uh here the mask actually has also been defined over here see this is the upper
41:10
triangular mask which is all the elements about the diagonal one then they are replaced with negative infinity
41:16
and that's applied to the attention scores so this will ensure that all the elements above the diagonal of the
41:21
attention scores are equal to negative infinity and we are only considering context length here why context length
41:29
because let's say context length is three it means that maximum if three words are given we can make prediction
41:36
of the next word so when we implement this Dropout mask we only Implement a mask of context length comma context
41:43
length there is no point in implementing a bigger mask because anyway we are not going to look at more tokens than the
41:50
context length at a time and if it happens that we are looking at a batch where the number of tokens are less than
41:56
the context size this statement makes sure that then the mask stops at number of tokens but this is a detail which
42:04
probably uh you can Overlook right now if you're understanding all the other things that's what the most important if
42:10
you understand this Minor Detail it's awesome then the next step is to apply soft Max but as I told you before
42:17
applying soft Max we defi we divide every element with the square root of the head Dimension if you look at the
42:23
keys. shape let's look at keys. shape uh this is going to be the keys do shape so
42:29
keys do shape of minus one which means that we are going to look at the last Dimension which is the head Dimension so
42:36
we are essentially dividing by square root of head Dimension here and then we apply the soft Max y Dimension equal to
42:43
minus one because we need to make sure that all The Columns of a row sum up to one and then as I said we can even
42:50
Implement Dropout if needed towards the end so up till now we have reached a stage where we have obtained the
42:58
attention weights is basically and I hope you understand the meaning behind this final attention weight matrix it's
43:05
not just important to understand how the dimensions work so to make sure you understand the meaning let me go through
43:11
the meaning of this attention weight Matrix once more um this what I'm
43:16
highlighting right now is the attention weights for the first head this second block is the attention weights for the
43:21
second head in each attention uh head block you will see that the size is
43:28
number of tokens rows and the number of tokens columns so each value is essentially the
43:35
attention weight between let's say this is the attention weight between this is the attention weight between the second
43:40
row which is the second token and the second token this is the attention weight between uh the third token as the query
43:49
and the first token as the key so basically you'll see that every single
43:55
element here has some meaning it essentially encodes the attention weight between the query and the particular key
44:03
okay now let's go ahead the last step which we are going to implement is that
Finding multi-head context vectors
44:08
we have to calculate the context Vector remember the aim of all the attention mechanisms is to ultimately compute the
44:14
context Vector Matrix and this is exactly what we are going to do and to compute the context Vector Matrix we
44:20
take the attention weights and we multiply them with values remember the value Matrix was The
44:27
Matrix which we had computed earlier let me show you where the value Matrix was in case you have forgotten it because we
44:33
have done so many things uh yeah so here was the value
44:38
Matrix which we had computed we have not used it until now it will only be used in this last
44:44
step okay so the attention weights will be multiplied by the values Matrix to get the context Vector Matrix so let's
44:50
see how the dimensions work out here okay the attention weights as we looked earlier over here
44:57
the attention weights have the dimensions of B comma number of heads
45:03
comma number of tokens comma number of tokens and as we saw earlier the values
45:08
Matrix has the dimensions B comma number of heads uh comma number of tokens and head
45:15
Dimension so effectively let's see whether these matrices can be multiplied so this is number of tokens and number
45:22
of tokens and that will be multiplied by number of tokens and head dim so the
45:27
number of columns here is number of tokens and the number of rows here is number of tokens so the number of
45:34
columns in the first Matrix are matching the number of rows in the values Matrix
45:39
so we can see that these two matrices can essentially be multiplied so multiplication is possible and now let
45:46
us see how the multiplication will actually work in practice this is the final attention weights Matrix here it's
45:53
mentioned attention scores but I should have called it attention weights remember there is a difference between
45:58
scores and weights attention weights in in the attention weights each row sums up to one that's not the case with
46:04
attention scores okay so this is the attention weights and this is my values
46:11
this my values Matrix so this 1A 2A 3A 3 and this 1A 2A 3A 3 and when we multiply
46:19
the resultant output will be B comma number of heads comma number of
46:24
tokens uh comma the head Dimension so here you can see that the
46:31
context Vector output is B comma number of heads comma number of tokens comma
46:36
head Dimension which is 1 comma 2 comma 3 comma 3 Let's interpret this again uh
46:42
so here you can see that there are two heads so this is head number one and this is head number two and in each head
46:48
there are number of tokens so if you look at each head there are three rows so each row corresponds to one token but
46:55
now if you look at what what each row represents each row represents the context Vector for that particular token
47:02
and it has the dimensions equal to head dim because head dim is equal to three
47:08
so that's the meaning of this uh context Vector Matrix which we have reached but
47:14
now remember there is a problem here right or not a problem uh but we have to
47:19
somehow merge this number of heads and head Dimension back together because the
47:25
resultant context Vector Matrix remember what we saw earlier the let me scroll up
47:30
a bit so if you if you if you looked at the goal which we had when we started
47:36
this lecture the goal was that the resultant context Vector Matrix should
47:42
have the dimensions of uh yeah as I mentioned to you the
47:48
goal was that the resultant context meor Matrix should have D out right as the
47:54
dimension so we should again pull back the head Dimension and the
47:59
number of heads together so that we can get the resultant Matrix which has the D out Dimension
48:05
preserved uh whereas let's see what we have obtained until
48:10
now well until now the context Vector Matrix which you have
48:20
obtained yeah the context Vector Matrix which we have obtained has number of heads and head dimensions in separate
48:26
positions so first what we'll do is that we'll bring them closer together so that we can then merge them to get the D out
48:33
so what we are going to do is now we are going to swap this this number of tokens
48:38
index with the number of heads index so that the dimension of the context Vector Matrix is so that the
48:46
shape of the context Vector Matrix is changed so the next step is basically
48:51
step number 10 and that is to reformat the context vectors So currently the
48:57
context Vector shape is B comma number of heads comma number of tokens comma head Dimension right and I want the
49:02
number of heads to come here so that they're closer to the Head dim and I want the number of Tok number of tokens
49:08
to go here so I want the resultant Matrix to be B comma number of tokens
49:15
comma number of heads comma head Dimension so essentially what I will do
49:22
is after I compute the context Vector Matrix I'll do a transpose of the first
49:27
index and the second index and so the resulting context Vector Matrix now which has the
49:33
dimensions of B comma number of tokens comma number of heads comma head Dimension looks like this so here you
49:40
see now the interpretation is different now this is my first token now the grouping is with respect to tokens this
49:46
is my second token and this is my third token and in each token there are two heads so if you look at the first token
49:52
there are two heads and if you look at the first head this is the vector with respect to the first head the context
49:58
Vector context vector and the second row is the context Vector with respect to the second head for the first
50:05
token now let's see how all of this is implemented in code actually all of what we saw right now is just implemented in
50:11
one line of code but to understand this we really have to understand first of all how the attention weights are
50:16
multiplied with values the multiplication really makes sense and why do we do this transpose 1A 2 the
50:23
reason we do this transpose 1 comma 2 is to get the context Vector mat in this shape the reason we get it in this shape
50:29
is now you can see the number of heads and head Dimension are closer together so we can merge them um into the D out
50:37
more easily so here you can see this is what we have reached until now where the
50:42
context Vector is obtained and it's in the correct format now the last step what we have to do is that we have
50:50
to um let me show you the last step what we have to do is
50:56
essentially we have to combine the results from multiple heads so see this is the context Vector Matrix which we
51:02
have obtained right now right so if you look at the first token which I've highlighted over here this is the first
51:08
head and this is the second head now what I will do is that when I look at the first token I will combine these two
51:14
together into one row so that it will be uh six the
51:19
dimension will be six so so here these are three and these are three right so
51:25
I'll combine the outputs from both of these heads into one output so let's see how this looks like so then the first
51:31
row will so then we'll flatten this is called flattening will flatten each token output into each row so the head
51:39
one and head two outputs are combined together so if you look at the final output the first row consists of merging
51:46
of the two heads for the first token the second row consist of the merging of the
51:52
two heads for the second token so for the second row we merge these two out outputs into one single row and for the
51:58
third token we merge these two outputs into a single row so you'll see that the
52:04
F the this is the third row so this now what I what I'm showing on the screen here is my final context Vector Matrix
52:12
and how to interpret this if you look at the first row the first row is the context Vector
52:19
context Vector for the first token first row is the context Vector for the first token why does it have six elements
52:25
because d out is equal to 6 the second row is the context Vector for the second token and the third row is the context
52:32
Vector for the third token so overall you see we first split the D out into number of heads and head Dimension and
52:39
now we brought it back together to get the D out so in the final shape you will not see the number of heads it's all
52:45
merged into this D out so this is my final answer right now and the shape of
52:51
this is 1A 3 comma 6 which is B comma number of tokens comma D out
52:57
so this is exactly what is done here what we do is that we uh we take this
53:02
context vector and we reshape it into B comma tokens comma D out why this
53:07
continuous is needed is because we want to make sure that when we reshape matrices they are in the same blocks of
53:13
memory so when we reshape uh tensors let's say and if they're in different memory blocks it becomes difficult so
53:19
first we make sure that using this continuous they in the same memory block then we reshape them so that the final
53:26
output is B which is the batch size number of tokens which is equal to three in the example we saw and D out which is
53:33
the output Dimension which is equal to six and then there is an optional projection layer towards the end so if
53:39
you look at the out out output projection it's again a linear layer and whose parameters can be learned this is
53:46
not really necessary but sometimes it is implemented in practice now this is exactly the entire procedure for how the
53:53
multi-ad attention is implemented from scratch and here we saw the multi-head attention for the example which we have
54:01
so the first token is again the the the second token is
54:11
cat and the third token is sleeps me write this
54:18
again yeah the third token is sleeps so you see through this entire procedure we obtained the enriched context Vector
54:25
representation for these tokens similarly when you deal with large volumes of text you take sentences you
54:32
break them down into tokens then into token IDs then into input embeddings and similar to this procedure you get
54:39
context vectors for each token which you have ideally when we run the actual code
54:44
we will have multiple batches but I showed only one batch right now for Simplicity so uh to whoever who have
Hands on example testing
54:52
reached until this stage I want to say that thank you for following with me for so to many lectures I know these
54:57
lectures are becoming very long but unless I explain every single thing in detail it's very difficult for you to
55:04
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



