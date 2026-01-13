#### A Simplified Self Attention Mechanism withoug Trainable Weights

* In this section, we will implement a simple variant of the self-attention mechanism, free from any trainable weights.

```
your journey starts with one step
```

1. pre-processing 
2. convert sentence into individual tokens GPT uses
3. bite-pair encoder which is a (word, sub-word, character)-tokenizer
4. convert tokens into token IDs
5. convert token ID into a vector representation called __vector embedding__
6. these vectors capture the __semantic meaning__ of each word 

* the main problem here ...how a particular word relates to other words in the sentence...
* we really need to know the __context__, i.e., which word in this sentence is closely related to other words
* how much attention should we pay to each each of these words when we look at "Journey" that's where attention mechanism comes into the picture
*  Goal of the attention mechanism is to take the __embedding Vector__ for word "Journey" and then transform it into another Vector which is called as the __context Vector__, which can be thought of as an __enriched embedding Vector__ because it contains much more information.
*  It not only contains the __semantic meaning__, which the embedding Vector also contains but it also contains information about how that given word relates to other words in the sentence. This contextual information really helps a lot in predicting the next word in LLM.

***

* One such context Vector will be generated from each of the embedding vectors
* The goal of all of these mechanisms is the same we need to convert the __embedding Vector__ into __context vectors__, which are then inputs to the LLMS.

```
\{x^{1}, x^{2}, x^{3}\}

z^{2}= \alpha_{21} \times x^{1} + \alpha_{22} \times x^{2} + \alpha_{23} \times x^{3}
```

* Alphas are called "attention weights" or "attention scores"
* the goal of today's lecture or the goal of any attention mechanism is to basically calculate a context Vector for each element in the input and as I mentioned before the context Vector can be thought of as an enriched embedding Vector which also contains information about how that particular word relates to other words in the sentense

***


* we'll be finding a context Vector for each element
* query
* What are attention scores?


***

15:06
right now what we have to quantify basically is that how much importance
15:11
should be paid to each of these other words or how much attention should be paid to each of the input word for the
15:19
query journey and this is Quantified by a mathematical metric which is called as
15:25
the attention score so the attention score exist exists between the query and
15:30
every input Vector so there will be an attention score between the query and the first input Vector there will be an
15:37
attention score between the query and the second input Vector there will be an attention score between the query and
15:44
the third input vector and finally there will be an attention score between the query and the final input
15:50
Vector the first step of getting to the context Vector is to find these attention scores which will help us
15:57
understand how much importance should be given to each of the tokens in the input okay for a moment assume that you
16:05
do not know anything about large language models you do not know anything about machine learning or natural language processing just assume that you
16:13
are a student who is Guided by intuition and Mathematics and try to think of this
16:19
question you have this input query X2 and you have these other embedding input
16:26
embedding vectors how would you find the importance of every input Vector
16:32
with respect to the query that's the question so you have the query vector and you have each of
16:39
these embedding vectors X1 X2 dot dot dot right up till the final input Vector
16:44
how would you find a Score which quantifies the importance between the
16:49
embedding between the query vector and the input embedding Vector can you try
16:55
to think about this intuitively forget about everything else forget about ml forget about attention just try to think
17:02
from the basics what is that is that mathematical operation which you will
17:08
consider to find the importance between two
17:15
vectors let me ask you this question in another way as a hint take a look at this Vector embedding you know that
17:21
vectors are embedded in like this in a three-dimensional space now let me ask
17:26
you we have the query Vector which is and we have all these other input vectors for step your with one and
17:34
starts how would you find the importance of all the other vectors with respect to
17:41
the query Vector which is Journey you can pause the video here for
17:48
a while while you think about it let me give you a hint if I rephrase this question in another manner I'm sure
17:55
many of you will be able to answer what is that mathematical operation
18:01
which gives you the alignment between the two vectors which mathematical
18:06
operation lets you find whether two vectors are aligned with each other or whether they are not aligned with each
18:15
other keep that thought in your mind now let me nudge you more towards the answer
18:22
now we know that uh the embedding vectors encode meaning right so if two
18:27
vectors are align to each other which means that they are parallel to each other or if their angles are closer to
18:33
each other like journey and start it implies that they have some sort of similarity in their meaning so it makes
18:41
sense that more importance should be paid to start Vector because it seems to be more aligned with the journey Vector
18:48
whereas if you take the vector corresponding to one this purple Vector at the bottom you'll see that it's
18:53
almost perpendicular to the journey Vector right the vector one is like this the vector journey is like this so it's
18:59
almost perpendicular which means that they do not have that much similarity in meaning so when we assign importance
19:06
probably the vector corresponding to one should have less importance than the
19:11
vector corresponding to starts for the query Journey because starts is more aligned to
19:17
Journey now can you think which mathematical operation would quantify this
Dot product and attention scores
19:23
alignment let me reveal the answer it's the dot product between the vectors this this is an awesome idea because Let Me
19:31
Now go to Google and type dot product formula if you see the dot product formula it basically the if you take the
19:39
dot product of the two vectors it's the product of the magnitude of the vectors multiplied by the cosine of the angle
19:45
between them so if the two vectors are aligned with each other that means the angle between them is zero and if the
19:52
angle between them is zero cost of 0 will be one so the dot product will be maximum whereas if the two vectors are
19:58
not all aligned with each other that means they are perpendicular to each other the angle between the two vectors
20:03
is now equal to 90° and COS of 90 is equal to
20:09
0 so there is no similarity between these vectors and the dot product is zero so higher the dot product more
20:16
aligned the vectors are lower the dot product the vectors are not aligned so the dot product actually encodes the
20:23
information about how aligned or how not aligned the vectors are and that's exactly
20:29
that's exactly what we need so the dot product between journey and starts might be more so it's because they are aligned
20:36
so what if I use the dot product to find the attention scores that's the first
20:41
key Insight which I want to deliver from this lecture what if I use the dot product to find the attention score
20:48
between my query vector and the input Vector this is a key step in understanding the attention mechanism
20:54
the dot product is that fundamental operation because it encodes the the meaning of how aligned or not how not or
21:01
how far apart the vectors are and this is exactly what we are going to do next so the intermediate
21:08
attention scores are essentially calculated between the query token and
21:13
each of the input token and the way the attention scores are calculated is that we take a DOT
21:20
product between the query token and every other input token why do we take a DOT product because the dot product
21:26
essentially quantifies how much two vectors are aligned if two vectors are aligned their
21:33
dot product is higher so more attention should be paid to this pair of vectors so their attention score will be
21:39
higher so in the context of self attention mechanisms dot product determines the
21:47
extent to which elements of a sequence attend to one
21:52
another this is just a fancy way of saying how different elements of the
21:57
sequence are more Alik with each other so higher the dot product higher
22:03
the dot product higher is the similarity and the attention scores
22:09
between the two elements this is very important higher the dot product higher is the similarity between the two
22:15
elements or the two vectors and higher is the attention score a fancy way of describing two vectors which are aligned
22:22
to each other is saying that these two tokens attend more to each other whereas
22:27
if the two vectors are not not aligned we can say that these two attend less to each other for example just visually we
22:33
can see that journey and starts are aligned right so these two vectors attend more to each other so more
22:39
attention should be paid to starts when the query is Journey and if you look at
22:44
the vector one and the vector for Journey you'll see that they are not aligned they have a 90° angle between
22:50
them so they do not attend to one another this is the key idea behind
22:55
calculation of the attention scores and this is exactly what we are going to implement in the code right now so to
Coding attention scores in Python
23:02
compute the attention score between the query vector and the input Vector we simply have to take the dot product
23:08
between these two vectors so let's say the query is inputs of one why inputs of one because python
23:15
has a zero based indexing system and since our query is the second input token uh the second input token remember
23:23
is for a journey it will be indexed with one because python has a zero indexing
23:28
system so inputs of one will be the vector for Journey so let's say the
23:34
query is the inputs of one and then the attention scores need to be calculated
23:39
so first we initialize the attention scores as an empty tensor then what we'll do is that we Loop over the inputs
23:46
we'll Loop over the inputs and we'll take the dot product between every input vector and the query Vector that's it
23:54
and then we'll populate the attention scores tensor so the first element of the attention scores tensor is the dot
24:01
product between the first input embedding vector and the query Vector the second element of the attention
24:07
scores tensor is the dot product between the second input vector and the
24:12
embedding Vector similarly the Sixth Element in the attention score tensor is
24:17
the dot product between the sixth input vector and the query Vector so each of
24:24
these element each of the elements of the attention score is a DOT product between the input vector and the query
24:31
Vector now let us actually look at these attention scores a bit and try to uh look at their magnitudes right so which
24:40
which of these have the largest magnitude so we can see that this second second value the third value and the
24:48
sixth value have the largest attention scores so second third and six so let's see the which which words they
24:54
correspond to so second is the word journey Third is the word starts and sixth is the word step so of course
25:01
Journey has the high attention score right because the query itself is Journey so if the query itself is
25:06
Journey it will be aligned with the vector for Journey and so it will have the highest dot product but the second
25:13
and the third highest dot products are for starts and step and let's see whether that follows our
25:19
intuition so as we had earlier seen starts is also very closely aligned with journey so it makes sense that the dot
25:26
product between journey and starts is higher so the attention score for starts is higher similarly when you see step
25:33
you'll see that step also seems to be closely aligned with journey their angles seem to be similar and so the
25:40
attention score between step and journey will also be higher now let's look at the elements
25:45
with the lowest attention score so it seems to be the fifth element and the fifth element is the word one and let's
25:54
see whether that makes sense with our intuition so you can see the vector for the word one and the vector for Journey
26:00
almost have a 90° angle between them they are not at all aligned with each other and that is exactly captured in
26:06
the attention score that seems to be the least between journey and one so every
26:12
time you deal with attention scores try to have a mental map of why exactly is the attention score higher or why
26:18
exactly is the attention score lower now we'll move to the third
Simple normalisation
26:24
step or the next step rather before we comp compute the context vector and that
26:30
step is normalization why do we really need
26:36
normalization the most important reason why we need normalization is because of
26:41
interpretability what does that mean well what it means is that when I look at the attention scores I want to be
26:48
able to make statements like okay give 50% attention to starts give uh 20%
26:55
attention to step and give only 5% ATT attention to the vector one so when the query is Journey I want to make these
27:02
kind of interpretable statements in terms of let's say percentages that if the query is Journey of course Journey
27:08
will receive 20% attention 30% attention but starts will also receive higher
27:13
attention maybe 30% maybe step receives 20% attention and the rest of the vectors remain receive less percent if
27:21
I'm conveying this to someone they will get a much better idea right if I convey in terms of these percentages so I want
27:27
my attention scores to be interpretable and they are not right now because if
27:33
you sum up the attention scores they are more than one so we cannot express these in terms of
27:40
percentages and that's why we are now going to normalize the attention
27:45
scores uh now first okay so the main goal behind the normalization is to
27:51
obtain attention weights that sum up to one that's the main goal and why do we do this as I mentioned it's useful for
28:00
interpretability um and for making statements like in terms of percentages how much attention should be given Etc
28:07
but the second reason why we do normalization is because generally it's good if things are between zero and one
28:13
so if the summation of the attention weights is between zero and one it helps
28:18
in the training we are going to do back propagation later so we need stability during the training procedure and that's
28:24
why normalization is integrated in many machine learning framework It generally helps a lot when you are back
28:30
propagating and including gradient descent for example so uh what is the simplest way
28:37
to implement normalization can you try to think about it remember we want all
28:44
of these weights to sum up to one so what is the best way you can normalize this you can pause the video for a while
28:51
if you want okay so the simplest way to normalize this is to just sum up these
28:57
weights and then divide every single element by the sum so what that will do is that will
29:04
make sure that the summation of all the attention scores will be equal to one and this is exactly what we are going to
29:10
implement as the simplest way to normalize so what we'll do is that we'll maintain another tensor which is
29:17
attention weights to that will just be the attention scores divided by the summation so what will happen is that
29:23
every element of the attention score tensor will be divided by the toal total
29:29
summation and so the final tensor which we have which are also called as the attention weights will be this it will
29:35
be 0455 2278 2249 Etc and you'll see that
29:40
they sum up to one here I would like to mention one terminology and that is the difference between attention scores and
29:47
attention weights both attention scores and attention weights represent the same thing intuitively they mean the same the
29:54
only difference is that all the attention weights sum up to one whereas that's not the case for attention
30:00
scores so if you take the summation of these attention weights you'll see that it is equal to
30:06
one um okay this is great so you might think awesome what's the next step well it's not that great because uh there are
30:14
better ways to do normalization many of you might have heard about soft Max right if you have not it's fine I'm
30:20
going to explain right now but when we consider normalization especially in the
30:26
machine learning context it's actually more common and advisable to use the softmax function for
30:33
normalization and why is this the case um I think I need to write this down on
30:38
the Whiteboard to explain um why soft Max is preferred
30:44
compared to let's say the normal summation especially when you consider extreme values so let me take a simple
30:50
example right now and I'm going to switch the color to Black so that you can see what I'm writing on the screen
30:57
so let's say the element which we have are 1 2 3 and nine uh or let's say
31:08
400 let's say these are the elements
31:13
400 now if you do the normal summation uh and normalize it that way what will
31:19
what will happen is that 1 will be divided by all the entire summation right so the first element will be 1
31:24
divided by 1 + 2 + 3 + 400 which is
31:30
406 uh so the denominator here will be 4
31:37
0 6 then the last element similarly would be the highest element which is
31:44
the extreme value which we are considering here this highest element will be
31:52
400 divided by 406
31:59
so I'm just writing the denominator right now yeah so the last element which is the extreme value will be 400 divided
32:05
46 and this element will be 2 divided by 406 and this element will be 3 divided
32:11
46 so you might think that okay what is the problem here the problem here is that when we look at the inputs 400 is
32:19
extremely high right so the normalization should convey this information that you should completely
32:26
neglect these other values so ideally when such a situation occurs we want the
32:31
normalized values to be zero for these smaller values and we want the normalized value to be one for this
32:38
extremely high value and that's not the case when you use the summation operation when you do the summation this
32:44
normalized will be around let's say uh I not I'm not calculating this
32:50
exactly but let's say it's 0 point uh
32:55
0 0
33:01
25 and let's say this this calculation for the extreme value is let's say
33:06
around uh Point uh
33:15
9 let me write this again so this will be let's say around
33:21
0.9 9 just as an example so you might think
33:27
that okay this is almost close to one right but it should not be almost close to one it should almost be exactly equal
33:33
to one the reason why this is a problem is that when you get values like this
33:39
they are not exactly equal to zero so when you are doing back propagation and gradient descent it confuses the
33:45
optimizer and the optimizer still gives enough or weight some weightage to these values although we should not give any
33:51
weightage to these values so ideally the normalization scheme should be such that
33:56
when we normalize the small values should be close to zero in such a case and the extremely large values should be
34:01
close to one and that is not achieved through this summation based normalization however this exact same
Softmax normalisation
34:08
thing is achieved if we do a softmax based normalization so let me explain what actually happens in the softmax
34:15
based normalization so we currently have the attention scores like these right we
34:20
have X1 X2 dot dot dot up till X6 in the softmax what happens is that we take the
34:28
exponent of every element and then we divide by the summation of the exponents so the denominator the summation is e to
34:36
X1 plus e to X2 plus e to X3 plus e to X4 plus e to X5 plus e to X6 so the
34:44
first element will be e to X1 divided by the summation the second element will be
34:49
e to X2 divided by the summation and similarly the last element will be e to X6 divided by the summation now if you
34:57
add add all of these elements together you'll see that they definitely sum up to one that's fine but the more
35:03
important thing is when you look at these extreme cases if you in if you use soft Max 400 now when you do e to 400
35:11
that will almost be like Infinity so this value when normalized will be very close to one and the smaller values when
35:19
normalized using softmax will be very close to zero so it's much better to use softmax when dealing with such extreme
35:26
values which may occur when we do large scale uh models like llms that's why
35:32
it's much more preferable to be using soft Max now we can easily code such
35:37
kind of a soft Max in Python ourself but it's much more recommended to use the
35:42
implementation provided by py torch so what py torch does is that instead of doing e to x divided by the summation it
35:50
actually subtracts the maximum value from each of the values before
35:56
doing the exponential operation so the first element will be e to X1 minus the
36:01
maximum value among X1 X2 X3 X4 X5 X6 divided by the summation and the
36:08
summation will be e to X1 - Max Plus e to X2 - Max Etc so the only difference
36:14
between pytorch implementation and the naive normal implementation we saw before is that pytorch includes this
36:20
minus maximum so it subtracts all the values by the maximum value uh before
36:26
doing the exponent IAL operation now mathematically you'll see that when you expand this e to X1 minus
36:34
the maximum it can also be written as e to X1 divided e to maximum so the E to
36:40
maximum in the numerator and denominator actually cancels out and the final thing what we get is actually the same thing
36:47
as what we had written previously so you might be thinking then why is this even implemented this subtracting by the
36:54
maximum the reason it's implemented is to avoid void numerical instability when dealing with very large values or very
37:01
small values if you have very large values as in the input or very small values It generally leads to numerical
37:08
instability and leads to errors which are called overflow errors this is a computational problem so theoretically
37:15
we can get away with this implementation which is on the screen right now but when you implement computation in Python
37:21
it's much better to subtract with the maximum to prevent overflow problems I have seen some companies in
37:28
which this is also asked as an interview question so it's better to be aware of how pytorch Implement softmax and we'll
Coding attention weights in Python
37:34
be doing both of these implementations in the code right now so let me jump to the coding interface as I said the first
37:41
thing which we'll be implementing is naive soft Max without dealing with the Overflow issues without subtracting the
37:46
maximum value um so what we'll be doing is simply taking the exponent of each
37:54
attention score and dividing by the summation of the exponent the dimension equal to Z is used because we are taking
38:01
the summation for a full row so we are essentially summing all the entries in a row and that's why we have the dimension
38:08
equal to zero why are we summing the entries in a row because if you look at
38:13
these Matrix over here um yeah this X1 X2 X3 X4 X5 and X6 this is one row right
38:22
and when we calculate the summation we are going to do e to X1 plus e to X 2
38:27
Etc so we are essentially summing all the entries in a row and that's why we have to give Dimension equal to zero as
38:34
an exercise you can try out Dimension equal to one and see the error which you get so this is how we'll implement the
38:41
knive soft Max and uh which will basically take the exponent of each attention score and divide by the
38:47
summation so we'll print out the attention weights obtained using this method and you'll see that it's 1385 dot
38:54
dot do1 1581 of course these scores are different than the summation because
38:59
here we are using the exponent operation but if you sum up these scores you'll see that they also sum to one that's
39:05
awesome and now what we'll be doing is we'll be using the P torch implementation of soft
39:11
Max one more key point to mention is that the soft Max attention weights
39:16
which are obtained are always positive because we always deal with the exponent operation and that's positive the
39:24
positivity is important to us because it makes the output interpret so if we want to say that give 50% attention to this
39:32
token give 20% attention to this token we need all the outputs to be positive
39:37
right if it's not positive well uh it's it becomes very difficult to interpret in that
39:45
case uh now um so note that knife softmax implementation May encounter
39:51
numerical instability problems which I mentioned to you such as overflow for very large values and underflow for very
39:57
small values therefore in practice it's always advisable to use the pytorch
40:02
implementation of soft Max so it just one line of command torch. softmax and we pass in the
40:09
attention score tensor and what this one line of command torge do softmax
40:14
actually does is that it implements um what I'm what I've shown
40:19
on the screen right now it implements this e to X1 minus maximum Etc this it
40:25
converts the input into this normalized format so this is tor. softmax and you
40:32
can quickly go to the documentation to see tor. softmax and you'll see the py
40:38
torge documentation for for softmax I'll be sharing this link also with you uh in
40:44
the information section of the YouTube video okay so this is basically pytorch
40:53
softmax and we have got these attention weights and we'll see that the attention weight up to one one thing I want to
40:59
show is that our knife soft Max results in this attention weight tensor and the
41:04
pytorch softmax also results in the exact same attention weight tensor since
41:09
we don't have any large values or any small values we don't have any overflow or underflow issues here and so both our
41:15
knife implementation and the pytorch implementation give the same results but the reason it always advisable to use py
41:23
torch is that later we'll be dealing with very large parameters and some of
41:28
those might be huge so it's better to um have numerical or better to deal with
41:33
numerical instability awesome so remember the one reason why we calculated the attention
41:40
weights is for interpretability right now let's try to interpret so the attention weight for the first Vector is
41:48
around .13 the attention weight for the second is 2379 attention for the third
41:54
is 233 Etc this means that we should pay
41:59
about 13% attention to the word your about 23% attention to
42:05
Journey about 23% attention to starts 12% attention to with uh with one 10%
42:14
attention to one and 15% attention to step so high attention is being paid to
42:20
journey and starts so as you can see here High attention is paid to journey and starts
42:25
and if someone asks how high we can can say that well about 20% and low attention is paid to one and
42:31
if someone asks how low we can say that well only 10% we are able to make this interpretable statements only because we
42:38
converted the attention scores into attention weights and remember that's the difference between attention scores
42:44
and attention weights attention weights sum up to one so it's much easier to make this kind of interpretable
42:53
statements so we have computed the attention weights right now and we are
42:58
actually ready to move to the next step which will which is actually the
43:04
final step of uh Computing the context Vector so let me take you to that
43:10
portion of the Whiteboard right now so after Computing the attention weights
43:16
now we have actually come to the last step of finding the context Vector so
43:22
let me just show you the image of what all we have covered up till now so we had this input query we computed the
43:29
attention scores by taking the dot product between the input query and each of the embedding vectors and from the
43:36
attention scores we actually got the normalized attention scores which are called as attention weights and these
43:42
attention weights actually sum up to one awesome so we have reached this
Context vector calculation visualised
43:48
stage right now and now we are ready to get the context Vector I just want to
43:53
intuitively and Visually show you how we calculate the context vector before coming to the mathematical
43:59
implementation so let's say these are the embedding vectors for the different words in the sentence and here I've also
44:06
mentioned the relative importance of each so Journey carries 25% importance because that's the query
44:14
so it should carry the highest importance but starts carries 20% importance
44:20
um then step carries 15% importance and one with and your carry less importance
44:28
now how do we get these how do we use these attention weights to compute the
44:34
ultimate context Vector so the way this is done is that let's say we use starts
44:40
so the starts Vector is now multiplied by the attention contribution and that
44:46
is equal to 02 right because 20% is important so it's importance is 20% so
44:52
the starts Vector is multiplied by 02 which means it scaled by by the
44:58
corresponding attention attention weight so the starts Vector is multiplied by 02
45:03
so it will be scaled down like this the width vect width Vector is scaled down by 15 because it carries 15% importance
45:11
the step Vector carries 15% importance the journey Vector carries 25% importance so it will it will be scaled
45:18
down by 1/4 uh and the one vector carries only 10% importance so it will be scaled down
45:25
by a lot which is about 1110th now what will and similarly your carries 15%
45:31
importance so it will be scaled down by multiplied by5 so what we do is that we multiply
45:39
each of the input embedding vector by the corresponding attention weights and we scale them down that much so now we
45:45
have uh the multiplied weights for each of these and we take the vector summation of all of these and you add
45:52
the vector summation and that gives the final context Vector like this so this is now my cont context Vector for
46:00
Journey so this is the context
46:07
Vector let me write it down again okay so let me explain this again
46:14
so we have calculate we have multiplied each of the input embedding Vector with the corresponding attention weight and
46:21
we have got these six vectors right we sum up all of these six vectors and that
46:27
uh now describes the context vector and this I'm just writing here context this
46:33
is the context Vector for the embedding Vector of Journey now look at how this context
46:40
Vector is calculated the context Vector has some contributions from all other vectors so it's an enriched embedding
46:47
Vector it has 25% contribution from the embedding Vector but it has all the
46:52
other contribution from other vectors and those contributions symbolize something those contributions symbolize
46:59
how much importance is given to the other vectors for example we have about 20% contribution from starts because the
47:06
attention weight to starts is 0.2 we have only 10% contribution from one
47:11
because the attention weight for one is 0.1 and that's why context vectors are
47:17
so important I wanted to show this to you visually because the context Vector which will calculate for modern llms for
47:24
large scale models like GPT they carry the exact same meaning they are enriched embedding vectors so the context Vector
47:32
for Journey looks like this and we can also see this in code so I have just written a small code to find the to plot
47:39
the context Vector I'll show the mathematical derivation but for now just take a look at this context Vector for
47:45
Journey which has been shown in Red so the vector embedding for Journey has been shown in green but the context
47:53
Vector is shown in red which is the summation of all the other vectors which I just showed to you and this is how it
47:58
looks like in the three-dimensional space ultimately we are interested in these context vectors and it was only
48:05
possible to get this context Vector because of the attention mechanism that's why the attention mechanism is so
48:11
important we would have been stuck at the embedding Vector if we did not have the attention
48:17
mechanism now let us look at the mathematical representation um regarding how we
48:22
actually compute the context Vector from the attention weights and if you have
48:29
understood the Whiteboard description which I just showed you understanding the this mathematical operation will
48:34
actually be very easy okay so we have reached this stage
48:40
now where we have computed these attention weights and after Computing the normalized attention weights what
48:46
we'll do is that we'll compute the context vector and currently we're looking at the context Vector for
48:51
Journey so it's Z2 and to do that we'll multiply all the embedded in input
48:57
tokens with the corresponding attention weights this is very important and so
49:02
that was the scaling which I showed you in the figure and then we will sum up all of the resultant
49:07
vectors when we sum up all of the resultant vectors that will give us the final context Vector for the token
49:14
journey and this has been showed in this schematic right now let me just rub rub
49:20
this here so that I can explain this to you in a better manner okay so we have the attention
49:26
weights for every token right so we have1 2 Etc so what we'll do is that for
49:33
the first input embedding we'll multiply the attention weight with this Vector for the second input embedding
49:40
we'll multiply the attention weight for the second input embedding with the second Vector we'll multiply the third
49:47
attention weight with the third input embedding and similarly we'll multiply
49:52
the sixth attention weight with the sixth input embedding this is scaling down the vectors in the victorial
49:57
representation and then we'll add all of them together when we add all of them together we'll ultimately get the
50:03
context Vector for Journey and this is the final answer this is the context Vector for Journey and I've have plotted
50:11
this context Vector over here which has been calculated through this uh mathematical operation which I just
50:16
showed you on the screen and now we'll be implementing this operation in Python to calculate
Coding context vectors in Python
50:22
the context vector and it's pretty simple it's only two to three lines of code first we have the query which is
50:28
the inputs index by one because the word which we are looking at is Journey then we initialize uh tensor context Vector
50:36
two why two because we are looking at the second token journey and we are finding the context Vector for that so
50:43
what we'll be doing is that we'll be looping through all the inputs and uh what we'll be doing is
50:49
that we'll scale each input with the corresponding attention weight and then we'll add all the scaled vectors
50:56
together to give the final context Vector that's it so let's say we are looking at the first input Vector which
51:03
is the first input embedding we'll multiply it with the first attention weight then we'll look at the second
51:08
input Vector we'll multiply it with the second attention weight then we'll look at the sixth input Vector at the end and
51:14
multiply it with the sixth attention weight and we'll add all of these vectors together which ultimately leads
51:20
to the final context vector and that's the one which I've showed here in the red arrow I've also shown the any
51:26
context here as a ping Dot and how it's different from the other vectors awesome so we have reached this
51:34
step where we have calculated the context Vector for Journey right however the task is not yet over because we have
51:41
to calculate a similar context Vector for all the other tokens right we have to calculate the similar context Vector
51:48
for your journey starts with one step all of these six
51:53
words and now if you have understood this computer comput which we did for Journey we can actually extend the exact
52:00
similar computation to compute the attention weight and context Vector for all the other
52:07
inputs and this is actually represented very nicely with this which is called as the attention weight Matrix so what the
52:14
Matrix which you're seeing on the screen right now is called as the attention weight Matrix and let me explain uh this
52:20
Matrix in a very simple manner so if you look at the rows each
52:26
row represents the attention weights for one particular word So currently we have
52:31
calculated the attention weights for Journey right so the first value here 13 is the
52:39
attention score or the attention weight between journey and your the second value here 23 is the
52:47
attention score or the attention weight between journey and
52:52
journey the second value here is the attention
52:57
weight between journey and starts the fourth value here is the
53:05
attention weight let me yeah the fourth value here is the attention weight
53:12
between journey and width the fifth value here is the
53:18
attention weight between journey and one and the sixth value here is the attention weight between journey and
53:24
step so these are the sixth atten ention weights which we also computed over here
53:29
so these are the six attention weights which have been computed here we have just rounded off the values so the values might not be exactly similar but
53:37
these are the uh these are the six attention weights okay now um let
53:45
us go next okay so similarly what we have to do is we have to find
53:52
essentially similar attention weights for all the other words like let's say if we look at starts we have to find six
53:59
attention weights for starts we have to find six attention weights for width we have to find six attention uh weights
54:05
for step and we have to find six attention weights for your and one so for every every query we have to
54:13
find six attention weights so all of these which I'm highlighting with star right now all of these are the queries
54:20
currently we only looked at the journey query but now we have to essentially replicate the exact same computation for
54:27
all the other queries as well so how will we do this let's say if the query is Step we'll find the
54:34
attention weight between step and all the other words and then we'll find the context vector by doing the summation
54:41
operation like we did at the end for the query of Journey so essentially we are going to
54:48
follow the exact same steps as before for all the other tokens also we are
54:54
first going to compute the attention scores then we are going to compute the attention weights and then we are going
55:00
to compute the context Vector remember these are the exact same steps which we followed uh for uh the query of Journey
55:09
and these are the exact same steps which we will follow for other queries as well so let me take you through code right
55:16
now and let us start uh implementing the attention or let us
55:22
start calculating the these three steps for the other queries as well so as we
Coding attention score matrix for all queries
55:29
discussed we have to follow three steps the first step is to find the attention scores and remember how do we find the
55:37
attention score if we have a particular query we'll just take the do product of that with all the other input vectors so
55:44
let's say if the query is in inputs we'll take the dot product of the query
55:50
with all the other vectors in the input so one way to find the attention scores
55:55
is to just Loop through the input two times and essentially find the dot product uh I'll show you what that
56:03
actually means uh right so let me rub these these
56:09
things over here so that I can show you this one method of finding the attention scores so let's say you Loop over the
56:16
input Vector right so the first you'll encounter your then you will find the dot product between your and all these
56:24
other uh all the other inputs so that will be uh the result of
56:31
the first inner loop so what you do is that first you fix an i in the outer
56:37
loop and in the Inner Loop you go through the input entirely so when we fix an i in the outer loop it means we
56:43
fix this sarey then we go through the inner loop entirely and find these six
56:49
dot products now change the outer loop so then the outer loop changes to journey and then similarly find the dot
56:55
product between Journey and all the other vectors now change the outer loop once more so similarly we'll change the
57:01
outer loop and in each outer loop we'll go through the inner loop so that is essentially finding the dot products
57:07
which are the attention scores uh the problem with this approach is that this will take a lot of
57:15
computational time so if you look at the output tensor so this is a 6x6 and each
57:21
element in this tensor represents an attention score between two pairs of inputs so for example this 6x6 Matrix
57:29
which you just saw in the code is very similar to this here I'm showing the normalized attention scores but even the
57:35
attention scores look like the 6x6 so if you look at the first row all of those
57:40
are the dot products between the first query and all the other queries if you look at the second row all of these are
57:46
the dot products between the second query and uh all the other inputs so
57:53
this second this second row actually will be exactly same to the attention uh
57:59
scores which we had calculated earlier see because the second row is
58:05
9444 1.49 ETC so if you look at the second row here that is also 9544
58:12
1.49 because the second row represents the dot product between the second query
58:17
and all the other input vectors similarly the last row represents the dot product between the last query and
58:24
all the other input vectors okay now here we have used two for Loops
58:30
right and that's not very computationally efficient for Loops are generally quite slow and that's the
58:35
reason why matrix multiplication needs to be understood the reason I say that linear algebra is actually the core
58:42
Foundation of every machine learning concept which you want to master is this for someone who does not know about
58:49
linear algebra and matrix multiplication they'll just do these two rounds of four Loops but if you actually know linear
58:55
algebra you'll see that instead of doing this you can just take the uh multiplication of inputs and the
59:02
transpose of the inputs and you will actually get the exact same answer so what this does is that you
59:08
take the input uh input Matrix and what the input Matrix looks like is this you
59:15
take the input Matrix and then you multiply with the transpose of the input Matrix and you'll get the exact same
59:21
answer as uh you'll take you'll get the exact same answer as doing this dot product in the
59:28
for Loop format and you can verify this so if you just take the product between
59:34
inputs Matrix and the inputs transpose what we'll see is the exact same thing
59:40
as the previous answer why because when we multiply two matrices what it essentially does is it just computes a
59:47
bunch of dot products between the rows of the first Matrix and The Columns of the second Matrix which is exactly what
59:53
we are doing here in these two for Loops it just that this matrix multiplication operation is much more efficient than
59:59
using these two for Loops so Step One is completed right now we have found the
1:00:05
attention scores I hope you have understood why this is a 6x6 Matrix here and why each what each row represents
1:00:12
each row represents the dot product between that particular query and all
1:00:18
the other input um input embedding vectors now we'll implement the
Coding attention weight matrix for all queries
1:00:24
normalization so remember how we did normalization here we did torch. soft Max right so similarly what
1:00:32
we are going to do here is we are going to do torge do softmax of this attention scores Matrix and what this will do is
1:00:39
that it will implement the soft Max operation to each row so the first row
1:00:44
we'll do the soft Max like we learned before then the second row we do the softmax like we learned before similarly
1:00:51
the last row will do the soft Max like we learned before so if you look at each individual row you'll see that entries
1:00:57
of each row sum up to one so you can look at the second row here1 385 2379
1:01:04
it's the same U attention weights which we have got for the
1:01:10
journey query uh one key thing to mention here
1:01:15
is that what is the dim parameter over here so the dim here I'm saying minus
1:01:21
one and the reason is explained below the dim parameter in functions like like tor. softmax specifies the dimension of
1:01:28
the input function input tensor along which the function will be computed so
1:01:34
by setting dim equal to minus1 here we are instructing the softmax function to
1:01:40
apply the normalization along the last dimension of the attention score tensor and what is the last dimension of
1:01:47
the attention score tensor it's essentially The Columns so if the attention scores is a 2d tensor it's a
1:01:53
2d 6x6 tensor right and it has the shape of rows and columns uh the last
1:01:59
Dimension is the column so Dimension equal to minus one will normalize across the
1:02:05
columns so what what will happen is that for the first row look at the columns so
1:02:10
this is the first column this is the second column actually I have to show here this is the First Column this is
1:02:15
the second column this is the third column so we are normalizing essentially along the columns right because we are
1:02:21
going to take the exponent of what all is there in the First Column second column third column Etc we are going to
1:02:26
sum these exponents that's why it's very important to uh write this dim equal to minus1
1:02:34
because we are normalizing across a column um and that's why the values in
1:02:39
one row sum up to one since we are normalizing in the each column that's
1:02:45
why the values in each row sum up to one it's very important to note that so Dimension equal to minus1 means that we
1:02:51
have to apply the normalization along the last Dimension and for a two dimensional t sensor like this the last
1:02:56
Dimension is the columns so the soft Max will be applied across the columns and
1:03:02
that's why for each row you will see that all the entries sum up to one so these are the attention weights
1:03:09
which we have calculated and the last step which is very important is calculating the context vector and uh I
1:03:16
want to uh show some things to you but before that let's verify that all the rows indeed sum up to one in this
1:03:23
attention weights so what I'm doing here is that I'm looking at the second row here and I'm just going to sum up to one
1:03:30
and I'm just going to sum up the entries of the second row and you will see that uh the second row sums up to one and I
1:03:37
have also included a print statement below which prints out the summation of all the rows and you'll see that the
1:03:44
first row the second row similarly the sixth row all of the rows essentially sum up to one this means that the
1:03:50
softmax operation has been employed in a correct manner for you to explore this
1:03:55
dim further you can try with dimm equal to Z dim equal to 1 also from the errors you will learn a
1:04:02
lot these small details are very important other students who just apply llm Lang chain and just focus on
1:04:10
deployment will never focus on these Minor Details like what is this dim operator over here Etc but I believe the
1:04:17
devil always lies in the details so the students who understand these Basics will really Master large
1:04:24
language models much more than other students now we come to the final step which is essentially Computing the
Coding context vector matrix for all queries
1:04:30
context vectors right and I will take you to code but the final step is
1:04:35
actually implemented in elegant one line of code let me take you to the final
1:04:41
step before what we had done in the final step remember what we simply did was uh we just
1:04:48
uh where was that yeah in the final step what we simply did was we just
1:04:53
multiplied the attention weights for each Vector for each input Vector with
1:05:01
that corresponding Vector right I can show this to you in the Whiteboard
1:05:06
also okay so what we did for the final step of the context Vector was something
1:05:12
like this yeah so what we did was we we got the attention weights for
1:05:18
each input embedding vector and we multiplied those attention weights with each with the corresponding input vector
1:05:25
and we those up now this is exactly what we have to do for the
1:05:30
other uh for the other tokens also but we have to do this in a matrix manner
1:05:35
because we cannot just keep on looping over and use for loops and there is a very elegant Matrix operation which
1:05:42
actually helps us calculate the context vectors it's just one line of matrix
1:05:47
product essentially we have to multiply the uh attention scores Matrix or the
1:05:56
attention weight Matrix with the input right and we have to do some summations can you think of the matrix
1:06:02
multiplication operation which will directly give us this answer um it's fine if you don't know
1:06:09
the answer but the simplified matrix multiplication is just essentially
1:06:14
multiplying the attention weights with the inputs that's it uh this last step of finding the
1:06:21
context Vector is just taking this attention weight Matrix and multiplying it with the input Matrix and the claim
1:06:27
is that it will give us the context vectors it will give us the six context vectors which we are looking for
1:06:33
remember we need a context Vector for every token right and there are six tokens so here are the six context
1:06:39
vectors now I'm going to try to explain why this matrix multiplication operation
1:06:44
really works so let's go to the Whiteboard once more all right so this is the first
1:06:51
Matrix which we have and that's the attention weights right here
1:06:57
and this is the second Matrix which we have which is the inputs uh so keep in mind here that the
1:07:04
attention weights is a 6x6 Matrix so we have six rows and six columns and the inputs is a 6x3 matrix now we have
1:07:13
already looked at how to find the uh context Vector for the
1:07:18
second uh for the second row right which which essentially corresponds to the word journey and uh so let's see what we
1:07:26
exactly did here so the final attention Matrix will be a 6x3 matrix because so
1:07:33
sorry the final context Vector Matrix will be a 6x3 matrix because every row of this will be a context Vector so the
1:07:41
first row will be the context Vector for the first word the second row will be the context Vector for the second word
1:07:48
so let's look at the second word which is essentially the context Vector for Journey now uh if we take a product of
1:07:57
these two Matrix so let's say if we take the product of the attention weight Matrix and the input Matrix first let's
1:08:02
check the dimensions so this is a 6x6 Matrix and the inputs is a 6x3 so 6X 6
1:08:09
can be multiplied with 6x3 so taking the product is completely possible and it will result in a 6x3
1:08:15
matrix uh so let's look at the second row if you look at the second row it will be uh something like we we'll take
1:08:22
the second row uh so the second row First Column would be the dot product
1:08:29
between the second row of this and the First Column of this the second row second column will be the dot product of
1:08:34
this with the second column of this and the second row third column will be the dot product of this second row and the
1:08:40
third column here so that's what I've written here in the output Matrix so the first element of the
1:08:47
second row will be the dot product between the second row and the First Column the second element of the second
1:08:54
row will be the dot product between the uh second row and the second column and
1:08:59
the third element of the second row will be the dot product between the third row
1:09:04
of the first Matrix and the third column of the second Matrix right now when you compute these dot
1:09:11
products uh very surprisingly you will see that the answer is actually equal to
1:09:17
this the answer is 138 the answer is actually 138 which is
1:09:24
138 multiplied by the first row over here plus 237 multiplied by the second
1:09:32
row over here plus 233 which is multiplied by the third row over here
1:09:37
Etc uh it's just a trick of matricis but what it it turns out that this second
1:09:45
row second row can also be represented by this formulation where you take
1:09:52
the uh first element of this second row multiply it with the first
1:09:59
row of the input Matrix plus the second element of the second row multiply with
1:10:04
the second row of the input Matrix plus the third element multiply it with the
1:10:09
third row of the input Matrix can you see what we are essentially doing here we essentially scaling every input
1:10:15
Vector right we take the first input Vector we take the first input Vector we scale it by 138 we take the second input
1:10:23
Vector we scale it by 237 we take the third input Vector we scale it by 233 isn't this the exact same
1:10:30
scaling operation which we saw uh when we looked at the visual representation of the uh context Vector calculations
1:10:39
remember we have seen seen the scaling operation to calculate the context Vector here where we had taken each of
1:10:45
the input vectors and we had scaled it by the attention weight values and then
1:10:50
we summ them to find the final context Vector this is the exact same thing which I which we are doing over here so
1:10:56
when we take the product of the uh when we take the product of the attention weights and the inputs another way to
1:11:03
look at it is that if you look at the second row it's actually scaling the first input by 138 scaling the second
1:11:11
input by 237 dot dot dot and scaling the sixth input by 0158 so it's the exact same operation as
1:11:18
we performed before for finding the context vector and that's why finding the context vectors is as simple as
1:11:25
multiplying the attention weights with the inputs the first row of this answer will give you the context Vector for the
1:11:32
first uh input embedding Vector the second row will give you the context
1:11:38
Vector for the second token the third row of this product will give you the context Vector for the second token for
1:11:44
the third token and right up till the very end the sixth row will give you the context Vector for the final token and
1:11:51
this is how the product between the attention weights and the inputs will give give you the final context Vector
1:11:57
Matrix which contains the context Vector for all of the tokens which you are looking for and uh with this final calculation
1:12:05
we calculate the context Vector for all um of the input
1:12:11
tokens and this is exactly what I've have tried to do here so finally uh we
1:12:16
generate a tensor which is called as the all context vectors and we multiply the attention weight Matrix with the input
1:12:22
Matrix and then we get this all context Vector tensor and if you look at the
1:12:28
second row here 4419 6515 56 you'll see
1:12:33
that this is exactly the same value of the context Vector which we had obtained over here uh when we looked at Journey
1:12:41
so this again implies that whatever we are doing here uh with the matrix multiplication is leading to the correct
1:12:48
answer so based on this result we can see that the previously calculated context Vector 2 for journey matches the
1:12:55
second row in the in this tensor exactly so remember this operation to get the
1:13:00
final context Vector we just multiply the attention weights with the inputs if you did not understand the matrix
1:13:06
multiplication which I showed on the Whiteboard I encourage you to do it on a piece of paper because this last matrix
1:13:14
multiplication is very important to get the context Vector we just have to multiply the attention weight Matrix
1:13:20
with the input Matrix and what all you are learning right now will Direct L extend to the
1:13:26
key query and value concept which we'll cover when we when we come to causal attention and multi-head attention and
1:13:32
even in the next lecture in the next lecture we are going to look at this exact mechanism but with trainable
1:13:38
weights then we'll come to the concept of key query and value but these operations which we are looking at here
1:13:45
so for example uh here we are taking the matrix product between attention weights and inputs right in the key query value
1:13:52
this will be replaced this inputs will be replaced by value we'll also have a key and a query but
1:13:59
the underlying intuition and the underlying mechanism is exactly the same so if you understand what's going on
1:14:04
here you'll really understand key query value very easily okay one last thing which I want
Need for trainable weights in the attention mechanism
1:14:11
to cover um at the end of today's lecture is that okay so you might think that we already then find the context
1:14:19
vectors like this right then what's the need for trainable weights
1:14:25
we just take the dot product and we then find these context vectors the main problem with the
1:14:31
current approach is that think about how we found the attention weights to find
1:14:37
the attention weights all we did was to just take the dot product right so currently uh in our world the reason why
1:14:45
we are giving more attention to starts is that the alignment between starts and journey is maximum so the only reason
1:14:52
why we are giving more attention to starts is because because it semantically matches with
1:14:57
journey because we are only getting the attention scores and attention weight from the dot product however that is not
1:15:04
correct right because two vectors might not be semantically aligned but maybe
1:15:09
they are more important in the context of the current sentence so for example journey and one are not semantically
1:15:17
related to each other but what if in the current context one is the is the vector
1:15:23
which is more important so apart from the meaning you also need to capture the information of the
1:15:30
context right what is happening in the current sentence and without trainable weights
1:15:35
it's not going to happen we are not going to capture the context effectively right now we did manage to capture the
1:15:41
context somewhat but we only give attention to Words which are similar in meaning to the query but even if a word
1:15:49
is not similar in meaning it still might deserve attention in the context of the current sentence
1:15:55
so let's take a simple example here okay so the example is the cat sat
1:16:01
on the mat because it is warm and let's say our query is warm so in the first
1:16:07
case let's say we do not use trainable weights like what we have done in today's lecture if we don't use
1:16:13
trainable weight we only take the dot product between the query warm and each words embedding and we'll find that warm
1:16:20
is most similar to itself and maybe somewhat related to mat words like the cat and sat might have
1:16:27
low similarity scores because they are not semantically related to or so with with this so if we don't consider
1:16:34
trainable weights we'll only look at Words which are more similar to this query which is warm now with trainable
1:16:40
weights the model can learn that warm should pay more attention to mat even if mat is not semantically related to warm
1:16:47
so what will happen without trainable weights is that mat and warm might be vectors which are like this which have a
1:16:53
90° angle and they might not be related because their meaning is not related but that does not mean we should not pay
1:17:00
attention to mat because in this context probably Matt is the most important if the query is warm because the mat is
1:17:06
warm but the meaning of mat and warm are not not related right that's why we need
1:17:12
trainable weights with trainable weights the model can learn that warm should pay
1:17:17
more attention to Matt even if mat isn't semantically similar to warm in traditional embedding space so this is
1:17:24
where important the trainable weight allows the model to learn that warm often follows mat in context like this
1:17:31
one and that's how it captures long range dependencies that is the reason why we need trainable weights without
1:17:38
trainable weights this meaning would be lost and we would only be looking at Words which are similar to The query by
1:17:45
trainable weights we get more more of this information that okay Matt might not be related in meaning but in the
1:17:52
current context mat is the word which is more important because the mat is warm
1:17:57
this is how trainable weights allow us to capture context and in the next lecture we'll specifically devote the
1:18:04
next lecture to uh the simplified self attention mechanism but with trainable weights so here is the lecture notes for
1:18:12
the next lecture self attention mechanism with trainable weights we'll introduce the concept of key query value
1:18:18
and then slowly we'll move to the concept of uh causal attention and then
1:18:26
we'll move to the concept of multi-head attention so up till now we have covered simplified self attention I know this
1:18:32
lecture became a bit long but it was very important because uh I have seen no
1:18:37
other lecture or no other material which covers this much detail visually theoretically and in code about the
1:18:44
attention mechanism I could have directly jumped to key query and value which will come later but then you would
1:18:50
not have understood the meaning but this lecture allowed me to build your intuition I hope you're liking these set
1:18:57
of lectures if you have any doubts or any questions please ask in the YouTube comment section and I'll be happy to
1:19:03
reply thank you so much everyone and I really encourage you to take notes while I'm making these lectures I'll also
1:19:09
share this code file with you um thanks everyone and I look forward to seeing you in the next lecture

***






