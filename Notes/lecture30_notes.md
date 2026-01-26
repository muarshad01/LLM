* 10:00

* Logits --> Top-K --> Logits/Temp --> Softmax --> Sample from Multinomial

***

* 15:00
  
retain those values where the
15:08
logits where the only return the top K
15:12
values in the logit tensor and we
15:14
replace all the other values with
15:16
negative
15:17
Infinity that's what is being done in
15:19
this part of the code then as I
15:21
mentioned so after top K is applied then
15:24
we'll scale with
15:26
temperature so in the next step what we
15:29
are going to do is that if temperature
15:30
is greater than zero we are going to
15:32
scale the logits with temperature then
15:34
we are going to take the soft Max and
15:36
then we are going to do the next token
15:38
ID using the multinomial probability
15:40
distribution so exactly these three
15:42
steps you scale with the temperature you
15:45
take the soft Max and then you sample
15:47
from multinomial this is how you select
15:49
the next token ID and what you predict
15:51
the next token ID to
15:53
be now if temperature is not specified
15:56
so if the user does not specify the
15:58
temperature
15:59
we will just uh do the sampling as
16:02
before we'll just get the token with the
16:06
maximum probability and we'll say that
16:08
uh we'll get the token ID with the
16:10
maximum probability and we'll say that
16:12
this corresponds to the next token in my
16:14
sequence but mostly when GPT
16:17
architectures are developed and built
16:19
the temperature value is specified I'll
16:21
come to that in a
16:22
moment there is one more uh loop here
16:25
which is basically if end of sequence
16:27
token so this EOS is actually end of
16:29
sequence if end of sequence token is
16:33
encountered then uh we can stop
16:37
early however if end of sequence is end
16:40
of sequence ID is not specified then we
16:43
can just uh keep on generating the next
16:47
token and then what we do is that uh we
16:50
append the new generated token ID to the
16:53
current ID until we reach this maximum
16:56
number of new tokens which are to be
16:58
generated so here you can see the flow
17:00
we have the logit tensor we apply the
17:03
top K sampling then we scale the logits
17:05
with the temperature and then we sample
17:07
the next token from a multinomial
17:09
probability distribution uh if
17:11
temperature is not specified then we
17:13
just use the argmax which was done
17:16
previously then we choose the next token
17:18
uh as that one which has the maximum
17:20
probability score and then we just keep
17:23
on appending the next token ID until the
17:26
maximum number of new tokens limit is
17:28
reached
17:30
this is exactly what is happening in the
17:32
uh generate function which we have
17:34
defined right now to generate the new
17:36
token and now what I'm going to do is
Testing top-k + temperature scaling on demo example
17:38
that I'm going to generate 25 new tokens
17:42
and let's compare the performance with
17:44
what we had earlier so earlier when the
17:46
KN decoding strategy was used of just
17:49
using the maximum probability you'll see
17:51
that the next tokens were was one of the
17:53
xmc laid down across the seers and
17:55
silver of an exquisitely appointed these
17:58
words we're not making too much sense so
18:01
let's say if the decoding has improved
18:03
using this new generate
18:05
function so I'm going to use this new
18:07
generate function now which has been
18:09
defined over here and we are going to
18:11
pass in the GPT model and then we have
18:14
to pass in the input token IDs which is
18:16
now every effort moves you maximum
18:19
number of new tokens I'm setting to be
18:20
15 the top ke tokens I'm setting to be
18:23
25 uh so I'm just going to look at the
18:26
25 tokens so remember the vocab size is
18:30
50257 and I'm going to look at the 25
18:33
tokens which have the highest
18:34
probabilities in those 50 to
18:37
57 and the temperature value have set to
18:40
1.4 this is a good trade-off because if
18:42
the temperature value is too low we'll
18:44
have sharper probability distributions
18:46
if it's too high then we'll get a
18:48
non-sensical output so 1.4 seems to be
18:51
like a good tradeoff you can even
18:52
experiment with this further so when I
18:54
run this I can see that the next output
18:57
which the next tokens which are
18:59
generated are every effort moves you
19:01
stand to work on surprise one of us had
19:03
gone with
19:04
random it is uh very different from the
Role of decoding strategies in reducing overfitting
19:07
one we had previously generated so
19:09
although here also we can see that uh
19:13
the next token does not make too much
19:15
sense but at least the overfitting
19:16
problem is solved what was happening
19:18
earlier is that we were making a
19:20
deterministic prediction right so when
19:22
the next tokens were selected it was
19:24
overfitting which means that the next
19:26
tokens were just being uh uh just being
19:29
taken from the text which we had so here
19:31
if you see the next tokens were directly
19:34
some of the next tokens were directly
19:36
taken from the text which is a classic
19:37
sign of overfitting the decoding
19:39
strategies which we have implemented are
19:41
probabilistic in nature which really
19:43
makes sense to avoid overfitting since
19:47
the sampling is probabilistic we it the
19:50
GPT model will not memorize the passage
19:52
when predicting the next words and
19:53
that's very important for us you will
19:55
have noticed that when you interact with
19:57
chat GPT it always gives you new output
19:59
right doesn't really memorize what you
20:01
have written and that is because of this
20:03
decoding strategies such as topk
20:05
sampling and temperature scaling they
20:07
really help to avoid overfitting and
20:09
that's one of the main purposes why we
20:11
also learn about decoding
20:13
strategies so you'll see that these new
20:15
words uh moves us stand to work on
20:18
surprise were not present in the
20:19
original text and we can check that
20:25
actually so this is the original text
20:28
and let's contrl F to see whether this
20:30
these words are actually present in the
20:32
original text moves you stand to work so
20:36
let me control F moves you it's not
20:39
there right let me search stand to work
20:42
it's not there which means that the next
20:45
tokens which have been generated are
20:46
completely new now it is a true



***


20:48
generative AI model because it's not
20:50
just memorizing what was there in the
20:52
original text it is actually generating
20:54
new tokens it is generating some new
20:56
sentences
20:59
awesome so that's the reason why we
21:01
learned about decoding strategies in
21:03
these modules you have now learned about
21:05
two decoding strategies first you
21:07
learned about temperature scaling and
21:09
that was very important to help us
21:12
intuitively get either sharper
21:14
distributions or uniform distributions
21:16
and it also helps to prevent overfitting
21:19
since we use multinomial probability
21:21
distribution to sample then we use topk
21:24
sampling because it addresses the issue
21:26
in temperature scaling where random also
21:29
or random tokens have the opportunity of
21:31
being the next token in top Cas sampling
21:34
we restrict the sample tokens to the top
21:37
K most likely tokens and we prevent
21:39
other tokens from even getting an
21:40
opportunity to become the next token and
21:43
that improves the meaning in the
21:45
generated next tokens it prevents us
21:47
from getting nonsensical or random
21:50
output uh as we conclude this lecture
21:52
please keep this workflow in mind where
21:54
you first uh you first obtain the logic
21:57
tensor from the GPT model then you apply
22:01
the topk sampling then what you do is
22:04
after topk sampling those tokens which
22:07
are not in the top K will be replaced
22:08
with negative Infinity then you do
22:10
temperature scaling so you divide the
22:12
logits with the temperature value then
22:14
you apply soft Max and then finally you
22:16
sample from the multinomial to predict
22:18
the next token this will make sure that
22:22
uh
22:25
overfitting overfitting is reduced or
22:27
overfitting is is
22:30
minimized uh thanks everyone this brings
22:33
us to the end of today's lecture where
22:34
we learned about the second decoding
22:36
strategy in the next lecture we'll look
22:39
at some interesting examples where we
22:41
will start to load pre-trained weights
22:42
from open AI itself and we'll learn at
22:45
we'll look at some other pre-training
22:47
strategies and after these set of
22:50
lectures are over we'll go to fine
22:51
tuning and look at applications thank
22:54
you so much everyone I hope you are
22:55
liking these series of lectures I have a
22:58
combination of whiteboard approach as
23:00
well as taking you through code I'm sure
23:04
that there are no other YouTube videos
23:05
right now which explain about large
23:07
language models in such a detailed
23:09
fashion so if you have reached this
23:11
stage I congratulate you for making this
23:13
way this much uh Headway into the course
23:17
if you are watching this for the first
23:18
time I encourage you to look at all the
23:20
previous videos which have been uploaded
23:22
in the course to develop your
23:24
understanding thanks everyone and I look
23:25
forward to seeing you in the next
23:27
lecture




