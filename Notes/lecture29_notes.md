next token generation
0:00
[Music]
0:05
hello everyone and welcome to this
0:08
lecture in the build large language
0:10
models from scratch Series today we are
0:13
going to learn about a very important
0:15
concept and that concept is called
0:18
temperature
0:20
scaling we'll understand what
0:22
temperature scaling is and why it is
0:24
used in large language
0:26
models first let me recap what all we
0:29
have done in the previous lecture in the
0:31
previous lecture we actually trained a
0:33
large language model completely from
0:35
scratch so here's the training process
0:38
which we had defined and we ran the
0:40
large language model for 10 EPO and we
0:43
saw the next Words which were predicted
0:45
so every effort moves was the input
0:48
token which we had or the input sentence
0:50
which we had given to the large language
0:52
model and then we recorded the output
0:55
tokens and then here you can see I
0:57
printed out the output tokens you know
0:59
was one of the XM he laid down across
1:02
dash dash dash so there are 50 output
1:04
tokens and as you can see the output
1:07
tokens don't really make too much of
1:09
sense right now and that's what we are
1:11
aiming to do in today's lecture our
1:14
whole goal in today's lecture is how to
1:16
make sure that the randomness in the
1:19
output tokens is
1:22
reduced how to make sure that the output
1:25
tokens eventually start making sense we
1:28
will learn techniques to do this in
1:31
today's lecture and one such technique
1:33
is called as the technique of
1:35
temperature
1:37
scaling this is the technique which we
1:39
are going to learn about today so let's
1:41
get started with today's
1:43
lecture Okay so until now what we have
1:47
seen is that the generated token is
1:49
selected corresponding to the largest
1:52
probability score among all the tokens
1:54
in the vocabulary what do I mean with
1:57
mean by this so this is the process
1:58
which we are following until now every
2:01
effort moves that's my input which is
2:04
fed to the GPT architecture and then I
2:07
get a logic tensor which is ultimately
2:10
passed through the soft Max and then I
2:12
get a tensor of probabilities like this
2:14
what we are doing until now to predict
2:16
the next token is we are looking at that
2:18
index or that token ID which has the
2:21
maximum probability and then we decode
2:24
that token ID which gives us the next
2:26
token so for example when every is the
2:30
input the token ID corresponding to this
2:34
second index where probability is 6
2:36
which is the highest that's the output
2:38
and that token ID is one and so the next
2:41
token is effort similarly when every
2:44
effort moves is the input we look at
2:46
this third row we trying we try to find
2:48
that entry which has the highest value
2:51
that entry is 34 the index number is
2:54
five or the token ID is five and then we
2:56
see that for token ID five the token
2:58
which corresponds to that is is you so
3:00
when every effort moves is the input U
3:02
should be the output right so until now
3:06
what we have seen is that the generated
3:08
token is selected corresponding to the
3:10
largest probability score among all the
3:12
tokens in the vocabulary what this leads
3:15
to is that this leads to a lot of
3:18
Randomness and diversity in the
3:20
generated text so since all of these are
3:23
probability scores why are we choosing
3:26
the next token in a deterministic manner
3:28
like this what if we
Probabilistic next token generation
3:31
use what if we choose the next token in
3:34
a probabilistic manner what if we sample
3:36
the next token from a probability
3:39
distribution this is exactly what is
3:41
explored in the concept of temperature
3:44
scaling there are actually two
3:46
techniques which really help to control
3:48
the randomness and the two techniques
3:50
are used together with each other first
3:53
is temperature scaling and the second is
3:55
top case sampling this top case sampling
3:58
Technique we are going to look at in the
4:00
next lecture today we are going to focus
4:02
on temperature scaling so the main uh
Multinomial probability distribution
4:06
main idea behind temperature scaling is
4:09
that first what we do is that instead of
4:11
taking the maximum probability and just
4:14
looking at the index which corresponds
4:16
to the ma maximum probability we replace
4:19
this ARG Max with a probability
4:21
distribution so for example here instead
4:24
of just choosing this token with a
4:26
maximum probability we will look at this
4:29
probability scores and then based on
4:31
this we'll sample from a probability
4:34
distribution and that probability
4:36
distribution turns out to be the
4:38
multinomial probability distribution so
4:40
the next token is then sampled according
4:44
to the probability score we don't just
4:47
blindly or we don't just choose the
4:49
token ID with the highest probability
4:52
value we sample the next token according
4:55
to the probability score this
4:57
distinction in terminology is very
4:59
important are sampling the next token so
5:01
it's not clear to us what the next token
5:04
is going to be because we are sampling
5:06
it so what the sampling mean to give you
5:09
an example let's look at the goian
5:11
distribution right and if I ask you to
5:14
sample values from a goian distribution
5:17
you I cannot tell you right now what the



***



5:19
value will be because I'm sampling it
5:20
from this distribution what I can tell
5:23
you is that most of my samples will
5:24
probably be close to the mean Point like
5:27
here very few will probably be at the
5:29
tail end of the goian distribution this
5:32
is what sampling from a distribution
5:34
means in this case we are sampling from
5:36
this distribution which is multinomial
5:39
distribution and if you go to
5:41
Wikipedia you will see that the
5:43
multinomial distribution essentially uh
5:47
is for a number of different mutually
5:50
exclusive outcomes so when we have K
5:54
possible mutually exclusive outcomes
5:56
with corresponding probabilities think
5:58
of these as the next token here also we
6:00
have the outcomes equal to the
6:03
vocabulary size right K possible
6:05
outcomes because anything can be the
6:06
next token and all of them have a
6:08
probability associated with them right
6:11
uh and let's say we conduct thousand or
6:14
n independent trials to predict the next
6:16
token then we will have a probability
6:19
distribution that is essentially called
6:21
as the multinomial probability
6:23
distribution and I'm going to explain
6:25
that to you in just a moment so that
6:27
it's going to be clear to you right now
6:29
just know that we are going to sample
6:31
the next token from this distribution
6:33
which is called as multinomial
6:35
distribution before coming to
Predicting next token from multinomial
6:37
temperature let's actually go to code to
6:40
understand until this point what we are
6:42
going to do okay so the whole topic of
6:45
today's lecture is decoding strategies
6:47
to control
6:49
Randomness let us first briefly revisit
6:51
the generate text simple function which
6:53
we have been using up till now and then
6:55
we will cover techniques such as
6:57
temperature scaling and top Cas sampling
7:00
and here first what I'm doing is I'm
7:02
switching the model to the CPU and I'm
7:04
running it in the evaluation mode so
7:06
here you can see that uh this is the GPT
7:08
model which we are going to use we have
7:10
the embedding layers then the
7:12
Transformer block then the after coming
7:15
out of the Transformer block we have
7:18
uh let's see when I come out of the
7:20
Transformer block
7:22
here I have another normalization layer
7:25
and then I have output head this is my
7:27
architecture awesome now I just just
7:29
want to show you currently we have this
7:31
generate text simple it takes in the
7:33
input token or the input sentence which
7:36
we have it passes it through the
7:38
architecture which I just showed you and
7:40
then it predicts the next 25 tokens
7:43
using the maximum probability so we are
7:45
not currently doing the sampling as I
7:47
mentioned and so here you can see when
7:49
every effort moves you is the input this
7:52
this is the output these are the 25
7:54
tokens and the randomness here is pretty
7:57
high right so previous ly inside the
8:00
generate text simple function we always
8:02
sample the token with the highest
8:04
probability as the next token using tor.
8:06
argmax right this is called as greedy
8:10
decoding now to generate text with more
8:13
variety what we are going to do is that
8:15
we are going to replace the argmax with
8:17
a function that samples from a
8:19
probability distribution that's the main
8:21
thing which we are going to do and to
8:23
illustrate this probabilistic sampling
8:25
with a concrete example let us actually
8:28
first take a very small vocabulary size
8:30
and let us demonstrate here what's going
8:32
on so let's say my vocabulary size only
8:35
consists of the tokens which are closer
8:37
every effort forward inches moves Piza
8:41
towards and you so these are the eight
8:45
uh these are the nine tokens in my
8:46
vocabulary and each has a token ID from
8:49
0 to 8 I also maintain an inverse
8:51
vocabulary dictionary which gives me the
8:54
token ID and then it decodes it back to
8:56
the Token awesome now uh let's say the
9:00
input to this input to my large language
9:03
model and remember the model is this
9:05
let's say the input to this model is uh
9:09
every effort moves you what this input
9:12
will do when pass through the model is
9:14
that we'll get the output logit sensor
9:17
and uh the output logit sensor will have
9:19
uh the number of columns which is equal
9:22
to the vocabulary size so the number of
9:24
columns here in the output logic sensor
9:28
will be nine right
9:30
now what we do here is that we then pass
9:32
this through soft Max so that this is
9:35
converted into probabilities right and
9:37
so when you pass this into a soft Max
9:39
you can print out the probabilities and
9:41
you'll see that the probabilities look
9:43
like this now all of these add up to one
9:46
so earlier what we used to do is that we
9:48
just looked at these probabilities and
9:51
we said that the next token ID is going
9:53
to be that token ID with the maximum
9:55
probability so let's see and that
9:57
maximum probability is 5721 and this is
10:01
column number 0 1 2 and 3 this is column
10:04
number three so the next token ID is
10:06
equal to three which we have printed and
10:08
then using the inverse vocabulary we'll
10:10
print the next token and that is forward
10:13
so then the next token will be every
10:15
effort moves you and then forward will
10:17
be my predicted
10:18
token here is where we are going to make
10:21
the change to implement a probabilistic
10:22
sampling process we can now replace the
10:25
ARG Max with the multinomial function in
10:28
P torch so here here you can see we are
10:30
replacing the AR Max with a multinomial
10:32
function here and this multinomial
10:35
function is uh applied to this probas
10:38
which is the probabilities uh uh which
10:42
is the tensor of probabilities this is
10:45
where the multinomial function is
10:47
applied and then what we can do is that
10:49
we can get the next token ID based on
10:52
what we sample from this probability




***


10:54
distribution and then we predict the
10:56
token corresponding to the next token ID
10:59
here you can see that this is just equal
11:00
to forward right so what really happened
11:02
it's the same as the previous one there
11:05
is no change so you might be thinking
11:07
why did we do this
11:09
multinomial so what happens is that the
11:11
multinomial function samples the next
11:13
token proportional to its probability
11:15
score this is the important thing here
11:18
so let's say currently we did only one
11:21
trial here right in that one trial what
11:23
the multinomial function did is that it
11:25
looked at all these probabilities and
11:27
then it will sample from this so there
11:29
is a high chance that it will choose
11:31
forward and that's why it chose forward
11:33
we'll really see the difference when we
11:35
do more number of Trials so see in this
11:38
definition there is n independent trials
11:41
which need to be performed right that is
11:43
what we are going to do right now so the
11:46
multinomial function samples the next
11:48
token proportional to its probability
11:50
score in other words forward is still
11:52
like still the most likely token so we
11:55
are not changing the most likely token
11:57
it Still Remains the most likely token
11:59
token but we will also select other
12:01
tokens
12:02
sometimes so forward is still the most
12:04
likely token and will be selected by the
12:06
multinomial most of the time but not all
12:08
the time so to illustrate this we will
12:11
repeat this process this sampling
12:13
thousand times so to simp the
12:16
multinomial function is very intuitive
12:19
so when you do thousand trials in each
12:21
trial it will try to choose
12:24
that token which has the highest
12:26
probability so now it's a bit of a
12:28
random proc right it's not deterministic
12:30
sometimes the multinomial function might
12:32
even choose Piza although it will be
12:35
very very rare uh sometimes it might
12:38
choose but most of the times it will
12:40
choose uh uh forward only so let's try
12:44
to do this now so what we are doing is
12:46
that we are running the same procedure
12:48
now uh we are applying the multinomial
12:50
function to this probas tensor and then
12:53
uh we are going
12:55
to uh take the
12:59
sample which is we are going to take the
13:01
token ID which is sampled in that
13:03
particular iteration and then we are
13:05
going to print out that what we are also
13:07
doing is that in these thousand
13:08
iterations I'm going to print out how
13:11
many times each token is chosen so
13:13
that's this frequency we print print out
13:15
how many times every single token is
13:17
chosen when you are doing this thousand
13:19
iterations so here are the results so
13:22
when you do this number of iterations
13:24
these many times you will see that uh
13:27
you'll see that the word for forward is
13:29
sampled most number of times it's
13:31
sampled 582 times out of the Thousand
13:34
but other tokens such as closer inches
13:37
and toward so closer inches and toward
13:40
they are also sampled some of the time
13:42
so in fact closer is sampled 73 times
13:45
inches is sampled two times and towards
13:47
is sampled 343 times this means that
13:51
since we replace the AR Max function
13:53
with the multinomial function the llm
13:56
would sometime generate text such as
13:58
every effort moves you toward every
13:59
effort moves you inches and every effort
14:02
moves you closer instead of every time
14:04
generating every effort moves you
14:06
forward so now integration of this
14:10
multinomial function has made sure that
14:12
we are not sampling the same token each
14:14
time sometime we are also giving
14:18
more uh we are giving more chance for
14:20
other tokens to be the next token and
14:23
that's what improves the creativity of
14:25
the large language model it leads to
14:27
more uh exploration it leads to more
14:30
creativity and sometimes it can also
14:33
lead to better outputs instead of just
14:35
sampling a deterministic prediction
14:37
every single
14:39
time so you might be thinking that okay
What is temperature scaling?
14:41
this looks fine but why is this method
14:43
called as temperature scaling where does
14:46
temperature come into the picture and
14:47
why is it called temperature so
14:49
basically you see this logic tensor over
14:52
here right before we apply the soft Max
14:55
there is a Logics tensor what what what
14:59
is meant by temperature is that
15:00
temperature is basically just a fancy
15:02
term for dividing the Logics tensor by a
15:06
number which is greater than zero so the
15:08
only thing which is done when we
15:10
introduce temperature is that we have
15:11
this thing called scale Logics and all
15:14
the logic values which we have are
15:15
divided by another number which is
15:17
called as the temperature value so what
15:20
this does is that so see when you divide
15:22
it by the temperature value you get the
15:24
scale logic then you apply soft Max and
15:26
then you get the tensor of probabilities
15:28
every
15:29
the rest of the process stays the same
15:31
but what this introduction of this
15:33
temperature does is that it changes the
15:35
distribution a bit it changes the
15:37
distribution of
15:38
probabilities so for example let's see
Coding the temperature scaling
15:41
what this dividing by this temperature




***



15:43
does through
15:44
code um so we can further control the
15:47
distribution and selection process via a
15:49
concept called temperature scaling where
15:52
temperature scaling is just a fancy
15:54
description for dividing the logits by a
15:56
number greater than zero now we'll see
15:58
two things we'll see temperature is
16:00
greater than one what happens when
16:01
temperature is greater than one and
16:03
we'll see what happens when temperature
16:04
is smaller than one okay so here what
16:07
I'm doing is that I'm scaling the Logics
16:10
with dividing by temperature and then
16:11
I'll apply the soft Max exactly like
16:14
what we had done over here and then uh
16:18
even before applying the multinomial
16:20
distribution uh then we'll get the
16:23
scaled probab we'll get the scaled
16:25
probabilities as applying soft Max with
16:28
temperature first I just want to show
16:30
you without even going to the
16:31
multinomial function what happens when
16:33
you scale the logits with temperature
16:35
and what happens as the result when you
16:37
apply soft Max so look at this plot
16:40
first I want you to see the plot with
16:42
temperature equal to one which are the
16:43
blue so when you see the blue you will
16:46
see that the probability for forward is
16:48
around 0.5 probability for closer is
16:51
around 0.1 and probability for towards
16:53
is around3 this is exactly what we saw
16:55
in this case right uh so probability for
16:59
forward is 05
17:02
probability uh for towards so
17:05
probability for towards is35 and
17:08
probability um for there is one more
17:12
probability for closer so probability
17:14
for closer is 0.06 right so the blue the
17:18
blue line there is when we have not
17:20
changed anything when temperature is
17:22
equal to 1 now let's see what happens
17:24
when temperature is small so when you
17:27
actually when you take these value when
17:29
you take these logits and you divide
17:32
each of them by 0.1 and then you take
17:34
the soft Max then you will get
17:36
probabilities which look something like
17:38
this in the orange you'll see there is a
17:40
sharper probability for forward now and
17:43
almost all the other probabilities have
17:45
been shifted to zero and we can test
17:47
this out a bit in the code right now so
17:49
what I'm going to do is that I'm going
17:51
to say that
17:53
uh uh next
17:56
logits next token to logits is equal to
18:01
next token logits or let's say next
18:04
token logits 2 is equal to next token
18:06
logits divided by
18:07
0.1
18:09
right okay so now I have a next to on
18:12
logits 2 where every logit is divided by
18:15
0.1 and here what I will do is that I
18:17
will print out I'll print out the
18:20
probabilities which correspond to
18:22
the uh when you apply the soft Max so
18:25
now I'll just do this and print out the
18:27
probabilities
18:30
so here you see what happened when we
18:32
divided by 0.1 and then you applied the
18:34
soft Max almost all the probabilities
18:36
all other probabilities are brought down
18:38
to zero but there is a sharper
18:40
probability very high probability now
18:42
for forward this is in contrast to these
18:44
probabilities right where even here this
18:46
value had a significant amount which was
18:49
toward uh I think that was toward yeah
18:52
this value also had a significant amount
18:53
that was for closer but now all these
18:56
values are turned to zero the only value
18:57
which matters the most most is uh
19:00
forward so that's the first conclusion
19:02
when temperature value is very low then
19:05
there is a peak in the probability
19:07
distribution which means that it becom
19:09
sharper for specific values now let me
19:12
try this that let me divide by
19:15
five or as shown in the graph uh yeah
19:18
temperature equal to five and here
19:19
you'll see the probability becomes a bit
19:22
flatter which means that there is kind
19:25
of high values for all the tokens so
19:27
let's see what is happening here when
19:29
the when I divide by five so next token
19:33
logits three I'm dividing the next token
19:35
logits by five and then what I'll do is
19:38
that I'll print out
19:40
the probabilities for next token logits
19:43
3 and let me print it out in a separate
19:46
cell actually so that it might be
19:48
cleaner so I have a separate cell where
19:50
I will print
19:52
out where I'll print out the
19:56
probabilities for
19:58
this tensor where I've divided the
20:00
logits by five and then I run it so now
20:03
you see this is my probability tensor
20:05
and you can see that it has just become
20:07
a bit flattened and a bit more uniform
20:09
now for every token we have certain
20:11
values of course the value for forward
20:13
is still the highest but there are other
20:15
values such as toward which has now
20:17
become very close to forward even the
20:20
probability value for closer has become
20:22
very similar to forward now so if the
20:25
temperature value is very high it means
20:28
means that the probability distribution
20:30
kind of flattens out a bit and then




***



20:33
every token kind of has uniform
20:36
probability as being the next token can
20:38
you think about what this would mean for
20:40
generating the next word or generating
20:42
the next
20:44
token this means that there is a lot of
20:47
variability now in our output token
20:48
which is also sometimes called
20:50
creativity but in some cases the
20:52
creativity can be too high because now
20:54
you'll see that there is certain output
20:56
for pizza also which means that every
20:58
every effort moves you Piza our LM is
21:01
also predicting this as an output so
21:03
ideally we don't want the temperature
21:05
value too high and we don't want the
21:07
temperature value too low also if the
21:09
temperature value is too low then there
21:11
is only one token which is predicted we
21:13
need to place emphasis on other tokens
21:15
also which might be making sense so as
Interpreting temperature scaling
21:18
we saw applying very small temperatur
21:20
such as01 will result in sharper
21:22
distributions such that later when we
21:24
apply the multinomial function it will
21:26
select the most likely token which is
21:28
forward now um please keep in mind that
21:32
after this we are going to apply
21:33
multinomial so multinomial sampling is
21:36
going to happen after the soft Max so if
21:39
the probability distribution looks
21:40
something like in the orange the
21:42
multinomial will always sample this
21:44
because the probability remember
21:47
multinomial samples according to the
21:49
probabilities right so if the
21:50
probability is
21:51
991 it will always sample this so that's
21:55
why when the temperature values are
21:58
small such as 0.1 it results in sharper
22:00
distributions and the multinomial
22:03
function then selects the most likely
22:04
token year forward Almost 100% of the
22:07
time and then we approach the behavior
22:09
of the argmax function so when the
22:11
temperature values are low it's almost
22:13
like going back to the earlier approach
22:15
where we selected the token ID with the
22:18
highest probability score now if you
22:21
look at the higher temperature if
22:22
temperature value is equal to 5 this
22:25
actually results in a more uniform
22:26
distribution where other tokens are also
22:28
selected more often such as what is
22:30
shown in the green color here this can
22:32
add more variety to the generated texts
22:35
but also it leads to non sensical text
22:37
so for example uh Pizza is now 4%
22:41
probability so we will get every effort
22:43
moves you Pizza about 4% of the time if
22:45
you use temperature equal to five that's
22:47
not good so ideally there needs to be a
22:50
good balance in the temperature which we
22:52
used there is a very nice graph which
Temperature scaling visualised
22:55
actually shows why this has the name
22:56
temperature so here you see if the
22:59
temperature becomes high as is shown on
23:01
the right side we saw that the
23:02
distribution becomes more uniform right
23:04
every token has some probability whereas
23:06
for low temperature we have something on
23:08
the right something on the left here
23:09
where we have sharper distributions only
23:11
one token makes sense so low temperature
23:14
can be corresponding to low entropy
23:16
whereas when you do increase the
23:19
temperature things generally be become
23:21
more unstable Things become more chaotic
23:23
become more creative that's why every
23:25
token has certain probability here of
23:27
being the next token
23:29
so that's why it's called as temperature
23:30
because increasing the temperature makes
23:33
makes uh makes sure that all tokens have
23:36
certain value which makes it a
23:40
bit uh unstable I would say or rather I
23:44
would call it a bit more creative a bit
23:45
more randomized it's almost like the
23:47
temp entropy has increased so that's why
23:51
conceptually it's called as temperature
23:53
because increase in temperature leads to
23:55
a more flattened diffused distribution
23:58
whereas lowering the temperature means
24:00
lower entropy so only one token value
24:03
would make more sense now here is a
24:06
graphic which actually shows what
24:07
happens as you increase the temperature
24:08
to very very high values if temperature
24:10
becomes very high almost every next
24:13
token will have equal probability if the
24:15
temperature becomes very low we have
24:17
sharper distributions as we have seen
24:19
before awesome so this brings us to the
Recap and next steps
24:22
end of the lecture where we covered
24:24
about temperature scaling and how
24:26
temperature scaling can be actually used
24:28
to predict the next token in a
24:30
probabilistic sense so instead of just
24:32
choosing the next token according to the
24:34
maximum value we will use a multi
24:36
multinomial probability distribution to
24:39
sample the next token that's the keyword
24:42
so first you take the logit so the
24:44
procedure is like this first you take
24:46
the logits U then you scale it you scale
24:50
the logits with the
24:52
temperature uh and then what you do is
24:55
then you apply the soft
24:57
Max and then you get a tensor of
24:59
probabilities and then you sample using
25:01
the multinomial
25:04
distribution this is the sequential
25:06
workflow for application of temperature
25:09
and we later saw that the reason it's
25:10
called temperature is because as the
25:12
temperature becomes high it's like the
25:14
entropy increases every next token has
25:16
some probability of being the next token
25:19
uh whereas if the temperature is very
25:21
low the entropy is very low it's like
25:23
everything is concentrated to one single
25:25
token and we have a sharper probability
25:27
distribution
25:29
so small token means that the llm output
25:31
does not have any creativity it will




***


25:33
always give the same thing as an output
25:35
but large token means that it's a
25:36
flatten distribution more variety but
25:39
also it has scope for more nonsense so
25:41
you need to be careful when choosing the
25:43
temperature
25:44
value uh in the next lecture we'll study
25:47
another technique for
25:50
uh reducing the randomization for while
25:55
decoding the tokens or reducing the
25:59
um reducing the randomization when
26:02
predicting the next
26:03
token and that strategy is called as
26:06
topk sampling so usually topk sampling
26:10
is used along with temperature scaling
26:12
and so we'll also see how top SK
26:14
sampling integrates with what we learned
26:16
today which is temperature scaling
26:18
thanks a lot everyone uh this was a
26:20
short lecture but I hope you understood
26:22
the concept of temperature scaling
26:24
thanks a lot and I'll see you in the
26:25
next lecture

***
