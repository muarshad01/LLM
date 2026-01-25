## Temperature Scaling

* How to reduce the randomness in the output tokens
* Until now, the generated token is selected corresponding to the largest probability score among all the tokens in the vocabulary.
* this leads to a lot of randomness and diversity in the generated text

#### Techniques to controlling randomness
* We'll learn two techniques to control this randomness:
1. Temperature scaling
2. Top-k sampling 

#### Temperature scaling
*  Replace `argmax` with __probability distribution__
* multi-nomial probability distribution samples next token according probability score.

***

* 10:00

#### What is temperature
* Fancy term for dividing the logits by a number greater than zero.
* $$\text{Scaled ~logits} = \frac{Logits}{Temperature}$$
* Small Temperature value: Sharper distribution
* Large Temperature value: Flatter distribution (more variety, but also more non-sense)

***

* 15:00

***

* 20:00

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





