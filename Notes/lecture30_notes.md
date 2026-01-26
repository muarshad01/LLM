* 10:00

* Logits --> Top-K --> Logits/Temp --> Softmax --> Sample from Multinomial

***

* 20:00
  
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
