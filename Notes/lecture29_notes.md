## Temperature Scaling

* How to __reduce the randomness__ in the output tokens?
* Until now, the generated token is selected corresponding to the __largest probability score__ among all the tokens in the vocabulary.
* This leads to a lot of randomness and diversity in the generated text

#### Techniques for Controlling Randomness
* There are two main techniques:
1. Temperature Scaling
2. Top-k Sampling 

#### Temperature scaling
* Replace `argmax` with __probability distribution__
* __Multi-nomial probability distribution__ samples next token according probability score.

***

* 10:00

#### What is temperature
* Fancy term for dividing the logits by a number greater than zero.
* $$\text{Scaled ~logits} = \frac{Logits}{Temperature}$$
* Small Temperature value: Sharper distribution
* Large Temperature value: Flatter distribution (more variety, but also more non-sense)

***

