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
