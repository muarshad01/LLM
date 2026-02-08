## Token Embeddings

* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)

* Also called Vector Embeddings / Word Embedding
* Preferred token embeddings 

* What are token embeddings?

* small Hands-On demo where we'll play

* Conceptual understanding of why token embeddings are needed

* Semantic meaning between these individual words so cat and kitten are close

* Convolutional neural networks (CNNs) work so well because convolutional neural networks don't just use the pixel values and stretch it out as one input vector. They actually encode the spatial relation between the pixels so the these two eyes are close to each other right.

* One-hot encoding

* 10:00 
* we have to train a neural network to create Vector embedding

* 15:00 
* I want to prove to you that well trained Vector embeddings actually the semantic meaning right so king plus

* Embedding weight Matrix

#### GPT-2

| Tokens| Embedding Dimension|
|---|---|
|50,257  | 768 |

* matrix size = 50,257 (rows - vocabulary size) x 768 (cols -  vector dimension)

* 30:00
* Embedding layer weight Matrix

* initially when this weight Matrix is initialized

* initialize the embedding weights with random values
* these weights are then optimized as part of
the llm training process
* embedding layer weight Matrix

* Embedding layer consists of small random values initially and these are the values which are optimized during llm training as part of the llm optimization itself

***

* 45:00

* (e) - lookup operation that retrieves rows from the embedding layer weight Matrix using a token ID 

* (f) what an embedding does is actually?

* Why is NN linear not used used to define the embedding Matrix?
*  the reason is it's not used is because so both embedding layer and NN layer lead to the same output. Both embedding layer and the NN linear layer lead to the same output but
*  embedding layer is much more computationally efficient because in the NN layer we have many unnecessarily multiplications with zero.
*   so you could just use NN do linear operation over here and do the X into W transpose but
* output = X.W^T
* exploited spatial similarities between features in vctor embeddings we have already exploited the semantic relationship and the semantic meaning between words
***

