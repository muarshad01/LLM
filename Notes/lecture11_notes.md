#### Positional Encoding
* Positional Encoding Layer
* Token Embedding Layer

* (4) It is helpful to inject additional position-information to LLM.
* (5) There are two types of positional embeddings:
* (5.1) Absolute positional embeddings
* For each position in the input sequence, a unique embedding is added to the token's embedding to convey its exact location.
* (5.2) Relative positional embeddings
* In this type of embedding, the emphasis is on the relative position or the distance between the tokens. Th model essentially learns the relationships in terms of "how-far-apart" rather than at which "exact-position".
* Advantage: model can generalize better to sequence of varing lengths, even if it has not seen such lengths during training.

* They enable the LLMss to understand the order-and-relationship between the tokens and this actually ensures more accurate and context-aware predictions.
* The choice between the two types of positional embeddings really depends on the specific application and the nature of the data being processed.

* __Absolute__: Suitable when fixed order of tokens is crucial, such as for sequence generation.
* __Relative__: Suitable for tasks like language modeling over long sequences, where the same phrase can appear in different parts of the sequence.

* (7) OpenAI's a GPT models use absolute positional embeddings. This optimization is part of the training model itself.

* sinusoidal and cosine formula over here uh so you can read a bit about what they have written
here since our model contains no recurrence in order for the model to make use of the order of the sequence we must inject some information about the relative or absolute position of the to tokens so they have added absolute

***

30:00

* 8 X 4 X 256 dimensional tensor
* Generate positional embeddings. Similar to the  token embedding, we also have to create another embedding layer for the positional encoding.
* 4 X 256 dimensional tensor

***

35:00

* we only need four positional vectors four positional embedding vectors and then each Vector will of course have size 256 so then the
* positional embedding Vector Matrix which we have will be a 4 X 256 Matrix

***
