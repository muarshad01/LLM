#### Implementing Multi-head Attention With Splits

* $$\text{head-dim} = \frac{d_{out}}{n_{heads}}$$

#### Example
* __Step-1__: b, num_tokens, d_in = (1, 3, 6)
  * context_vector = 3 x d_out

* __Step-2__: Decide d_out, num_heads = (6, 2)
  * Usually d_out = d_in
  * $$\text{head-dim} = \frac{d_{out}}{num_{heads}}$$ = 6/2 = 3

* __Step-3__: Initialize trainable weight matrices for key, query, and value $$(W_k, W_q, W_v)$$ = (d_in, d_out) = 6 x 6

* __Step-4__: Calculate Keys, Queries, Values Matrices $$(W_k, W_q, W_v)$$ = (d_in, d_out) = 6 x 6
  * Input X $$W_k$$, Input X $$W_q$$, Input X $$W_v$$
  * Input = (1, 3, 6) = (b, num_tokens, d_out)
  * $$W_k = W_q = W_v = (6, 6)$$

* __Step-5__: Unroll last dimension of Keys, Queries, and Values to include num_heads and head_dim
  * $$\text{head-dim} = \frac{d_{out}}{num_{heads}} =\frac{6}/{2} = 3$$
  * (b, num_tokens, d_out) = (b, num_tokens, num_head, head_dim)
  * (1, 3, 6) = (1, 3, 2, 3)

* __Step-6__: Group matrices by number of heads
  * (b, num_tokens, num_head, head_dim) -> (b, num_head, num_tokens, head_dim)
  * (1,3,2,3) = (1,2,3,3)

* __Step-7__: Find Attention Scores
  * Queries X Keys.Transpose (2, 3)

* __Step-8__: Find Attention Weights
  * Mask attention scores to implement Causal Attention
  * Divide by $$\sqrt{\text{head-dim}} = \sqrt{\frac{d_{out}}{n_{heads}}} = \sqrt{\frac{6}{2}} = \sqrt{3}$$
  * Attention weights (1, 2, 3, 3) = (b, num_head, num_tokens, num_tokens)

* __Step-9__: Context Vector
  * Attention Weights X Values
  * (b, num_heads, num_tokens, num_tokens) = (b, num_heads, num_tokens, head_dim)

* __Step-10__: Reformat the Context Vector
  * (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)

* __Step-11__: Combine Heads
  * (1, 3, 6) = (b, num_tokens, d_out)


| LLM | Parameters | Attention heads | Context Vector Embedding Size| 
| ---| --- | --  | ---|
| GPT-2 | 117 Million | 12  | 768 |
| GPT-3 | 117 Million | 96  | |

***

