#### 
1. Text Generation
2. Text Evaluation
3. Training & Validation Losses
4. LLM Training Function

***

* 15:00

* __Main step__: Find loss gradients using `loss.backward()`

* Input --> GPT model --> Logits --> Softmax --> Cross Entropy Loss (Output, Target)

#### Number of Parameters
* 161 Million

```
* Embedding Parameters = Token Embedding  + Positional EMbedding 
                       = (vocab_size x embedding_dim ) + (context_size x embedding_dim)
                       = (50,257 x 768) + (1,024 x 768)
                       = 38.4 Million
```

```
* Transformer Block Parameters = *Multi-head attention
                               = (Q, K, V) weights
                               = 3 x 768 x 768 = 1.77 Million
                               = Output head = 768 x 768
                               = 0.59 Million
                               = Total = 2.36 Million
```

```
* Feed-forward NN = 768 X (4 x 768) + 768 x (4 x 768)
                  = 4.72 Million
```

```
* Total for 12 Transfomer blocks = 12 x (2.36 + 4.72)
                                 = 85.2 Million
``` 

***

* 25:00

* ADAM

***

* 30:00
  
* AdamW Optimizer

***





