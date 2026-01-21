#### Generating Text from output Tensors

***

* 5:00


#### Output logits 
* Number of Tokens (4) X Vocubalry Size (50,257)
* 1-batch = 1 X 4 x 50,257
* 2-batch = 2 X 4 x 50,257

* __Step-1__:
  * Number of Tokens (4) X Vocubalry Size (50,257)
  * 1-batch = 1 X 4 x 50,257
    * Every [...|...|...]
    * Effort [...|...|...]
    * Moves [...|...|...]
    * You [...|...|...]
* __Step-2__:
  * Extract the last Tensor
    * You [...|...|...] -- Logits Vector
* __Step-3__:
  * Convert logits into probabilities by applying softmax
* __Step-4__:
  * Identity the index position (token ID of the largest value)
* __Step-5__:
  * Apply token ID to previous inputs for the next round

***

* 20:00

#### 4.7 Generating text

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
```

***

* 25:00

***

* 30:00

***

* 36:00

```python
start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
```

```python
model.eval() # disable dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
```

```python
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
```

***
