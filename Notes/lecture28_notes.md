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
                               = Total
                               = 2.36 Million
```

```
* Feed-forward NN = 768 X (4 x 768) + 768 x (4 x 768)
                  = 4.72 Million
```

```
* Total for 12 Transfomer blocks = 12 x (2.36 + 4.72)
                                 = 85.2 Million
``` 

```
* Final layer (softmax) = 50,257 x 768
                        = 38.4 Million
```

```
* Total parameters to be optimized = 38.4 + 85.2 + 38.4
                                   = ~161 Million
```

#### Weight Tying
* GPT-2 has 124 Million parameters

***

* 25:00

* AadmW (Adaptive Moment Estimation) optimizer

***

* 30:00
  
* AdamW Optimizer

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
```

***

```python
# Note:
# Uncomment the following code to calculate the execution time
# import time
# start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

***
