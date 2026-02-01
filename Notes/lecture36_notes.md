## Calculating the classification loss and the accuracy 

* [Welcome to the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
* Training batch = 8 samples
* 8 x 120
* (batch_size, num_tokens, num_classes)

***

* 5:00

```python
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())
```

***

* 15:00

* $$\text{Loss} = -\sum_{i}y_i\log(p_i)$$

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

***
