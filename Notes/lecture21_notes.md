#### GELU Activation Function & Feed Forward Neural Network
* We'll implement a small NN sub-module that is a part of LLM transformer block.

### Two activation functions commonly implemented in LLMs:
1. GELU -> $$x^{*}\phi(x)$$ --> CDF of standard Gussian distribution
2. Swi GLU??


#### Dead Neuron Problem
* __Dead neuron problem__, which means that if the output from one layer is negative and if RuLU activation function is applied to it then the output becomes zero and stays at zero because because we cannot do any learning of after that. So the neurons which are associated with that particular output they don't contribute anything to the learning process once the output of the neuron becomes negative and that's called as the dead neuron problem. So learning essentially stagnates of course.
* RuLU has a huge number of other advantages this non-linearity, which is introduced over here makes NN expressive.
* It gives the power to NN but the reason we we are looking at the disadvantages of ReLU is that understanding the disadvantages of ReLU will open an opportunity for us to learn about the GELU activation function and why it is used in LLMs?

***

* 5:00 

#### Approximation for the GELU activation 
* The following approximation is used for training GPT-2:

* $$\text{GELU(x)}\approx 0.5.x.(1+tanh[\sqrt{\frac{2}{\pi}}.(x+.055715.x^3)])$$

***

* 10:00

* __First Advantage__: ReLU discontinuity at $$x=0$$, which makes it not differentiable. JELU activation on the other hand is smooth throughout so it's differentiable across all X.
* __Second Advantage__ :is that it's not zero for Negative X so that SOLVES the dead neuron problem. Even if the output of a neuron is negative once it goes through JELU it will not become zero. So the neuron won't become dead, it will still keep on contributing to the learning process that's the second reason.


1. First reason is differentiability.
2. Second reason is it prevents the dead neuron problem
3. Third reason is that it just seems to work better than ReLU


* When we do experiments with LLMs so as always __activation functions are hyperparameters__ right. So we need to test out multiple activation functions to see which one performs better. We have generally seen that JELU performs much better in the context of LLM compared to ReLU.


```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

```python
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
```

***

* 15:00 

* __Expansion and contraction allows__ for a rich exploration space.

***

* 20:00

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

```python
print(GPT_CONFIG_124M["emb_dim"])
```

```python
ffn = FeedForward(GPT_CONFIG_124M)

# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) 
out = ffn(x)
print(out.shape)
```

***
