## Shortcut Connections
* Also know as skip connections or residual connections
* They were first proposed in the field of computer vision to solve the __problem of vanishing gradients.__
* gradients become progressively smaller as we propagate backwards through a NN and when gradients become very small -> making it difficutl to train earlier laryers.
* The weight updates are not made and the learning becomes stagnant. So convergence is delayed.

***

* 5:00

* $$W^{*}=W^{old} - \alpha\frac{\partial L}{\partial W}$$

* Shortcut connections create alternative an path for gradient to flow, by skipping one or more layers.
* This is achieved by adding output of one layer to the output of a later layer.
* They are also called skip connections.
* Play a crucial role in preserving flow of gradients during training backwards pass
* [Visualizing the Loss Landscape of Neural Nets - 2028](https://arxiv.org/abs/1712.09913)

***

* 20:00

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

***

```python
layer_sizes = [3, 3, 3, 3, 3, 1]  

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)
```

***
