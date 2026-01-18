## Shortcut Connections
* Also know as skip connections or residual connections
* They were first proposed in the field of computer vision to solve the __problem of vanishing gradients.__
* gradients become progressively smaller as we propagate backwards through a neural network and when gradients become very small -> making it difficutl to train earlier laryers.
* The weight updates are not made learning becomes stagnant and so convergence is delayed.

***

* 5:00

* becomes extremely small because a product of small quantities becomes even more smaller so by the time the training is finished and by the time we get the gradients of the loss with respect to all the weights of this layer we find that this gradient becomes very small and then it even starts approaching zero do you see the problem which happens when the gradient start approaching zero so let's say if you have a particular weight uh the gradient update or the weight update rule looks something like the the weight in the new iteration is the weight in the old iteration minus Alpha which is the step size multiplied by partial derivative of loss with respect to the weights right um so partial derivative of L with respect to

* Shortcut connections create alternative an path for gradient to flow, by skipping one or more layers.
* This is achieved by adding output of one layer to output of a later layer.
* They are also called skip connections.
* Play a crucial role in preserving flow of gradients during training backwards pass
* [Visualizing the Loss Landscape of Neural Nets - 2028](https://arxiv.org/abs/1712.09913)

***

* 10:00

***

* 15:00

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

* 25:00


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

30:00

***
