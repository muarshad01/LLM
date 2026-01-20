#### Transformer
* Transformer block is the fundamental building block of GPT and other llm architectures.
* The transformer block is repeated 12 times in GPT-2 small (124M parameters)

#### Trransformer Block
1. Multi-ead attention
2. Layer normalization
3. Dropout
4. Feed forward layer
5. GELU activation

***

* 15:00

***

* 20:00

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

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

***

* 25:00

```python
# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch03 import MultiHeadAttention

from previous_chapters import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
```

***

* 30:00

will take you to this documentation I'll also share the link to this in the YouTube description
30:49
section okay so now when you look at the forward method I have explained to you the norm one uh which is the object Norm
30:58
normalization first normalization layer object then at drop shortcut shortcut
31:03
which is the Dropout layer we do not need to create a separate class for the shortcut connection because what we do
31:09
is that we just add the output of this back to the original input so when you
31:14
look at the shortcut mechanism look at where the arrows are there so if you look at this
31:21
Arrow if you look at this Arrow over here what we are doing is we are adding the this input over here
31:28
uh this input over here to the output from the Dropout right so the output
31:34
from the Dropout is added to this input which is over here let me actually mark
31:39
this with yellow so that you can get a clear understanding so the input is being marked with an yellow star here
31:45
and the output from the Dropout is marked with another yellow star and we are adding these two yellow stars
31:50
together that's what this first shortcut connections ISS so what we are doing is
31:56
that um shortcut is initially in initialized to X which is the input and
32:02
then X gets modified to the output of the Dropout so we are essentially adding the Dropout output to the input which is
32:09
exactly what we saw on the white board in the second shortcut layer let's see what happens in the second shortcut
32:16
Connection in the second shortcut connection what is actually happening is that uh here you see there is an input
32:23
to the second normalization layer and there's an output from the second dropout so we are adding the input to
32:29
the second normalization layer with the output from the second Dropout and you will see in the code what we are doing
32:36
so shortcut equal to X where X right now is the input to the second normalization layer and uh when we reach this step X
32:44
is equal to the output so here we are actually adding the output to the input of the second normalization layer which
32:51
is exactly what we saw on the Whiteboard so after all these operations are performed in the forward method the
32:58
Transformer block Returns the modified input which is the same dimensions as the input Vector remember that if
33:05
you keep in mind or visualize this Blue Block which I'm showing on the screen
33:10
right now you will easily understand what is happening in the code because in the code we have just followed a
33:16
sequential workflow of all of these different modules together okay so the Transformer block
33:23
is as simple as this once we have understood about the previous modules we just stack them together to build the
33:29
Transformer block now I think hopefully you would have understood why I have spent so many lectures on the feed
33:35
forward neural network we had one full lecture on the feed forward neural network and and Jou we had one full
33:41
lecture on layer normalization and we had one full lecture on shortcut connections as well the reason I spent
33:47
so much time separately on those lectures is that all of those different aspects come together beautifully when
33:53
you try to code out the Transformer block from scratch okay so here I have just written
Transformer block code summary
33:59
an explanation of what we have done in the code so the given code defines a Transformer block class in py torch that
34:06
includes a multi-head attention mechanism and a feed forward neural network so this is the multi-head
34:11
attention mechanism and this is the feed forward neural network layer normalization is applied before each of
34:17
these two components so before the multi-ad attention mechanism and once which is before the feed forward neural
34:24
network that's why it's called as pre-layer Norm older architecture such as the original
34:30
Transformer model applied layer normalization after the self attention and feed forward neural network that was
34:36
called as post layer Norm post layer Norm so researchers later discovered
34:43
that post layer layer Norm sometimes leads to worse or many times leads to worse training Dynamics that's why
34:50
nowadays pre-layer Norm is used so we also implement the forward
34:55
path where each component is followed by shortcut connection that adds the input of the block to its output so here you
35:01
see this shortcut connection adds the input of this whole block to the output this shortcut connection over here adds
35:08
the input of this whole block to the output now what we can do is that we can
Testing the transformer class using simple example
35:14
initiate a Transformer block object and let's feed it some data and let's see what's the output so here what I'm
35:21
defining is that I'm defining X which is my input it has two batches each batch has four tokens and the embedding
35:28
dimension of each token is 768 now I'm passing this this through the Transformer model let's first visualize
35:35
what will happen once this x passes through this Transformer model or Transformer class rather when X first
35:41
passes through the Transformer class uh normalization layer is applied so now try to visualize this try to visualize
35:48
every token as a row of 768 columns so every row is normalized which
35:55
means that the mean of every row and the standard deviation of every row will be one that will be done for all the four
36:01
tokens of one batch and then the same thing will be done for all the four tokens of the second batch so then the
36:07
normalization layer is applied to X and then all of the four tokens of one batch will be transformed so that the mean of
36:15
every row and the standard deviation or the variance of every row will be equal to one mean will be zero sorry the mean
36:22
of every row will be zero and the standard deviation will be equal to one after that that we pass every token of
36:29
both the batches to the self attention mechanism or the multi-head attention rather and the output of this is that
36:36
every token embedding Dimension is converted into a context Vector of the same size so if you look at the first
36:43
token of the first batch that has 768 dimensions that's a embedding Vector which does not encode the attention of
36:49
how that should relate to the other input vectors when we implement this the
36:55
resultant is the embedding Vector which essentially has four tokens and each
37:01
token has 768 Dimensions but now the resultant will be context vectors main
37:06
aim after this attention mechanism or after this attention block is to convert the embedding vectors into context
37:12
vectors of the same size then we apply a Dropout layer which randomly drops off
37:17
some U uh some parameter values to zero and then we add a shortcut layer this is
37:24
the first block you can see in the second block the output of the previous block passes through a second
37:30
normalization layer then through a feed forward neural network where the dimensions are preserved so after coming
37:36
out from the feed forward neural network the dimensions would again be uh two batches multiplied by four
37:44
tokens multiplied by 768 which is the dimension of each token and then we again have a Dropout
37:51
layer and then we again add the shortcut mechanism to prevent the vanishing gradient so when we return the X we
37:57
expect the output to be 2x 4X 768 which is the same size as the input now let's
38:04
check whether that's the case so this is my X and now I'm creating an instance of the Transformer block but remember I
38:10
need to pass in this configuration so when I create when you create an instance of the Transformer block you
38:15
have to pass in the configuration and remember again this is the configuration which I'm using over here which defines
38:21
the context length embedding Dimension number of attention heads number of Transformer blocks and the dropout rate
38:29
okay so now we pass in this configuration and then we just print out the output um and the input shape is 2x 4X
38:37
768 and you'll see that the output shape is exactly the same 2x 4X 768 what I
38:43
really encourage all of you to do is uh when you watch this lecture try to understand the dimensions try to write
38:50
down the 2x 4X 768 on the Whiteboard apply the layer normalization try to see
38:55
how the dimensions work out through all of these different building blocks and try to see that when you
39:01
reach the end the dimension is exactly preserved which is 2x 4X
39:06
768 okay so I have just added some notes here so that we can conclude this
39:11
lecture so as we can see from the code output the Transformer block maintains the input
39:17
Dimensions indicating that the Transformer architecture processes sequences of data without altering their
39:23
shape throughout the network this is very important the Transformer block processes the data without altering the
39:30
shape of the data the preservation of shape throughout the Transformer block
39:35
architecture is not incidental but it is a crucial aspect of the design of Transformer block itself this design
39:42
enables its effective application across a wide range of sequence to sequence tasks where each output Vector directly
39:49
corresponds to an input Vector maintaining a on toone relationship however the output is a
39:55
context Vector that encapsulates in information from the entire input sequence remember that the output
40:01
contains so much information it's very rich output Vector because it contains information about how the uh in the
40:08
input sequence how every token relates to the other tokens of the input sequence that's the whole idea of the
40:14
attention mechanism which we looked about before or which you understood before so the GPT architecture is the
40:20
broad level within that there is a Transformer model within that there is a attention mechanism so these three
40:26
things power each other other it starts from the attention mechanism which is a key component of the Transformer block
40:33
the and the Transformer block is the key component of the whole GPT architecture so finally this means that
40:40
while the physical dimensions of the sequence length and feature size remain unchanged as it passes through the Transformer block the content of each
40:47
output Vector is reint is re-encoded to integrate contextual information from
40:52
across the entire input sequence so this is just saying that although the input and the output dimensions are the same
40:59
the output contains a lot more information since it also contains information about how each token relates
41:05
to the other tokens in the input that actually brings us to the end
Lecture summary and next steps
41:10
of this lecture I just want to show you one thing as we are about to conclude
41:16
um I want to show you um how what all we have learned so far relates to each
41:23
other so when we looked at the GPT architecture we saw that there are four
41:29
things which are important the layer normalization the J
41:36
activation uh let me draw that again the layer normalization which is here the J
41:44
activation which is here the feed forward neural network the feed forward neural network
41:50
component and finally the shortcut connections and we also saw today how all of these four come together together
41:57
to build the entire Transformer block and we coded out this Transformer block together now in the next lecture what we
42:04
are going to see is that how the Transformer block uh leads to the entire GPT GPT
42:11
architecture so remember what I said earlier it all starts from this
42:16
attention which is the mass multi-ad attention that forms the core of the Transformer block now that we have coded
42:23
the Transformer block our task is not over because remember that after the Transformer block there are lot of
42:29
pre-processing steps and then finally we have to use the output Vector in order to predict the next next token and we
42:37
still have to see how that is done so the last step is still remaining and the last step is
42:42
the final GPT architecture but try to understand this sequence here it all
42:49
starts from the attention block then so it all starts here the magic starts here
42:55
at the attention block then that forms the core of the Transformer
43:01
block which I'm marking over here and the Transformer block essentially forms
43:06
the core of the entire GPT
43:12
architecture I hope you have got this sequence in mind so that's why for us it was very important to first spend a lot
43:18
of time understanding attention we have covered five to six lectures on that then it was very important to spend a
43:24
lot of time on every single component of this Transformer Block in the next lecture we are finally going to see uh
43:31
how the Transformer block leads or fits in the entire GPT architecture or rather
43:37
put in other words how can we pre-process the or postprocess the output from the Transformer block so
43:43
that we can predict the next word remember the whole goal of GPT style models is that given an input given an
43:50
input let's say the input is every effort moves you how to predict the next word up till now we have just seen that
43:56
the Transformer block retains the input shape right the Transformer block output but how is this converted to next word
44:03
prediction that's what we are going to see in the next lecture okay so now with the Transformer
44:10
block implemented we have all the ammunition or building blocks needed to implement the entire GPT architecture
44:17
and here you can see the next lecture which I've already planned is coding the entire GPT
44:24
model so I hope everyone you are liking this style of lectures where I first cover intuition plus theory on the
44:30
Whiteboard and then I take you to the code please follow with me and then try
44:35
to implement the code on your own try to write things on the Whiteboard Because
44:40
unless you understand the nuts and bolts of how large language model works it will be very difficult to invent
44:46
something new in this field it will be very difficult to truly be a researcher or engineer and such kind of fundamental
44:52
understanding will help you as you transition in your career as well thanks thanks everyone and I look forward to
44:58
seeing you in the next lecture

***






