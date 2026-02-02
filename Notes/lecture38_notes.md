## Organizing data into training batches 

* Apply Alpaca prompt style template
* [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

***

* 45:00


45:45
implementing the batching in the training data set there are very few videos which explain this batching
45:50
process in detail but as I say so many times the devil lies in the details so if you directly go to the model training
45:57
right now I could have shown you that directly without explaining this lecture but there is so much information to
46:03
learn here this padding tokens this minus 100 then creating Target token IDs
46:08
which are just shifted uh to the right hand side by one all of this information would have been lost if I directly
46:14
jumped to the fine tuning process so in the instruction fine tuning creating
46:19
these batches is very important to learn because as I explained to you there is a specific fstep process to do this first
46:26
you you have to convert the data to the alpaka prom template then you have to tokenize the formatted data into token
46:32
ID then you have to append padding tokens to all the tokenized input sequences in in each batch remember we
46:39
are doing this separately for each batch so that in each batch the length of all input sequences should be the same then
46:47
we create the target token IDs which is just the input token ID tensor shifted to the right by one and then we replace
46:53
the padding tokens with uh 50256 withus 100 except for the first 50256 which
47:00
indicates the end of text okay one last thing which I want to cover is that sometimes some researchers
Masking target token IDs
47:07
also mask the target token IDs so if this is
47:12
my this is my prompt and that's tokenized into this right as input IDs
47:19
in Target we we shift to the right by one right so as I told you here we shift to the right by one so the target tensor
47:25
is now the in put shifted to the right which means only this part is the target
47:32
text but now as I mentioned to you before why should I learn all these other things in the Target all I should
47:39
learn is the response right so what if I mask the entire other thing in the
47:45
target with the tokens minus 100 so only the response matters to me I
47:50
want my llm to learn the instruction input and then produce this response why should I keep the instruction and the
47:56
input as I mentioned here in the Target text isn't that doing unnecessary
48:02
computations so why don't we replace the token ID is in the Target text with minus 100 so that they are ignored in
48:08
the loss function calculation and in fact this part is actually still not yet
48:14
finalized so let me explain this part a bit better in addition to masking out
48:19
padding tokens it is common to mask out the target token IDs that correspond to the instructions as I mentioned over
48:25
here this is called mask asking the target token IDs um by masking out the target token
48:31
IDs that correspond to the instruction the llm cross entropy loss is only calculated for the generated response
48:37
Target IDs right and by masking out the instruction tokens the model is trained to focus on generating accurate
48:44
responses rather than memorizing instructions and that helps with overfitting or reducing overfitting so
48:51
if this instruction is not even present in the Target the llm won't memorize this as the output so it will reduce
48:58
overfitting now there is still there are some researchers who are trying this but
49:04
it's not yet confirmed which approach works the best so as is mentioned here currently researchers are divided on
49:10
whether masking the instructions is universally beneficial during instruction fine tuning for instance a
49:16
recent paper titled instruction tuning with loss over instructions demonstrated that not masking the instruction
49:22
benefits the llm performance so let me actually display this paper paper so that you can see see this paper um so
49:30
this paper is instruction tuning with loss over instructions this paper demonstrated that uh fine
49:37
tuning with masking is actually not good so this paper demonstrated that not
49:42
masking actually benefits the llm performance so it's not yet finalized
49:47
which is the best method and it's and it's a subject for open research which all of you who are listening to this
49:53
lecture can also actively contribute by testing out master and no masking as I
49:58
mentioned it's very easy to mask you just replace the token IDs of the target text with minus 100 as is seen over here
50:04
on the white board uh okay so in this implementation
50:10
in this series we are not going to apply the masking and we will leave it as an optional exercise for you but the more
50:16
optional exercises like these which you perform the more confident you will be of the subject and the stronger llm
50:22
engineer or machine learning engineer you will become so at many places I'm introducing these open areas of research
50:29
to you so you can even do this research and publish an impactful paper if you thoroughly investigate the effect of
50:35
masking Target token IDs in instruction fine tuning awesome right so this brings
Recap and summary
50:42
us to the end of the lecture where we covered batching the data set and it
50:47
sounds simple batching the data set you might think what's so complicated in this but it took me the full lecture to




***


50:53
explain the second part itself which is batching the data set and the reason is because this batching
51:00
itself involved five detailed steps which I needed to explain to you in a lot of detail along with the code so I
51:07
hope you I hope you liking the style of whiteboard Plus Code and I'm trying my best to explain this to you in as simple
51:13
manner as possible in the next lectures we'll create data loaders then we'll load the pre-trend llm then will
51:20
instruction fine tune the llm inspect the loss accuracy generate the responses
51:26
and and even do an evaluation and score the responses uh we'll ultimately even package this into a chat GPT prompt
51:32
style so you'll get the feeling that you have built your own chat GPT thanks a lot everyone I'm making
51:39
this nuts and bols approach of learning large language models deliberately because I feel that's the strongest way
51:44
to build machine learning Engineers rather than just doing applications without understanding the basics
51:50
foundations are the most important thanks a lot everyone and I look forward to seeing you in the next lecture

***
