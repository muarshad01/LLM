## Organizing data into training batches 

* Apply Alpaca prompt style template
* [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

***

* 5:00

***

* 10:00

***

* 15:00

***

* 20:00

***

* 25:00

25:36
alpaka format is created and then what we do is that as I mentioned in the next step over here we are going to tokenize
25:43
the formatted data so we uh we first Define a empty list and then we start
25:50
appending the token IDs to this Mt list so let's say if you have a prompt which looks like this each of the tokens here
25:57
are conver ConEd into token IDs and then appended to a list so for every prompt corresponding to each input output pair
26:04
now we have a list of token IDs which is mentioned in the step one over here see here what we are doing is that every
26:11
prompt we are converting it into token IDs and for that the tokenizer which we are going to use is we also need to pass
26:19
this to the instruction data set class but it's going to be the tick token Library I'll share the link to this uh
26:26
in the chat we have had a separate lecture on bite pair encoder in this lecture Series so if you want to
26:32
understand about this library in detail I highly encourage you to uh watch that
26:37
lecture right so now what does the instruction class ENT instruction data
26:42
set class essentially return return well it returns for every uh for every data
26:48
which is in this format it converts it into the alpaka style prompt and then it returns uh a bunch of token IDs for
26:56
every entry awesome so uh let's go to the next part
27:03
now before coming to the next part I just want to show you the end of text token and its corresponding token ID as
27:11
we had seen on the Whiteboard and it's indeed 5256 so that's why we are using this
27:17
50256 token ID because it conveys the end of text okay now as I mentioned what
27:23
we are going to do here is that we are going to define a custom colate function what this custom colet function does is
27:29
that the name may sound complex but it actually does a very simple thing it takes the inputs in each data set that's
27:36
the first thing it takes the input in each batch it finds that input with the maximum length and then it appends the
27:43
50256 or pads the 50256 token ID to all other inputs that's the only thing which
27:49
it is doing so this custom colate draft one it takes the batch so you can think of the
Coding the custom collate padding function
27:55
batch as coming in this format like this uh and then it has you have to give the
28:01
padding token ID which is 50256 and the device which is CPU so this function
28:06
implements four steps first it finds the longest sequence in the batch and then it pads the other sequences so that the
28:12
length is equal to the longest sequence that's it and then it converts the list of inputs into a tensor and transfers to
28:18
our Target device which is the CPU so this entire thing is converted into a tensor what Ive marked with this orange
28:25
color over here that's the function of of the custom colla draft so let's see how it does it the first thing this
28:32
function does is that it will find the longest sequence in the batch and it will add it by one so let's say if you
28:39
have these three it if you have these three the longest sequence length is five and then it will add it by one so
28:45
then it will be six there is a reason why you add it by one and I'll come to that later but after you add it by one
28:51
what you do is that for every item in the batch you first add a token ID so even for the first one even for for the
28:58
first item you add this 50256 token that's the first thing which you do and then you pad the 50256 again so that the
29:06
length is equal to the maximum length and then what you do is you
29:11
remove uh you remove the extra added token so here essentially what we are
29:16
doing is that let's say if you have uh I'll actually remove this in the code
29:22
what we are trying to do is that we add first a 50256 token ID to all of these
29:28
so even to the first one we add this 50256 token and to the other ones we add the 50256 token then this will be added
29:36
one 2 three three more times so total it will be added four times and here we'll add it a total of three times right but
29:43
then you might think why are we adding an extra 50256 token because here we don't need to add 50256 here also we
29:50
need to add three times here also we need to add it two times so then we get rid of that extra token later the reason
29:57
we do this EXT extra addition is that it later helps us to create the target token because if you already add an
30:03
extra token creating the target creating the target is just simple because then you just use this much to create the
30:09
target as we saw before the target is just the input you remove the first element and then you add
30:15
50256 so earlier adding the 50256 token to all of the inputs in this part of the
30:22
code it's important because it easily helps us to create the target ID uh to
30:27
create the target for every inputs so essentially what we do is we add an extra 50256 and then get rid of it later
30:34
and then we pad everything all the other inputs with 50256 so that the length is equal to the maximum token ID length or



***


30:42
the that input which has the maximum length so essentially uh when you reach this part
30:49
of the code every item in the batch so all of these three items essentially
30:54
will have the same length that's what is happening in the code and ultimately we convert this um into a tensor and the
31:02
tensor is the input tensor which is returned by this function custom colate draft one now let's see a practical
31:09
application of this if you have uh these three inputs like this if inputs one has the size of five inputs
31:16
two has the size of two and inputs three has the size of three the batch you create a batch with these three inputs
31:22
and then you pass these you pass this batch into the custom Cola draft one now let's see the output as you can see the
31:29
first input remains unchanged it has five token IDs because those are the maximum length in the second inputs we
31:35
pad 50256 three times so that the length becomes same as the first input sequence
31:40
and in the third input we pad 50256 two times so that the length becomes the same as the first two so now you can see
31:47
we have an input stenor in which every row has the L has five columns
31:52
awesome so as we can see here all inputs have been ped to the
31:58
length of the longest input list inputs one which contains five token IDs
32:03
awesome uh so until now we have reached this stage where we have padded the inputs with the token IDs and now we
Coding target token IDs
32:10
have to implement the next part of this process the next part is essentially creating Target token IDs for
32:17
training so until now we have just implemented our first custom colate function to create batches from list of
32:24
inputs however as you have learned in previous lessons we also need to create batches with the target token IDs right
32:30
because we need to know what the real answer is the target token IDs are crucial because they represent what we
32:36
want the model to generate and based on the target token IDs itself we'll get the loss function
32:43
ultimately so as I explained to you thoroughly on the Whiteboard the way to get the target token ID is just to shift
32:50
the input uh to the right by one and then add an additional padding token towards
32:56
the end and that's exactly what we are going to do in the code so if you see in the code until this part it Remains the
33:03
Same we have the inputs um and they're padded by the 50256 token and now if you
33:08
see the targets token it just shifted to the right by one so here you see one colon which means that you forget the
33:15
first entry and you take the remaining entries uh and here you don't even need to add the 50256 token we because we
33:22
have already added added an extra 5256 token and this is why we add that extra
33:27
5 0256 token as I showed you earlier because here you see actually in the
33:33
first input you don't need to add the 50256 token but if you add it it makes it very easy to create the target um it
33:42
makes it very easy to create the the target sensor why because you just ignore the first element and take
33:48
everything from the second element so here you ignore the first element which is zero and take everything from the
33:55
second element so the target will be 1 2 3 4 4 5256 in the second the target will be 6
34:01
50256 50256 50256 and one more 5256 so it's the inputs which are
34:07
shifted to the right by one so that is how you create the target tensor and
34:12
then you just return the input tensor and the target tensor so the simplest way to think
34:19
about this code is that until now we have made sure that the inputs are of the same length due to the padding which
34:24
we have done and the targets is the inputs which are shifted to the right by one this is the very important process
34:30
and as I mentioned to you this is the non-intuitive step because uh the true
34:37
value is just the input which is shifted to the right by one and that is what is nonintuitive you might think that the
34:43
response needs to be given in the True Value right why is the instruction and input also given in the True Value but
34:50
it's given because in the next word prediction task the llm automatically learns that when you have the instruction and the input you have to
34:56
predict the response this was a bit harder for me to explain but I hope you have got this idea um in
35:04
the code if it's difficult to understand just try imagining it through the visual representation which I showed to you on



***



35:09
the Whiteboard you can even try going back as you are learning this lecture to see the Whiteboard explanation before
35:15
you try to understand the code so there are actually only two things which are happening in this custom colate draft 2
35:22
it takes it truncates the last token for the inputs so that everything is of the same length and it shifts
35:28
the input to the right by one to get the target tens and we can check this now let's say we have these three inputs as
35:34
before we create a batch of these three inputs and then we call the custom col draft two function on this batch and
35:40
we'll print the inputs and the targets so let's see as we learned before the inputs is just the first row Remains the
35:47
Same the second row has three 50256 tokens padded the third row has 250 256
35:53
tokens padded and let's look at the Target if you look at the first row of the targets







***


35:58
it's the first dra of the inputs you shift to the right by one so you take the remaining four and then you have 50
36:04
256 token similarly if you look at the second row of the target it's basically the second row of the input you shift to
36:11
the right by one which means you take only the remaining four values and then you add an extra
36:17
5256 similarly if you take the third row of the targets it's essentially the
36:22
third input but you shift to the right by one so you ignore the first and you take all the four and then you append an
36:28
extra 50256 token ID towards the end that's how you get the inputs and the target tensor so the first tensor
36:35
represents the inputs and the second tensor represents the target
36:41
awesome now after this step is implemented we come to the next step which is essentially creating uh or
Coding the padding token replacement with “ignore index = -100”
36:47
replacing the padding tokens with placeholders which means that except for the first 50256 we'll replace all the
36:54
remaining with minus 100 uh so in the next step we assign a minus 100
37:00
placeholder to all the padding tokens this special value allows us to exclude these padding tokens from contributing
37:06
to the training loss calculation as we saw on the white board
37:12
um okay so in the following code okay one more thing to mention is that as I
37:17
told you on the Whiteboard when we replace this 5025 tokens with minus 100 we retain one 50256 token and the reason
37:25
we retain one end of text token is because it allows the llm to learn when to generate an end of text token in the
37:31
response to instructions which we use an which we use as an indicator that the generated
37:37
response is now complete so you need one 5256 token ID to say that or to
37:42
represent that this is indeed the end of text so now what we have to do is that
37:47
we have to take this custom colate draft two and then we have to modify it further so most of the function is the
37:53
same which now I'm calling Custom colate function which takes in my batch my padding token ID and my ignore index so
38:01
that's minus 100 so what this does now is that until now here the steps are the same you get
38:08
the inputs and the target sensor but now what you do is that you take the target sensor only and all the indexes except
38:16
for the first 50256 you replace it with the ignore index so you first create a
38:22
mask and that mask has all the indexes which has the padding token ID then you ignore the first index which has the
38:28
padding token ID that's the first 50256 value and then you replace all the remaining ones with ignore index which
38:34
is minus 100 uh okay so this now creates my input
38:41
this now creates my input sensor and this creates my Target stenor and these
38:46
are both returned by this function which is custom colate
38:52
function okay until now if you note through the Whiteboard what we have
38:57
implemented is that we have implemented this part of the code where you can see in the figure that
39:04
uh here so except for the first 50256 we replace all of the remaining 50256 with
39:11
the value of minus 100 and now we are going to see why we replace the remaining 50256 token IDs
39:19
with minus 100 so to to see why we
39:24
replace the remaining token IDs with minus 100 we are going to see some implementations using pytorch but before
39:32
that let's actually see whether our custom colate function is really working
39:37
so to test that you take three inputs you have the inputs one as 0 1 2 3 4 You have the inputs two as 5 comma 6 and you
39:44
have the inputs three as 7 8 and 9 you create a batch with these three inputs
39:49
similar to the batch we have created before and then you create the inputs and targets based on the custom collate
39:55
function now if you see the inputs and targets tensor which we had obtained before the inputs and targets tensor
40:02
which we obtain now is actually exactly the same except that in the Target
40:08
sensor uh except for the first 50256 all the remaining 50256 values
40:14
have been replaced with minus 100 this is the only change which has been done there is no changes to the input sensor
40:21
the only change happens in the Target sensor where except for the first 50256
40:27
all the remaining have been replaced with minus 100 and now you can appreciate this code
40:34
which is the custom colate function where in this 15 to 20 lines of code what we are essentially doing is that we
40:40
we are implementing all of the steps which we have learned on the Whiteboard we implementing all of the steps over
40:47
here so essentially we are implementing the we are implementing the padding we
40:54
are implementing creating Target token IDs we are also implementing replacing the padding tokens with the placeholder



***


41:00
value of minus 100 and that's also called ignore index so up till now it's working right
Significance of PyTorch “ignore index = -100”
41:07
but now you might be thinking that the modified colate function works as expected altering the target list by
41:13
inserting the token ID of minus 100 but what is the logic behind this adjustment
41:18
why do we replace with minus 100 so let us take a small demonstration uh and I'm
41:24
going to show you the categorical or the cross entropy loss calculation right and the cross entropy loss calculation is
41:31
based on a logic sensor and a Target sensor I'm not going into details of this Logics and targets because I have
41:38
just taken two sample examples but for now you can think of it that the true answer is 0a 1 and the logits predicted
41:46
are minus minus one and one for the first training example and for the second training example the Logics
41:52
predictor are Min -.5 and 1.5 so this this is my predicted value and these are
41:57
the targets to calculate the loss between the prediction and the target we use the cross entropy loss so I'll share
42:05
the link to the pytorch Cross entropy loss which which calculates this loss between the prediction and the Target
42:11
and if you print out the loss you'll get it to be 1.1 1269 that's good then what
42:16
we do is that we add one more additional token ID in the prediction so in the
42:21
logits two now we have three training examples with three predictions and we have three true answers
42:27
so then we are using the categorical or the cross entropy function to calculate the loss between the prediction and
42:35
between the actual value and here if you print out the loss you'll see that the loss is 7936 it differs from the
42:41
previous loss because we added one more example now what I want to show you is that let's say instead of the targets
42:48
two being 0 1 and one what if the targets three is 0 1 and minus
42:53
100 so the Logics two will remain the same so the logits which are my predictions will remain the same but now
43:00
the targets are 0 1 and minus 100 now I want to show you a interesting thing
43:06
even if you add a minus 100 here and if you have a third example when you calculate this new loss you'll see that
43:13
the loss for this is the same as the first loss which you had obtained so
43:18
it's almost like adding the third training example made no difference at all so the loss here is 1.1 1269 and if
43:26
you saw the the loss in the first logits one and targets one that was also 1.12 69 so essentially there was no effect of
43:33
this third prediction and the reason there was no effect of this third prediction is that the targets had minus
43:39
100 that is the effect of ignore index equal to minus 100 so in other words the cross entropy
43:46
loss function ignored the third entry in the targets three Vector the token ID corresponding to minus 100 so you can
43:54
try reading replacing the minus 100 with another token and then the loss will not be the same it's only for minus 100 so
44:01
what's special aboutus 100 that it's ignored by the cross entropy loss well the default setting of the Cross entropy
44:08
function in pytorch is cross entropy ignore index equal to minus 100 and you
44:14
can see this over here also if you look at Cross entropy loss formulation you'll see that the ignore index is equal to
44:20
minus 100 over here that's the default setting of the pytorch Cross entropy loss this means that pytorch ignores all
44:27
the targets which are labeled with minus 100 so now if my target tensor has minus
44:32
100 over here it will ignore all of those predictions corresponding to these indices which have minus 100 in the
44:38
targets and that's good for us because these anyways don't represent anything meaningful that's why they won't
44:44
contribute to our loss function um so in this chapter we
44:50
actually take advantage of this ignore index to ignore the additional end of text padding tokens that we that we use
44:57
to pad so we we take advantage of this ignore index to ignore the additional
45:03
end of text tokens that we used to pad the training examples to have the same length in each batch however as I
45:09
mentioned we kept one 15256 token ID so this I will reiterate again because
45:15
sometimes students forget this we we we retain this this first
45:20
50256 we retain the first 50256 in all of the targets so even if
45:26
you see in the even if you replace the other 50256 with minus 100 you retain the first
45:32
one and you retain the first one because it helps the llm to learn to generate end of text tokens and that's an
45:38
indicator that the response is complete so this is the entire process and this is the entire workflow for




***


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






