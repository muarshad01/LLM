0:00
[Music]
0:05
hello everyone welcome to this lecture in the large language models from
0:10
scratch Series today we are going to learn about a very important topic which is called
0:17
as bite pair encoding in many lectures or even video content which you see on
0:23
large language models when tokenization is covered the concept of bite pair
0:29
encoding is rarely explained however behind modern algorithms such as gpt2
0:36
gpt3 Etc the tokenizer which these modern llms use is usually bite pair
0:45
encoding so you really need to understand what this means and how encoding how this encoding is done for
0:52
modern L large language models so let's get started with today's
0:57
lecture first let me take you to the Google collab
1:03
notebook um until now if you have followed the previous lecture what we did in the previous lecture is that we
1:10
implemented a simple tokenization scheme if you remember what we covered in the
1:16
previous lecture let me just uh take you quickly through that in the previous
1:21
lecture what we did was we basically took sentences we converted them into tokens and then we converted this into
1:29
vocabul so then every word of the token was arranged in an ascending order and then
1:35
a ID or a numerical ID or a token ID was assigned to each token so here every word was a unique
1:44
token along with special characters like comma full stop exclamation mark
1:50
Etc um now in today's lecture we are going to cover a much more sophisticated
1:56
tokenizing tokenization scheme why sophistic ated I'll come to that in a moment but just remember that today we
2:03
are going to learn about the bite pair encoding or BP tokenizer this BP tokenizer which we are
2:10
going to cover today was used to train large language models such as gpt2 gpt3
2:17
and the original model used in Chad GPT if you have not seen the previous
2:22
lecture on tokenization I highly encourage you to watch that so that this lecture will become very clear to you
2:29
because in the previous lecture we implemented our own tokenizer completely from scratch and today we are going to
2:36
learn about a bit more advanced concept and this scheme this bpe scheme is used
2:42
in all modern llms and so it's very important for us to learn so let me take
2:48
you to the Whiteboard right now and explain the concept of bite pair tokenizer and why do we even need it in
2:54
the first place so if you look at the tokenization algorithms there are
Word and character level tokenizers
2:59
essentially three types of tokenization algorithms the first is the word based
3:05
tokenizer U the second type is the subword based tokenizer and the third
3:11
type is the character based tokenizer let me walk you through these step by step in the word based tokenizer what we
3:18
usually do is every word in the sentence is usually one token the tokenizer which
3:24
we saw in the previous lecture such as for example in this example the um the
3:30
fox chased the dog the tokens were the fox chased the dog so every word was one
3:37
token right so that's why it's an example of word based
3:42
tokenizer uh here I'm taking one more example to illustrate this concept to
3:48
you if the sentence is my hobby is playing cricket and if the tokens are
3:53
individual words such as my then hobby then is then playing and then Cricket
3:59
that that is a word based tokenizer awesome now can you think of the
4:04
problems associated with the word based tokenizer
4:10
first uh the main problem is what do you do with Words which are not present in the vocabulary so let's say if you have
4:16
huge amount of training data and you will break it down into sentences right and then you'll break down the sentences
4:23
into individual words and then you will assign a token ID to each of these words
4:28
but when the user is interacting with the llm let's say and the user inputs a word which was not present in the
4:34
vocabulary these words are also called as out of vocabulary words so you can think of as preparing for an exam but in
4:42
the exam a question is asked which is completely different from what you have prepared for uh in word based tokenization
4:49
schemes it's very difficult to know what do we do with out of vocabulary words because consider this case itself my
4:56
hobby is playing Cricket let's say if the data set was small and only one sentence if a new word is given such as
5:03
football and that is not present in this data set then usually these tokenization
5:10
schemes run into an error this is a very simplified example I have constructed
5:15
but uh this problem shows up when you do word based tokenization a lot usually
5:21

***

it's very difficult to deal with out of vocabulary or oo words if they are not
5:26
present in the vocabulary and what do I mean by vocabulary vocabulary is just a dictionary with a collection of tokens
5:33
arranged in ascending order and every token corresponds to a token ID great now another problem with this
5:41
word based tokenization scheme is that uh let's say boy and boys are two words
5:48
in the vocabulary each of them will have different tokens maybe the token IDs for
5:53
both of them will be far apart based on uh where they appear or they might be similar also but but the problem is that
6:01
uh these words are very similar right and when we do this tokenization this similarity is not captured so boys if
6:10
you think of the word boys the root word is boy so ideally both of these words
6:15
are very similar to each other but with this tokenization they are treated as separate words and that similarity is
6:22
not captured these are the main problems with word based tokenization now let's
6:27
look at the other end of the spectrum which is character based tokenization this is also a very popular tokenization
6:34
scheme where what we do is that if the sentence is again my hobby is playing Cricket instead of having individual
6:41
words as tokens in this kind of character based tokenization individual characters are
6:47
considered as tokens so for example here let's look at which characters are there m is the first character why is the
6:54
second character uh I'm ignoring white spaces for now so H is the third
7:00
character o is the fourth character Etc so then the vocabulary or the tokens would contain
7:06
m y h o b Etc these are the tokens instead of
7:13
having individual words now can you think of what will be the vocabulary size in this
7:19
case just think about this for a moment if you are not able to answer this just
7:24
pause and think okay so this will actually lead to a very small vocabulary because every
7:32
language has a fixed number of characters if you look at the English language it has 256 characters on the
7:38
other hand if you look at the total words in the English language it has about 170,000 to 200,000 words uh but
7:47
what is one of the advantages of the character based tokenization is that it has a very small vocabulary size which
7:55
means that uh either so if you look at the Engish Eng language there are fixed number of
8:01
characters right there are 256 characters so we will never have the out of vocabulary problem because let's say
8:06
any new sentence is given by the user you can always break it down into characters even if you don't know the
8:12
words in that sentence it's fine you break it down into characters and it will be either of the 256 characters
8:17
which are already present in your vocabulary or dictionary so the out of vocabulary problem will not come into
8:23
the picture and uh it solves one more problem if you look at the word BAS
8:29
tokenization right as I told you English language is about 170,000 to 200,000
8:35
words so if you really want to include everything in the vocabulary you need a vocabulary size which is huge and that
8:42
is one big problem in word based tokenization this problem is completely solved in the character based
8:48
tokenization because the vocabulary is based on characters and the vocabulary length is pretty
8:54
small but then you might think oh this sounds amazing right it literally solves all of the problem problem it solves the
9:00
out of vocabulary problem it's also computationally and memory efficient because the vocabulary size is very
9:06
small and uh then that's great then what's the issue there are some problems
9:11
with character based tokenization and the first major problem is that the meaning which is associated with words
9:17
is completely lost essentially the advantage of dealing with language models is that words have meanings right
9:23
so different words and different sentences might be related to each other boy and boys have a common meaning this
9:29
is completely lost since you're breaking it down into individual characters that's one of the first
9:35
biggest problem with uh character level tokenization the second problem is that
9:41
the tokenized sequence is much longer than the initial raw text so for example
9:47
if there is a word in the let's say there is a word in the
9:52
text which is dinosaur now in word based tokenization
9:58
this will be treated as one single token right but in character based tokenization what will happen is that
10:03
every every single character here d i n o s a UR will be treated as separate
10:10
token so in character based tokenization the word dinosaur will be actually broken down or split into eight
10:17
tokens and that is another major problem the tokenized sequence is much longer
10:22
than the initial raw text now uh so as we can see the word
10:27
based tokenization has its advantages and disadvantages the disadvantages is
10:33
that we don't know what to do with out of vocabulary words and the vocabulary size is pretty large the advantage is
10:39
that uh of course the words every word is a token
10:45
so the tokenized sequence length will be small like dinosaur it will be just one
10:51
token so when we tokenize the paragraph it's very small uh because every word
10:57
will be one token unlike character based tokenization where the tokenized sequence is much longer than the initial
11:03
text that's the disadvantage of character sequence character tokenization the advantages of character
11:09
tokenization is that they have very small vocabulary because every language has fixed number of characters and we
11:16
solve we completely solve the out of vocabulary problem and both of these approaches have a disadvantage that the
11:22
meaning between words is not captured boy and boys have a common root right
11:27
that root is cap is not captured tokenization and modernization both have the isation which is common this meaning
11:34
is completely lost so now so what do we do then we turn to another tokenization algorithm
Sub-word tokenization
11:42
which is called as the subword based tokenization and the bite pair encoding which we are going to see is an example
11:48
of subword based tokenization algorithm the subword based tokenization is kind of like a Best of Both words words and
11:56
let's see how it uh why it is the best best of both words so the first thing to
12:02
remember about subword based tokenization is it does capture some root words which come in many other
12:08
words so boy and boys it will treat boy as a common root word uh and let's see
12:14
how it does that okay so in subword based tokenization there are essentially two rules the first rule of this
12:21
tokenization is that when you get the data set you do not do not split
12:27
frequently used word into smaller subwords so if there are some words which are coming frequently you should
12:34
retain those words as it is right so then it retains this from the word
12:41
tokenization then the rule two is that if there are some words which are very
12:46
rare which are not occurring too many times then you split these words into smaller meaningful subwords this is
12:53
extremely important this second part basically says that if there are some words which
12:59
are rare not appearing too many times you can go on splitting it further you can even go down to the Character level
13:05
so you can see why it is a mix between word tokenizer and character tokenizer the first rule implies that if the words
13:12
are occurring many times you return it as a word so this is taken this is a feature taken from the word tokenizer
13:18
the second rule implies that if the word is rare you can go on splitting it into further subwords and if needed you can
13:25
drop down to the Character level we don't always drop down to the character level we even stay in the middle but
13:31
this is a feature from the character level because we are breaking down the word further so to give you an example
13:38
if the word boy is appearing multiple times in the data set it should not be split further that's from rule number
13:44
one so then boy is retained as a token but boys if the word boys is encountered
13:51
uh that should be split into boy and it should be split into
13:57
s uh because boys might not be appearing too many times and again boys also derives from the word boy so we should
14:05
divide this word boys into smaller meaningful subwords so boys is divided
14:10
into boy and S and this is the main essence of subword tokenization words
14:17
are sometimes broken down into smaller elements and why smaller elements because those smaller elements appear
14:24
very frequently so for example why is boys broken down into boy and S because boy appears more frequently and S is
14:31
also another token which appears very frequently this is what is basically
14:37
done in a subord tokenization scheme at a very high level so let me explain some advantages of subord tokenization the
14:45
subword splitting helps the model learn that different words with the same root word such as for example token tokens
14:54
and tokenizing all of these three words essentially have the same root word right token so subord splitting helps
15:01
the model understand that these different words essentially have the same root word and they are similar in
15:07
meaning this meaning is lost in word based tokenization and even character based tokenization that is number one
15:14
the second advantage of subord tokenization is that it also helps the
15:21
model learn that let's say tokenization and modernization are made up of different root words token tokenize and
15:28
modernize but have the same suffix isation and are used in same syntactic
15:35
situations so basically it derives these patterns like tokenization and modernization maybe zation z a t i o n
15:43
is a is a subword token in this tokenization and then token and then
15:48
modern and another tokens so then it learns that this isation is a common suffix which appears in both of these
15:54
words all of these are advantages of breaking down one word into subwords this is what is majorly done in subword
16:02
tokenization so why are we learning about subword tokenization and what's the relation between bite pair encoding
Byte Pair Encoder (BPE) Algorithm
16:08
and subo tokenization well the main relation is that uh bite pair encoding
16:15
or bpe is a subord tokenization algorithm this is very important and
16:20
that's why we are learning about bpe and that's why modern llms like gpt2 and
16:26
gpt3 employ bite pair encoding so let us look at the bit of a history
16:31
of the BP algorithm and let and then we'll see how it is implemented in practice so the bite pair algorithm was
16:39
actually introduced in 1994 and uh it does a very simple thing
16:45
it is basically a data compression algorithm uh and what is done is that we
16:50
we scan the data and then mostov common pair of consecutive bytes so let me
16:56
actually use a different color here this sentence is very important most
17:01
common pair of consecutive bytes of data is replaced with a bite that does not
17:07
occur in data so we identify pair of consecutive bites which occur the most
17:13
so we find these pairs which occur the most frequently and then we replace them with a bite which does not exist in the
17:19
data and we keep on doing this iteratively I'll show an example for this so don't worry if you don't
17:25
understand this explanation I want to quickly show you the paper so this is the the paper which was published in
17:30
1994 a new algorithm for data compression which introduced bite pair tokenizer or bite pair encoder rather at
17:37
that time it was not known that this will be so useful for modern large language models but it is quite
17:44
useful okay so now let us see a practical demonstration of this algorithm so I've taken this example
17:51
from Wikipedia because I think it is an awesome illustration so let's say we have this original data right and the
17:58
original data looks like this a a a b d a a a b a c okay and now in bite pair
18:05
encoding we are going to compress this data right and let us see what do we do we first identify the most common pair
18:12
of consecutive bytes so let's see let us see the pair which occurs the most so if
18:18
you scan this sequence from left to right you will see that the bite pair AA occurs the most right you might say a AA
18:25
also occurs the most but that's not a bite pair because that three characters the bite pair AA occurs the most it
18:32
occurs here it occurs here it occurs here and it occurs the here so it occurs
18:37
four times really so what we will do next is that we have identified the bite pair which occurs the most we replace
18:44
this bite pair with a bite that does not exist in the data so what we'll do is
18:49
that we will replace it with zed and why Z because just Zed does not occur in the
18:55
data you can use any variable here so then we'll take these a a and wherever AA shows up we'll replace it by Zed so
19:03
then this first AA will be Zed so then this new sequence will be Zed a b d then
19:08
again Zed to replace this a a so then it will be Zed and then a b a c so then we
19:14
have a b a c so remember what has happened here is that this AA which I'm highlighting right now in circle that
19:20
has been replaced by Zed and this AA has been replaced by Zed so now we have a compressed data correct very good so now
19:28
now let us move next next what we do is we keep on repeating this sequentially
19:33
now we again look at this sequence and find the next common bite pair so we can see that a is ur occurring once and a is
19:42
occurring twice so it is repeating two times and which is the most frequent bite pair so we will replace AB by y so
19:49
this AB will be replaced by Y and this AB will be replaced by y why why because
19:54
it's a bite which does not occur anywhere in the U data
20:00
so then this AB will be replaced by y as you can see here and here also the ab will be replaced by y so then my new
20:06
compressed data will be z y d z y a great this is a compressed data compared
20:13
to the original data and this is exactly how the bite pair algorithm works now we do this recur do this again and again
20:20
right so now let us look at the bite pair so here AC is the only bite pair which is left all others are special
20:26
tokens but AC only appears once so we we don't need to encode it this process of
20:32
replacing common bite pairs with another variable is called encoding that's where the word encoder comes from so we will
20:39
not encode this further because it only appears once if a c were appearing twice then we would have encoded it with
20:45
another variable so this compression actually stops here you can go one more layer Deeper by further com replacing
20:52
the zy with another variable which is let's say w so then it will be WD w a and then you
21:01
will stop so you will compress it further like this so you see the original data has been compressed to
21:07
this right uh to this compressed version U using the bite pair algorithm so the
21:12
algorithm itself is pretty simple you scan the data from left to right you identify the bite pairs which occur the
21:18
most and then you replace them with a bite which does not exist in the data and you do this iteratively until you
21:25
reach a stage where no bite pair occurs more than once that's it that is the simple bite pair encoding
21:32
algorithm now you might think okay what has that got to do with large language models right I understand this algorithm
BPE for Large Language Models
21:39
and I understand how it compresses a given data sequence but what has that
21:44
got to do with large language models well it turns out that the we slightly
21:49
tweak the bite pair encoding algorithm and use this to convert our entire
21:55
sentence into subwords which will be very useful for us and I'll show you exactly how I'm going to do that so the
22:03
bite pair encoding for llm ensures that the most common words in the vocabulary
22:08
are represented as a single token remember rule number one and rare words
22:14
are broken down into two or more subord tokens this is exactly the same rules which we had looked at the rule number
22:20
one and rule number two so rule number one is that most commonly used words
22:26
should not be split and second is that that rare words should be split into meaningful subwords now let's see how uh
22:33
how it's related to The Bite pair encoding algorithm which we saw and we will be looking at a practical example
22:39
for this uh demonstration okay I will use a color which is a bit different from the green
BPE practical demonstration
22:46
one here so that uh there will be a good contrast so let me use the orange color
22:52
okay so for the Practical example we are going to look at a vocabulary or rather I should say we are going to look at the
22:58
data set of Words which is also called as Data Corpus which is this so we have
23:03
these words in our data set old older finest and lowest right let's say
23:12
we have these words in the data set right now and I'm going to show you how we are going to use the bite pair
23:19
encoding algorithm to break these down into tokens and you will also see why this is
23:24
a subword tokenization scheme so first of all if we are to use a word based
23:30
tokenization then this will have four tokens old older finest and lowest
23:35
that's it similarly if we were to use the Character level tokenization then the tokens will be individual characters
23:42
like o o will be one token then L will be another token then D will be another
23:48
token Etc so this is how the word based tokenization and the character based
23:53
tokenization will work but right now we are going to see the subw based tokenization using the bite pair
24:00
algorithm before we uh proceed further we'll need to do a pre-processing step
24:05
and this is actually done uh even when we train large language models and when we are using these tokenizers so
24:12
basically when you look at these different tokens there should be some ending right so for example when this
24:18
old token appears we should have another end token so that we know that this word
24:24
has ended over here so I'm going to augment every token here with this additional end token which is called
24:32
slw so whenever the algorithm or the model comes across slw we know that this
24:37
word ends over here so I'm going to replace the tokens in my data set with adding a w/w at the end like old becomes
24:45
old slw older becomes older slw finest becomes finest slw and lowest becomes
24:53
lowest /w now remember here that if we use the word based
24:59
tokenization uh there is no meaning which is captured so the fact that old
25:04
is the common root between old and older is not captured number one EST is the
25:09
common root between finest and lowest that's not captured so word based
25:15
tokenization character based tokenization have so many problems because they don't capture these meanings or root words and towards the
25:22
end of this section we'll see how subword based tokenization using the bite pair encoding algorithm actually
25:27
captures these root words like old uh like EST Etc okay so
25:35
let's get started with the individual steps the first step is basically to split all these words into their
25:43
characters um and then make a frequency table so what we are going to do is that we are going to take all these words old
25:50
older finest and lowest so old appears seven times in the data set older
25:55
appears three times in the data set these are the frequencies finest appears nine times in the data set and lowest
26:02
essentially appears four times in the data set so what we are going to do right now is we are going to split these
26:08
words into individual characters so then here is the table which I have made remember slw is also there uh so old so
26:17
all words have/ W right and totally how many words do we have 7 + 3 10 + 9 19+ 4
26:23
23 and since all words have slw at the end of it slw comes 23 times similarly
26:29
we see that o comes 14 times L comes 14 times D comes 10 times
26:36
e comes 16 times Etc and we make this frequency table list so here you can see
26:42
that we have 12 tokens and if we did we use character level tokenization our
26:47
tokenization would end here because all these characters would be individual tokens now what we will do similar to if
26:54
you remember the bite pair encoding algorithm we looked at the most frequent pairing right uh so let me take you back
27:02
to this yeah we looked at AA because AA appeared the most so we looked at that
27:07
bite pair which occurred the most this is exactly what we'll be doing here we'll be look for the mo we'll be
27:13
looking for the most frequent pairing in the data set and then what we'll do is that this
27:19
is the modification compared to the original algorithm when we look for the most frequent pairing we will merge them
27:26
we will merge the most fre pairing and then perform the same iteration again and again and
27:32
again so let me show you how this is done so in the iteration one as I mentioned we start with the uh finding
27:40
the most frequent pairing right so it so you the way to do it is look at the first character which appears the most
27:47
so e is that character which appears 16 times right so if you want to look at the pairing which appears most it most
27:53
probably starts with e so it turns out if you look at these words e and s is
27:59
the pairing which appears the most number of times so e and s here appears nine times in finest and E and S appears
28:06
four times in lowest so e and s is that pairing which appears 13 number of times
28:12
right so uh most common bite pair starting with e is e and s so what we'll
28:18
now be doing is that we'll be going through the data set again uh and we'll
28:23
be merging these two tokens e and s so now e s will be one token and that's why it's called subword es will be one token
28:31
so now let me show you my token table again everything else is the same up token number 12 but look at this token
28:38
number 13 which has been added in token number 13 we have added one more token which is
28:44
es because it's the most frequent pairing and Es appears 13 times but
28:50
remember when we add es we have to subtract something from E and we have to subtract from s because now uh ES has
28:58
been included so we subtract 13 from the e count so now the the number of time only e appears is three the number of
29:05
time only s appears is zero this is very interesting to know so the number of time only s appears is zero so s it
29:12
seems always appears with e so es is a subword see this we would not have discovered if we just did character
29:18
level tokenization or uh Word level it seems that e and s always so s only
29:24
comes with e in this data set we have already obtained our first site so now this is my new uh this is my
29:32
new token library and this is my additional token and now we are going to actually continuously keep on doing this
29:39
process to find uh frequency or to find tokens which appear the most number of
29:45
times so in the previous iteration we saw that e and s was the bite pair right
29:51
which occurred most number of times ands it appeared 13 times but now es is a separate token for us so now using that
29:58
token we see that EST is again a bite pair because es is one token and T is
30:04
another token so es s and t becomes a bite pair in the second iteration and EST appears 9 + 3 which is again 13
30:13
times so now in the next iteration what we'll be doing essentially is that U let
30:19
me show you iteration number two in the iteration number two we'll merge the tokens es s and t because they have
30:26
appeared 13 times in the data set see we are doing the same thing what we did in the earlier bite pair encoding for that
30:33
character so let me just showed you show you here remember here what we did after
30:38
AA so AA was done right and then we merged this and then we looked at the second sequence which appeared the most
30:45
which is AB that is actually very similar to what we are doing here es appears the most so we created one more
30:51
token for ES then we looked at another bite pair which is appearing the most and that is es s and t so now what we'll
30:58
be doing is that we'll merge es andt into one token and uh so EST comes to be 13 times
31:06
and we'll then subtract 13 from the previous token es so now only es appears
31:11
zero times and EST appears 13 times see we have constructed a new token EST so
31:18
now remember what I said earlier the previous World level tokenizers and the
31:23
Character level tokenizers could not identify that EST is a common roote between finest and lowest but with our
31:30
bite pair encoding algorithm we have already created a new token for EST so this algorithm has already identified
31:37
that EST is a common root World great so up till now we have done
31:43
iteration number three and uh we see that okay I think yeah up till now we
31:49
have done I think two iterations and now up till now we have merged es and T into one common token
31:55
awesome now let us take look at this slw token now we can see that EST and /w
32:03
basically appears 13 times again it's the same thing over here so EST always
32:09
comes with slw now EST is one token so EST and slw forms a bite pair and this
32:15
bite pair again occurs 9 + 4 which is 13 times which is much more than any other bite pair in these words so we'll
32:22
combine EST and /w into one more token so in the third iteration we are going
32:28
to combine EST and /w into one more token so now this becomes our
32:33
word do you understand why we combined slw with one more token we could have
32:39
just left it at EST right but if we left it at EST then essentially there would have been no difference between words
32:46
like estimate and highest so estimate and highest both have EST but the words
32:52
in our data set are highest and lowest so EST is the ending sequence in all of
32:58
our words in our data set and we need to encode that information that it is an ending sequence so estw allows us to
33:05
encode this information so now the tokenizer knows that whenever EST comes
33:10
it's always followed by slw which means the word ends after estd so now our
33:16
algorithm or the tokenizer can differentiate between estimate and highest because in estimate EST does not
33:23
end with a /w so now if you look at the Tok which we have earlier we had all these 12
33:30
tokens but now we created es then we merged it to EST and finally we created this
33:36
estw so now these two tokens are actually not needed es and EST so now
33:41
let's look at another other bite pairs which occur a lot so it turns out that o and L is another bite pair which occurs
33:47
10 times because it's present in old and older so what we'll do is that we'll create one more we'll merge these two
33:54
and create one more token for o and L it appears 10 times and we'll subtract that count 10 from the O and L so o
34:02
individually or with some other character appears 14 times so we subtract 10 because now we have created
34:07
one more token for o which appears 10 times similarly L appears 14 times
34:13
overall and we subtract 10 from it because L comes with o 10 times right
34:19
and now what we do is that o l is one token so now we see that o l and D has
34:25
appeared 10 times so this bite pair has appear 10 times so we merge this bite pair so then o d now becomes another
34:32
token which appears 10 times so you see the meaning which our bite pair encoder has captured we have constructed one
34:39
token which is old we have another token which is estw these tokens are subwords so they
34:47
are neither full words nor characters they are subwords but they encode the root representation so old is one token
34:53
and this actually tells us that uh old is the root word which comes in
34:59
Old it which comes in old as well as older and our BP algorithm has actually
35:04
captured that perfectly that's awesome right all these root words were not captured by just the word uh word em or
35:13
the word encoder word tokenizer and the Character level tokenizer now you might be seeing here
35:20
that these f i and N appear nine times uh it's fine that they appear nine
35:26
times but we we just have one word with these characters so if you look at our data
35:32
set again let's see where f i and N appear so the words are finest so all of
35:40
these words actually only appear in finest so it's no it does not make sense to merge them into one um one token
35:48
because that token does not the frequency of that token appearing in other words is not too much why did we
35:53
merge EST into one token because it appears in multiple words why why did we merge old in one token because it
35:59
appears in multiple words so see the rule of uh subord tokenization the first
36:05
rule is that that word which occurs multiple times you keep it as it is old
36:10
so we kept old as it is right it is a separate token uh but that word which is
36:16
not used too many times like older it needs to be split into old and then e and then R are separate tokens similarly
36:23
in finest EST is one token fi and N are separate tokens in lowest EST is one token and L O and W
36:31
are separate tokens and this helps us to retain root wordss in sub
36:37
tokenization I hope you are I hope you have understood this concept now what we can do is that uh this EST and old are
36:45
final tokens which we constructed by merging now let us actually remove all of those tokens whose frequency is zero
36:51
so we can remove s we can remove so let me Mark them with a different color so s
36:57
has a frequency of z t has a frequency of Z es EST has frequency of z o has a
37:03
frequency of Z so let's remove this so then our final table looks like these these are the final tokens in our subord
37:10
tokenizer which is obtained using the bite pair encoding algorithm how did we obtain these tokens we just looked at
37:16
bite pairs which occur the most then we merged them into one and then we repeated this process uh until we
37:24
obtained until we reached a stage where enough number of tokens have been created or until we reached a stage
37:31
where our iterations have stopped and this is the final tokenized uh final uh tokens which we'll
37:37
be using for next steps of the large language model training which are vector embedding Etc this is how the subword
37:44
tokenizer works and this is exactly how bite pair encoder which is a subword
37:49
tokenizer it works for uh training models like gpt2 or gpt3 very few people
37:55
have this understanding but I hope this lecture has intuitively made it clear for you how the bite pair tokenizer
38:01
actually works so now this list of 11 tokens will serve as our vocabulary why
38:07
is it called subword tokenizer because these are subwords right EST is not a full word neither it's a character it's
38:13
a subword but it's the root of many words o is also the root of many
38:18
words now you must be thinking when do we stop this uh merging when do we stop
38:24
these iterations so you have to specify a stopping criteria usually if you look at gpt2 or gpt3 you run this bite par
38:31
coding algorithm on a huge number of tokens right so the stopping criteria
38:36
can be if the token count becomes a certain number then you stop or just the number of iterations can be the stop uh
38:42
stopping count so I'm in this lecture I'm not covering the stopping criteria in too much detail but I just want to
38:49
give you an intuition of how the stopping criteria can actually be formulated or computed it's based on the
38:55
token count or the number of iterations to give you an example uh or to mention
39:01
one more advantage of bite pair encoding here you can see that we it's actually better than Word level encoding right
39:08
because uh let's say if you have two words boy and boys both won't be separate tokens in this just boy will be
39:15
a token so bite pair encoder also reduces the number of tokens which we have compared to word encoding let's see
39:23
uh and that usually helps so in gpt2 or gpt3 when it was trained I think it used
39:29
about 50,000 more than around 57,000 tokens I think when when bite pair
39:34
encoding was done uh for gpt2 or gpt3 so bite pair
39:40
encoding solves the out of vocabulary problem why does it solve the out of vocabulary problem because we also have
39:46
subwords now and some of it might even be characters so character level
39:51
tokenization solves the out of vocabulary problem right and bite pair encoding or subord encoding retains some
39:58
properties from it so you can see some tokens here are characters which is good but it also solves the problem of the
40:04
word level encoding and one of the major problem is that root meanings are not captured but now they are captured over
40:10
here and it also solves the problem of the word level encoding being just a
40:15
huge bunch of numbers here that is not the problem because we break it down into characters and subwords so the
40:22
total length of the vocabulary is also shorter than if you would have just considered Word level embedding so
40:27
subord embedding solves it it solves the out of vocabulary problem it also gives us a vocabulary size which is
40:35
manageable and it also retains the root words meanings such as EST and
40:40
old awesome right and that's why because of all these advantages this is the bite
40:46
pair encoding algorithm is used for tokenization in gpt2 and gpt3 now let me return back to the code
Implementing BPE in Python
40:53
I could have just taken you through the code today but then you would not have UND OD how the bite pair algorithm actually
40:59
works now implementing BP from scratch is relatively complicated so we will be
41:05
using a python open source Library called tick token so if you I'll share the link of this library in the chat so
41:12
this is a library which is essentially a bite pair encoder it's a bite pair encoder which is used for open a models
41:19
so open a themselves use tick token for tokenizing the sentences from the data
41:25
you can see that it has about 11,000 stars and about 780 Forks so it's a pretty popular
41:32
repository okay so uh now we are going to implement the bite pair encoder in
41:38
Python and I'm going to show you this implementation in a Hands-On demonstration the first thing to do is
41:44
install tick token because this is we'll be using the bite pair encoder from uh
41:50
tick token and here you can see that it takes a it takes some time to install this uh for me when I first install
41:57
insted it it took about 1 minute but now it's a bit faster so requirements are already satisfied um and you'll see that
42:04
it will get installed now in some time see it's already installed so now I have installed tick token and let me just
42:10
print out the version of tick token which I'm using here so you can see that the origion of tick token is 6 which is
42:17
good enough now what we have to do is that we once the tick token is installed we can
42:23
instantiate the bite pair encoding tokenizer from tick token so the way to do this is just uh use tick token do get
42:32
encoding gpt2 and we store it in this object called tokenizer so now tokenizer
42:38
essentially has is that is similar to the simple tokenizer version two class
42:43
or version one class which we defined in python in the last lecture remember we defined this uh simple tokenizer version
42:51
one class and uh we defined the simple tokenizer version two class here which
42:56
also include special tokens so now in just one line of code we have essentially defined this tokenizer which
43:03
is something similar but now it will it's initiated an instance is created
43:08
through this tick token Library so this is a bite pair tokenizer and uh similar to the
43:14
tokenizer class which we had created if you see every tokenizer class has an encode method and a decode method what
43:21
the encode method does is basically it uses that tokenizer and it converts words into token IDs what the decode
43:29
method does is that it decodes those token IDs back into individual words so let's see how this uh a tokenizer
43:37
actually works so this is the bite pair encoding tokenizer from gpt2 and let's
43:42
give it some text and try to encode and decode right and I'm testing uh this
43:48
tokenizer because I've given it a complex sentence see so the first part of the sentence is hello do you like T
43:55
then we have given one more thing which is called end of text which means that a new text is starting here this is the
44:00
end of text is usually done to separate entire documents but here I'm just using it to illustrate and in fact gpt2
44:08
actively uses end of text so just to give a demonstration to you when gp2
44:13
when gpt2 or gpt3 is trained on large amounts of data sets here is how the
44:19
data sets are loaded so see uh what gpt2
44:24
does is that when it has data from different text sources it usually shows when a particular text has ended and
44:30
when the other text has started so end of text is actually a part of the gpt2 vocabulary itself so I have given this
44:38
end of text here and then the second sentence is in The sunl Terraces of some unknown place now look at this if we are
44:45
using a word level tokenizer this this will lead to the out of vocabulary problem because this word cannot be in
44:52
the vocabulary of a word level tokenizer and that's the advantage of bite pair encoding in bite pair encoding we have
44:59
characters as tokens and we even have subwords as tokens so definitely this will be encoded because might be some
45:07
character can encode it so some unknown and place might be subwords which are individual tokens they can also be used
45:13
to encode this so bite pair encoder will take care of this this entire word which
45:18
usually does not exist as a single word so let us test this actually and uh I've
45:24
run this right now and let us see the different tokens so see hello do you like T it all of these have been
45:30
converted into token IDs I want you to take a look at this 50256 token so this
45:35
is the token ID of end of text and this is also the vocabulary size of the
45:41
tokenization scheme used in gpt2 or gpt3 so if we were to use a World level
45:46
tokenizer English language has about 170,000 to 200,000 words so the
45:51
vocabulary size would have been this much but now that we have used the
45:57
bite pair tokenizer or bite pair encoding the vocabulary size has reduced by around 13 from
46:04
150,000 and with with a vocabulary of around 50,000 subwords uh we are able to get the
46:10
amazing performance from gpt2 gpt3 or GPT 4 I think has much higher level of
46:16
uh tokens but even gpt2 has good enough performance and it has 50,2 56 tokens
46:23
awesome and here you can see that I did not not get an error because some unknown place was not encoded the
46:30
tokenizer is able to encode random words like these which also look wrong and here you can see these are the encodings
46:37
for this sub some unknown place so the end of text is this in the sunlight Terraces of some unknown place so I
46:43
think some unknown place is broken down into subwords three or four subwords and then it's tokenized by
46:49
gpt2 so this is how subord tokenization can handle out of
46:54
vocabulary awesome so the code prints these IDs which we just saw and now what
47:00
we can do is we can convert the token IDs back into text using the decode method remember every tokenizer is
47:06
encode method as well as decode method so we can do tokenizer do decode and put all of these integers and here you see
47:14
it decodes exactly the same sentence what we gave to it so the decoded sentence is hello do you like T then end
47:20
of text in The sunl Terraces of some unknown place exactly similar to the uh
47:26
text which it had G which we had given which means that the encoder and decoder is working very well uh and the bite
47:33
pair encoding has its advantages of dealing with out of vocabulary so here we see two advantages of bite pair
47:39
encoding first is that it reduces the vocabulary length second is that it knows how to deal with unknown text
47:46
awesome so we can make two noteworthy observations based on the token IDs and the decoded text first as I already
Key takeaways
47:53
mentioned the end of text token is assigned to a relatively large token token ID which is
47:59
50256 uh that is the last token in the BP tokenizer used for gpt2 so in fact
48:06
the BP tokenizer which was used to train models like gpt2 gpt3 and the original
48:11
model used in chat GPT has a total vocabulary size of 50257 keep this in mind now you know what this means and
48:19
end of text is the last largest token ID and it's also the last token in the
48:25
vocabulary the second uh thing to note which I already mentioned is that the BP tokenizer encodes and decodes unknown
48:32
words such as some unknown Place correctly the BPA tokenizer can handle
48:37
any unknown word and how does it achieve this without mention so we did not give special tokens for unknown words right
48:44
uh how did it achieve this and this this is because of how the tokenization was done and because of how the uh BP
48:52
tokenizer actually goes down to the level of uh let me show this the BP tokenizer goes down to the level of
48:59
characters and subwords that's why it is able to deal with unknown text so the algorithm underlying BP breaks down
49:06
words that aren't in its predefined vocabulary to smaller subword units or even individual characters this is
49:13
exactly what we saw over here uh and this enables it to handle
49:19
out of vocabulary words so thanks to the BP algorithm if the tokenizer encounters
49:25
an unfamiliar word during tokenization it can represent it as a sequence of subword tokens or
49:31
characters that is the advantage of BP tokenizer and that's why we are having this lecture in the first place great
49:39
now let us actually take one last example to illustrate how the BP tokenizer deals with unknown tokens
49:45
right so let's say we have this this sentence a k w i r w i e r completely
49:51
random words which do not make any make any sense but if you pass these random words to
49:57
uh the tick token gpt2 encoder bite pair encoder you'll see that it does not show
50:02
an error in fact it even encodes these so a k w i r w i e r is encoded as these
50:09
tokens because it broken down into subwords maybe e r is a commonly occurring subword in all the vocabulary
50:15
which we have so then the token for this is 25959 maybe AK is a commonly occurring
50:21
subo maybe W is just a simple character maybe IR is a commonly occurring subo
50:26
maybe I is just a character so you can see this what underneath the hood what
50:33
this tokenizer must have done is that it must have broken this down into words and subwords based on what is present in
50:39
the vocabulary and then to each of these word to each of the subword or character
50:45
it would have assigned a token ID so the tokenizer would have scanned this from left to right and broken it down into
50:51
characters or subwords and then assigned each character or subword a unique token ID that's how the encoding works and if
50:58
you decode the integers you get back exactly the same sentence which you started with and the reason I covered
51:05
this lecture in so much detail is because in the previous lecture we saw how to do tokenization from scratch and
51:11
today we saw how what is bite pair encoding and how you can install tick token and Implement your own bite pair
51:18
encoder or tokenizer with gpt2 is using and we saw how it handles unknown words
51:24
uh how it captures the root meaning and how the vocabulary size is about
51:29
50,000 uh this brings us to the end of today's lecture in the next lecture we'll be looking at data sampling batch
51:36
sizes context length Etc before we feed the embed or before we feed the
51:42
tokenizers into the word embedding so in our progression of learning right now
51:48
let me show you um one plot so that I can illustrate up till where we have
51:53
covered until now so if you look at uh
51:58
if you look at yeah if you look at this schematic right here yeah if you look at this schematic
52:05
right here until now we have looked at how to tokenize an input text and how to
52:10
convert it into token IDs and we saw implementing our own tokenizer in the previous lecture and using a bite pair
52:17
encoder in this lecture we still have to see token embeddings so we'll get to this point and before that we'll also
52:23
see a bit about data sampling context length and batch sizes this brings us to the end of this
52:30
lecture thank you so much everyone I really wanted to have a good mix of uh whiteboard writing and whiteboard
52:36
understanding which we did a lot in today's lecture so uh if you see our whiteboard notes we did a number of
52:43
itations of the BP encoder I explained it to you intuitively I hope you understood this please mention in the
52:50
chat if something is unclear and I'll be happy to explain it in the comment section and then we also saw how to code
52:57
the bite pair encoder uh using uh the tick token library in Python so now what
53:04
I encourage you all to do is take some unknown sentences and try using the uh
53:10
this tick token library and the bite pair encoder and try to just see what are the results uh if you if you
53:16
encounter any error for any sentence if the tokenizer is not able to encode it will be awesome and I'll highlight that
53:23
comment in the next video thank you so much everyone and I look forward to seeing you in the next lecture


Show chat replay
