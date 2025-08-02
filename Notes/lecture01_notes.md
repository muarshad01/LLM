* Hello everyone welcome to the introductory lecture in this series, which is titled __Build an LLM from Scratch__.

* My name is Dr Raj Dander. I graduated from IIT Madras with a BTech in ME in 2017. After that, I did my PhD from the MIT. I graduated from MIT with a PhD in ML in 2022.

* Since then I've come back to India and we are all on a mission to basically make AI accessible to everyone U at viua which is our YouTube channel. We have made several playlists now on ML on DL 
and uh the main approach or the philosophy, which we really follow when we teach, is to teach-everything-from-the-basics not to assume anything. Teach you the nuts-and-bolts of every single concept. 

* The real reason behind this series is that as we all know that LLM and GenAI are transforming everything around us. Startups are forming in this space. Companies are switching to LLMs for various tasks. 

* The jobs in this field are rising a lot but many people when they are learning about this field are making one-common-mistake that they directly-switch to the applications. They directly-run some __Google-Collab__ notebook. They follow some YouTube video and directly-run a code uh but very few people really understand how LLM actually work. 

* Very few people actually know the nuts-and-the-bolts, which make LLM really so powerful. Wouldn't be it amazing if you can build an LLM completely from scratch? Wouldn't that make you feel very confident about this subject? Very few people have this knowledge right now and I'm making this YouTube playlist which will be a very comprehensive playlist showing you everything about building an
LLM from scratch.

* The way I'm making this playlist or the way I will make videos in this playlist is to teach to you everything from the basics as a beginner without assuming anything and at the end of this playlist you will have built an LLM from scratch successfully all by yourself.

* You'll see that after this point everything which comes later all the application Parts everything will just start seeming extremely easy to you so that's the whole philosophy behind making this lecture series it takes a huge amount of effort on our part to make this series because as you you will see and I'll show you in some time to make every lecture we are going to make detailed lecture notes.

I'll share those lecture notes with you and all the videos in this series will be
available completely for free.  Okay so now I'm going to tell you a bit about my story of learning LLMS then I'm going to tell you a bit about what exists right now the material which is already available on the internet for learning LLMs why it is so inadequate? why it is so insufficient and what we are trying to do new with this playlist?  so let's get into the video so let me Demonstration
take you back in time a bit to show you how LLMS looked like in maybe 1960s uh or where the field of NLP was really about 50 to 60 years back so this is one of the first __Chatbots__ which humans developed it's called as Elisa it was supposed to be a therapist and let's see how it works so first this interface asks me to choose my language so I'm choosing English. let's
go so then Elisa tells me how do you do please tell me your problem so then let
me type I am trying to learn about LLMs but I am finding it difficult could you provide some resources for me to start with then Elisa asks is it because
you're trying to learn about LLM that you came to me then I say yes and then Alisa says You seem to be quite positive and then and then Elisa
says you are sure you see this conversation is proceeding nowhere this
was the state of LLMs 50 to 60 years 50 to 60 years back it's
not very good right.

* Fast forward to __ChatGPT__ and you ask what are LLMs or let me ask the same thing which I asked over here I am trying to learn about LLMs tell me me some
resources and then you'll see the response by chat GPT it's extremely
useful it's to the point and uh it gives me books it gives me online courses it
gives me research papers this is exactly what I need so you see this simple
illustration shows that we are living in an age where we should be very lucky
that the research on NLP and LLMs is at such a stage where LLM such as GPT are very powerful they are very sophisticated if you're not familiar with chat GPT that's fine we are going to be building our own GPT in this playlist so you'll learn about it along the way but I showed you this demonstration for you to appreciate the times we are living in right now LLMS have become really powerful and uh that's the first motivation to really learn about them there are several more Open Source vs Closed Source things which are happening as I'm making
this video __Facebook released their Lama 3.1__ which is one of their most capable LLMs up till date and this is an open-source model which means that the entire architecture of the model is available for free uh or rather it's available to the public anyone can see the architecture the models released
by __Open AI__ are usually closed source, which means they don't release the
weights uh the architecture not too many things are known about the model itself
so here is a graph which shows closed Source versus open source models in 2022
when the field of LLMs was booming most of the things were pretty closed Source when gp4 was released in 2023 it blew the world uh away everyone was surprised everyone was happy to see the functionality but still it was a closed Source model now in 2024 can you see the gap between the open source and closed Source model is slowly decreasing and when Lama 3.1 is released you can see that it performs at the same level as __GPT-4__ which is closed Source this is to say that all the information which you need is available right now as open source models you should just be willing to learn if you are not sure what's open source close Source models what's this Lama 340 5B

* I'll explain to you about all of these things as we proceed with the
lectures Now text is one thing right

* __What is generative AI__ there are many other things which
generative AI as a field is capable ofthere is also a lot of confusion with
many people regarding what is generativeAI what are LLMS really but GenAI is a broader subset and it includes language it includes video audio 3D models all the things so have a look at some of these videos don't these videos look
incredibly realistic this video then this video. The Waves video this particular video you'll be surprised to know that all of these videos are made
by AI these videos are not shot on camera they are made by AI this is the power of GenAI currently finally uh when we work with schools we have developed our own a application so this is viu's AI application uh or application on llms
here you can see that there are a huge number of functionalities for example you can click on McQ generator and you can just type in the topic let's say gravity and you can click on generate now you'll see that within a matter of seconds the LLM which is powering this application will generate a set of multiple choice questions see right here we are providing this application to teachers all of this was impossible to even
imagine uh just four to 5 years back but now LLMs and generative AI are changing every single thing around us in fact what is probably more relevant to all of you watching this video is the global generative AI job market and just look at this growth it's
incredible uh and this is the pro projected job market it's expected to grow about five to six times in the next 5 years generate and LLMs is an extremely useful skill and the need for this skill is only going to increase in the future so this again
brings me to the question that if someone wants to learn about LLMs how do they go about doing this so let's say you go to Google and you search build llms learn which say which means you want to learn about LLMs there are a number of courses which show up really over here now if you go to many of these courses you will see let let say build llm apps it's about app development it does not teach you __how to build a LLM from scratch__ here is another course Master SL

* Master LLM Concepts if you look at this course description
right they don't teach you how to build an llm from scratch at all they don't teach you the nuts-and-bolts it's a pretty quick course this is also not what I'm looking for what I'm looking
for is one course which teaches me the foundations in a lot of detail and in a lot of depth I want to know about the foundations because I want to build an llm from scratch so that uh my skill set is improved and so that I feel confident
about giving GenAI and even LLM job interviews this is what I'm looking for really I don't want a quick crash course I want something in depth then I go to YouTube and search about llm from scratch this is the first video which shows up or rather it's a first playlist you'll see that there are 18 chapters in this playlist but each of these chapters is again only 10 15
minutes and again I'm a bit demotivated seeing this because I'm not looking for this either I'm looking for a massive deep technical course which teaches mehow to build an llm right from the very Basics there is Andre karpat is building a GPT course but if you look at this course it's actually quite complex it's not an easy course at all he starts right in the middle of uh a concept and it's not meant for beginners and it's just 90 minutes again this is not what I'm looking for here is another create a LLM from scratch but you see the red bar here I tried taking this course and I finished three hours of this but again this is not explained well it's not explained from scratch I want to make a course which people understand right from the very Basics and none of the courses on YouTube on Google are satisfying that need luckily very luckily I came across this book by an author called "Sebastian Rashka" I think it's one of the best books on LLM I purchased thisbook and this is going to serve as the reference material for our course what.

I'm going to do is that this is a 48page book I'm going to convert every
single aspect of this book into a set of
videos I'll probably have 35 40 or maybe
50 videos in this playlist similar to my
playlist for ML and DL and uh whatever is given here right I'll convert it into
uh I'll convert it into video format
so for example here are some of the
notes which I've already started to make
we'll we'll start covering this when we
move to uh the next lecture look at
these notes what I've done over here is
that I have built my understanding from
this book over here and to to transfer
my understanding to all of you I have
started writing on a white board every
single thing in detail look at this and
I'm trying to make it as interesting as
possible and as fundamental as possible
so this is going to be the next lecture
uh intro to llms so I finished making
the notes for this and I finished making
the notes for the lecture after that as
well stages of building and llm so
basically I'm making I'm in the process
of making all these notes but I'm making
the videos simultaneously so that I am
also motivated and I am also on track
eventually this will be a massive
lecture notes of maybe 200 to 300 pages
and uh there will be 50 to 60 videos
based on this you'll learn everything
right from the very Basics nothing will
be assumed everything will be spelled
out and these set of videos will be
perfect for serious people who want to
transition to LLMs
really and we want to understand llms
the right
way uh okay so this is the main idea for
the course

another thing which I want to say is that you might have heard of
applications like LangChain this is a tool which helps you build llm apps but
again it's not useful if you don't know how to build an llm yourself many
students directly start deploying apps as I mentioned before but I personally
don't think that's the right way to learn about llms you need to know how to
build a entire LLM from scratch and we'll teach you that in this
playlist finally the goal is to equip
all of you so that you feel confident in
front of an interviewer right if you
have just deployed applications by
cloning some GitHub repository then if
someone asks you a detailed question
about key query and values let's say or
positional embedding or positional
positional encoding you need to be in a
position to say that I have built this
from scratch so I know the details
better than anyone
else that's really going to position you
in a different boot than all the other
people so this is the main objective of
this course this is going to be the
outline of this course we are going to
follow uh all the top table of contents
which have been mentioned over here and
I'm going to convert those into video
lecture formats and it's going to be an
awesome series it's going to be a
detailed Series so if this seems of
interest to you please uh mention in the
comments so that I also feel motivated
to build subsequent videos and as I
mentioned the plan is to release
everything all of this content
completely for free so that you all can
benefit from
it okay thank you so much everyone uh
and I look forward to seeing you in the
next lecture where we will cover
introduction to LLMs
and we'll cover all these things what
you are seeing in the board right now
thanks everyone see you in the next



