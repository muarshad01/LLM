* Hello everyone welcome to the introductory lecture in this series, which is titled __Build an LLM from Scratch__.

* My name is Dr Raj Dander. I graduated from IIT Madras with a BTech in ME in 2017. After that, I did my PhD from the MIT. I graduated from MIT with a PhD in ML in 2022. Since then I've come back to India and we are all on a mission to basically make AI accessible to everyone at __Vizura__, which is our YouTube channel, we have made several playlists now on ML on DL. The main approach or the philosophy, which we really follow when we teach, is to __teach everything from the basics__, i.e., not to assume anything. Teach you the __nuts and bolts__ of every single concept. 

* The real reason behind this series is that as we all know that LLM and GenAI are transforming everything around us. Startups are forming in this space. Companies are switching to LLMs for various tasks. The jobs in this field are rising a lot. But many people when they are learning about this field are making __one common mistake__: they directly-switch to the applications, they directly-run some __Google-Collab notebook__. They follow some YouTube video and directly-run a code.

* But very few people really understand how LLM actually work? Very few people actually know the __nuts and the bolts__, which make LLM really so powerful.
    * Wouldn't be it amazing if you can build an LLM completely from scratch?
    * Wouldn't that make you feel very confident about this subject?
    * Very few people have this knowledge right now and I'm making this YouTube playlist, which will be a very comprehensive playlist showing you everything about building an LLM from scratch.

* The way I'm making this playlist or the way I will make videos in this playlist is to teach to you everything from the basics as a beginner without assuming anything and at the end of this playlist you will have built an LLM from scratch successfully all by yourself.

* You'll see that after this point everything which comes later, all the application parts,  will just start seeming extremely easy to you. It takes a huge amount of effort on our part to make this series because as you you will see and I'll show you in some time to make every lecture we are going to make detailed lecture notes. I'll share those lecture notes with you and all the videos in this series will be
available completely for free.  

* Okay so now I'm going to tell you a bit about my story of learning LLMS then I'm going to tell you a bit about:
   * What exists right now ( the material which is already available on the internet for learning LLMs)?
   * why it is so inadequate?
   * why it is so insufficient and what we are trying to do new with this playlist?

#### How LLMS looked like in maybe 1960s or where the field of NLP was really about 50 to 60 years back?
* One of the first __Chatbots__, which humans developed, is called as __ELIZA__. It was supposed to be a therapist, and let's see how it works?
   * First choose your language, so I'm choosing English.
   * Eliza: How do you do. Please tell me your problem?
   * You: I am trying to learn about LLMs. But I am finding it difficult. Could you provide some resources for me to start with.
   * Eliza: It is because you're trying to learn about LLM that you came to me.
   * You: yes 
   * Eliza: You seem to be quite positive.
   * You: yes
   * Eliza: You are sure.
* You see this conversation is proceeding nowhere. Thiswas the state of LLMs 50 to 60 years 50 to 60 years back. iIt's not very good right.


#### 5:15: Fast forward to __ChatGPT__
* What are LLMs?
* I am trying to learn about LLMs tell me some resources?
* You'll see the response by chat GPT it's extremely useful, it's to the point and it gives me books, it gives me online courses, it gives me research papers, this is exactly what I need.
* We are living in an age where we should be very lucky that the research on NLP and LLMs is at such a stage where LLM such as GPT are very powerful they are very sophisticated. If you're not familiar with __ChatGPT__ that's fine. We are going to be building our own GPT in this playlist, so you'll learn about it along the way. I showed you this demonstration for you to appreciate the times we are living in right now. LLMS have become really powerful and that's the first motivation to really learn about them.
* There are several more Open Source vs Closed Source things which are happening as I'm making. __Facebook released their Llama 3.1__, which is one of their most capable LLMs up till date and this is an open-source model, which means that the entire architecture of the model is available for free or rather it's available to the public anyone can see the architecture the models released by __Open AI__ are usually closed-source, which means they don't release the the architecture not too many things are known about the model itself.
* Graph which shows closed Source versus open source models in 2022,
when the field of LLMs was booming, most of the things were pretty closed Source. When __GPT4__ was released in 2023, it blew the world away, everyone was surprised and was happy to see the functionality, but still it was a closed Source model. Now in 2024, you can see the gap between the open-source and closed-source model is slowly decreasing and when __Lama 3.1__ is released. You can see that it performs at the same level as __GPT-4__, which is closed Source.
* All the information which you need is available right now as open source models, you should just be willing to learn, if you are not sure what's open-source close-ource models. What's this __Llama 3 405B__. Now text is one thing right

#### What is generative AI?
* There are many other things which GenAI as a field is capable of. There is also a lot of confusion with many people regarding what is GenAI? what are LLMS really? But GenAI is a broader subset and it includes language, video, audio, 3D models, all the things.
* So have a look at some of these videos don't these videos look incredibly realistic. You'll be surprised to know that all of these videos are made
by GenAI. These videos are not shot on camera they are made by AI. This is the power of GenAI.
* Currently, we work with schools. We have developed our own a application (https://vizuara-one.vercel.app/tool). Here ere you can see that there are a huge number of functionalities for example you can click on __MCQ generator__ and you can just type in the topic. Let's say gravity and you can click on generate now you'll see that within a matter of seconds the LLM, which is powering this application, will generate a set of MCQs. See right here we are providing this application to all teachers. This was impossible to even to imagine just four to 5 years back, but now LLMs and GenAI are changing every single thing around us.
* Infact what is probably more relevant to all of you watching this video is the __global GenAI job market__ and just look at this growth it's incredible and this is the projected job market. It's expected to grow about five to six times in the next 5 years __GenAI and LLMs__ is an extremely useful skill and the need for this skill is only going to increase in the future.
* __How to build a LLM from scratch?__
* A course which teaches me the foundations in a lot of detail and in a lot of depth.  I want to know about the foundations because I want to build an LLM from scratch. So that my skill-set is improved and so that I feel confident about giving GenAI and even LLM job interviews. This is what I'm looking for really.

* __Book: Build a Large Language Model (From Scratch) by Sebastian Raschka (Author)__
* Purchase this book as this is going to serve as the reference material for our course.

* Serious people who want to transition to LLMs really and who want to understand LLMs the right way.
*  __LangChain__, is a tool which helps you build LLM apps, but again it's not useful.
*  You need to know how to build a entire LLM from scratch.
*  __positional embedding__ or __positional encoding__ you need to be in a position to say that I have built this from scratch so I know the details better than anyone else.
