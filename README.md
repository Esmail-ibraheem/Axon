# Axon 🧠🔬
# <center style="font-family: consolas; font-size: 32px; font-weight: bold;">  Prompt Engineering Best Practices: Building An End to End Customer Service System </center>

***


Prompt engineering plays a pivotal role in crafting queries that help large language models (LLMs) understand not just the language but also the nuance and intent behind the query and help us build complex applications with ease. 

In this notebook, we will put into action what we covered in previous notebooks and build an end-to-end customer service assistant. Starting with checking  the input to see if it flags the moderation API then extracting the list of products searching for the products the user asked about answering the user question with the model and checking the output with the moderation API. 

Finally, we will put all of these together and build a conversational chatbot the takes the user input passes it through all of these steps, and returns it back to him.

#### <a id="top"></a>
# <div style="box-shadow: rgb(60, 121, 245) 0px 0px 0px 3px inset, rgb(255, 255, 255) 10px -10px 0px -3px, rgb(31, 193, 27) 10px -10px, rgb(255, 255, 255) 20px -20px 0px -3px, rgb(255, 217, 19) 20px -20px, rgb(255, 255, 255) 30px -30px 0px -3px, rgb(255, 156, 85) 30px -30px, rgb(255, 255, 255) 40px -40px 0px -3px, rgb(255, 85, 85) 40px -40px; padding:20px; margin-right: 40px; font-size:30px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(60, 121, 245);"><b>Table of contents</b></div>

<div style="background-color: rgba(60, 121, 245, 0.03); padding:30px; font-size:15px; font-family: consolas;">
<ul>
    <li><a href="#1" target="_self" rel=" noreferrer nofollow">1. Setting Up Working Environment </a> </li>
    <li><a href="#2" target="_self" rel=" noreferrer nofollow">2. Chain of Prompts For Processing the User Query </a></li>
    <li><a href="#3" target="_self" rel=" noreferrer nofollow">3. Building Conversational Chatbot </a></li> 
</ul>
</div>

***
