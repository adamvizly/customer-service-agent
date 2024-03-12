<b>the process</b>
<br  />in this script I am trying to fine tune a pre trained model in our case "dialogpt".<br  />
I have chosen this model because of it's light weight and need for much lower processing power<br  />
first I load the dataset conversation.json for fine tuning model and knowledge-base.md for knowlegde embedding.<br  />
I do fine tuning with conversation.json because the purpose of fine tuning a pre trained model is to make the model more task specific<br  />
in this case I want my model to talk like a customer service agent so I feed customer support conversation to the model.<br  />
then because I want my ai agent to be able to understand rules of charge back I create a embedding layer with knowlegde-base data.<br  />
this will be in training and as a result agent will be able to know our rules.<br  />
then I have user pytorch trainer to train this modified model and save it.<br  /><br  />
after that I have created an Interface for the agent to be able to talk to it interactively.<br  />
in this step I have loaded the live-datasource.json file for some data and after that I created a function for responsing based on knowlegde base data<br  />
for given transaction and a generate response function that gets user input and tokenizes it then gives it to the model and gets model's response<br  />
in this part I implemented some string proccessing to be able to find transaction amount and date and information and used that data to find the transaction from datasource<br  />
then generate model's answer based on that transaction.<br  />
that is all.<br  />
