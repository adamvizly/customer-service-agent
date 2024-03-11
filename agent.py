import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import datetime


def train():
    # Load conversations
    with open('conversations.json', 'r') as f:
        conversations = json.load(f)['conversations']

    # Preprocess conversations
    conversation_data = []
    for conv in conversations:
        messages = [msg['message'] for msg in conv['messages']]
        conversation_data.append('\n'.join(messages))

    # Load knowledge base content
    with open('knowledge-base.md', 'r') as f:
        knowledge_text = f.read()

    # Combine conversations and knowledge base
    training_data = conversation_data + [knowledge_text]

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    embedding = nn.Embedding(len(tokenizer), model.config.n_embd, padding_idx=tokenizer.pad_token_id)

    # Replace the embedding layer in the model
    model.transformer.wte = embedding

    # Tokenize and encode training data
    def encode_data(text):
        return tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

    encoded_data = [encode_data(text) for text in training_data]

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        gradient_checkpointing=True,
        # fp16=True,
    )

    # Create trainer
    class CustomDataset(Dataset):
        def __init__(self, encoded_data):
            self.encoded_data = encoded_data

        def __len__(self):
            return len(self.encoded_data)

        def __getitem__(self, idx):
            return self.encoded_data[idx]

    dataset = CustomDataset(encoded_data)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model('./fine_tuned_model')
    tokenizer.save_pretrained('./tokenizer')

def conversation():
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('./fine_tuned_model', ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

    # Load the live data source
    with open('live-datasource.json', 'r') as f:
        live_data = json.load(f)

    def apply_knowledge_base(transaction, response):
        """
        Apply knowledge base rules to the response based on the transaction details.
        """
        # Check if the transaction is eligible for chargeback based on the rules
        transaction_date = datetime.strptime(transaction['date'], '%d-%b-%Y')
        if (datetime.now() - transaction_date).days > 90:
            return "Sorry, chargebacks beyond 90 days are not possible."
        if transaction['amount'] > 1000:
            return "Sorry, chargebacks above $1000 are not allowed."
        if transaction['2FA_authorization']:
            return "Sorry, chargebacks for transactions with a valid 3D secure are not allowed."

        return response

    def generate_response(input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Check if the user wants to request a charge back
        if "charge back" in input_text.lower():
            response = "May I know the date, amount and merchant name?"
            print("Agent:", response)
            user_input = input("User: ")
            words = input_text.lower().split()
            amount = date = merchant = transaction = None
            try:
                amount_idx = words.index("amount") + 1
                amount = float(words[amount_idx])
                date_idx = words.index("date") + 2  # Assuming "date is:" format
                date = words[date_idx].replace("'", "")
                merchant_idx = words.index("from") + 1
                merchant = " ".join(words[merchant_idx:])
            except (ValueError, IndexError):
                try:
                    amount_idx = words.index("amount") + 1
                    amount = float(words[amount_idx])
                    merchant_idx = words.index("from") + 1
                    merchant = words[merchant_idx]
                    date_str = " ".join(words[merchant_idx + 1:])
                    date = datetime.strptime(date_str, '%B %d').replace(year=datetime.now().year).strftime('%d-%b-%Y')
                except (ValueError, IndexError):
                    response = "Sorry, I couldn't extract the transaction details from your input. Please provide the amount, date, and merchant name."

            # Find the transaction based on the extracted details
            if amount and date and merchant:
                transaction = next((t for t in live_data if t['amount'] == amount or t['date'] == date or t['merchant'].lower() == merchant.lower()), None)

            if transaction:
                response = apply_knowledge_base(transaction, response)
            else:
                response = "Sorry, I could not find a matching transaction for the provided details. is there other transaction I can help you with?"

        return response

    # Conversation
    greeting = "Hi, What can I help you with?"
    print("Agent:", greeting)

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "bye"]:
            print("Agent:", "Good Bye!")
            break

        response = generate_response(user_input)
        print("Agent:", response)

if __name__ == "__main__":
    train()
    conversation()
