from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import pandas as pd

  # Load dataset
df = pd.read_csv("explanation_dataset.csv")
with open("training_data.txt", "w") as f:
    for _, row in df.iterrows():
        f.write(f"INPUT: {row['input']}\nOUTPUT: {row['output']}\n\n")

  # Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Add padding token

  # Prepare dataset
def load_dataset(file_path, tokenizer):
      return TextDataset(
          tokenizer=tokenizer,
          file_path=file_path,
          block_size=128
      )

dataset = load_dataset("training_data.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

  # Training arguments
training_args = TrainingArguments(
      output_dir="./custom_llm",
      overwrite_output_dir=True,
      num_train_epochs=3,
      per_device_train_batch_size=4,
      save_steps=10_000,
      save_total_limit=2,
      logging_dir='./logs',
  )

  # Initialize trainer
trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=dataset,
  )

  # Train the model
trainer.train()
model.save_pretrained("./custom_llm")
tokenizer.save_pretrained("./custom_llm")