import os
import transformers
from peft import get_peft_model_state_dict


def train(
    tokenizer,
    model,
    train_data,
    eval_data=None,
    output_dir: str = "output",
    load_checkpoint: str | None = None,
):

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        # fp16=True,
        fp16=False,
        optim="adamw_torch",
        logging_steps=10,
        # evaluation_strategy="steps",
        save_strategy="steps",
        # eval_steps=200,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train(load_checkpoint)
    model.save_pretrained(output_dir)
