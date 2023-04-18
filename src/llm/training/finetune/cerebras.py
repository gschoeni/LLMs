import os
import transformers
from peft import get_peft_model_state_dict


def train(
    tokenizer,
    model,
    train_data,
    epochs: int = 3,
    save_steps: int = 200,
    device: str = "cuda",
    output_dir: str = "output",
    load_checkpoint: str | None = None,
):

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=1e-4,
        fp16=True if device == "cuda" else False,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=3,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
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
