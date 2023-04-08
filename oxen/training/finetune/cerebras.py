import os
import transformers
from peft import get_peft_model_state_dict

def train(
    tokenizer,
    model,
    train_data,
    eval_data=None,
    enable_wandb: bool = False,
    output_dir: str = "output",
    load_checkpoint: str | None = None
):
    # print("Training... with wandb: ", enable_wandb)
    # if enable_wandb:
    #     import wandb
    #     wandb_run_name = f"{wandb.util.generate_id()}"

    #     # set the wandb project where this run will be logged
    #     os.environ["WANDB_PROJECT"]="cerebras-fine-tune"

    #     # save your trained model checkpoint to wandb
    #     os.environ["WANDB_LOG_MODEL"]="true"

    #     # turn off watch to log faster
    #     os.environ["WANDB_WATCH"]="false"

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=8,  
        num_train_epochs=3,  
        learning_rate=1e-4, 
        # fp16=True,
        fp16=False,
        optim="adamw_torch",
        logging_steps=10, 
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=output_dir, 
        save_total_limit=3,
        # report_to="wandb" if enable_wandb else None,
        # run_name=wandb_run_name if enable_wandb else None,
    )
    
    trainer = transformers.Trainer(
        model=model, 
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args, 
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )

    model.config.use_cache = False
    
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train(load_checkpoint)
    model.save_pretrained(output_dir)

    # if enable_wandb:
    #     import wandb
    #     wandb.finish()