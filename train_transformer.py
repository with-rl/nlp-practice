import torch

from transformers import (
    T5TokenizerFast,
    AutoConfig,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# dataset class
class NMTDataset(torch.utils.data.Dataset):
    def __init__(self, src_fn, tgt_fn, bos="<s>", eos="</s>"):
        self.bos = bos
        self.eos = eos
        self.src_data = []
        self.tgt_data = []

        with open(src_fn) as f:
            for line in f:
                self.src_data.append(line.strip())

        with open(tgt_fn) as f:
            for line in f:
                self.tgt_data.append(line.strip())

        assert len(self.src_data) == len(self.tgt_data)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return (
            self.src_data[idx],
            f"{self.bos}{self.tgt_data[idx]}",
            f"{self.tgt_data[idx]}{self.eos}",
        )


# collator class
class NMTCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc_batch, dec_batch, tgt_batch = zip(*batch)

        encoder_input = self.tokenizer(
            enc_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        decoder_input = self.tokenizer(
            dec_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        decoder_labels = self.tokenizer(
            tgt_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        decoder_labels = decoder_labels["input_ids"]
        decoder_labels[decoder_labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoder_input["input_ids"],
            "attention_mask": encoder_input["attention_mask"],
            "decoder_input_ids": decoder_input["input_ids"],
            "labels": decoder_labels,
        }


def main():
    # train dataset
    train_dataset = NMTDataset("data/aihub_koen/train.ko", "data/aihub_koen/train.en")
    # valid dataset
    valid_dataset = NMTDataset("data/aihub_koen/valid.ko", "data/aihub_koen/valid.en")

    # 학습된 tokenizer 로딩
    tokenizer = T5TokenizerFast.from_pretrained("data/aihub_koen_32k")
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    # 모델 config 로딩
    model_config = AutoConfig.from_pretrained(
        "google-t5/t5-small",
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 파라미터가 랜덤 초기화된 모델 로딩
    model = T5ForConditionalGeneration(model_config)

    # train args class
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/t5-nmt",
        overwrite_output_dir=True,
        eval_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=5.0,
        num_train_epochs=3,
        lr_scheduler_type="linear",
        warmup_steps=1000,
        logging_strategy="steps",
        logging_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        # bf16=True,
        # bf16_full_eval=True,
        fp16=True,
        fp16_full_eval=True,
        half_precision_backend="auto",
        eval_steps=5000,
        load_best_model_at_end=True,
        report_to="none",
    )

    # trainer class
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=NMTCollator(tokenizer),
    )

    # train
    trainer.train()


if __name__ == "__main__":
    main()
