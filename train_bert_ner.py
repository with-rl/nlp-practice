import json
from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data_fn, tokenizer, label2idx):
        super().__init__()

        with open(data_fn, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]

        self.tokenizer = tokenizer
        self.labels = [row["labels"] for row in dataset]
        self.tokens = [row["tokens"] for row in dataset]

        self.label2idx = label2idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        self.n_classes = len(self.label2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokens[idx])
        labels = [self.label2idx[t] for t in self.labels[idx]]
        assert len(input_ids) == len(labels)
        return (
            torch.tensor(input_ids),
            torch.tensor([0] * len(input_ids)),  # token_type_ids
            torch.tensor([1] * len(input_ids)),  # attention_mask
            torch.tensor(labels),
        )


class NERCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        input_ids, token_type_ids, attention_mask, labels = zip(*samples)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    # label to index
    with open("./data/NER/label2idx.json", "r") as f:
        label2idx = json.load(f)

    # train dataset
    train_dataset = NERDataset("./data/NER/train_dataset.jsonl", tokenizer, label2idx)
    # valid dataset
    valid_dataset = NERDataset("./data/NER/valid_dataset.jsonl", tokenizer, label2idx)

    # model
    model = AutoModelForTokenClassification.from_pretrained("klue/roberta-base", num_labels=train_dataset.n_classes)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./results/bert-ner",  # 결과 저장 경로
        num_train_epochs=5,  # 학습 횟수
        per_device_train_batch_size=32,  # 학습 배치 크기
        per_device_eval_batch_size=32,  # 평가 배치 크기
        warmup_steps=1000,  # 학습률 워밍업
        weight_decay=0.01,  # 가중치 감쇠
        eval_strategy="epoch",  # 에포크마다 평가
        save_strategy="epoch",  # 에포크마다 저장
        load_best_model_at_end=True,  # 최적 모델 로드
        save_total_limit=3,  # 최대 저장 파라미터 수
        report_to="none",  # 결과 리포트
    )

    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=NERCollator(tokenizer),
    )

    # 학습 실행
    trainer.train()

    # valid dataloader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=NERCollator(tokenizer),
    )

    # eval mode & device
    model.eval()
    device = next(model.parameters()).device

    # evaluate
    total_correct_cnt = 0
    total_sample_cnt = 0

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )
            probs = torch.functional.F.softmax(output.logits, dim=-1)
            prob, pred = torch.max(probs, dim=-1)

            correct_cnt = (pred.cpu() == batch["labels"]).sum().item()
            sample_cnt = (batch["labels"] != -100).sum().item()

            total_correct_cnt += correct_cnt
            total_sample_cnt += sample_cnt

    print(f"Test Accuracy: {total_correct_cnt / total_sample_cnt * 100:.2f}%")
    print(f"Correct / Total: {total_correct_cnt} / {total_sample_cnt}")


if __name__ == "__main__":
    main()
