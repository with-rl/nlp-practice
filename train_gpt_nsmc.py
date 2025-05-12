from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


class NSMCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return row["document"], row["label"]


class NSMCCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        documents, labels = zip(*batch)

        inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    # hugging face dataset
    dataset = load_dataset("e9t/nsmc")
    # taain dataset
    train_dataset = NSMCDataset(dataset["train"])
    # test dataset
    test_dataset = NSMCDataset(dataset["test"])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    tokenizer.pad_token = "<pad>"

    # model
    model = AutoModelForSequenceClassification.from_pretrained("skt/kogpt2-base-v2", num_labels=2)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./results/gpt-nsmc",  # 결과 저장 경로
        num_train_epochs=3,  # 학습 횟수
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
        eval_dataset=test_dataset,
        data_collator=NSMCCollator(tokenizer),
    )

    # 학습 실행
    trainer.train()

    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=NSMCCollator(tokenizer),
    )

    # test
    model.eval()
    device = next(model.parameters()).device

    total_correct_cnt = 0
    total_sample_cnt = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.functional.F.softmax(output.logits, dim=-1)
            prob, pred = torch.max(probs, dim=-1)

            correct_cnt = (pred.cpu() == batch["labels"]).sum().item()
            sample_cnt = len(batch["labels"])

            total_correct_cnt += correct_cnt
            total_sample_cnt += sample_cnt

    print(f"Test Accuracy: {total_correct_cnt / total_sample_cnt * 100:.2f}%")
    print(f"Correct / Total: {total_correct_cnt} / {total_sample_cnt}")


if __name__ == "__main__":
    main()
