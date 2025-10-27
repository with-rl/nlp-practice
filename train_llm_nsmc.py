import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


# 학습을 위한 prompt를 생성 함수.
def gen_train_prompt(example):
    doc = example["document"]
    label = "긍정" if example["label"] == 1 else "부정"
    prompt = r"""<bos><start_of_turn>user
다음 문장은 영화리뷰입니다. 긍정 또는 부정으로 분류해주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(
        doc, label
    )
    return prompt


def main():
    MODEL_ID = "google/gemma-3-1b-it"
    # dataset
    dataset = load_dataset("Blpeng/nsmc")

    # declare 4 bits quantize
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    # load 4 bits model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", quantization_config=quantization_config, attn_implementation='eager'
    )
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"

    # lora config
    lora_config = LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # train args
    training_args = TrainingArguments(
        output_dir="./results/gemma-nsmc",  # 결과 저장 경로
        # num_train_epochs=3,                       # epoc으로 할 경우 너무 많이 걸리 수 있음
        max_steps=1000,  # 학습 step 수 (1000 ~ 2000 사이)
        per_device_train_batch_size=4,  # gpu당 입력 batch_size
        gradient_accumulation_steps=4,  # gradient 누적 후 학습
        optim="paged_adamw_8bit",  # optimizer (QLoRA)
        warmup_steps=500,  # learning rate warmup step
        learning_rate=1e-4,  # learning rate
        # bf16=True,                               # bf16 사용 여부 (3090 이상에서 가능)
        fp16=True,  # fp16 사용 여부 (예전 GPU에서 사용 가능, T4)
        logging_steps=100,  # 얼마만에 한번 씩 중간 결과를 확인할 것인가?
        save_steps=500,  # 얼마만에 한번 씩 파라미터를 저장할 것인가?
        report_to="none",  # 결과 리포트
    )

    # trainer 정의
    trainer = SFTTrainer(
        model=model,  # 학습할 모델
        train_dataset=dataset["train"],  # 학습할 데이터 셋
        args=training_args,
        peft_config=lora_config,  # QLoRA config
        formatting_func=gen_train_prompt,  # 프롬프트 생성 함수
    )

    # train
    trainer.train()

    # save lora (delta weight)
    trainer.model.save_pretrained("./results/gemma-nsmc/lora_weight")

    # original model load (before finetuned)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16, attn_implementation='eager')

    # merge : original + delta wieght
    model = PeftModel.from_pretrained(
        model, "./results/gemma-nsmc/lora_weight", device_map="auto", torch_dtype=torch.float16
    )
    model = model.merge_and_unload()

    # save fine-tunned model
    model.save_pretrained("./results/gemma-nsmc/merged")


if __name__ == "__main__":
    main()
