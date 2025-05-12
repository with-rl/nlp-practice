import os

from tokenizers import ByteLevelBPETokenizer  # CharBPETokenizer


def main():
    # reserved 100 tokens
    unused_tokens = [f"<unused_{i}>" for i in range(100)]

    # tokenizer 학습
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[
            "data/aihub_koen/train.ko",
            "data/aihub_koen/train.en"
        ],
        vocab_size=32000,
        min_frequency=2,
        special_tokens=[
            "<pad>",  # padding
            "<s>",  # start of sentence
            "</s>",  # end of sentence
            "<unk>",  # unknown
        ]
        + unused_tokens,
        show_progress=True,
    )
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    # save tokenizer
    output_dir = "data/aihub_koen_32k"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))


if __name__ == "__main__":
    main()
