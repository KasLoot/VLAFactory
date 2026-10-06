"""Small cached generation loop supporting greedy and temperature/top-p/min-p sampling."""

import torch
from LFM_ACT.cache import ModelCache
from LFM_ACT import ModelConfig, LFMModel, load_weights, LFMTokenizer
from LFM_ACT.processor import Processor

@torch.inference_mode()
def generate(
    model,
    input_ids,
    *,
    attention_mask=None,
    max_new_tokens=64,
    temperature=0.0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    generator=None,
    **image_inputs,
):
    if (
        max_new_tokens < 0
        or temperature < 0
        or not 0 < top_p <= 1
        or not 0 <= min_p <= 1
        or repetition_penalty <= 0
    ):
        raise ValueError("Invalid generation parameters")
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if not attention_mask[:, -1].all():
        raise ValueError("Generation requires left-padded prompts")
    if not max_new_tokens:
        return input_ids
    if input_ids.shape[1] + max_new_tokens > model.config.context_length:
        raise ValueError("Requested generation exceeds context_length")
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        cache = ModelCache()
        sequence = input_ids
        current = input_ids
        done = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.bool
        )
        for step in range(max_new_tokens):
            output = model(
                current,
                attention_mask=attention_mask,
                cache=cache,
                logits_to_keep=1,
                **(image_inputs if step == 0 else {}),
            )
            logits = output.logits[:, -1].float()
            if repetition_penalty != 1:
                for row in range(len(sequence)):
                    seen = sequence[row][attention_mask[row].bool()].unique()
                    scores = logits[row, seen]
                    logits[row, seen] = torch.where(
                        scores < 0,
                        scores * repetition_penalty,
                        scores / repetition_penalty,
                    )
            if temperature == 0:
                token = logits.argmax(-1)
            else:
                probabilities = (logits / temperature).softmax(-1)
                probabilities.masked_fill_(
                    probabilities < min_p * probabilities.amax(-1, keepdim=True), 0
                )
                probabilities = probabilities / probabilities.sum(-1, keepdim=True)
                if top_p < 1:
                    sorted_probs, indices = probabilities.sort(descending=True)
                    remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
                    sorted_probs.masked_fill_(remove, 0)
                    probabilities = torch.zeros_like(probabilities).scatter(
                        -1, indices, sorted_probs
                    )
                token = torch.multinomial(
                    probabilities, 1, generator=generator
                ).squeeze(-1)
            token = torch.where(done, model.config.pad_token_id, token)
            sequence = torch.cat((sequence, token[:, None]), dim=-1)
            attention_mask = torch.cat((attention_mask, (~done)[:, None]), dim=-1)
            done |= token == model.config.eos_token_id
            if done.all():
                break
            current = token[:, None]
        return sequence
    finally:
        for module, training in training_states:
            module.training = training



def main():
    checkpoint = "/home/yuxin/workspace/data/models/LFM2.5-VL-450M"
    config = ModelConfig.from_pretrained(checkpoint)
    model = LFMModel(config).to(device="cuda", dtype=torch.bfloat16)
    print(load_weights(model, checkpoint))
    # loaded=349, missing=0, unused=0, skipped=0, incompatible=0

    tokenizer = LFMTokenizer.from_pretrained()  # LFM_ACT/tokenizer/, entirely local
    processor = Processor(tokenizer, config.image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image/butterfly.jpg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    batch = processor.apply_chat_template(messages).to("cuda", torch.bfloat16)
    output_ids = generate(model, **batch, max_new_tokens=1024)
    print(
        tokenizer.decode(
            output_ids[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    main()