from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()
model.to("cpu")

# langs = tokenizer.additional_special_tokens
# print(f"Total languages: {langs}")


def translate(text, src="eng_Latn", tgt="hin_Deva"):
    tokenizer.src_lang = src

    inputs = tokenizer(text, return_tensors="pt")

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=64,
        num_beams=1,
        do_sample=False,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# print(translate("I love pizza", tgt="ace_Arab"))
