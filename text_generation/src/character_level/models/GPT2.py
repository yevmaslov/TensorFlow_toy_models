from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

EOS_TOKEN = 50256


def get_pretrained_model():
    gpt2 = TFGPT2LMHeadModel.from_pretrained("gpt2",
                                             pad_token_id=EOS_TOKEN)
    return gpt2


def generate_text_greedy(model, start_string):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(start_string, return_tensors='tf')
    greedy_output = model.generate(input_ids,
                                   max_length=150,
                                   top_k=25,
                                   temperature=2,
                                   no_repeat_ngram_size=3,
                                   early_stopping=True)
    return tokenizer.decode(greedy_output[0], skip_special_tokens=True)

