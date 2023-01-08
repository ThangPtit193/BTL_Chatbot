if __name__ == "__main__":
    import time
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer

    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    prompt = '''Ask: translate to Vietnamese: Who are you? Ans:'''
    time_start = time.time()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=1.0, max_length=50)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
    time_end = time.time()
    cost = time_end - time_start
    # convert to seconds
    print("Cost: %.2f seconds" % cost)