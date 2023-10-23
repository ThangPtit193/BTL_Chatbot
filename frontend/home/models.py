import torch

def get_answer(outputs, tokenizer):
    answer = ""
    for output in outputs:
        id = torch.argmax(output).item()
        if output[id] == 0:
            continue
        answer = answer + " " + tokenizer.get_id(id)

    return answer
