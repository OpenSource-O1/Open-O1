import spaces

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "O1-OPEN/OpenO1-Qwen-7B-v0.1"
# model_name = "O1-OPEN/OpenO1-LLama-8B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@spaces.GPU
def api_call(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def call_gpt(history, prompt):
    return api_call(history+[{"role":"user", "content":prompt}])

if __name__ == "__main__":
    messages = [{"role":"user", "content":"你是谁？"}]
    print(api_call(messages))
    breakpoint()
