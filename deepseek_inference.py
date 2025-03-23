import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel , get_peft_model

#Clear gpu cache
#torch.cuda.empty_cache()

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main(input_text):
    
    #Load the tokenizer and model from the saved directory
    base_model_name = "deepseek-ai/deepseek-llm-7b-chat"
    lora_path = r'.\finetuned_deepseek'

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the LoRA adapter into the base model
    model = PeftModel.from_pretrained(base_model, lora_path)
    #Merges LoRA into the base model (no need for LoRA layers anymore)
    model = model.merge_and_unload()

    #Calculate the Number of Parameters in the model being used for inference
    total_params = get_model_parameters(model)
    print(f"Total number of paramerers: {total_params}")

    #Model and inputs should be on same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda available ?: ", device)
    print("Model device: ", model.device)
    model.to(device)

    #Prepare the input text you want to generate predictions for
    inputs = tokenizer(input_text, return_tensors='pt').to(device)

    #Generate Text
    outputs = model.generate(**inputs, max_length=1560, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    #Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)


text = "Can you tell me a story"
main(text)


