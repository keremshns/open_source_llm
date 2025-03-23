import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main(input_text):
    
    # Load the tokenizer and model from the saved directory
    model_path = '/mnt/disks/disk1/results/model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Calculate the Number of Parameters in the model being used for inference
    total_params = get_model_parameters(model)
    print(f"Total number of paramerers: {total_params}")

    # Prepare the input text you want to generate predictions for
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate Text
    outputs = model.generate(**inputs, max_length=250, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)


text = "Can you tell me a story"
main(text)

