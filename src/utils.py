import os
import json
import torch

def save_model_and_tokenizer(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
