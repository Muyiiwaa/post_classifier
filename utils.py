import torch
from typing import Tuple

def classifier_function(text: str, tokenizer, model, device) -> Tuple:
    # Tokenize the text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # Perform inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Convert logits to probabilities using softmax
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)

    # Get the predicted class (0 for not spam, 1 for spam)
    prob, prediction = torch.max(probs, 1)
    probs = prob.item()

    # Return the prediction
    return probs, prediction