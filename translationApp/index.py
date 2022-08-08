from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", output_attentions=True)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

app = Flask(__name__)
CORS(app)

@app.route("/translate", methods=['GET', 'POST'])
def translate():
    text = request.json["text"]
    print(text)
    english = text

    tokenizer.src_lang = "en_XX"
    encoded_en = tokenizer(english, return_tensors="pt")
    print(encoded_en)
    generated_tokens = model.generate(
        **encoded_en,
        forced_bos_token_id=tokenizer.lang_code_to_id["ne_NP"]
    )
    print(generated_tokens)
    np = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(np)
    encoder_input_ids = tokenizer(english, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = tokenizer(np, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)

    encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
    decoder_text = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
    print(decoder_text)
    include_layers = list(range(len(outputs.cross_attentions)))
    attentionsList = format_attention(outputs.cross_attentions, include_layers)
    attn_data = []

    attn_data.append(
        {
            'attn': attentionsList,
            'source': decoder_text,
            'target': encoder_text
        }
    )



    result = {
        'attention': attn_data,
    }

    return jsonify(result)


def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    return torch.stack(squeezed)

if __name__ == "__main__":
    app.run()
