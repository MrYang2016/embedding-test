from flask import Flask, request, jsonify
from typing import Union
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
# import os

# proxies = {
#     "http": "http://localhost:1087",
#     "https": "http://localhost:1087",
# }

# 设置环境变量
# os.environ['HTTP_PROXY'] = proxies['http']
# os.environ['HTTPS_PROXY'] = proxies['https']

model = AutoModel.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0", proxies=proxies)
tokenizer = AutoTokenizer.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")


def get_embedding(text: Union[str, list[str]], mode: str = "sentence"):
    model.eval()

    assert mode in ("query", "sentence"), f"mode={mode} was passed but only `query` and `sentence` are the supported modes."

    if isinstance(text, str):
        text = [text]

    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output = model(**inp)

    # The model is optimized to use the mean pooling for queries,
    # while the sentence / document embedding uses the [CLS] representation.

    if mode == "query":
        vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
        vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
    else:
        vectors = output.last_hidden_state[:, 0, :]

    return vectors


# texts = [
#     "Illustration of the REaLTabFormer model. The left block shows the non-relational tabular data model using GPT-2 with a causal LM head. In contrast, the right block shows how a relational dataset's child table is modeled using a sequence-to-sequence (Seq2Seq) model. The Seq2Seq model uses the observations in the parent table to condition the generation of the observations in the child table. The trained GPT-2 model on the parent table, with weights frozen, is also used as the encoder in the Seq2Seq model.",
#     "Predicting human mobility holds significant practical value, with applications ranging from enhancing disaster risk planning to simulating epidemic spread. In this paper, we present the GeoFormer, a decoder-only transformer model adapted from the GPT architecture to forecast human mobility.",
#     "As the economies of Southeast Asia continue adopting digital technologies, policy makers increasingly ask how to prepare the workforce for emerging labor demands. However, little is known about the skills that workers need to adapt to these changes"
# ]

# Compute embeddings
# embeddings = get_embedding(texts, mode="sentence")

# # Compute cosine-similarity for each pair of sentences
# scores = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
# print(scores.cpu().numpy())

# # Test the retrieval performance.
# query = get_embedding("Which sentence talks about concept on jobs?", mode="query")

# scores = F.cosine_similarity(query, embeddings, dim=-1)
# print(scores.cpu().numpy())

# embeddings2 = get_embedding([
#     "Which sentence talks about concept on jobs?", 
#     "Which sentence talks about concept on jobs, Joe?"
# ], mode="sentence")

# # 打印embeddings2长度
# print(len(embeddings2[0]))
# print(len(embeddings2[1]))

# scores2 = F.cosine_similarity(embeddings2[0], embeddings2[1], dim=-1)
# print(scores2.cpu().numpy())

# embeddings3 = get_embedding("Which sentence talks about concept on jobs, Joe?", mode="sentence")
# print(embeddings3)

app = Flask(__name__)

@app.route('/embedding', methods=['GET'])
def get_embedding_api():
    text = request.args.get('text')
    mode = request.args.get('mode', 'sentence')
    embeddings = get_embedding(text, mode)
    return jsonify(embeddings[0].tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)