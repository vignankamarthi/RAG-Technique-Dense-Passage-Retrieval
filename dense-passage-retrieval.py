from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Loading the pre-trained dense-passage-retrieval
# models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

# Encoding a query
query = "Who invented the first compiler?"
#query2 = "African capital cities?"
question_inputs = question_tokenizer(query, return_tensors="pt")
question_embedding = question_encoder(
    input_ids=question_inputs["input_ids"], 
    attention_mask=question_inputs["attention_mask"]
    ).pooler_output

# Encoding Passages
passages = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "Maputo is the capital of Mozambique.",
    "To be or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "Grace Hopper was an American computer scientist and United States Navy rear admiral. who was a pioneer of computer programming, and one of the first programmers of the Harvard Mark I computer. inventor of the first compiler for a computer programming language.",
]

# Generating the embeddings for the context
context_embeddings = []
for passage in passages:
    context_inputs = context_tokenizer(passage, return_tensors="pt")
    context_embedding = context_encoder(
        input_ids=context_inputs["input_ids"],
        attention_mask=context_inputs["attention_mask"]
    ).pooler_output
    context_embeddings.append(context_embedding)

context_embeddings = torch.cat(context_embeddings, dim=0)

# Compute the cosine similarities of the question embeddings 
# and the context embeddings
similarities = cosine_similarity(
    question_embedding.detach().numpy(), context_embeddings.detach().numpy()
)
print("Similarities:", similarities)

# Get the most relevant passage
most_relevant_idx = np.argmax(similarities)
print("Most relevant passage:", passages[most_relevant_idx])
