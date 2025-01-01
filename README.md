# Dense Passage Retrieval: Advanced RAG Techniques

This project demonstrates a simplified implementation of **Dense Passage Retrieval (DPR)**, an effective approach for extracting relevant passages from a document collection based on a query. By utilizing **Hugging Face’s DPR models** and **cosine similarity**, this project identifies and retrieves the most relevant passage to answer a given query.

### Workflow Overview:
1. **Encode the Query**: The query is encoded into a dense vector representation using a pre-trained DPRQuestionEncoder model.
2. **Encode the Passages**: Each passage in the document collection is similarly encoded using a DPRContextEncoder model to produce dense vector representations.
3. **Compute Similarities**: Cosine similarity is computed between the query vector and each passage vector to determine relevance.
4. **Retrieve the Most Relevant Passage**:  The passage with the highest similarity score is selected as the most relevant.

This simplified pipeline highlights the power of dense vector representations in information retrieval tasks.

## Workflow Overview:

Here is an example output from the project:
- Query: “Who invented the first compiler?”
- Most Relevant Passage: “Grace Hopper was an American computer scientist and United States Navy rear admiral, who was a pioneer of computer programming, and one of the first programmers of the Harvard Mark I computer. She was the inventor of the first compiler for a computer programming language.”

## Getting Started

Follow these steps to set up the project and run the example:

1. **Clone the repository**:
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install Dependencies
    Install the required dependencies using the requirements.txt file:
    ```bash
    pip install -r requirements.txt

3. Run the Project
    ```bash
    python dense-passage-retrieval.py

## Key Concepts

**Dense Vectors**
Dense vectors are numerical representations of text that capture semantic meaning. In this project, the DPRQuestionEncoder and DPRContextEncoder transform the query and passages into dense vector formats.

**Cosine Similarity**
Cosine similarity is used to measure the relevance between the query and passages by comparing their vector representations. The passage with the highest similarity score is retrieved.

## Conclusions
This implementation demonstrates how **dense passage retrieval (DPR)** can effectively identify the most relevant passage for a given query. By leveraging pre-trained models and cosine similarity, the system achieves high retrieval precision. The results show a clear alignment between the query and the retrieved passage, as evidenced by the most relevant passage output.

This simplified example serves as a foundation for more advanced retrieval-augmented generation techniques.
