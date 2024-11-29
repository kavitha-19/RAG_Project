# RAG Project with ChromaDB from Scratch which runs locally.

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to enhance the ability of large language models (LLMs) to generate accurate and contextually relevant responses. The system integrates ChromaDB for embedding storage . This solution uses both Gemma and Azure API LLMs for query understanding and response generation. The interface is developed using **Streamlit**, allowing users to interact with the RAG pipeline seamlessly.


## About Dataset 
The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles. The answer to every question is a segment of text, or span, from the corresponding reading passage. There are 100,000+ question-answer pairs on 500+ articles.

There are two files to help you get started with the dataset and evaluate your models: train-v1.1.json and dev-v1.1.json
Dataset link : https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset

## Features
- **Efficient Data Storage**: Uses ChromaDB to manage embeddings.
- **RAG Pipeline**: Retrieves the most relevant context from text chunks to augment responses from LLMs.
- **Interactive UI**: A user-friendly Streamlit interface for querying and visualization.
- **Data Preprocessing**: Includes techniques like sentence segmentation and chunking.
- **Impact Analysis**: Assesses the effects of chunking on question clarity and model coherence.

## Project Workflow
1. **Data Initialization**:
   - **ChromaDB** is initialized to store and manage embeddings efficiently.
2. **Data Exploration**:
   - Analyzed the dataset for structure, quality, and patterns.
   - Identified preprocessing needs, including sentence segmentation and chunking.
3. **Preprocessing**:
   - Performed text formatting to clean and standardize the data.
   - Applied sentence segmentation to split large text into smaller units.
   - Chunked relevant context fields to ensure coherent retrieval.
4. **Embedding Generation**:
   - Generated semantic embeddings using **Gemma LLM** and **Azure API LLM**.
5. **Embedding Storage**:
   - Stored the embeddings in **ChromaDB** with indexing for efficient similarity searches.
6. **Similarity Search for Retrieval**:
   - Queries are embedded and matched against stored embeddings in ChromaDB to retrieve relevant context.
7. **RAG Pipeline**:
   - Augmented user queries with the retrieved context.
   - Used **Gemma LLM** for initial response generation.
   - **Azure API LLM** refined the responses for higher accuracy.
8. **Avoiding Over-Chunking**:
   - Carefully balanced chunking to avoid fragmented information and maintain coherence.
9. **Interactive Streamlit UI**:
   - Provides a user-friendly interface to input queries and view responses.
   - Displays the retrieved context used to generate the final response.

## Technologies Used
- **ChromaDB**: For embedding storage and similarity search.
- **Gemma LLM**: A smaller model used for initial embeddings and query responses.
- **Azure API LLM**: A larger model for refined and accurate responses.
- **Streamlit**: For building an interactive and dynamic user interface.
- **SpaCy**: For text segmentation and preprocessing tasks.
- **Hugging Face Transformers**: Integrated for model experimentation and embedding generation.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   
2. Install required dependencies:
    ```bash
   pip install -r requirements.txt

4. Run the Streamlit app:
   ```bash
   streamlit run app.py


