from pyexpat import model
from sre_parse import Tokenizer
from typing import Collection
from sentence_transformers import SentenceTransformer, util # type: ignore
import torch
import os
import openai

# Define the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to retrieve the relevant resources based on the query
def retrieve_relevant_resources(query: str,
                                collection,  # Collection to query from
                                model: SentenceTransformer = embedding_model,
                                n_resources_to_return: int = 5,
                                print_time: bool = True):
    """
    Embeds a query with model and returns top k results from the collection using embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Perform similarity search with the query embedding
   
    results = collection.query(
        query_embeddings=query_embedding.cpu().detach().numpy(),  # Convert to NumPy format
        n_results=n_resources_to_return,  # Top k results
        include=["embeddings", "documents", "metadatas"],  # Include embeddings, documents, and metadata
    )
    
    return results

# Function to print the top results and scores
def print_top_results_and_scores(query: str,
                                 collection,  # The collection to query
                                 n_resources_to_return: int = 5):
    """
    Takes a query, retrieves the most relevant resources, and prints them in descending order.
    """

    # Retrieve the relevant resources (documents, scores, and metadata)
    results = retrieve_relevant_resources(query=query, collection=collection, n_resources_to_return=n_resources_to_return)

    print(f"Query: {query}\n")
    print("Similarity Search Results:")

    # Loop through the results and print the documents with their metadata
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        print(f"\nResult {i + 1}:")
        print(f"Document: {doc}")
        print(f"Metadata: {metadata}")

# Example usage:

# collection = collection()  # Replace with your actual collection object
# print_top_results_and_scores(query=query, collection=collection, n_resources_to_return=5)

def prompt_formatter(query: str,
                     context_items: list[dict],tokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'..
Make sure your answers are as explanatory as possible.Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: Where is the Grotto located in Notre Dame?
Answer: Immediately behind the basilica
\nExample 2:
Query: What is the purpose of the Grotto at Notre Dame?
Answer: The Grotto is a Marian place of prayer and reflection.
\nNow use the following cont.ext items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt


async def ask(query,device,collection,tokenizer,model,
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True,
        return_answer_only=True,reference_count:int=1): # type: ignore
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # Get just the scores and indices of top related results
    results = retrieve_relevant_resources(query=query, collection=collection, n_resources_to_return=reference_count)

    # Create a list of context items
    documents = results['documents'][0]

# Convert the documents into the required format (list of dictionaries)
    context_items = [{"sentence_chunk": doc} for doc in documents]

# Now call the prompt_formatter function with the query and context_items
    
    

    # Add score to context item
    #for i, item in enumerate(context_items):
      #  item["score"] = scores[i].cpu() # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items,tokenizer=tokenizer)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate an output of tokens
    outputs = model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text

    return output_text, context_items


async def openapi_llm(query, tokenizer, reference_count=1, collection=None):
    results = retrieve_relevant_resources(query=query, collection=collection, n_resources_to_return=reference_count)

    # Create a list of context items
    documents = results['documents'][0]

    # Convert the documents into the required format (list of dictionaries)
    context_items = [{"sentence_chunk": doc} for doc in documents]

    # Format the prompt
    prompt = prompt_formatter(query=query, context_items=context_items, tokenizer=tokenizer)

    # Use ChatCompletion for chat models like GPT-4
    response = openai.ChatCompletion.create(
        engine=deployment_name,  # Replace with the correct model ID
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    print(response)

    # Extract and clean the response text
    text = response['choices'][0]['message']['content'].strip()
    return text



async def two_llms(query,model,tokenizer,device,collection):
    
    responses = []
    response = await ask(query,device,collection,tokenizer,model,temperature=0.7,max_new_tokens=512,format_answer_text=True,return_answer_only=True,reference_count=1)
    
    responses.append(response)
    response = await openapi_llm(query,tokenizer,1,collection)
    responses.append(response)
    print(responses)
    return responses




    





openai.api_key = "your_apikey"
openai.api_base = "https://openai-eus-poc-1.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = "2024-08-01-preview" # this might change in the future

deployment_name='gpt-4o' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

# Send a completion call to generate an answer
