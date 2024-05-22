import random
import hnswlib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import pickle
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Embed words from a dictionary.")
parser.add_argument('--num-words', type=int, default=None, help='Number of words to use from the dictionary')
args = parser.parse_args()

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Load the model and tokenizer
print("Loading model and tokenizer...")
model_name = "Alibaba-NLP/gte-large-en-v1.5"
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

def get_embedding(texts, mode="sentence"):
    model.eval()
    if isinstance(texts, str):
        texts = [texts]

    inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output = model(**inp)

    if mode == "query":
        vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
        vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
    else:
        vectors = output.last_hidden_state[:, 0, :]

    return vectors.cpu().numpy()

# Normalize embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Read words from dictionary.txt
print("Reading words from dictionary.txt...")
with open('dictionary.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]

# Filter words longer than 6 characters
words = [word for word in words if len(word) > 6]

# Use only the specified number of words if provided
if args.num_words is not None:
    words = random.sample(words, args.num_words)

print(f"Total words read: {len(words)}")
# Compute embeddings for each word in batches
print("Computing embeddings for each word...")
batch_size = 64  # Adjust batch size based on GPU memory
embeddings = []
embeddings_file = "embeddings.pkl"
for i in tqdm(range(0, len(words), batch_size), desc="Embedding words"):
    batch_words = words[i:i + batch_size]
    batch_embeddings = get_embedding(batch_words, mode="sentence")
    embeddings.append(batch_embeddings)
embeddings = np.vstack(embeddings)
embeddings = normalize_embeddings(embeddings)  # Normalize embeddings
print("Embeddings computed and normalized successfully.")

# Save embeddings to file
print("Saving embeddings to file...")
with open(embeddings_file, 'wb') as f:
    pickle.dump({'words': words, 'embeddings': embeddings}, f)
print("Embeddings saved successfully.")

# Initialize the HNSW index
print("Initializing HNSW index...")
dim = embeddings.shape[1]
num_elements = len(words)
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_elements, ef_construction=200, M=16)
index.add_items(embeddings, list(range(num_elements)))
print("HNSW index initialized and items added.")

# Select start and end words randomly
start_word = random.choice(words).strip()
end_word = random.choice(words).strip()

# Ensure start_word and end_word are not empty
while not start_word:
    start_word = random.choice(words).strip()
while not end_word:
    end_word = random.choice(words).strip()

print(f"Start word: {start_word}")
print(f"End word: {end_word}")

current_word = start_word

while current_word != end_word:

    # Find the nearest 5 words to the current word
    current_embedding = get_embedding(current_word, mode="sentence")
    current_embedding = normalize_embeddings(current_embedding)  # Normalize current embedding
    labels, distances = index.knn_query(current_embedding, k=10)  # Get more than 5 to filter out duplicates and current word
    nearest_words = []
    seen_words = set()
    for label in labels[0]:
        word = words[label]
        if word != current_word and word not in seen_words:
            nearest_words.append(word)
            seen_words.add(word)
            if len(nearest_words) == 5:
                break

    # Calculate and print the distance to the end word
    end_embedding = get_embedding(end_word, mode="sentence")
    end_embedding = normalize_embeddings(end_embedding)  # Normalize end embedding
    distance_to_end = index.knn_query(end_embedding, k=1, num_threads=1)[1][0][0]
    print(f"Distance to end word: {distance_to_end:.4f}")
    # Print the nearest words
    print(f"\nCurrent word: {current_word}\nEnd word: {end_word}\nNearest words:")
    for i, (word, distance) in enumerate(zip(nearest_words, distances[0][:len(nearest_words)])):
        print(f"{i + 1}. {word} (distance: {distance:.4f})")

    # Prompt the user to select one of the nearest words
    choice = int(input("Select a word (1-5): ")) - 1
    next_word = nearest_words[choice]

    # Move to the next word
    current_word = next_word
print(f"Congratulations! You reached the end word: {end_word}")
