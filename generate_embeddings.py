"""
This code is adapted from the source code used in the paper
'Multi-label Style Change Detection by Solving a Binary Classification Problem---Notebook for PAN at CLEF 2021'

Title: Multi-label Style Change Detection by Solving a Binary Classification Problem---Notebook for PAN at CLEF 2021
Authors: Eivind Strom
Date: 2021
Availability: https://github.com/eivistr/pan21-style-change-detection-stacking-ensemble
"""
import random
import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utilities import load_documents
from split_into_sentences import par_into_sentences

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_sentence(sentence, tokenizer, model):
    # Tokenize input
    sentence = tokenizer.tokenize("[CLS] " + sentence + " [SEP]")

    if len(sentence) > 512:
        sentence = sentence[:512]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
    # In our case we only have one sentence, i.e. one segment id
    segment_ids = [0] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    token_tensor = torch.tensor([indexed_tokens]).to(device)
    segment_tensor = torch.tensor([segment_ids]).to(device)

    with torch.no_grad():
        # Output state of last 4 layers
        output = model(token_tensor, segment_tensor, output_hidden_states=True)["hidden_states"][-4:]
        token_embeddings = torch.stack(output, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = torch.sum(token_embeddings, dim=0)
        sentence_embedding_sum = torch.sum(token_embeddings, dim=0)

    return sentence_embedding_sum


def generate_embeddings(documents):

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased').to(device)
    model.eval()

    embeddings_per_document = []
    embeddings_per_paragraph = []

    with tqdm(documents, unit="document", desc=f"Generating embeddings") as pbar:
        for doc in pbar:

            doc_embedding = torch.zeros(768)
            par_embeddings = []
            sentence_count = 0
            for par in doc:

                par_embedding = torch.zeros(768)
                sentences = par_into_sentences(par)
                if(len(sentences)==0):
                    sentences.append(par)
                for sent in sentences:
                    sentence_count += 1
                    sent_embedding = embed_sentence(sent, tokenizer, model)
                    par_embedding.add_(sent_embedding)

                doc_embedding.add_(par_embedding)
                par_embeddings.append(par_embedding)

            embeddings_per_document.append(doc_embedding / sentence_count)
            embeddings_per_paragraph.append(par_embeddings)

    # Convert lists to numpy arrays
    embeddings_per_document = np.stack(embeddings_per_document)

    for i in range(len(embeddings_per_paragraph)):
        embeddings_per_paragraph[i] = np.stack(embeddings_per_paragraph[i])

    return embeddings_per_document, embeddings_per_paragraph



def main():

    # Load documents
    train_dataset1_docs, train_dataset1_doc_ids = load_documents('train_dataset1')
    train_dataset2_docs, train_dataset2_doc_ids = load_documents('train_dataset2')
    train_dataset3_docs, train_dataset3_doc_ids = load_documents('train_dataset3')
    val_dataset1_docs, val_dataset1_doc_ids = load_documents('val_dataset1')
    val_dataset2_docs, val_dataset2_doc_ids = load_documents('val_dataset2')
    val_dataset3_docs, val_dataset3_doc_ids = load_documents('val_dataset3')

    # Save results
    if not os.path.exists('./features/dataset1'):
        os.makedirs('./features/dataset1')
    if not os.path.exists('./features/dataset2'):
        os.makedirs('./features/dataset2')
    if not os.path.exists('./features/dataset3'):
        os.makedirs('./features/dataset3')
        
   
    
    train_dataset1_doc_emb, train_dataset1_par_emb = generate_embeddings(train_dataset1_docs)
    with open('./features/dataset1/' + 'doc_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset1_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset1/'+ 'par_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset1_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_dataset2_doc_emb, train_dataset2_par_emb = generate_embeddings(train_dataset2_docs)
    with open('./features/dataset2/' + 'doc_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset2_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset2/' + 'par_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset2_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    train_dataset3_doc_emb, train_dataset3_par_emb = generate_embeddings(train_dataset3_docs)
    with open('./features/dataset3/' + 'doc_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset3_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset3/' + 'par_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset3_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    val_dataset1_doc_emb, val_dataset1_par_emb = generate_embeddings(val_dataset1_docs)
    with open('./features/dataset1/' + 'doc_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset1_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset1/' + 'par_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset1_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    val_dataset2_doc_emb, val_dataset2_par_emb = generate_embeddings(val_dataset2_docs)
    with open('./features/dataset2/' + 'doc_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset2_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset2/' + 'par_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset2_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    
    val_dataset3_doc_emb, val_dataset3_par_emb = generate_embeddings(val_dataset3_docs)
    with open('./features/dataset3/' + 'doc_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset3_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset3/' + 'par_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset3_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    

if __name__ == '__main__':
    main()