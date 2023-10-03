from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import torch.nn.functional as F
import torch
import sentencepiece

# Initialize model and tokenizer
model = AutoModelForMaskedLM.from_pretrained('roberta-large').eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
embedding_matrix = model.roberta.embeddings.word_embeddings.weight

dataset_to_paraphrase = ['Eric Wallace is a PhD student at UC Berkeley working on natural language processing.']

# Paraphrase parameters
number_of_paraphrases_to_generate = 20
number_of_paraphrases_to_select = 20
embedding_dropout_amount = 0.7
dropout_percentage = 0.25

all_paraphrases = []

with torch.inference_mode():
    for sentence in dataset_to_paraphrase:
        tokens = tokenizer.encode(sentence, return_tensors='pt')
        num_tokens_to_replace = max(1, int(dropout_percentage * (len(tokens[0]) - 2)))
        
        # Create batched inputs for the model
        all_inputs_embeds = []
        all_positions_to_replace = []
        for _ in range(number_of_paraphrases_to_generate):
            positions_to_replace = np.random.choice(np.arange(1, len(tokens[0]) - 1), size=num_tokens_to_replace, replace=False)
            all_positions_to_replace.append(positions_to_replace)
            
            inputs_embeds = embedding_matrix[tokens].clone()
            for pos in positions_to_replace:
                inputs_embeds[0, pos] = F.dropout(inputs_embeds[0, pos], p=embedding_dropout_amount)
            all_inputs_embeds.append(inputs_embeds)
        
        all_inputs_embeds = torch.cat(all_inputs_embeds).cuda()
        all_logits = model(inputs_embeds=all_inputs_embeds)['logits'].cpu()
        
        paraphrase_probs = []
        for i in range(number_of_paraphrases_to_generate):
            logits = all_logits[i]
            positions_to_replace = all_positions_to_replace[i]
            
            paraphrased_tokens = tokens[0].clone().cpu().tolist()
            chosen_log_probs = []
            for j, pos in enumerate(positions_to_replace):
                logits_at_position = logits[pos]
                logits_at_position[tokens[0, pos].item()] = -9999
                log_probs = F.log_softmax(logits_at_position, dim=-1)
                chosen_token = torch.multinomial(F.softmax(logits_at_position, dim=-1), 1).item()
                chosen_log_probs.append(log_probs[chosen_token].item())
                paraphrased_tokens[pos] = chosen_token

            total_log_prob = sum(chosen_log_probs)
            paraphrased_sentence = tokenizer.decode(paraphrased_tokens[1:-1])
            paraphrase_probs.append((paraphrased_sentence, total_log_prob))

        # Sorting by log probabilities and taking top N
        sorted_paraphrases = sorted(paraphrase_probs, key=lambda x: x[1], reverse=True)[:number_of_paraphrases_to_select]
        all_paraphrases.extend([p[0] for p in sorted_paraphrases])
        
        for p in all_paraphrases:
            print(p)
        print()
