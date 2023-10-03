from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy
import scipy

model = AutoModelForMaskedLM.from_pretrained('bert-base-cased')
embedding_matrix = deepcopy(model.bert.embeddings.word_embeddings) # the word embedding matrix
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

dataset_to_paraphrase = ['hello my name is Eric Wallace and I like to eat pie',
						 'UC Berkeley is a school in the bay area']

# we will randomly sample a token position to swap out, doing so with replacement
number_of_paraphrases = 50
embedding_dropout_amount = 0.7 # per Zhou et al. 2019

all_paraphrases = []
with torch.inference_mode():
	for sentence in dataset_to_paraphrase:
		curr_paraphrases = []
		tokens = tokenizer.encode(sentence, return_tensors='pt')
		# sample random tokens, but dont sample [CLS] or [SEP]
		positions_to_replace = np.random.randint(1, len(tokens[0]) - 1, size=number_of_paraphrases)
		
		# get word embedding at each sampled position and drop it out
		inputs_embeds = embedding_matrix(tokens)
		inputs_embeds = inputs_embeds.repeat(number_of_paraphrases, 1, 1)
		for batch_element in range(len(inputs_embeds)):
			embedding_to_dropout = inputs_embeds[batch_element][positions_to_replace[batch_element]]
			embedding_to_dropout = F.dropout(embedding_to_dropout, p=0.7)
			inputs_embeds[batch_element][positions_to_replace[batch_element]] = embedding_to_dropout
		
		logits = model(inputs_embeds=inputs_embeds.cuda())['logits'].cpu()
		# zero out the logit for the original token
		for seq_idx, seq in enumerate(logits):
			seq[positions_to_replace[seq_idx]][tokens[0][positions_to_replace[seq_idx]].item()] = -9999

		# sample a random token
		probs = F.softmax(logits, dim=-1)
		for seq_idx, seq in enumerate(probs):
			sampled_token = torch.multinomial(seq[positions_to_replace[seq_idx]], 1).item()
			tokenized_paraphrase = deepcopy(tokens)
			tokenized_paraphrase[0][positions_to_replace[seq_idx]] = sampled_token
			curr_paraphrases.append(tokenizer.decode(tokenized_paraphrase[0]).replace('[CLS] ','').replace(' [SEP]',''))
		all_paraphrases.append(curr_paraphrases)
