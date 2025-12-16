import torch


def compute_word_log_prob(model, tokenizer, device, multi_token_word_idx, context_input_ids,
                          vocab_token_ids, vocab, batch_size, context_length):
    multi_token_probs = {}
    for start in range(0, len(multi_token_word_idx), batch_size):
        end = start + batch_size
        chunk_indices = multi_token_word_idx[start:end]

        chunk_input_ids = []
        for idx in chunk_indices:
            combined_ids = context_input_ids[0].tolist() + vocab_token_ids[idx]
            chunk_input_ids.append(combined_ids)
        padded_batch = tokenizer.pad(
            [{"input_ids": seq} for seq in chunk_input_ids],
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**padded_batch, use_cache=True, output_hidden_states=True)
        all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Calculate probabilities for each sequence in this chunk
        for batch_idx, idx in enumerate(chunk_indices):
            token_ids_for_word = vocab_token_ids[idx]
            log_p = 0.0
            for k, subtoken_id in enumerate(token_ids_for_word):
                pred_pos = context_length + k - 1
                if pred_pos < 0:
                    raise ValueError("The subtoken index is out of range. Check offsets.")

                token_logits = all_logits[batch_idx, pred_pos, :]
                token_prob = torch.softmax(token_logits, dim=-1)[subtoken_id]
                log_p += torch.log(token_prob).item()

            # Convert log-prob to probability
            word_probability = float(torch.exp(torch.tensor(log_p)))
            multi_token_probs[vocab[idx]] = word_probability
    return multi_token_probs