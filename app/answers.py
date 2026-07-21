candidates_data = {}

def candidate_data(candidate_id, raw_text):
    if candidate_id in candidate_data:
        candidate_data[candidate_id].append(raw_text)
    else:
        candidate_data[candidate_id] = raw_text

    return candidates_data