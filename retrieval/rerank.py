def filter_low_quality(docs, min_length=50):
    return [d for d in docs if len(d.page_content) > min_length]
