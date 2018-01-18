def ModelIt(doc_tokens):
    embedding = load_embedding_model()
    rgr = load_regression_model()

    newvec = embedding.infer_vector(doc_tokens)
    return rgr.predict(docvec)
