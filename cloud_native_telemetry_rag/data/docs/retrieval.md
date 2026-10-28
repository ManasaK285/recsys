# Retrieval Quality and Recall Drift

A retrieval-augmented generation system searches a document collection and passes relevant passages to a language model. Recall at k measures the fraction of labeled relevant documents present in the top k retrieved results. If the relevant set contains four documents and the top-k results contain two of them, recall is 0.5.

Retrieval quality can drift when documents change, embedding models are updated, queries shift, or index configuration changes. Track evaluation results over a fixed, representative set of labeled queries. Compare current Recall@k with a baseline and alert when the score falls below a minimum or drifts beyond a configured tolerance. Small or unrepresentative evaluation sets can create noisy signals.
