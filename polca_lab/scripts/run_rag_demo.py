from polca_lab.benchmarks.rag_benchmark import RAGBenchmark
from polca_lab.core.evaluator import StochasticEvaluator
import random
b=RAGBenchmark(); e=StochasticEvaluator(b,noise_std=.08,seed=7)
program='Use only the provided context. Answer directly from retrieved evidence. If the context is insufficient, say so instead of guessing. Cite relevant evidence. Keep answers concise.'
print(e.evaluate(program,None))
