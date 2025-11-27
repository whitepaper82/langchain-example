try:
    import ragas.testset.synthesizers as synthesizers
    print(f"Synthesizers dir: {dir(synthesizers)}")
except ImportError:
    print("Could not import ragas.testset.synthesizers")

try:
    from ragas.testset.evolutions import simple, reasoning, multi_context
    print("Found in evolutions")
except ImportError:
    print("Not found in evolutions")
