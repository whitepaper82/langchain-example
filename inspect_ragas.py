import ragas
import os

print(f"Ragas location: {ragas.__file__}")
print(f"Ragas dir: {dir(ragas)}")

try:
    import ragas.testset
    print(f"Ragas testset dir: {dir(ragas.testset)}")
except ImportError:
    print("Could not import ragas.testset")
