# The Next 700 ML-Enabled Compiler Optimizations (ML-Compiler-Bridge)

- **Authors**: S. VenkataKeerthy, Siddharth Jain, et al. (includes Albert Cohen, Mircea Trofin from Google DeepMind)
- **Date**: February 2024
- **URL**: https://arxiv.org/abs/2311.10800
- **Venue**: CC 2024 (ACM SIGPLAN)

## Key Approach

ML-Compiler-Bridge provides infrastructure for integrating ML models with optimizing compilers. It addresses the fundamental tension: ML model development happens in Python, but compilers are written in C++. The bridge provides:

- Inter-process and in-process model runners
- Serialization/deserialization mechanisms
- C++ and C APIs for compiler integration
- Python APIs for ML models
- Framework-independent APIs supporting LLVM, MLIR, and Pluto

This is infrastructure, not a specific optimization — it enables the "next 700" ML-enabled optimizations by making integration tractable.

## Results

- Demonstrated with multiple compiler optimization tasks
- Reduces engineering effort to integrate ML into compilers
- Co-authored by Google DeepMind researchers, suggesting production use

## Relation to KG-Optimization Idea

This paper provides critical infrastructure insight:
1. The KG agent needs to interface with real compilers — ML-Compiler-Bridge shows how
2. The serialization/deserialization layer is necessary for our KG to exchange data with MLIR/LLVM
3. The framework-independence principle should guide our KG API design
4. Having Google DeepMind co-authors validates the approach for production use

## What We Can Learn

- Compiler-ML integration is an engineering challenge, not just a research one
- Framework independence is essential — don't lock into one compiler
- Both inter-process (flexible) and in-process (fast) communication are needed
- The infrastructure layer is unglamorous but critical for practical systems
