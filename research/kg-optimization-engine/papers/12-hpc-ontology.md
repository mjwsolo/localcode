# A Unified Ontology for Scalable Knowledge Graph-Driven Operational Data Analytics in HPC Systems

- **Authors**: Junaid Ahmed Khan, Andrea Bartolini (University of Bologna)
- **Date**: July 2025
- **URL**: https://arxiv.org/abs/2507.06107
- **Venue**: arXiv

## Key Approach

Defines a unified ontology for operational data analytics across heterogeneous HPC systems. The ontology provides semantic interoperability for telemetry data from diverse supercomputers:

- **12 core classes**: DataCenter, HPCSystem, ComputeNode, Sensor, Job, etc.
- **23 object properties**: job-to-node, rack-to-node relationships
- **25 data properties**: timestamps, metrics, sensor readings

Validated on two major HPC datasets (Cineca's M100, Fugaku) within a single data model. Storage optimizations (blank nodes, class consolidation) reduce KG overhead by up to 38.84%.

## Results

- Single ontology successfully models two very different HPC systems
- 36 competency questions validated by stakeholders
- 38.84% storage reduction over previous approaches
- Enables cross-system analysis for the first time

## Relation to KG-Optimization Idea

This is the closest existing work to a "hardware knowledge graph":
1. The ontology structure (classes, object properties, data properties) is a template for our KG schema
2. Their 12-class design shows what's needed — we'd extend it with GPU-specific classes (Kernel, Quantization, MemoryHierarchy)
3. Cross-system analysis is exactly what we need — optimize for M4 today, transfer to future chips
4. The competency question validation methodology is a framework for testing our KG

## What We Can Learn

- 12-15 core classes is the right granularity for a hardware ontology
- Storage optimization matters at scale — blank nodes and class consolidation
- Competency questions are the right way to validate an ontology
- Cross-system interoperability should be designed in from day one
