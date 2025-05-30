# Code for ZeroFusion
## Abstract
CircRNA-miRNA interactions (CMIs) play a pivotal role in gene regulation, yet their prediction remains constrained by the scarcity of direct association data. Here, we present ZeroFusion, a novel zero-shot prediction framework leveraging multimodal graph cascade learning, which eliminates reliance on known CMIs. ZeroFusion integrates disease-mediated mechanisms and multimodal biological semantics by constructing a heterogeneous network that synergizes functional conservation features of circRNAs, ontology-driven semantic representations of diseases, and context-aware sequence embeddings of miRNAs. ZeroFusion employs a cascaded architecture to model cross-entity dependencies and topological patterns, where global attention mechanisms decipher long-range associations mediated by shared diseases, and local neighborhood propagation captures fine-grained circRNA-disease-miRNA triadic interactions. Through cascaded nonlinear transformations, the architecture fuses multimodal representations to enable zero-shot inference of unseen CMIs, circumventing the limitations of conventional methods dependent on direct observational evidence. Experimental results demonstrate that ZeroFusion achieves an AUC of 0.9539 on a newly constructed dataset and outperforms state-of-the-art methods across two benchmark datasets. In a hepatocellular carcinoma case study, 18 out of the top 20 predicted CMIs were experimentally validated.
## Framework
![image](workflow.png)
## Hardware requirements
Training the ZeroFusion model does not strictly require a GPU, but having one is highly desirable for efficient performance. Therefore, proper installation of GPU drivers, including CUDA integration, is recommended.
## Setup Environment
We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
## Depedencies:
python>=3.9
numpy>=1.26.4  
pandas>=2.0.1  
transformers>=4.36.2  
torch>=2.0.1+cu117  
lightgbm>=3.3.5  
scipy>=1.10.0  
tqdm>=4.65.0  
matplotlib>=3.7.1  
## Usage Steps
1. **Positive and negative sample processing**: Generates biologically plausible negative samples and merges them with positive samples.  
   *Execution Script*: `generate_negative_samples_and_merge.py`
2. **Learning representations for biologically heterogeneous graphs**: A three-level cascaded architecture of graph Transformer, GCN, and nonlinear layers to capture global dependencies and local topology.  
   *Execution Script*: `graph_cascade_learning.py`
3. **Zero-Shot CMIs Prediction**: Five-fold cross-validation for indirect inference of circRNA-miRNA interaction prediction.  
   *Execution Script*: `prediction.py`
## Dislaimer
This code was developed for research purposes only. The authors make no warranties, express or implied, regarding its suitability for any particular purpose or its performance.
## License
This library is MIT licensed.

<a href="https://github.com/591286260/ZeroFusion/blob/main/LICENSE"><img src="https://img.shields.io/npm/l/heroicons.svg" alt="License"></a>
