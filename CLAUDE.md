# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

**AWARE-NET** is a comprehensive deep fake detection framework built as a 10-stage academic research project. The project is currently in the planning/documentation phase with a sophisticated multi-stage implementation strategy.

### Core Directory Structure

```
AWARE-NET/
├── project_instruction/           # Complete project documentation
│   ├── stage/                    # 10 detailed stage specifications (stage_00.md - stage_09.md)  
│   ├── implementation/           # Implementation plans and guides
│   └── history/                  # Project evolution documentation
├── src/                          # Source code (to be implemented)
└── tool/                         # Development tools and utilities
```

### Project Phases

The project follows a **10-stage implementation strategy** designed for academic rigor:

1. **Stage 0**: Infrastructure & baseline model establishment
2. **Stage 1**: SupCon-based rapid filtering system  
3. **Stage 2**: Heterogeneous expert models (spatial/generative)
4. **Stage 3**: Temporal modeling expert
5. **Stage 4**: Feature fusion system with LightGBM meta-model
6. **Stage 5**: SAT (Self-supervised Adversarial Training) 
7. **Stage 6**: Cascade system integration
8. **Stage 7**: Continual learning mechanisms
9. **Stage 8**: Mobile deployment optimization
10. **Stage 9**: Comprehensive evaluation and academic analysis

### Academic Focus Areas

- **Paradigm Innovation**: Shift from "detect fake" to "model authenticity"
- **SAT Framework**: Novel self-supervised adversarial training approach
- **Heterogeneous Expert System**: Multi-modal detection specialists
- **Stage-Gate Methodology**: Rigorous validation at each development phase

## Key Documentation Files

### Primary References
- `project_instruction/implementation/implementation_plan.md` - Master implementation guide
- `project_instruction/stage/stage_00.md` through `stage_09.md` - Detailed stage specifications
- Each stage document contains:
  - Academic objectives and theoretical foundations
  - Detailed task breakdowns with code specifications
  - **Stage-Gate Criteria**: Quantified success metrics with Go/No-Go decision points
  - **Risk Mitigation Plans**: 4-6 risks per stage with 3-level contingency plans
  - Success milestones and deliverables

### Stage-Gate Methodology
Every stage implements a three-tier validation system:
- **Technical Gates**: Functional and performance requirements
- **Academic Gates**: Innovation and rigor standards  
- **System Gates**: Usability and scalability requirements

### Risk-First Development
Each stage includes comprehensive risk assessment:
- **Risk Identification**: 4-6 major risks per stage
- **Mitigation Strategies**: Concrete actionable responses
- **Contingency Plans**: 3-level fallback strategies (Adjustment → Simplification → Complete Rollback)
- **Success Indicators**: Green/Yellow/Red decision frameworks

## Development Guidelines

### Academic Standards
- All implementations must support **reproducible research**
- Comprehensive ablation studies and baseline comparisons required
- Statistical significance testing for all performance claims
- Code must be publication-ready with detailed technical documentation

### Baseline-Driven Validation  
- **Stage 0**: Establish EfficientNetV2-B3 + BCE Loss baseline (target AUC: 0.88-0.92)
- All subsequent innovations must demonstrate quantified improvements over this baseline
- Complex methods require justification through small-scale ablation studies

### Implementation Priorities
1. **Infrastructure First**: Complete Stage 0 before any model development
2. **Baseline Validation**: Prove superiority over simple methods
3. **Incremental Complexity**: Start simple, add complexity only when justified
4. **Risk Mitigation**: Plan for failures and prepare fallback strategies

## Technical Architecture

### Expected Technology Stack
- **Deep Learning**: PyTorch 2.6+, timm, torchvision
- **Data Processing**: OpenCV, albumentations, pandas
- **Academic Tools**: scikit-learn, matplotlib, seaborn
- **Meta-Learning**: LightGBM for ensemble methods
- **Deployment**: ONNX conversion, mobile optimization
- **Environment**: Docker containerization, conda environment management

### GPU Compatibility Management
- **RTX 30/40 Series**: Standard PyTorch 2.6+ with CUDA 12.4
- **RTX 50 Series (sm_120)**: PyTorch nightly builds with CUDA 12.6
- **Intelligent Fallback**: Automatic CPU training when GPU incompatible
- **Multi-GPU Support**: Automatic selection of best compatible GPU
- **Environment Isolation**: Separate conda environments for different GPU architectures

### Key Design Patterns
- **Configuration-Driven**: JSON-based dataset and model configuration
- **Modular Architecture**: Separate components for each detection expert
- **Academic Reproducibility**: Seed management, deterministic training
- **Stage-Isolation**: Each stage can run independently for testing

## Development Workflow

### Stage Implementation Process
1. **Read Stage Documentation**: Thoroughly review the specific stage_XX.md file
2. **Check Stage-Gate Criteria**: Understand quantified success metrics
3. **Review Risk Plans**: Prepare mitigation strategies before coding
4. **Implement with Testing**: Build comprehensive test suites
5. **Validate Against Criteria**: Ensure all gates are met before proceeding

### Code Organization
- Follow the 10-stage directory structure within `src/`
- Maintain clear separation between stages for academic validation
- Include comprehensive documentation for reproducibility
- Implement automated testing for all Stage-Gate criteria

## Important Notes

- **Project Status**: Currently in documentation/planning phase - no production code exists yet
- **Academic Focus**: This is primarily a research project aimed at top-tier conference publication
- **Risk-Aware**: Every stage has detailed failure scenarios and recovery strategies
- **Baseline-Driven**: All innovations must be validated against strong baseline models
- **Stage-Gate Enforcement**: Strict validation required before advancing between stages

When implementing any component, always refer to the relevant stage documentation for detailed specifications, success criteria, and risk mitigation strategies.