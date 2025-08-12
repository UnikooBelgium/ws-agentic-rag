# Agentic RAG - Music Production Assistant

An intelligent Retrieval-Augmented Generation (RAG) system built with LangGraph that searches through PDF documents and generates accurate answers using AWS Bedrock's Claude 4 Sonnet model.

![Workflow Diagram](resources/workflow_diagram.png)

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **uv** package manager
- **AWS Account** with Bedrock access
- **AWS CLI** configured with appropriate credentials

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### AWS Configuration

Ensure your AWS credentials are configured with access to Bedrock services:

```bash
aws configure
```

### Environment Configuration

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Configure your environment variables** in `.env`:
   ```properties
   # AWS Configuration
   AWS_DEFAULT_REGION="your-preferred-region"
   
   # LangSmith Configuration
   LANGSMITH_API_KEY="your_langsmith_api_key_here"
   LANGSMITH_TRACING="true"
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"  # or https://eu.api.smith.langchain.com for EU
   LANGSMITH_PROJECT="agentic-rag"
   ```

3. **Example configuration for EU region**:
   ```properties
   AWS_DEFAULT_REGION="eu-central-1"
   LANGSMITH_API_KEY="lsv2_pt_your_api_key_here"
   LANGSMITH_TRACING="true"
   LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
   LANGSMITH_PROJECT="agentic-rag"
   ```

### Required Bedrock Models

Ensure the following models are available in your configured AWS region:
- **Chat Model**: `eu.anthropic.claude-sonnet-4-20250514-v1:0` (EU regions) or equivalent Claude Sonnet model for your region
- **Embeddings**: `cohere.embed-multilingual-v3`
- **Reranking**: `cohere.rerank-v3-5:0`

**Note**: Model availability varies by AWS region. Check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) for model availability in your chosen region.

### Setup

1. **Clone and install**:
   ```bash
   git clone https://github.com/UnikooBelgium/ws-agentic-rag
   cd ws-agentic-rag
   uv sync
   ```

2. **Configure AWS** (if not already done above):
   ```bash
   aws configure
   ```

3. **Run the application**:
   ```bash
   uv run langgraph dev
   ```

This opens LangGraph Studio where you can visualize the workflow and test queries interactively.

## 🎯 Features

- **Intelligent Query Routing** with supervisor node that decides when to use retrieval vs. direct responses
- **Advanced Tool Integration** using LangGraph's built-in tools_condition for dynamic workflow control
- **Smart Document Search** with ChromaDB + Cohere embeddings and reranking
- **Comprehensive Retrieval Tool** with detailed music production knowledge base covering 74+ strategies
- **Enhanced Document Grading** with improved relevance filtering and lenient assessment criteria
- **Self-Correcting Queries** with semantic intent analysis and automatic rephrasing (max 3x)
- **Multi-Stage Answer Validation** with both hallucination detection and answer quality assessment
- **Intelligent Uncertainty Handling** when confidence is low or information is insufficient
- **Optimized State Management** with streamlined data flow and reduced complexity

## 📁 Architecture

**Improved Self-RAG Workflow:** **Supervise** → **Retrieve Documents** → **Grade Documents** → **Generate** → **Grade Answer** → **Wrap Up**

### Core Components

**🧠 Supervisor Node**: Intelligent routing system that analyzes queries and decides between:
- **Tool-based retrieval** for complex technical questions requiring detailed documentation
- **Direct responses** for simple questions or when comprehensive knowledge is sufficient
- **Non-music production filtering** to maintain domain focus

**🔄 Self-Correction Loop**:
- **Semantic Query Analysis**: Understanding of music production terminology and intent
- **Advanced Document Grading**: Lenient relevance assessment to maximize useful information
- **Multi-criteria Answer Validation**: Separate hallucination detection and answer quality assessment
- **Intelligent Rephrasing**: Context-aware query reformulation with previous attempt awareness

**🛤️ Smart Routing Logic**:
- Supervisor determines retrieval necessity → Retrieve & grade documents if needed
- No relevant docs found OR 3 rephrases reached → Generate answer anyway
- Quality answer generated → Wrap up successfully  
- Hallucinated content detected → Express uncertainty with caveats
- Poor but grounded answer → Rephrase query and retry (max 3x)

## 📄 License

MIT License - see LICENSE file for details.