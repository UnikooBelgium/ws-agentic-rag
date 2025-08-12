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

- **Smart Document Search** with ChromaDB + Cohere embeddings
- **Relevance Filtering** to grade document quality
- **Self-Correcting Queries** with automatic rephrasing (max 3x)
- **Answer Validation** with hallucination detection
- **Uncertainty Handling** when confidence is low

## 📁 Architecture

Self-correcting RAG workflow: **User Intent** → **Load Documents** → **Grade Documents** → **Generate** → **Grade Answer** → **Wrap Up**

**Smart Routing:**
- No relevant docs or 3 rephrases → Generate anyway
- Good answer → Wrap up
- Hallucinated answer → Express uncertainty
- Poor answer → Rephrase and retry

## 🛠️ Programmatic Usage

```python
from main import get_graph

graph = get_graph()
result = graph.invoke({
    "messages": [("human", "What is compression in music production?")]
})
print(result["messages"])
```

## 📄 License

MIT License - see LICENSE file for details.