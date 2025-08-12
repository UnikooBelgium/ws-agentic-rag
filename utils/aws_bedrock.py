from langchain_aws import ChatBedrockConverse, BedrockEmbeddings, BedrockRerank
from langchain_core.rate_limiters import InMemoryRateLimiter

aws_region = "eu-central-1"
model_arn_rerank = "arn:aws:bedrock:eu-central-1::foundation-model/cohere.rerank-v3-5:0"
model_id_embeddings = "cohere.embed-multilingual-v3"
model_id_chat = "eu.anthropic.claude-sonnet-4-20250514-v1:0"

chat_claude_4_sonnet = ChatBedrockConverse(
    model=model_id_chat,
    region_name=aws_region,
    temperature=0,
    max_tokens=4096,
    rate_limiter=InMemoryRateLimiter(
        requests_per_second=5, check_every_n_seconds=0.5, max_bucket_size=2
    ),
)

compressor = BedrockRerank(
    model_arn=model_arn_rerank,
    region_name=aws_region,
    top_n=10,
)

embeddings = BedrockEmbeddings(
    model_id=model_id_embeddings,
    region_name=aws_region,
)
