from langchain_aws import ChatBedrock, BedrockEmbeddings, BedrockRerank

aws_region = "eu-central-1"
model_arn_rerank = "arn:aws:bedrock:eu-central-1::foundation-model/cohere.rerank-v3-5:0"
model_id_embeddings = "cohere.embed-multilingual-v3"
model_id_chat = "eu.anthropic.claude-sonnet-4-20250514-v1:0"

chat_claude_4_sonnet = ChatBedrock(
    model_id=model_id_chat,
    region_name=aws_region,
    model_kwargs={"temperature": 0, "max_tokens": 4096},
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
