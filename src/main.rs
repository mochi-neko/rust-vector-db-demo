use std::collections::HashMap;

use anyhow::Result;
use qdrant_client::{
    prelude::{Payload, QdrantClient},
    qdrant::{
        vectors_config::Config, CreateCollection, Distance, Filter,
        PointStruct, ScoredPoint, SearchPoints, Value, VectorParams,
        VectorsConfig,
    },
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel,
    SentenceEmbeddingsModelType,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Setup sentence embeddings model
    let model = SentenceEmbeddingsBuilder::remote(
        SentenceEmbeddingsModelType::AllMiniLmL6V2,
    )
    .create_model()?;

    // Get embedding dimension
    let dimension = model.get_embedding_dim()? as u64;

    // Setup Qdrant client
    let qdrant = QdrantClient::from_url("http://qdrant:6334").build()?;
    qdrant.health_check().await?;

    // Create collection
    let collection_name = "collection_name".to_string();
    qdrant
        .create_collection(&CreateCollection {
            collection_name,
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: dimension,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;

    Ok(())
}

fn embed(
    model: &SentenceEmbeddingsModel,
    sentence: &String,
) -> Result<Vec<Vec<f32>>> {
    Ok(model.encode(&[sentence])?)
}

async fn upsert(
    client: &QdrantClient,
    collection_name: &String,
    model: &SentenceEmbeddingsModel,
    text: &String,
) -> Result<()> {
    let embedding = embed(&model, &text)?;
    let mut points = Vec::new();
    let mut payload: HashMap<String, Value> = HashMap::new();
    payload.insert(
        "text".to_string(),
        Value::from(text.clone()),
    );
    payload.insert(
        "datetime".to_string(),
        Value::from(
            chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S%.3f")
                .to_string(),
        ),
    );

    for vector in embedding {
        let point = PointStruct::new(
            uuid::Uuid::new_v4().to_string(),
            vector,
            Payload::new_from_hashmap(payload.clone()),
        );
        points.push(point);
    }

    client
        .upsert_points(collection_name.clone(), points, None)
        .await?;

    Ok(())
}

async fn search(
    client: &QdrantClient,
    collection_name: &String,
    model: &SentenceEmbeddingsModel,
    query: String,
    count_limit: u64,
    filter: Option<Filter>,
) -> Result<Vec<ScoredPoint>> {
    let embedding = embed(model, &query)?;
    let vector = embedding[0].clone();

    let result = client
        .search_points(&SearchPoints {
            collection_name: collection_name.clone(),
            vector,
            limit: count_limit,
            filter,
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            ..Default::default()
        })
        .await?;

    Ok(result.result)
}
