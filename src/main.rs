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
use tokio::task;

#[tokio::main]
async fn main() -> Result<()> {
    // Setup sentence embeddings model
    // NOTE: Run blocking operation in task::spawn_blocking
    let model = task::spawn_blocking(move || {
        SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL6V2,
        )
        .create_model()
    })
    .await??;

    // Get embedding dimension
    let dimension = model.get_embedding_dim()? as u64;

    // Setup Qdrant client
    let qdrant = QdrantClient::from_url("http://qdrant:6334").build()?;
    qdrant.health_check().await?;

    // Create collection
    let collection_name = "collection_name".to_string();
    qdrant
        .delete_collection(collection_name.clone())
        .await?;
    qdrant
        .create_collection(&CreateCollection {
            collection_name: collection_name.clone(),
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

    // Store corpus in Vector DB
    let corpus = vec![
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ];
    for sentence in corpus {
        upsert(
            &qdrant,
            &collection_name,
            &model,
            &sentence.to_string(),
        )
        .await?;
        println!("Upserted: {}", sentence)
    }

    // Search for similar sentences
    let queries = vec![
        "A man is eating pasta.",
        "Someone in a gorilla costume is playing a set of drums.",
        "A cheetah chases prey on across a field.",
    ];
    for query in queries {
        let result = search(
            &qdrant,
            &collection_name,
            &model,
            query.to_string(),
            5,
            None,
        )
        .await?;

        println!("Query: {}", query);
        for point in result {
            println!(
                "Score: {}, Text: {}",
                point.score,
                point
                    .payload
                    .get("text")
                    .unwrap()
            );
        }
        println!();
    }

    // Delete collection
    qdrant
        .delete_collection(collection_name.clone())
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
    collection_name: &str,
    model: &SentenceEmbeddingsModel,
    text: &String,
) -> Result<()> {
    let embedding = embed(model, text)?;
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
        .upsert_points(
            collection_name.to_string(),
            points,
            None,
        )
        .await?;

    Ok(())
}

async fn search(
    client: &QdrantClient,
    collection_name: &str,
    model: &SentenceEmbeddingsModel,
    query: String,
    count_limit: u64,
    filter: Option<Filter>,
) -> Result<Vec<ScoredPoint>> {
    let embedding = embed(model, &query)?;
    let vector = embedding[0].clone();

    let result = client
        .search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
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
