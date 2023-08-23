# rust-vector-db-demo
A demonstration to use vector database for text on Rust in Docker.

## How to run a sample

1. Build and run docker.

```shell
docker compose up -d --build
```

2. Start bash in docker.

```shell
docker compose exec rust bash
```

3. Build and run Rust code.

```shell
cargo run
```
