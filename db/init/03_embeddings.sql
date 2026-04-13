ALTER TABLE player_season_features
    ADD COLUMN IF NOT EXISTS embedding vector;

UPDATE player_season_features
SET embedding = z_vector::vector
WHERE embedding IS NULL;

CREATE INDEX IF NOT EXISTS idx_psf_embedding_hnsw
    ON player_season_features
    USING hnsw (embedding vector_cosine_ops);