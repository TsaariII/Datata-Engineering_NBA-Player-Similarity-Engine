

UPDATE player_season_features
SET embedding = z_vector::vector
WHERE embedding IS NULL
    AND z_vector IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_psf_embedding_per_game_v1
    ON player_season_features
    USING hnsw (embedding vector_cosine_ops)
    WHERE feature_set = 'per_game_v1';

CREATE INDEX IF NOT EXISTS idx_psf_embedding_advanced_v1
    ON player_season_features
    USING hnsw (embedding vector_cosine_ops)
    WHERE feature_set = 'advanced_v1';