CREATE TABLE IF NOT EXISTS player_seasons (
    player_key      text        NOT NULL,
    player_name     text        NOT NULL,
    season          int         NOT NULL,
    team            text        NULL,
    pos             text        NULL,
    g               int         NULL,
    mp              numeric     NULL,
    mpg             numeric     NULL,
    total_minutes   numeric     NULL,
    stats           jsonb       NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (player_key, season)
);

CREATE INDEX IF NOT EXISTS idx_player_seasons_season ON player_seasons (season);
CREATE INDEX IF NOT EXISTS idx_player_seasons_name ON player_seasons (player_name);

CREATE TABLE IF NOT EXISTS player_season_features (
    player_key      text                NOT NULL,
    season          int                 NOT NULL,
    feature_set     text                NOT NULL,
    feature_names   text[]              NOT NULL,
    z_scores        jsonb               NOT NULL,
    z_vector        double precision[]  NOT NULL,
    created_at      timestamptz         NOT NULL DEFAULT now(),
    PRIMARY KEY (player_key, season, feature_set),
    FOREIGN KEY (player_key, season)
        REFERENCES player_seasons (player_key, season)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psf_season_set ON player_season_features (season, feature_set);
CREATE INDEX IF NOT EXISTS idx_psf_zscores_gin ON player_season_features USING gin (z_scores);