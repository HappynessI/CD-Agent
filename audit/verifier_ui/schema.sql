-- MLLM verifier audit store. One SQLite file is the single source of truth;
-- metrics are computed on read, never stored. Only review fields are written by the UI.

PRAGMA foreign_keys = ON;

-- Exact rendered prompt text; id is the first 12 hex digits of its sha256.
CREATE TABLE IF NOT EXISTS prompt_version (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    text        TEXT NOT NULL,
    parent_id   TEXT REFERENCES prompt_version(id),
    note        TEXT,
    created_at  TEXT NOT NULL,
    grp         TEXT,
    title       TEXT,
    kind        TEXT CHECK (kind IN ('template', 'rendered'))
);

-- Any candidate shown to a model, as a query or as a demo.
-- split: train | val | test (of the source dataset).
-- inputs: server paths by key. Form A: t1, t2 (near view, red contour).
--   Form B: full_t1, full_t2, mask (whole image + binary change mask). Optional: thumb.
-- bbox: [x0, y0, x1, y1] of the candidate in the whole image (for the location thumbnail).
-- pseudo_type: for negatives, which pseudo-change it is (answer is still "no change").
-- change_kind: SECOND "t1 class -> t2 class" for positives.
-- semantic_text: SECOND two-date semantic description shown with form B.
CREATE TABLE IF NOT EXISTS sample (
    id            INTEGER PRIMARY KEY,
    dataset       TEXT NOT NULL,
    split         TEXT NOT NULL,
    cid           TEXT NOT NULL,
    image         TEXT,
    source        TEXT,
    label         INTEGER,
    rationale     TEXT,
    pseudo_type   TEXT,
    change_kind   TEXT,
    semantic_text TEXT,
    area          INTEGER,
    det_mean      REAL,
    m1_score      REAL,
    bbox          TEXT,
    inputs        TEXT NOT NULL,
    meta          TEXT,
    UNIQUE (dataset, split, cid)
);

-- Named sample collections: demo pools (purpose demo_pool, from train), dev sets (val)
-- and test sets (test). stratum: pos2 | pos_other | hard_neg | rand_neg; weight is the
-- inverse sampling probability. review_* are legacy (before reviews were shared by sample);
-- store.connect migrates them into sample_review once.
CREATE TABLE IF NOT EXISTS sample_set (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    purpose     TEXT NOT NULL CHECK (purpose IN ('demo_pool', 'dev', 'test')),
    dataset     TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_set_item (
    set_id        INTEGER NOT NULL REFERENCES sample_set(id),
    sample_id     INTEGER NOT NULL REFERENCES sample(id),
    stratum       TEXT,
    weight        REAL NOT NULL DEFAULT 1,
    review_status TEXT CHECK (review_status IN ('pass', 'reject')),
    review_note   TEXT,
    reviewed_at   TEXT,
    PRIMARY KEY (set_id, sample_id)
);

-- Review result per sample (pass / reject / note), shared by every demo pool that contains
-- the sample, e.g. the 64 and 256 pools of one dataset.
CREATE TABLE IF NOT EXISTS sample_review (
    sample_id   INTEGER PRIMARY KEY REFERENCES sample(id),
    status      TEXT CHECK (status IN ('pass', 'reject')),
    note        TEXT,
    reviewed_at TEXT
);

-- mode: zero | few | many | batched_many. form: A | B | AB (demo presentation).
-- id is a hash of mode, form and the ordered items.
CREATE TABLE IF NOT EXISTS demo_set (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    mode        TEXT NOT NULL CHECK (mode IN ('zero', 'few', 'many', 'batched_many')),
    form        TEXT NOT NULL DEFAULT 'A' CHECK (form IN ('A', 'B', 'AB')),
    k           INTEGER NOT NULL,
    pool_id     INTEGER REFERENCES sample_set(id),
    strategy    TEXT,
    seed        INTEGER,
    note        TEXT,
    created_at  TEXT NOT NULL
);

-- shown_label is the answer text the prompt gives for this demo ("yes"/"no").
CREATE TABLE IF NOT EXISTS demo_set_item (
    demo_set_id TEXT NOT NULL REFERENCES demo_set(id),
    position    INTEGER NOT NULL,
    sample_id   INTEGER NOT NULL REFERENCES sample(id),
    shown_label TEXT,
    PRIMARY KEY (demo_set_id, position)
);

-- template: the chat messages as sent, with images as refs such as
-- {"type": "image", "ref": "demo[3].full_t2"} or {"type": "image", "ref": "query[0].t1"}.
-- readout: token_prob | verbal_confidence | answer_only (how score is obtained).
-- batch_by: e.g. "image" when one call asks about all candidates of one image.
CREATE TABLE IF NOT EXISTS run (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    experiment    TEXT,
    model         TEXT NOT NULL,
    prompt_id     TEXT NOT NULL REFERENCES prompt_version(id),
    demo_set_id   TEXT NOT NULL REFERENCES demo_set(id),
    eval_set_id   INTEGER REFERENCES sample_set(id),
    readout       TEXT,
    view          TEXT,
    packing       TEXT,
    batch_size    INTEGER NOT NULL DEFAULT 1,
    batch_by      TEXT,
    api_calls     INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost          REAL,
    currency      TEXT,
    template      TEXT NOT NULL,
    git_commit    TEXT,
    config        TEXT,
    created_at    TEXT NOT NULL
);

-- One model call that answered several queries (batched ICL); raw text stored once.
CREATE TABLE IF NOT EXISTS call (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES run(id),
    idx           INTEGER NOT NULL,
    group_key     TEXT,
    raw_text      TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost          REAL,
    meta          TEXT,
    UNIQUE (run_id, idx)
);

CREATE TABLE IF NOT EXISTS output (
    run_id      INTEGER NOT NULL REFERENCES run(id),
    sample_id   INTEGER NOT NULL REFERENCES sample(id),
    call_id     INTEGER REFERENCES call(id),
    position    INTEGER,
    p_yes       REAL,
    logit_diff  REAL,
    yes_no_mass REAL,
    answer      TEXT,
    confidence  INTEGER,
    raw_text    TEXT,
    extra       TEXT,
    PRIMARY KEY (run_id, sample_id)
);

CREATE INDEX IF NOT EXISTS output_sample ON output(sample_id);
CREATE INDEX IF NOT EXISTS item_sample ON demo_set_item(sample_id);
CREATE INDEX IF NOT EXISTS set_item_sample ON sample_set_item(sample_id);
