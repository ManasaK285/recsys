# Data

The initial project creates a deterministic synthetic benchmark at runtime; no external download is required. The synthetic dataset has semantic-style numeric features, categories, difficulty, and error severity. Labels intentionally contain nonlinear and category-dependent patterns so that sampling strategies can be compared.

For real experiments, add a documented dataset adapter that produces:
`case_id`, `category`, `text`, numeric feature columns, `error_severity` in [0, 1], and `is_failure` derived from a declared threshold. Do not mix train/test labels when fitting the surrogate.
