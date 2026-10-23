PreferenceForge
Personalized Reward Modeling from Human Preferences
PreferenceForge is an end-to-end, research-inspired project exploring how AI systems can learn what different users prefer from pairwise comparisons between candidate responses.

The project is inspired by Google DeepMind's paper:
Barreto, A., Dumoulin, V., Mao, Y., Rowland, M., Perez-Nieves, N., Shahriari, B., Dauphin, Y., Precup, D., & Larochelle, H. (2025). Capturing Human Preferences with Reward Features. NeurIPS 2025.
Paper: https://deepmind.google/research/publications/141313/

This repository is an educational implementation of the paper's high-level motivation. It is not an exact reproduction of the paper's model, training procedure, dataset, or reported experiments.
1. Problem and motivation
A single global reward model assumes that all users value the same response qualities in the same way. In practice, preferences can differ: one user may prioritize detail, while another may prefer concise answers.

PreferenceForge investigates whether a model can combine:
- Shared reward features that describe candidate responses.
- User-specific reward weights that represent individual preferences.
- Pairwise feedback indicating which of two responses a user prefers.
At a high level, a response is represented by a feature vector:
\[
\phi(x) = [\text{helpfulness},\text{clarity},\text{detail},\text{concision},\text{safety}]
\]
A personalized reward score can be written as:
\[
r(x,u) = w_u^\top \phi(x)
\]
where \(x\) is a response and \(u\) is a user. The model compares the scores for Response A and Response B to estimate which one the user is more likely to prefer.

In this project, the qualities are explicit feature annotations. The app does not automatically infer these qualities from raw text using an LLM.
2. What the application does
Collect a pairwise preference
The Streamlit app lets a user:
1. Enter a user ID and a prompt.
2. Provide Response A and Response B.
3. Rate each response on five observable qualities:
   - Helpfulness
   - Clarity
   - Detail
   - Concision
   - Safety
4. Select which response they prefer.
The current example in the interface compares two answers to “Explain how a hash table works.” Response B includes more explanation of hashing, collisions, and lookup complexity, while Response A is shorter. A reasonable illustrative annotation might rate B higher for detail and helpfulness, while A could score higher for concision. These are subjective judgments, not ground-truth labels.

Important: Feature ratings and the preferred-response selection are different pieces of feedback. Ratings describe response qualities; the A/B selection records the user's overall preference. A meaningful experiment should collect multiple comparisons from multiple users, including users with differing preferences.
Model evaluation
The project includes a pipeline to train and compare several preference-modeling baselines:
1. Global reward model — learns one shared reward-weight vector for all users.
2. Independent per-user model — fits a separate model from each user's available training feedback.
3. Adaptive reward-feature model — combines shared feature learning with user-specific adaptation from a support set.
4. Preference-summary baseline — estimates preference direction from a user's small support set.
The evaluation pipeline reports metrics such as accuracy, log loss, and Brier score on held-out comparisons. Run the pipeline to generate your own results; this README does not claim any particular model wins.
User reward profile
The dashboard includes a user-profile view intended to help inspect the learned feature weights or preference profile for a selected user. Treat these weights as model estimates, not definitive statements about a person's preferences.
API
A FastAPI service exposes endpoints for health checking, submitting preferences, and comparing responses. Use the interactive API documentation to inspect the exact request schemas supported by the running version.
3. Architecture
Pairwise comparisons
        |
        v
Feature annotations for A and B
(helpfulness, clarity, detail, concision, safety)
        |
        +------------------------------+
        |                              |
        v                              v
Synthetic data generator       Streamlit preference UI
        |                              |
        +---------------+--------------+
                        |
                        v
              Training and evaluation
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Global       Independent     Adaptive
       reward        per-user     reward features
          |             |             |
          +-------------+-------------+
                        |
                        v
        Metrics, predictions, and user profiles
The repository separates data generation, feature definitions, model training, evaluation, UI, and API code so the experiment can be extended without rewriting the whole application.
4. Setup
Recommended: Python 3.10 or newer.
Windows PowerShell
Open a terminal in the extracted PreferenceForge project directory:
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
If PowerShell blocks activation, you can run the environment's Python directly:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
5. Run the experiment
From the project root, generate data, train models, and evaluate:
python -m preferenceforge.cli all
You can also run the steps separately:
python -m preferenceforge.cli generate --users 30 --pairs-per-user 80
python -m preferenceforge.cli train
python -m preferenceforge.cli evaluate
The generator creates simulated users with heterogeneous preferences and adds noise to their choices. These synthetic labels are useful for checking the pipeline and comparing methods under controlled assumptions; they are not evidence of real human behavior.
6. Launch the dashboard
From the project root, with the virtual environment activated:
streamlit run app.py
If streamlit is not recognized on Windows, try:
python -m streamlit run app.py
Open the local URL printed by Streamlit, usually http://localhost:8501.

In the preference collection screen, enter a user ID, prompt, two responses, feature ratings, and the preferred response. Use the app's submission control to save the comparison. Then inspect the evaluation and user-profile sections. For a credible human-feedback experiment, verify that collected rows are saved and included in the training/evaluation data path; merely changing slider values does not establish that a preference has been persisted.
7. Run the API
In a separate terminal:
uvicorn api.main:app --reload
Open:
- API root / health endpoint: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs
The API supports health checking, preference submission, and response comparison. Refer to the live /docs page for the exact endpoint schemas.
8. Data format
The generated preference dataset uses columns including:
- user_id
- prompt
- response_a
- response_b
- label — 1 means the user preferred A; 0 means the user preferred B
- a_helpfulness, a_clarity, a_detail, a_concision, a_safety
- b_helpfulness, b_clarity, b_detail, b_concision, b_safety
Check the generated CSV and the collection implementation for the exact schema used by your current run. Do not assume that manually entered UI feedback is automatically merged with the synthetic training dataset unless you verify that behavior in the code and generated files.
9. Outputs
After running the training and evaluation pipeline, the project is designed to write artifacts such as:
- artifacts/metrics.json
- artifacts/metrics.csv
- artifacts/predictions.csv
- artifacts/models.json
- artifacts/figures/model_comparison.png
The exact artifacts depend on the commands successfully completed. Review the output files from your run before reporting numerical results.
10. Tests
Run the included tests from the project root:
python -m pytest
Use these tests to check core feature, data, and model behavior after making changes.
11. Research questions
- Does personalization improve held-out pairwise preference prediction over a global reward model?
- Can shared reward features help when each user provides only a small amount of feedback?
- How does preference heterogeneity affect the value of personalization?
- How does the number of user-specific comparisons affect the quality of a personalized reward estimate?
- Do feature annotations align with the overall pairwise choice, or do users make trade-offs that the five-feature representation cannot capture?
12. Limitations and responsible use
- Hand-defined features: The five response qualities are explicit numeric annotations, not embeddings or judgments automatically produced by a language model.
- Synthetic evaluation: Generated preferences come from simulated users and assumptions encoded in the data generator.
- Subjective annotations: Ratings for helpfulness, clarity, detail, concision, and safety can vary across annotators.
- Simplified adaptive model: The implementation captures the general idea of shared features plus personalization; it is not the paper's exact architecture.
- Baseline simplification: The preference-summary baseline is not a faithful reproduction of an LLM's in-context learning.
- Persistence matters: Before claiming the app has collected real human feedback, confirm that the submission handler stores it and that training reads those stored examples.
- No unsupported performance claims: Report metrics only after running the experiment, and include the dataset setup, seed, and evaluation split.
- Not for high-stakes decisions: Do not use this educational prototype to make consequential decisions about people.
13. Suggested next steps
1. Collect pairwise comparisons from several participants using consistent annotation guidance.
2. Keep a clear separation between synthetic data, human-collected data, training data, and held-out test data.
3. Split by user and/or prompt as appropriate to the research question; avoid leakage between support examples and evaluation comparisons.
4. Run repeated seeds and report means and variability rather than relying on one run.
5. Compare performance as the number of support preferences per user increases.
6. Consider replacing manually annotated features with a validated feature extractor, while documenting how its scores are obtained.
7. Record consent and follow your institution's requirements before conducting a formal human-subject study.
Paper reference
Barreto, A., Dumoulin, V., Mao, Y., Rowland, M., Perez-Nieves, N., Shahriari, B., Dauphin, Y., Precup, D., & Larochelle, H. (2025). Capturing Human Preferences with Reward Features. NeurIPS 2025.
https://deepmind.google/research/publications/141313/