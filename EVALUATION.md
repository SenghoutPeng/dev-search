# Evaluation

8 test queries against the 37-document Git man-page index (top_k=3 unless noted).
For each: what was retrieved, whether the generated answer was correctly grounded
and cited, and any notes on failure modes.

| # | Query | Expected source(s) | Retrieval result | Generation result | Notes |
|---|-------|--------------------|--------------------|---------------------|-------|
| 1 | How do I undo the last commit but keep the changes staged? | git-reset | | | |
| 2 | What's the difference between git merge and git rebase? | git-merge, git-rebase | | | |
| 3 | How do I temporarily save uncommitted changes without committing them? | git-stash | | | |
| 4 | How do I rename a remote branch? | git-branch, git-push | | | |
| 5 | What does the -p flag do in git add? | git-add | | | |
| 6 | How do I find which commit introduced a bug? | git-bisect | | | |
| 7 | How do I squash multiple commits into one? | git-rebase | | | |
| 8 | What's the best way to bake a sourdough loaf? | (none — out of domain) | | | should trigger graceful failure |

## Dataset 2: Programming language docs

8 test queries against the framework-docs index (Go, Gin, GORM, Laravel, Next.js,
Docker, Tailwind, Fiber), top_k=3 unless noted.

| # | Query | Expected source(s) | Retrieval result | Generation result | Notes |
|---|-------|--------------------|--------------------|---------------------|-------|
| 1 | What's the difference between goroutines and threads? | go_effective_go | | | |
| 2 | How do I create middleware in Gin? | gin_middleware | | | |
| 3 | How do I define a model/schema in GORM? | gorm_models | | | |
| 4 | How does routing work in Laravel? | laravel_routing | | | |
| 5 | How do I fetch data in a Next.js App Router page? | nextjs_data_fetching | | | |
| 6 | What does a basic Dockerfile look like? | docker_getstarted | | | |
| 7 | How do I make a design responsive with Tailwind? | tailwind_responsive | | | |
| 8 | What's the best way to bake a sourdough loaf? | (none — out of domain) | | | should trigger graceful failure | 

## How to reproduce

1. `streamlit run app.py`
2. Pick a dataset from the sidebar ("Git documentation" or "Programming language docs").
3. For each query in the matching table above: set `top_k=3`, run in `extractive`
   mode first to check raw retrieval quality, then in `llm` mode (provider of
   choice) to check generation quality.
4. Fill in the retrieval/generation columns with: top doc + similarity score
   for retrieval; a one-line verdict (correct / partially correct / wrong /
   hallucinated / correctly refused) for generation.

## Discussion

_(Fill in after running the queries above: 3-5 sentences per dataset on overall
retrieval precision, where generation added value vs. introduced risk, and how
the system behaved on the out-of-domain query in each dataset.)_
