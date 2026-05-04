
Run these commands:







```
uv sync --extra test
uv run pytest tests/test_supra_pomdp_smoke.py -v
uv run pytest tests/test_supra_pomdp_*.py -v
uv run python verify_state_dependent.py && uv run pytest tests/test_supra_pomdp_agent.py

uv run pytest --cov=ibrl --cov-report=term-missing
```
