This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                        # All tests
uv run pytest tests/path/to/test_file.py -svv                # Single test file
pre-commit run --all-files                                   # Lint + format + mypy + bandit + typos
uv run mypy --config-file=pyproject.toml                     # Mypy only (no ruff/typos/bandit)
DEVICE=cuda make test-end-to-end                             # All E2E tests (GPU)
LEROBOT_TEST_DEVICE=cpu uv run pytest tests -svv             # Force CPU tests
```

All CLI tools (`lerobot-train`, `lerobot-eval`, etc.) must be run via `uv run` prefix in a uv-managed venv.

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`. Key policies: `act`, `diffusion`, `smolvla`, `pi0`, `pi05`, `wall_x`, `xvla`, `vqbet`, `tdmpc`, `groot`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`rollout/`** — Policy rollout execution (`context.py`, `ring_buffer.py`, `robot_wrapper.py`, inference strategies).
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py` (`require_cuda`, `require_x86_64_kernel`, `require_env`, `skip_if_package_missing`, etc.). E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (tiered pytest with base/dataset/hardware/viz extras, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval only), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`), CI (`Dockerfile.internal`), and per-benchmark images (`Dockerfile.benchmark.*`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **`scripts/ci/`** — CI helper scripts (extract task descriptions, parse eval metrics).
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. All other modules have `ignore_errors = true`. Add type annotations when modifying the strict modules.
- **`pre-commit` includes mypy**: running `pre-commit run --all-files` executes ruff format, ruff lint, mypy, bandit, typos, pyupgrade, gitleaks, and prettier (markdown). Mypy uses `--config-file=pyproject.toml` and excludes `examples/`, `benchmarks/`, `tests/`.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). Policy extras include `smolvla`, `pi`, `diffusion`, `wallx`, `xvla`, `groot`, etc. New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Checkpoint resume convention**: checkpoints are at `outputs/train/<job>/checkpoints/<step>/pretrained_model/` containing `train_config.json` + weights. Resume with `lerobot-train --config_path=<path>/train_config.json --resume=true`.
- **`uv run` for everything**: always use `uv run` to execute Python commands (not raw `python` or `pip`), including CLI entry points like `lerobot-train`.
- **Test tiers**: CI runs tests in dependency tiers (base → +dataset → +hardware → +viz). Tests gated behind missing extras auto-skip via `pytest.importorskip` or the decorators in `tests/utils.py`. Run `uv sync --locked --extra test` for base tier, add extras incrementally.
