# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added default `auto` value for `--best_weights_device`.

### Changed

- Moved postprocess() from NameTag3Model to NameTag3Dataset.

### Changed

- Upgrade dependencies: tensorboard 2.20.0.

### Changed

- Refactored best model checkpoint saving.

### Added

- LoRA.
- Transformer weights dtype.
- User-requested max tokenizer length (overrides HF max model length).
- GPU memory tracking for memory debugging.
- Added options for early stopping best weights storage device.

### Added

- Resolve HF tokenizer model max length when undeclared.

### Changed

- Polished client and improved server error processing on the client side.
- Added explicit server-side handling of oversized requests.
- Silenced HF tokenizer warning (splitting is handled manually).

### Fixed

- Added guard against words exploding into too many subwords.

### Added

- Learning rate decay command-line argument.
- Frozen pretraining warmup safeguard.

### Added

- Added gradient accumulation.

### Changed

- Fixed minor issues in nested NER evaluation scripts.

### Changed

- Upgrade dependencies: keras 3.13.1.

### Added

- Version number.

### Changed

- Upgrade dependencies: keras 3.12.0.

### Added

- SafeSparseCategoricalCrossentropy.

### Changed

- Upgrade dependencies: torch 2.8.0.
- Upgrade dependencies: transformers 4.53.0.

### Added

- Changelog.
