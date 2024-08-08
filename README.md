
## Setup

1. Install rust with [rustup](https://rustup.rs/): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Install [maturin](https://www.maturin.rs/): `cargo install maturin`
3. Build with: `maturin develop --release`
4. Test with: `pytest -sv tests/`

