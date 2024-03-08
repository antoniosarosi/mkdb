# MKDB

Toy database implemented for learning purposes. Almost all of the implementation
ideas come from these resources:

- [SQLite 2.X.X source code](https://github.com/antoniosarosi/sqlite2-btree-visualizer)
- [CMU Intro to Database Systems 2023](https://www.youtube.com/playlist?list=PLSE8ODhjZXjbj8BMuIrRcacnQh20hmY9g)
- [CMU Intro to Database Systems 2018](https://www.youtube.com/playlist?list=PLSE8ODhjZXja3hgmuwhf89qboV1kOxMx7)

Other less important resources are linked in the source code.

# Compilation

Install [`rustup`](https://rustup.rs/) if you don't have it and then install the
`nightly` toolchain:

```bash
rustup toolchain install nightly
```

Use `cargo` to compile the project:

```bash
cargo +nightly build
```

Alternatively, set your default toolchaing to `nightly` to avoid specifying
`+nightly` for every `cargo` command:

```bash
rustup default nightly
```

# Tests

Use [`cargo test`](https://doc.rust-lang.org/cargo/commands/cargo-test.html) to
run all the unit tests:

```bash
cargo +nightly test
```

## Unsafe

Most of the [unsafe](https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html)
code is located in [`./src/storage/page.rs`](./src/paging/page.rs), which is the
module that implements slotted pages. [`Miri`](https://github.com/rust-lang/miri)
can be used to test possible undefined behaviour bugs. Install the component
using `rustup`:

```bash
rustup +nightly component add miri
```

Then use `cargo` to test the `page` module:

```bash
cargo +nightly miri test storage::page::tests
```

The [`./src/paging/`](./src/paging/) and
[`./src/storage/btree.rs`](./src/storage/btree.rs) modules are not unsafe
themselves but they heavily rely on the slotted page module so it's worth
testing them with Miri as well:

```bash
cargo +nightly miri test paging
cargo +nightly miri test storage::btree::tests
```

If all these modules work correctly without UB then the rest of the codebase
should be fine.
