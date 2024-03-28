# MKDB

Toy database implemented for learning purposes. Almost all of the implementation
ideas come from these resources:

- [SQLite 2.8.1 source code](https://github.com/antoniosarosi/sqlite2-btree-visualizer)
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

Alternatively, set your default toolchain to `nightly` to avoid specifying
`+nightly` for every `cargo` command:

```bash
rustup default nightly
```

## Compiler Version

If you see any compilation errors it's probably because of the compiler
version. Run `rustc +nightly --version` and compare the output to the last
version used to compile and test the project:

```
rustc 1.78.0-nightly (9c3ad802d 2024-03-07)
```

# Tests

Use [`cargo test`](https://doc.rust-lang.org/cargo/commands/cargo-test.html) to
run all the unit tests:

```bash
cargo +nightly test
```

## Unsafe

All the [unsafe](https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html) code
is located in [`./src/storage/page.rs`](./src/paging/page.rs), which is the
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
cargo +nightly miri test storage::btree
```

If all these modules work correctly without UB then the rest of the codebase
should be fine. The ultimate UB test is the [`./src/db.rs`](./src/db.rs) module
since it's the entry point to SQL execution so it makes use of every other
module. It can be tested with Miri as well since tests are configured to run
completely in memory without files or system calls:

```bash
cargo +nightly miri test db::tests
```

The rest of modules don't make any use of unsafe code so it's not necessary to
test them with Miri.

# Running The Program

Use this command to start the TCP server on port `8000` (the default if not
specified):

```bash
cargo +nightly run -- file.db 8000
```

`file.db` can be any empty file or a file previously
managed by `mkdb`. It works like SQLite in that regard, the difference is that
in order to use `mkdb` you have to connect to the server with some TCP client
that implements the network protocol described at
[`./src/tcp/proto.rs`](./src/tcp/proto.rs). The [`./client`](`./client`) package
is a console client similar to that of MySQL or any other database and can be
used like this:

```bash
cargo +nightly run --package client -- 8000
```

This will connect to the `mkdb` server running on port `8000` and provide you with
a shell where you can type SQL and see the results of the queries.
