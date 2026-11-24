// SPDX-License-Identifier: Apache-2.0
//! Build scope only: the `kv/<format>` features forward to the spyre worker, whose models' bake
//! selects the shared KV codec fact from them (`KvCodec::Fp8E4m3` under `fp8`, the CLI's card
//! default (`spyre-hw`); `Dense` under `dense`, which wins). Nothing here is called.
