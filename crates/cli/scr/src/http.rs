// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The CLI's client for talking to a running vLLM server.
//!
//! `ureq`, not `reqwest`, for the same reason `scratchy-bench` moved: reqwest
//! costs ~53 crates here — its hyper / rustls-platform-verifier /
//! security-framework stack plus `url` and the ICU subtree `url` pulls for
//! internationalised domain names — to do what `--url http://host:8000`
//! needs, which is a POST and a `Read` over the response body. ureq is
//! already in the tree for the Hub downloader and already on ring, so it also
//! needs no process-wide crypto provider install.
//!
//! Blocking is the right shape for both callers: `scr chat --url` is a REPL
//! already blocked on `read_line`, and `scr top`'s SSE loop owns a thread
//! feeding a channel.

use std::io::Read;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// A client bound to one server, carrying the bearer token every request needs.
pub(crate) struct RemoteClient {
    agent: ureq::Agent,
    api_key: String,
}

impl RemoteClient {
    pub(crate) fn new(api_key: &str) -> Self {
        Self {
            // `http_status_as_error(false)` so a 4xx/5xx comes back as a
            // response we can read the body of and report, rather than a bare
            // transport error. The previous reqwest client did not check
            // status at all, which meant a rejected request streamed its error
            // body through the SSE parser and printed nothing.
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(3600)))
                .http_status_as_error(false)
                .build()
                .into(),
            api_key: api_key.to_string(),
        }
    }

    /// GET a URL and parse the body as JSON.
    pub(crate) fn get_json(&self, url: &str) -> Result<Value> {
        let resp = self
            .agent
            .get(url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .call()
            .with_context(|| format!("failed to GET {url}"))?;
        let status = resp.status();
        let body = resp
            .into_body()
            .read_to_string()
            .with_context(|| format!("failed to read body from {url}"))?;
        if !status.is_success() {
            bail!("{url} returned {status}: {body}");
        }
        serde_json::from_str(&body).with_context(|| format!("invalid JSON from {url}"))
    }

    /// POST a JSON body and hand back a reader over the streaming response.
    pub(crate) fn post_stream(&self, url: &str, body: &Value) -> Result<Box<dyn Read + Send>> {
        let resp = self
            .agent
            .post(url)
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", self.api_key))
            .send(body.to_string().as_str())
            .with_context(|| format!("failed to POST {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.into_body().read_to_string().unwrap_or_default();
            bail!("{url} returned {status}: {body}");
        }
        Ok(Box::new(resp.into_body().into_reader()))
    }
}
