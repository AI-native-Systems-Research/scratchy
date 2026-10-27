// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The local proxy behind `scr launch claude --auto-mode-only`: Claude Code's
//! auto-mode safety checks go to a local scratchy server, everything else to
//! the user's own gateway, unchanged.
//!
//! Claude Code sends every request to one base URL, so splitting its traffic
//! takes something listening at that URL. What makes the split exact is that
//! launch sets `CLAUDE_CODE_AUTO_MODE_MODEL` to [`CLASSIFIER_MODEL`], a name no
//! other request uses: a request naming it is a safety check, and anything
//! else is the user's own session. (Claude Code also tags requests with
//! `x-claude-code-request-class`, but `auxiliary` covers session titles and
//! summaries too, so the model name is the narrower signal.)
//!
//! A gateway request is forwarded byte for byte, headers included, so the
//! user's credentials reach their gateway exactly as Claude Code sent them and
//! are never read here. Only a classifier request is parsed, and its credentials
//! are dropped: the local server needs none.
//!
//! The classifier request is rewritten in two ways before it reaches the local
//! model:
//! - `model` becomes the served model's id.
//! - One text block is appended to the last user message: a reminder of the two
//!   answers Claude Code parses. Measured on Qwen3.6-35B-A3B-4bit with Claude
//!   Code 2.1.311's security prompt, the model otherwise follows `<block>` with
//!   an explanation instead of `yes`/`no`, which Claude Code cannot parse and so
//!   denies every action: 0/8 parseable without it, 24/24 with it. The
//!   security prompt itself is left as Claude Code wrote it.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use axum::response::Response;
use serde_json::Value;

/// The model name launch tells Claude Code to send its safety checks under.
/// Nothing else Claude Code sends uses it, which is what makes it a routing key.
pub(crate) const CLASSIFIER_MODEL: &str = "scratchy-classifier";

/// The answer formats Claude Code's classifier parser accepts (`<block>yes` or
/// `<block>no`, after any `<thinking>…</thinking>`), restated for a local model
/// that does not follow the security prompt's own Output Format section. The
/// `<thinking>` clause keeps it compatible with the second stage, which asks
/// the model to reason first.
const FORMAT_REMINDER: &str = "\nAfter any <thinking>...</thinking>, reply with EXACTLY one of \
these two forms and nothing else:\n<block>no</block>\n<block>yes</block><category>Rule \
Name</category><reason>[Rule Name] one short sentence</reason>\nThe very next characters after \
<block> must be `yes` or `no`.";

/// How long a forwarded request may take end to end. A main-session turn can
/// stream for a long time; a classifier check is short but may wait behind a
/// cold prefill.
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(3600);

/// Headers that describe one hop of the connection, not the request, so they
/// are not copied to the next hop (RFC 9110 §7.6.1), plus `host`, which names
/// this proxy, and `content-length`, which the client recomputes.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

/// Where each kind of request goes.
#[derive(Clone)]
pub(crate) struct Routes {
    /// The user's gateway, e.g. `https://litellm.example.com`, no trailing `/`.
    pub(crate) gateway: String,
    /// The local scratchy server, e.g. `http://127.0.0.1:8000`.
    pub(crate) local: String,
    /// The model id the local server serves.
    pub(crate) local_model: String,
}

struct Proxy {
    routes: Routes,
    agent: ureq::Agent,
}

/// A running proxy. Dropping it stops the listener.
pub(crate) struct ProxyHandle {
    pub(crate) addr: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for ProxyHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

/// Start the proxy on a free local port.
pub(crate) async fn start(routes: Routes) -> Result<ProxyHandle> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(UPSTREAM_TIMEOUT))
        // A non-2xx answer is the gateway's to give Claude Code, not an error
        // here.
        .http_status_as_error(false)
        // Ask for an uncompressed body. ureq decodes a gzip body whatever it
        // asked for, so `forward` also drops `content-encoding`, which would
        // otherwise describe bytes the client never receives.
        .accept_encoding("")
        .max_redirects(0)
        .build()
        .into();
    let state = Arc::new(Proxy { routes, agent });
    let app = axum::Router::new().fallback(handle).with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind the auto-mode proxy")?;
    let addr = listener.local_addr()?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await;
    });
    Ok(ProxyHandle {
        addr,
        shutdown: Some(tx),
    })
}

async fn handle(
    State(proxy): State<Arc<Proxy>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/")
        .to_string();
    // The proxy's own thread pool is tokio's; ureq is blocking.
    let proxy2 = proxy.clone();
    let result =
        tokio::task::spawn_blocking(move || route(&proxy2, &method, &path, &headers, &body))
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("proxy task failed: {e}")));
    result.unwrap_or_else(|e| {
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": format!("scratchy auto-mode proxy: {e:#}") }
                })
                .to_string(),
            ))
            .expect("static response parts are valid")
    })
}

fn route(
    proxy: &Proxy,
    method: &Method,
    path: &str,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Response> {
    if let Some(local_body) = classifier_body(method, path, body, &proxy.routes.local_model) {
        let url = format!("{}{}", proxy.routes.local, path);
        let local_headers = [
            ("content-type", "application/json"),
            ("anthropic-version", "2023-06-01"),
        ];
        return forward(
            &proxy.agent,
            method,
            &url,
            local_headers.iter().copied(),
            local_body,
        );
    }
    let url = format!("{}{}", proxy.routes.gateway, path);
    let passthrough = headers
        .iter()
        .filter(|(k, _)| !HOP_BY_HOP.contains(&k.as_str()))
        .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str(), v)));
    forward(&proxy.agent, method, &url, passthrough, body.to_vec())
}

/// The body to send the local server, when `body` is a classifier check: a
/// `POST /v1/messages` naming [`CLASSIFIER_MODEL`]. `None` means forward the
/// request to the gateway untouched.
fn classifier_body(method: &Method, path: &str, body: &[u8], local_model: &str) -> Option<Vec<u8>> {
    if method != Method::POST
        || !path.starts_with("/v1/messages")
        || path.starts_with("/v1/messages/")
    {
        return None;
    }
    let mut json: Value = serde_json::from_slice(body).ok()?;
    if json.get("model").and_then(Value::as_str) != Some(CLASSIFIER_MODEL) {
        return None;
    }
    json["model"] = Value::String(local_model.to_string());
    // Claude Code's identity and billing fields mean nothing to a local server.
    if let Some(obj) = json.as_object_mut() {
        obj.remove("metadata");
    }
    append_format_reminder(&mut json);
    serde_json::to_vec(&json).ok()
}

/// Append [`FORMAT_REMINDER`] to the last user message, as one more text block.
fn append_format_reminder(json: &mut Value) {
    let Some(last) = json
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|m| m.iter_mut().rev().find(|m| m["role"] == "user"))
    else {
        return;
    };
    let block = serde_json::json!({ "type": "text", "text": FORMAT_REMINDER });
    match &mut last["content"] {
        Value::Array(blocks) => blocks.push(block),
        Value::String(text) => text.push_str(FORMAT_REMINDER),
        _ => {}
    }
}

fn forward<'a>(
    agent: &ureq::Agent,
    method: &Method,
    url: &str,
    headers: impl Iterator<Item = (&'a str, &'a str)>,
    body: Vec<u8>,
) -> Result<Response> {
    let mut req = axum::http::Request::builder()
        .method(method.as_str())
        .uri(url);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let req = req
        .body(body)
        .context("could not build the forwarded request")?;
    let resp = agent
        .run(req)
        .with_context(|| format!("{method} {url} failed"))?;

    let (parts, body) = resp.into_parts();
    let mut out = Response::builder().status(parts.status.as_u16());
    for (k, v) in parts.headers.iter() {
        // ureq hands over a decoded body, so its encoding no longer applies.
        if HOP_BY_HOP.contains(&k.as_str()) || k.as_str() == "content-encoding" {
            continue;
        }
        if let (Ok(k), Ok(v)) = (
            HeaderName::from_bytes(k.as_str().as_bytes()),
            HeaderValue::from_bytes(v.as_bytes()),
        ) {
            out = out.header(k, v);
        }
    }
    // Stream the body through as it arrives: a main-session turn is an SSE
    // stream Claude Code renders token by token.
    let reader = body.into_with_config().limit(u64::MAX).reader();
    out.body(Body::from_stream(stream_reader(reader)))
        .context("could not build the proxied response")
}

/// A blocking reader as a byte stream: read on tokio's blocking pool, handed
/// over a channel chunk by chunk.
fn stream_reader(
    mut reader: impl std::io::Read + Send + 'static,
) -> tokio_stream::wrappers::ReceiverStream<std::io::Result<Bytes>> {
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    tokio::task::spawn_blocking(move || {
        let mut buf = vec![0u8; 16 * 1024];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx
                        .blocking_send(Ok(Bytes::copy_from_slice(&buf[..n])))
                        .is_err()
                    {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    break;
                }
            }
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

#[cfg(test)]
mod tests {
    use super::{CLASSIFIER_MODEL, FORMAT_REMINDER, Method, classifier_body};
    use serde_json::{Value, json};

    fn check(body: &Value) -> Option<Value> {
        classifier_body(
            &Method::POST,
            "/v1/messages?beta=true",
            &serde_json::to_vec(body).unwrap(),
            "local-model",
        )
        .map(|b| serde_json::from_slice(&b).unwrap())
    }

    /// A request naming the classifier model is the only kind that goes local,
    /// renamed to the served model and without Claude Code's metadata.
    #[test]
    fn a_classifier_check_goes_local_renamed() {
        let body = json!({
            "model": CLASSIFIER_MODEL,
            "max_tokens": 2112,
            "metadata": { "user_id": "someone" },
            "stop_sequences": ["</block>"],
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "<transcript>" }] }]
        });
        let out = check(&body).expect("a classifier check");
        assert_eq!(out["model"], "local-model");
        assert!(out.get("metadata").is_none());
        assert_eq!(out["stop_sequences"], json!(["</block>"]));
        assert_eq!(out["max_tokens"], 2112);
    }

    /// The format reminder is appended to the last user message, after Claude
    /// Code's own instruction, and nothing else in the message changes.
    #[test]
    fn the_format_reminder_follows_claude_codes_instruction() {
        let body = json!({
            "model": CLASSIFIER_MODEL,
            "messages": [{ "role": "user", "content": [
                { "type": "text", "text": "<transcript>" },
                { "type": "text", "text": "Your ENTIRE response MUST begin with <block>." }
            ]}]
        });
        let blocks = check(&body).unwrap()["messages"][0]["content"].clone();
        assert_eq!(blocks.as_array().unwrap().len(), 3);
        assert_eq!(
            blocks[1]["text"],
            "Your ENTIRE response MUST begin with <block>."
        );
        assert_eq!(blocks[2]["text"], FORMAT_REMINDER);

        let string_content = json!({
            "model": CLASSIFIER_MODEL,
            "messages": [{ "role": "user", "content": "check this" }]
        });
        let text = check(&string_content).unwrap()["messages"][0]["content"].clone();
        assert_eq!(text, format!("check this{FORMAT_REMINDER}"));
    }

    /// Everything that is not a classifier check stays on the gateway: the main
    /// session, other models, other routes, and bodies that are not JSON.
    #[test]
    fn everything_else_stays_on_the_gateway() {
        let main = json!({ "model": "claude-opus-5-5", "messages": [] });
        assert!(check(&main).is_none());
        assert!(check(&json!({ "messages": [] })).is_none());

        let classifier =
            serde_json::to_vec(&json!({ "model": CLASSIFIER_MODEL, "messages": [] })).unwrap();
        assert!(classifier_body(&Method::GET, "/v1/messages", &classifier, "m").is_none());
        assert!(
            classifier_body(&Method::POST, "/v1/messages/count_tokens", &classifier, "m").is_none()
        );
        assert!(classifier_body(&Method::POST, "/v1/models", &classifier, "m").is_none());
        assert!(classifier_body(&Method::POST, "/v1/messages", b"not json", "m").is_none());
    }
}
