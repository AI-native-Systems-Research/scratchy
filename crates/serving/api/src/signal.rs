// SPDX-License-Identifier: Apache-2.0
//! The process's termination signals, for the commands that end on them through their normal
//! teardown rather than being killed with device work in flight (`serve`, the in-process chat).

/// Resolves on the process's first SIGINT or SIGTERM after the call.
pub async fn termination_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
