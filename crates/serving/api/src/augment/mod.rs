pub(crate) mod embed;
mod index;
mod options;
mod retrieve;
mod sidecar;

pub use index::index;
pub use options::AugmentOptions;
pub use retrieve::retrieve;
pub use sidecar::SidecarManager;
