//! Small text helpers the Triton code generators share.

/// Indents every line of `lines` by four spaces.
pub(super) fn indented(lines: &str) -> String {
    lines.lines().map(|l| format!("    {l}\n")).collect()
}
