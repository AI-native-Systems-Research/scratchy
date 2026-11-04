# Homebrew formula for the scratchy CLI — and the tap lives in this very repo
# (Homebrew only needs the homebrew-* repo name for the `brew install
# org/tap/scr` shorthand; any repo works as a tap by URL). This file is the
# live formula users install AND the template: the release workflow in
# .github/workflows/release-macos.yml rewrites the version/sha256 lines and
# commits the update on every published release, with the built-in github
# token — no PAT, no second repo.
class Scr < Formula
  desc "LLM inference CLI for scratchy (metal backend, macOS arm64)"
  homepage "https://github.com/AI-native-Systems-Research/scratchy"
  version "__VERSION__"
  license :cannot_represent

  # Metal 4 needs macOS 26.2+; x86_64 Macs can't run it at all.
  depends_on macos: ">= 26"
  depends_on arch: :arm64

  url "https://github.com/AI-native-Systems-Research/scratchy/releases/download/__VERSION__/scr-__VERSION__-macos-arm64.tar.gz"
  sha256 "__SHA256__"

  def install
    bin.install "scr"
  end

  def caveats
    <<~EOS
      The binary is Developer ID-signed and notarized, so Gatekeeper accepts it
      without any xattr bypass. Models are downloaded on first use; run
      `scr model ls` to list cached ones.
    EOS
  end

  test do
    assert_match "Usage", shell_output("#{bin}/scr --help")
  end
end
