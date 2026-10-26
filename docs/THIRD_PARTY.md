# Third-party tools

Kira's original source, documentation, and released NNUE weights use the MIT license in the repository root. The engine has no external Cargo dependencies. This license does not relicense external tools.

- **Bullet:** optional training infrastructure from [jw1912/bullet](https://github.com/jw1912/bullet), upstream MIT ([original license notice](licenses/Bullet-MIT.txt)). The recorded training source commit is `579b78b37a694cacd6e9ede4706386c15c34837c`. Kira's training driver applies the recorded training-only patch and uses Kira-owned record and feature code. See `tools/train/README.md` and its training-only patch. The upstream checkout is not distributed in this release.
- **fastchess:** match runner from [Disservin/fastchess v1.8.2-alpha](https://github.com/Disservin/fastchess/releases/tag/v1.8.2-alpha). Its executable is not included in the Kira runtime distribution. Its own license applies.
- **Stockfish 19:** external perft oracle and match opponent, under its upstream license. Neither Stockfish source nor its executable is distributed in Kira's runtime package. It did not provide training labels or training openings.

Mentioning or hashing a tool does not make it part of Kira's MIT-licensed engine.
