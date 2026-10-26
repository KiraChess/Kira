use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use kira1a::eval::nnue_quantized::{QuantizedNnueMetadata, QuantizedNnueNetwork};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("kira-uci-{}-{id}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn executable(&self) -> PathBuf {
        let name = if cfg!(windows) { "kira1a.exe" } else { "kira1a" };
        let target = self.0.join(name);
        fs::copy(env!("CARGO_BIN_EXE_kira1a"), &target).unwrap();
        target
    }
}
impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

fn network(path: &Path) {
    let net = QuantizedNnueNetwork::<128>::from_float_parts(
        &vec![0.0; 768 * 128], &vec![0.0; 128], &vec![0.0; 256], 0.0, 400.0,
        QuantizedNnueMetadata::default(),
    ).unwrap();
    fs::write(path, net.to_bytes().unwrap()).unwrap();
}

fn run(exe: &Path, cwd: &Path, commands: &str) -> String {
    let mut child = Command::new(exe).current_dir(cwd).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(commands.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn bundled_network_loads_independent_of_working_directory() {
    let fixture = Fixture::new();
    let exe = fixture.executable();
    network(&fixture.0.join("kira1a-retrained.nnue"));
    let cwd = fixture.0.join("other directory");
    fs::create_dir(&cwd).unwrap();
    let output = run(&exe, &cwd, "uci\nisready\nucinewgame\nposition startpos\ngo depth 1\nquit\n");
    assert!(output.contains("option name Eval type combo default NNUE"), "{output}");
    assert!(output.contains("NNUE is active"), "{output}");
    assert!(output.contains("uciok\n"), "{output}");
    assert!(output.contains("readyok\n"), "{output}");
    assert!(output.contains("bestmove "), "{output}");
}

#[test]
fn missing_or_invalid_network_falls_back_and_explicit_override_works() {
    let fixture = Fixture::new();
    let exe = fixture.executable();
    let net = fixture.0.join("kira1a-retrained.nnue");
    let output = run(&exe, &fixture.0, "uci\nisready\nquit\n");
    assert!(output.contains("NNUE unavailable:") && output.contains("HCE is active"), "{output}");
    fs::write(&net, b"corrupt").unwrap();
    let override_file = fixture.0.join("override with spaces.nnue");
    network(&override_file);
    let commands = format!("uci\nsetoption name EvalFile value {}\nisready\nquit\n", override_file.display());
    let output = run(&exe, &fixture.0, &commands);
    assert!(output.contains("NNUE unavailable:") && output.contains("HCE is active"), "{output}");
    assert!(output.contains("EvalFile loaded") && output.contains("NNUE is active"), "{output}");
    assert!(output.contains("readyok"), "{output}");
    let commands = format!("uci\nsetoption name EvalFile value {}\nsetoption name Eval value NNUE\nsetoption name EvalFile value {}\nuci\nquit\n", override_file.display(), net.display());
    let output = run(&exe, &fixture.0, &commands);
    assert!(output.contains("EvalFile load failed; current settings unchanged"), "{output}");
    assert!(output.contains("NNUE is active"), "{output}");
}
