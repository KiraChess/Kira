use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use kira1a::eval::{Evaluator, Hce, nnue_quantized::QuantizedNnueNetwork};
use kira1a::search::{SearchInfo, SearchLimits, SearchResult, Searcher, MATE_SCORE};
use kira1a::{play_position_command_with_history, Board};

type Output = Arc<Mutex<BufWriter<io::Stdout>>>;
type SearchWorker = (JoinHandle<()>, Arc<AtomicBool>);

struct SharedEvaluator(Arc<dyn Evaluator>);

impl Evaluator for SharedEvaluator {
    fn evaluate(&self, board: &Board) -> i32 {
        self.0.evaluate(board)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvalMode {
    Hce,
    Nnue,
}

const DEFAULT_NET_NAME: &str = "kira1a-retrained.nnue";

fn bundled_network_path(executable: &Path) -> Option<PathBuf> {
    executable.parent().map(|directory| directory.join(DEFAULT_NET_NAME))
}

fn emit(output: &Output, line: &str) {
    if let Ok(mut out) = output.lock() {
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

fn stop_search(worker: &mut Option<SearchWorker>) {
    if let Some((handle, stop)) = worker.take() {
        stop.store(true, Ordering::Relaxed);
        let _ = handle.join();
    }
}

fn main() {
    let stdin = io::stdin();
    let output = Arc::new(Mutex::new(BufWriter::new(io::stdout())));
    let mut board = Board::startpos();
    let mut game_history = vec![board.hash()];
    let searcher = Arc::new(Mutex::new(Searcher::new(16)));
    let mut worker: Option<SearchWorker> = None;
    // Resolve against the executable, never against the GUI's working directory.
    let default_path = std::env::current_exe()
        .map_err(|error| error.to_string())
        .and_then(|path| bundled_network_path(&path).ok_or_else(|| "executable has no parent directory".to_string()));
    let mut eval_file = default_path.as_ref().map(|path| path.to_string_lossy().into_owned()).unwrap_or_default();
    let default_eval_file = eval_file.clone();
    let mut want_nnue = true;
    let loaded = default_path.and_then(|path| QuantizedNnueNetwork::<128>::load(&path).map(Arc::new));
    let (mut eval_mode, mut nnue_network, mut evaluator, startup_status): (_, _, Arc<dyn Evaluator>, _) = match loaded {
        Ok(network) => {
            let message = format!("info string NNUE loaded {}; NNUE is active", one_line(&eval_file));
            (EvalMode::Nnue, Some(Arc::clone(&network)), network, message)
        }
        Err(error) => {
            let message = format!("info string NNUE unavailable: {}; HCE is active", one_line(&error));
            (EvalMode::Hce, None, Arc::new(Hce), message)
        }
    };

    for input in stdin.lock().lines() {
        let Ok(line) = input else { break; };
        let command = line.trim();
        if command == "uci" {
            emit(&output, "id name Kira 1 Alpha");
            emit(&output, "id author whosnds");
            emit(&output, "option name Hash type spin default 16 min 1 max 4096");
            emit(&output, "option name Threads type spin default 1 min 1 max 1");
            emit(&output, "option name Eval type combo default NNUE var HCE var NNUE");
            let default_file = if eval_file.is_empty() { "<empty>" } else { &eval_file };
            emit(&output, &format!("option name EvalFile type string default {}", one_line(default_file)));
            if eval_mode == EvalMode::Nnue {
                emit(&output, &format!("info string NNUE loaded {}; NNUE is active", one_line(&eval_file)));
            } else if nnue_network.is_none() && eval_file == default_eval_file {
                emit(&output, &startup_status);
            } else {
                emit(&output, "info string HCE is active");
            }
            emit(&output, "uciok");
        } else if command == "isready" {
            emit(&output, "readyok");
        } else if command == "ucinewgame" {
            stop_search(&mut worker);
            board = Board::startpos();
            game_history = vec![board.hash()];
            if let Ok(mut engine) = searcher.lock() { engine.clear(); }
        } else if command.starts_with("position ") {
            stop_search(&mut worker);
            let mut candidate_position = board.clone();
            match play_position_command_with_history(&mut candidate_position, command) {
                Ok(history) => {
                    board = candidate_position;
                    game_history = history;
                }
                Err(error) => emit(&output, &format!("info string position error: {error}")),
            }
        } else if let Some(rest) = command.strip_prefix("setoption name ") {
            stop_search(&mut worker);
            let mut parts = rest.splitn(2, char::is_whitespace);
            let name = parts.next().unwrap_or("");
            let remainder = parts.next().unwrap_or("").trim_start();
            let value = if remainder == "value" {
                ""
            } else {
                remainder.strip_prefix("value ").unwrap_or(remainder)
            }.trim();
            if name.eq_ignore_ascii_case("hash") {
                if let Ok(requested) = value.parse::<usize>() {
                    if (1..=4096).contains(&requested) {
                        if let Ok(mut engine) = searcher.lock() { engine.resize_hash(requested); }
                    } else {
                        emit(&output, "info string Hash must be between 1 and 4096 MB");
                    }
                } else {
                    emit(&output, "info string invalid Hash value");
                }
            } else if name.eq_ignore_ascii_case("threads") {
                if value.parse::<u8>() != Ok(1) {
                    emit(&output, "info string Threads is fixed at 1");
                }
            } else if name.eq_ignore_ascii_case("eval") {
                if value.eq_ignore_ascii_case("hce") {
                    want_nnue = false;
                    eval_mode = EvalMode::Hce;
                    evaluator = Arc::new(Hce);
                    if let Ok(mut engine) = searcher.lock() { engine.clear(); }
                    emit(&output, "info string HCE is active");
                } else if value.eq_ignore_ascii_case("nnue") {
                    want_nnue = true;
                    if nnue_network.is_none() && !eval_file.is_empty() {
                        match QuantizedNnueNetwork::<128>::load(Path::new(&eval_file)) {
                            Ok(network) => nnue_network = Some(Arc::new(network)),
                            Err(error) => emit(&output, &format!("info string Eval NNUE load failed: {}", one_line(&error))),
                        }
                    }
                    if let Some(network) = nnue_network.as_ref() {
                        eval_mode = EvalMode::Nnue;
                        evaluator = network.clone();
                        if let Ok(mut engine) = searcher.lock() { engine.clear(); }
                        emit(&output, "info string NNUE is active");
                    } else {
                        emit(&output, "info string Eval NNUE unavailable; HCE remains active");
                    }
                } else {
                    emit(&output, "info string Eval must be HCE or NNUE");
                }
            } else if name.eq_ignore_ascii_case("evalfile") {
                if value.is_empty() {
                    eval_file.clear();
                    nnue_network = None;
                    if eval_mode == EvalMode::Nnue {
                        eval_mode = EvalMode::Hce;
                        evaluator = Arc::new(Hce);
                    }
                    if let Ok(mut engine) = searcher.lock() { engine.clear(); }
                    emit(&output, "info string EvalFile cleared; HCE is active if NNUE was selected");
                } else {
                    match QuantizedNnueNetwork::<128>::load(Path::new(value)) {
                        Ok(network) => {
                            let network = Arc::new(network);
                            eval_file = value.to_string();
                            nnue_network = Some(Arc::clone(&network));
                            if want_nnue {
                                eval_mode = EvalMode::Nnue;
                                evaluator = network;
                            }
                            if let Ok(mut engine) = searcher.lock() { engine.clear(); }
                            emit(&output, &format!("info string EvalFile loaded {}; {} is active", one_line(&eval_file), if eval_mode == EvalMode::Nnue { "NNUE" } else { "HCE" }));
                        }
                        Err(error) => emit(&output, &format!("info string EvalFile load failed; current settings unchanged: {}", one_line(&error))),
                    }
                }
            }
        } else if let Some(rest) = command.strip_prefix("go perft ") {
            stop_search(&mut worker);
            match rest.parse::<u8>() {
                Ok(depth) if depth <= 10 => {
                    let mut position = board.clone();
                    let divide = position.perft_divide(depth);
                    for (move_uci, count) in &divide {
                        emit(&output, &format!("{move_uci}: {count}"));
                        emit(&output, &format!("info string perft divide {move_uci} {count}"));
                    }
                    let nodes = if depth == 0 { 1 } else { divide.iter().map(|(_, count)| count).sum() };
                    emit(&output, &format!("Nodes searched: {nodes}"));
                    emit(&output, &format!("info string perft depth {depth} nodes {nodes}"));
                }
                _ => {
                    emit(&output, "info string invalid perft depth");
                    emit(&output, "Nodes searched: 0");
                }
            }
        } else if command == "d" {
            emit(&output, &format!("Fen: {}", board.to_fen()));
            if board.is_in_check(board.white_to_move) {
                emit(&output, "Checkers: check");
            } else {
                emit(&output, "Checkers:");
            }
        } else if command.starts_with("go ") || command == "go" {
            stop_search(&mut worker);
            let limits = parse_go_limits(command, board.white_to_move);
            let position = board.clone();
            let history = game_history.clone();
            let searcher = Arc::clone(&searcher);
            let output = Arc::clone(&output);
            let evaluator = Arc::clone(&evaluator);
            let stop = Arc::new(AtomicBool::new(false));
            let thread_stop = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                let result = if let Ok(mut engine) = searcher.lock() {
                    let evaluator = SharedEvaluator(evaluator);
                    engine.search(&position, &history, &evaluator, limits, &thread_stop, |info| {
                        emit(&output, &format_info(&info));
                    })
                } else {
                    SearchResult { best_move: None, score: 0, depth: 0, nodes: 0 }
                };
                emit(&output, &format!("bestmove {}", result.best_move.map_or_else(|| "0000".to_owned(), |mv| mv.uci())));
            });
            worker = Some((handle, stop));
        } else if command == "stop" {
            stop_search(&mut worker);
        } else if command == "quit" {
            stop_search(&mut worker);
            break;
        }
    }
    stop_search(&mut worker);
}

fn one_line(value: &str) -> String {
    value.replace('\r', " ").replace('\n', " ")
}

fn parse_go_limits(command: &str, white_to_move: bool) -> SearchLimits {
    let fields: Vec<&str> = command.split_whitespace().collect();
    let mut depth = None;
    let mut nodes = None;
    let mut movetime = None;
    let mut wtime = None;
    let mut btime = None;
    let mut winc = 0u64;
    let mut binc = 0u64;
    let mut infinite = false;
    let mut index = 1;
    while index < fields.len() {
        let token = fields[index];
        if token == "infinite" {
            infinite = true;
            index += 1;
            continue;
        }
        let value = fields.get(index + 1).and_then(|s| s.parse::<u64>().ok());
        match token {
            "depth" => depth = value.map(|n| n.min(127) as u8),
            "nodes" => nodes = value,
            "movetime" => movetime = value,
            "wtime" => wtime = value,
            "btime" => btime = value,
            "winc" => winc = value.unwrap_or(0),
            "binc" => binc = value.unwrap_or(0),
            _ => { index += 1; continue; }
        }
        index += 2;
    }

    let budget_ms = if infinite {
        None
    } else if let Some(ms) = movetime {
        Some(ms.max(1))
    } else {
        let remaining = if white_to_move { wtime } else { btime };
        let increment = if white_to_move { winc } else { binc };
        remaining.map(|ms| {
            let reserve = 10u64.min(ms / 4);
            let usable = ms.saturating_sub(reserve);
            let target = usable / 30 + increment.saturating_mul(8) / 10;
            target.max(1).min(usable.max(1) / 2)
        })
    };
    let deadline = budget_ms.map(|ms| Instant::now() + Duration::from_millis(ms));

    SearchLimits {
        depth: depth.or_else(|| {
            if infinite || nodes.is_some() || budget_ms.is_some() { None } else { Some(8) }
        }),
        nodes,
        deadline,
    }
}

fn format_info(info: &SearchInfo) -> String {
    let score = if info.score.abs() >= MATE_SCORE - 256 {
        let plies = MATE_SCORE.saturating_sub(info.score.abs());
        let mate_moves = (plies + 1) / 2;
        format!("score mate {}", if info.score < 0 { -mate_moves } else { mate_moves })
    } else {
        format!("score cp {}", info.score)
    };
    let nps = info.nodes.saturating_mul(1000) / (info.elapsed_ms.max(1) as u64);
    format!(
        "info depth {} {score} nodes {} nps {nps} time {} pv {}",
        info.depth,
        info.nodes,
        info.elapsed_ms,
        info.pv.join(" "),
    )
}
