// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream parser on BATCH samples: feed each batch fixture's full
//! `model_text` to the streaming parser and assert the assembled tool calls match
//! the BATCH parser's `expected.dynamo`. This is the streaming-vs-batch
//! consistency check — the stream parser, given the complete output, must land on
//! the same calls as the batch parser.

use std::collections::BTreeMap;

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolParseResult, assemble_tool_calls,
    create_tool_parser_for_family,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    family: String,
    mode: String,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    model_text: Option<String>,
    #[serde(default)]
    expected: Option<Expected>,
    // The schema-dependent parsers (glm47, kimi_k2, qwen3_coder, minimax_m2, …)
    // need the tool schema to coerce argument types the way the v1 batch parser
    // did; the batch fixture carries it per case.
    #[serde(default)]
    tools: Vec<Tool>,
}

#[derive(Deserialize)]
struct Expected {
    dynamo: EngineExpected,
}

#[derive(Deserialize)]
struct EngineExpected {
    #[serde(default)]
    calls: Vec<ExpCall>,
    #[serde(default)]
    normal_text: String,
}

#[derive(Deserialize)]
struct ExpCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[test]
fn toolcalling_batch_via_stream_parity() {
    // Versioned corpus (inputs/ + <impl>-<version>/): read the shared inputs and fold
    // Dynamo v1's `expected.dynamo` from the (single) dynamo-<version>/ dir back in.
    let batch_root = common::ensure_fixtures().join("toolcalling/fixtures-batch-v1");
    let inputs_root = batch_root.join("inputs");
    let dyn_dir = std::fs::read_dir(batch_root)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("dynamo-"))
        })
        .expect("no dynamo-<version> dir under fixtures-batch-v1");
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();

    // Batch samples where the streaming parser deliberately differs from the
    // strict batch parser. Removing an entry asserts that stream and batch now
    // agree on that sample. This compares BOTH calls and normal_text; the HTML
    // batch-on-stream tab compares calls only, so the `normal_text`-only entries
    // below still render green there.
    //
    // The DSv4 stream parser now buffers each invoke until `</｜DSML｜invoke>` and
    // drops a call truncated before its close (v1 parity), so the former 5.c /
    // 5.e truncation divergences are gone. Remaining entries:
    //   deepseek_v4 5.g: bare invoke after prose — the stream parser recovers it
    //        while the strict batch parser drops it (recovery divergence).
    //   gemma4 / kimi_k2 / glm47 / minimax_m2 / qwen3_coder (calls IDENTICAL,
    //        normal_text only): the streaming parser faithfully emits the model's
    //        text AROUND the tool calls VERBATIM, while the v1 batch parser trims
    //        surrounding whitespace that would otherwise be the entire (or a
    //        trailing) normal_text. The calls always match; only whitespace-only
    //        surrounding text differs. e.g. gemma4 8.b stream "  Let me know if you
    //        need more." vs batch ""; the XML families' 2.a/2.b/2.d/10 (parallel
    //        calls separated only by a newline) stream "\n" vs batch ""; qwen3 5.d/
    //        5.e keep one extra trailing "\n" the batch parser trims. The streaming
    //        peers stream this surrounding text the same way, and the HTML
    //        batch-on-stream tab compares calls only, so these render green there.
    let known_divergences: std::collections::BTreeSet<&str> = [
        "deepseek_v4:TOOLCALLING.batch.5.g",
        "gemma4:TOOLCALLING.batch.5.g",
        "gemma4:TOOLCALLING.batch.8.a",
        "gemma4:TOOLCALLING.batch.8.b",
        "gemma4:TOOLCALLING.batch.8.d",
        "kimi_k2:TOOLCALLING.batch.5.g",
        "minimax_m2:TOOLCALLING.batch.2.b",
        "qwen3_coder:TOOLCALLING.batch.2.a",
        "qwen3_coder:TOOLCALLING.batch.2.d",
        "qwen3_coder:TOOLCALLING.batch.5.d",
        "qwen3_coder:TOOLCALLING.batch.5.e",
        "qwen3_coder:TOOLCALLING.batch.10",
    ]
    .into_iter()
    .collect();

    let mut total = 0usize;
    let mut consistent = 0usize;
    let mut diverged = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut unexpected_match: Vec<String> = Vec::new();

    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let mut fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{}: YAML parse error: {e}", path.display()));
                continue;
            }
        };
        if fx.mode != "batch" {
            continue;
        }
        // Data-driven coverage (reuse the family registry, no hardcoded list):
        // harmony runs the token/text Harmony path; every other family is
        // exercised iff `create_tool_parser_for_family` can build a v2 parser for
        // it. Registering a new family there auto-adds it to this stream-on-batch
        // consistency check.
        if fx.family != "harmony" && create_tool_parser_for_family(&fx.family, &[]).is_err() {
            continue;
        }
        let rel = path.strip_prefix(&inputs_root).unwrap();
        let dyn_fx = std::fs::read_to_string(dyn_dir.join(rel))
            .ok()
            .and_then(|t| serde_yaml::from_str::<Fixture>(&t).ok());
        if let Some(dfx) = dyn_fx {
            for (cid, dcase) in dfx.cases {
                if let (Some(c), Some(exp)) = (fx.cases.get_mut(&cid), dcase.expected) {
                    c.expected = Some(exp);
                }
            }
        }
        eprintln!("fixture {}", fixture_name(path));

        for (cid, case) in &fx.cases {
            let (Some(text), Some(expected)) = (case.model_text.as_ref(), case.expected.as_ref())
            else {
                continue; // placeholder case
            };
            total += 1;

            let got = parse_stream_result(&fx.family, text, &case.tools).unwrap();
            let want = EngineResult {
                calls: expected
                    .dynamo
                    .calls
                    .iter()
                    .map(|c| (c.name.clone(), c.arguments.clone()))
                    .collect(),
                normal_text: expected.dynamo.normal_text.clone(),
            };

            let known_id = format!("{}:{cid}", fx.family);
            let known = known_divergences.contains(known_id.as_str());
            if got == want {
                consistent += 1;
                if known {
                    // It now agrees — the allowlist entry is stale.
                    unexpected_match.push(known_id);
                }
            } else {
                diverged += 1;
                if !known {
                    failures.push(format!(
                        "{} {cid}:\n        stream got {got:?}\n        batch want {want:?}",
                        fx.family
                    ));
                }
            }
        }
    }

    eprintln!(
        "Dynamo stream-on-batch: {consistent}/{total} consistent, {diverged} diverged \
         ({} are known/documented)",
        diverged - failures.len(),
    );
    for f in &failures {
        eprintln!("UNEXPECTED DIVERGENCE {f}");
    }
    for c in &unexpected_match {
        eprintln!("STALE ALLOWLIST (now agrees, drop it): {c}");
    }
    assert!(
        failures.is_empty(),
        "{} batch samples newly diverged between stream and batch (not in the \
         known-divergence allowlist)",
        failures.len()
    );
    assert!(
        unexpected_match.is_empty(),
        "{} allowlist entries now agree — remove them",
        unexpected_match.len()
    );
}

#[derive(Debug, PartialEq, Eq)]
struct EngineResult {
    calls: Vec<(String, Value)>,
    normal_text: String,
}

fn parse_stream_result(
    family: &str,
    text: &str,
    tools: &[Tool],
) -> Result<EngineResult, Box<dyn std::error::Error>> {
    if family == "harmony" {
        let mut parser = HarmonyToolStreamParser::new()?;
        let mut result = parser.parse_tool_call_streaming_text(text);
        let finish = parser.finish_tool_call_stream();
        result.normal_text.push_str(&finish.normal_text);
        result.tool_call_chunks.extend(finish.tool_call_chunks);
        return Ok(EngineResult {
            calls: assemble_tool_calls(&result.tool_call_chunks)
                .into_iter()
                .map(|(n, a)| {
                    let v = serde_json::from_str(&a).unwrap_or(Value::String(a));
                    (n, v)
                })
                .collect(),
            normal_text: result.normal_text,
        });
    }

    let mut parser = create_tool_parser_for_family(family, tools)?;
    let mut result = parser.push(text)?;
    result.append(parser.finish()?);
    Ok(EngineResult {
        normal_text: result.normal_text.clone(),
        calls: assemble_trait_calls(result),
    })
}

fn assemble_trait_calls(result: ToolParseResult) -> Vec<(String, Value)> {
    let mut names = BTreeMap::<usize, String>::new();
    let mut args = BTreeMap::<usize, String>::new();
    for ToolCallDelta {
        tool_index,
        name,
        arguments,
    } in result.calls
    {
        if let Some(name) = name {
            names.entry(tool_index).or_default().push_str(&name);
        }
        args.entry(tool_index).or_default().push_str(&arguments);
    }
    names
        .into_iter()
        .map(|(idx, name)| {
            let raw = args.remove(&idx).unwrap_or_default();
            let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            (name, value)
        })
        .collect()
}
