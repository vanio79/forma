"""
LongMemEval Benchmark: Parallel Turn-by-Turn Feeding with Sliding Window

Both models receive identical conversation history (within window limit).
Forma builds RAG memory during feeding phase.
When history exceeds limit, oldest messages are discarded from both models.
"""

import json
import requests
import time
import sys
import re
import traceback
from datetime import datetime
from pathlib import Path

# Configuration
FORMA_URL = "http://localhost:8000/v1/chat/completions"
DIRECT_URL = "http://192.168.68.10:1234/v1/chat/completions"
MODEL = "gemma-4-e4b-it"
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data" / "longmemeval_oracle.json"
OUTPUT_DIR = SCRIPT_DIR / "results"
MAX_QUESTIONS = 50

# Token limit
MAX_CONTEXT_TOKENS = 1000

OUTPUT_DIR.mkdir(exist_ok=True)


def estimate_tokens(text: str) -> int:
    """Estimate tokens (approximation: ~4 chars per token)."""
    return len(text) // 4


def count_history_tokens(history: list) -> int:
    """Count total tokens in history."""
    total = 0
    for msg in history:
        total += 4  # Role overhead
        total += estimate_tokens(msg.get("content", ""))
    return total


def trim_history(history: list, max_tokens: int) -> tuple[list, int, int]:
    """Trim oldest messages to stay within token limit. Returns (trimmed_history, overflow_tokens, removed_messages)."""
    total = count_history_tokens(history)
    trimmed = list(history)
    removed_messages = 0

    # First, trim for token limit
    if total > max_tokens:
        # Keep at least 2 messages (one user + one assistant pair)
        while count_history_tokens(trimmed) > max_tokens and len(trimmed) > 2:
            trimmed.pop(0)
            removed_messages += 1

    # Always ensure history starts with 'user' (chat templates require this)
    # If first message is 'assistant', remove it
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
        removed_messages += 1

    # If we removed everything, keep at least one user message if available from original
    if not trimmed and history:
        for msg in history:
            if msg.get("role") == "user":
                trimmed = [msg]
                removed_messages = len(history) - 1
                break

    overflow_tokens = total - count_history_tokens(trimmed)
    return trimmed, overflow_tokens, removed_messages


def send_to_model(
    url: str, history: list, user_message: str, max_tokens: int = 150, debug: bool = False
) -> dict:
    """Send history + new user message to a model."""
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    # Debug: print payload structure
    if debug:
        print(f"    [DEBUG] Payload to {url}:")
        print(f"    [DEBUG]   model: {payload['model']}")
        print(f"    [DEBUG]   messages count: {len(payload['messages'])}")
        print(f"    [DEBUG]   message roles: {[m['role'] for m in payload['messages']]}")
        for i, m in enumerate(payload["messages"]):
            content_preview = m.get("content", "")[:50] if m.get("content") else "EMPTY"
            print(f'    [DEBUG]   [{i}] {m["role"]}: "{content_preview}..."')
        sys.stdout.flush()

    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return {"status": "ok", "content": content}
        else:
            # Include FULL response body for debugging
            try:
                error_body = response.json()
            except:
                error_body = response.text

            # Print error details immediately
            print(f"    [ERROR] HTTP {response.status_code} from {url}")
            print(f"    [ERROR] Response body: {json.dumps(error_body, indent=2)[:500]}")
            sys.stdout.flush()

            return {
                "status": "error",
                "code": response.status_code,
                "body": error_body,
                "payload": payload,  # Include payload for debugging
            }
    except requests.exceptions.Timeout:
        print(f"    [ERROR] Timeout (60s) from {url}")
        sys.stdout.flush()
        return {"status": "error", "message": "Timeout after 60s", "payload": payload}
    except Exception as e:
        print(f"    [ERROR] Exception: {type(e).__name__}: {str(e)}")
        print(f"    [ERROR] Traceback: {traceback.format_exc()[:300]}")
        sys.stdout.flush()
        return {
            "status": "error",
            "message": f"{type(e).__name__}: {str(e)[:100]}",
            "payload": payload,
        }


def feed_session(session: list, shared_history: list, debug: bool = False) -> tuple[list, dict]:
    """
    Feed a session to build shared history and trigger Forma extraction.

    - Use original session messages for history (not model-generated responses)
    - Send user messages to Forma to trigger extraction (build RAG memory)
    - Returns a SINGLE shared history used by both models

    Returns: (updated shared_history, stats)
    """
    stats = {"turns_processed": 0, "overflow_total": 0, "removed_messages": 0, "errors": []}

    # Build shared history from original session messages
    for turn_idx, turn in enumerate(session):
        role = turn.get("role")
        content = turn.get("content")

        # Validate message structure
        if content is None:
            print(f"    [WARN] Turn {turn_idx}: Missing content field")
            content = ""
        elif not isinstance(content, str):
            print(
                f"    [WARN] Turn {turn_idx}: Content is not string (type: {type(content).__name__})"
            )
            content = str(content) if content else ""

        # Append original message to shared history
        shared_history.append({"role": role, "content": content})

        # For user messages, send to Forma for extraction
        if role == "user":
            # Trim history before sending
            trimmed, overflow, removed = trim_history(shared_history, MAX_CONTEXT_TOKENS)

            if overflow > 0:
                stats["overflow_total"] += overflow
                stats["removed_messages"] += removed

            shared_history = trimmed

            # Send to Forma for extraction (response not used for history)
            # Pass history[:-1] because we just appended the user message
            forma_result = send_to_model(
                FORMA_URL, shared_history[:-1], content, max_tokens=10, debug=debug
            )

            if forma_result.get("status") == "error":
                stats["errors"].append(
                    {
                        "turn": turn_idx,
                        "phase": "feeding",
                        "url": FORMA_URL,
                        "error": forma_result,
                    }
                )

            stats["turns_processed"] += 1

    return shared_history, stats


def ask_question(shared_history: list, question: str, debug: bool = False) -> dict:
    """Ask question to both models with IDENTICAL histories."""
    # Trim history before asking
    shared_trimmed, overflow, removed = trim_history(shared_history, MAX_CONTEXT_TOKENS)

    # Validate question
    if not question:
        print(f"    [ERROR] Empty question!")
        sys.stdout.flush()

    # Debug print history structure before asking
    if debug:
        print(f"    [DEBUG] History for question:")
        print(
            f"    [DEBUG]   Turns: {len(shared_trimmed)}, Tokens: {count_history_tokens(shared_trimmed)}"
        )
        print(f"    [DEBUG]   Roles: {[m['role'] for m in shared_trimmed]}")
        sys.stdout.flush()

    # Ask Forma (with RAG augmentation from stored memory)
    forma_result = send_to_model(FORMA_URL, shared_trimmed, question, max_tokens=150, debug=debug)

    # Ask Direct (no RAG, only window context)
    direct_result = send_to_model(DIRECT_URL, shared_trimmed, question, max_tokens=150, debug=debug)

    history_tokens = count_history_tokens(shared_trimmed)

    return {
        "forma_answer": forma_result.get("content", "ERROR"),
        "direct_answer": direct_result.get("content", "ERROR"),
        "forma_result": forma_result,
        "direct_result": direct_result,
        "history_tokens": history_tokens,
        "history_turns": len(shared_trimmed),
        "overflow_tokens": overflow,
        "removed_messages": removed,
        "history_sample": shared_trimmed[:3]
        if len(shared_trimmed) > 3
        else shared_trimmed,  # Include sample of history for debugging
    }


def evaluate_answer(question: str, expected: str, answer: str) -> float:
    """Evaluate single answer against expected."""
    # Skip evaluation for error answers
    if answer == "ERROR" or not answer:
        return 0.0

    eval_prompt = f"""Rate how well this answer matches the expected answer.

Question: {question}
Expected: {expected}
Answer: {answer}

Score from 0 to 1:
- 1.0: Perfect match, all key info present
- 0.7-0.9: Mostly correct, minor gaps
- 0.4-0.6: Partial, some key info
- 0.0-0.3: Wrong or missing most info

Score (just the number):"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": eval_prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    try:
        response = requests.post(DIRECT_URL, json=payload, timeout=30)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
            # Extract number
            match = re.search(r"(\d+\.\d+|\d+)", text)
            if match:
                return float(match.group(1))
        else:
            print(f"    [WARN] Evaluation failed: HTTP {response.status_code}")
            sys.stdout.flush()
    except Exception as e:
        print(f"    [WARN] Evaluation error: {str(e)[:50]}")
        sys.stdout.flush()
    return 0.5


def run_question_test(question_data: dict, debug: bool = False) -> dict:
    """Run test for a single question."""
    question_id = question_data["question_id"]
    question = question_data["question"]
    expected = question_data["answer"]
    sessions = question_data["haystack_sessions"]

    print(f"\n  Question: {question[:80]}...")
    sys.stdout.flush()

    # Initialize empty shared history (used by both models)
    shared_history = []

    # Feed all sessions to build history and trigger Forma extraction
    print(f"  Feeding {len(sessions)} sessions...")
    sys.stdout.flush()

    total_stats = {"turns_processed": 0, "overflow_total": 0, "removed_messages": 0, "errors": []}

    for i, session in enumerate(sessions):
        print(f"    Session {i + 1}/{len(sessions)}: {len(session)} turns")
        sys.stdout.flush()

        shared_history, stats = feed_session(session, shared_history, debug=debug)

        total_stats["turns_processed"] += stats["turns_processed"]
        total_stats["overflow_total"] += stats["overflow_total"]
        total_stats["removed_messages"] += stats.get("removed_messages", 0)
        total_stats["errors"].extend(stats.get("errors", []))

        # Brief pause between sessions
        time.sleep(0.1)

    # Report any feeding errors
    if total_stats.get("errors"):
        print(f"    [WARN] {len(total_stats['errors'])} errors during feeding")
        for err in total_stats["errors"][:3]:  # Show first 3
            print(
                f"    [WARN]   Turn {err['turn']}: {err['error'].get('code', err['error'].get('message', 'unknown'))}"
            )
        sys.stdout.flush()

    print(
        f"  Total turns fed: {total_stats['turns_processed']}, Overflow: {total_stats['overflow_total']} tokens ({total_stats.get('removed_messages', 0)} messages trimmed)"
    )
    sys.stdout.flush()

    # Ask the question
    print(f"  Asking question to both models...")
    sys.stdout.flush()

    answers = ask_question(shared_history, question, debug=debug)

    print(
        f"    Shared history: {answers['history_tokens']} tokens, {answers['history_turns']} turns (overflow: {answers['overflow_tokens']} tokens)"
    )
    sys.stdout.flush()

    # Evaluate
    print(f"  Evaluating answers...")
    sys.stdout.flush()

    forma_score = evaluate_answer(question, expected, answers["forma_answer"])
    direct_score = evaluate_answer(question, expected, answers["direct_answer"])

    # Print answers for visibility
    print(f"\n  Expected: {expected[:150]}...")

    # Show Forma answer or error details
    forma_result = answers.get("forma_result", {})
    if forma_result.get("status") == "error":
        if "body" in forma_result:
            error_detail = forma_result["body"]
            if isinstance(error_detail, dict):
                print(
                    f"  Forma:    ERROR (HTTP {forma_result.get('code')}): {json.dumps(error_detail)[:300]}"
                )
            else:
                print(
                    f"  Forma:    ERROR (HTTP {forma_result.get('code')}): {str(error_detail)[:300]}"
                )
        elif "code" in forma_result:
            print(f"  Forma:    ERROR (HTTP {forma_result['code']})")
        elif "message" in forma_result:
            print(f"  Forma:    ERROR ({forma_result['message']})")
        else:
            print(f"  Forma:    ERROR (unknown)")
        # Print payload if available
        if "payload" in forma_result:
            print(f"    Payload messages count: {len(forma_result['payload'].get('messages', []))}")
    else:
        print(f"  Forma:    {answers['forma_answer'][:150]}...")

    # Show Direct answer or error details
    direct_result = answers.get("direct_result", {})
    if direct_result.get("status") == "error":
        if "body" in direct_result:
            error_detail = direct_result["body"]
            if isinstance(error_detail, dict):
                print(
                    f"  Direct:   ERROR (HTTP {direct_result.get('code')}): {json.dumps(error_detail)[:300]}"
                )
            else:
                print(
                    f"  Direct:   ERROR (HTTP {direct_result.get('code')}): {str(error_detail)[:300]}"
                )
        elif "code" in direct_result:
            print(f"  Direct:   ERROR (HTTP {direct_result['code']})")
        elif "message" in direct_result:
            print(f"  Direct:   ERROR ({direct_result['message']})")
        else:
            print(f"  Direct:   ERROR (unknown)")
        # Print payload if available
        if "payload" in direct_result:
            print(
                f"    Payload messages count: {len(direct_result['payload'].get('messages', []))}"
            )
    else:
        print(f"  Direct:   {answers['direct_answer'][:150]}...")

    print(f"  Scores: Forma={forma_score:.2f}, Direct={direct_score:.2f}")
    sys.stdout.flush()

    return {
        "question_id": question_id,
        "question_type": question_data["question_type"],
        "question": question,
        "expected_answer": expected,
        "forma_answer": answers["forma_answer"],
        "direct_answer": answers["direct_answer"],
        "forma_score": forma_score,
        "direct_score": direct_score,
        "history_tokens": answers["history_tokens"],
        "history_turns": answers["history_turns"],
        "overflow_tokens": answers["overflow_tokens"],
        "removed_messages": answers["removed_messages"],
        "total_turns_fed": total_stats["turns_processed"],
        "total_overflow": total_stats["overflow_total"],
        "total_removed_messages": total_stats.get("removed_messages", 0),
        "feeding_errors": len(total_stats.get("errors", [])),
        "forma_result": forma_result,
        "direct_result": direct_result,
    }


def main():
    """Run benchmark."""
    print("=" * 80)
    print("LongMemEval Benchmark: Parallel Turn-by-Turn with Sliding Window")
    print("=" * 80)
    print(f"\nMax context tokens: {MAX_CONTEXT_TOKENS}")
    print(f"Model: {MODEL}")
    print(f"Forma URL: {FORMA_URL}")
    print(f"Direct URL: {DIRECT_URL}")
    sys.stdout.flush()

    # Clear Forma storage first
    print("\nClearing Forma storage...")
    sys.stdout.flush()
    try:
        clear_resp = requests.post("http://localhost:8000/admin/clear", timeout=10)
        if clear_resp.status_code == 200:
            print(f"  Cleared: {clear_resp.json()}")
        else:
            print(f"  [WARN] Clear failed: HTTP {clear_resp.status_code}")
    except Exception as e:
        print(f"  [WARN] Clear error: {str(e)}")
    sys.stdout.flush()

    # Load data
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    test_data = data[:MAX_QUESTIONS]
    print(f"\nTesting {len(test_data)} questions")
    sys.stdout.flush()

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "forma_url": FORMA_URL,
            "direct_url": DIRECT_URL,
            "model": MODEL,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_questions": MAX_QUESTIONS,
        },
        "questions": [],
    }

    # Run tests (enable debug for first 2 questions)
    for i, q_data in enumerate(test_data):
        print(f"\n[{i + 1}/{len(test_data)}] {q_data['question_id']} ({q_data['question_type']})")
        sys.stdout.flush()

        # Debug mode disabled for production runs
        debug_mode = False

        result = run_question_test(q_data, debug=debug_mode)
        results["questions"].append(result)

        time.sleep(0.3)

    # Calculate totals
    forma_scores = [q["forma_score"] for q in results["questions"]]
    direct_scores = [q["direct_score"] for q in results["questions"]]

    avg_forma = sum(forma_scores) / len(forma_scores)
    avg_direct = sum(direct_scores) / len(direct_scores)

    forma_wins = sum(1 for f, d in zip(forma_scores, direct_scores) if f > d)
    direct_wins = sum(1 for f, d in zip(forma_scores, direct_scores) if d > f)
    ties = sum(1 for f, d in zip(forma_scores, direct_scores) if f == d)

    avg_overflow = sum(q["total_overflow"] for q in results["questions"]) / len(
        results["questions"]
    )
    avg_removed = sum(q["total_removed_messages"] for q in results["questions"]) / len(
        results["questions"]
    )
    total_errors = sum(q.get("feeding_errors", 0) for q in results["questions"])

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nAverage Forma Score:   {avg_forma:.3f}")
    print(f"Average Direct Score:  {avg_direct:.3f}")
    if avg_direct > 0:
        print(
            f"Forma Improvement:     {(avg_forma - avg_direct):.3f} ({((avg_forma / avg_direct) - 1) * 100:+.1f}%)"
        )
    else:
        print(f"Forma Improvement:     {(avg_forma - avg_direct):.3f}")
    print(f"\nWins:")
    print(f"  Forma:  {forma_wins} ({forma_wins * 100 / len(results['questions']):.1f}%)")
    print(f"  Direct: {direct_wins} ({direct_wins * 100 / len(results['questions']):.1f}%)")
    print(f"  Ties:   {ties} ({ties * 100 / len(results['questions']):.1f}%)")
    print(f"\nContext Overflow:")
    print(f"  Average overflow: {avg_overflow:.0f} tokens")
    print(f"  Average removed messages: {avg_removed:.1f}")
    print(f"  (This is the info Forma can retrieve via RAG but Direct model loses)")
    print(f"\nErrors:")
    print(f"  Total feeding errors: {total_errors}")

    # Sample results with full details
    print("\n" + "-" * 40)
    print("SAMPLE (first 5):")
    for q in results["questions"][:5]:
        print(f"\n{'─' * 40}")
        print(f"Question: {q['question'][:80]}...")
        print(f"Type: {q['question_type']}")
        print(
            f"Overflow: {q['total_overflow']} tokens, {q['total_removed_messages']} messages trimmed"
        )
        if q.get("feeding_errors"):
            print(f"Feeding errors: {q['feeding_errors']}")
        print(f"\nExpected: {q['expected_answer'][:80]}...")
        print(f"\nForma ({q['forma_score']:.2f}): {q['forma_answer'][:120]}...")
        print(f"\nDirect ({q['direct_score']:.2f}): {q['direct_answer'][:120]}...")

    # Save
    output_file = OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_file}")

    # Summary
    summary = {
        "timestamp": results["timestamp"],
        "avg_forma_score": avg_forma,
        "avg_direct_score": avg_direct,
        "improvement": avg_forma - avg_direct,
        "improvement_percent": ((avg_forma / avg_direct) - 1) * 100 if avg_direct > 0 else 0,
        "forma_wins": forma_wins,
        "direct_wins": direct_wins,
        "ties": ties,
        "avg_overflow_tokens": avg_overflow,
        "avg_removed_messages": avg_removed,
        "total_questions": len(results["questions"]),
        "total_errors": total_errors,
    }

    summary_file = OUTPUT_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
