"""
CogInfera — CLI Entry Point

Usage:
    python main.py ingest <pdf_path>
    python main.py query "<your question>"
    python main.py interactive
"""

import sys
import textwrap
from orchestrator import CogInfera


def print_answer(result: dict):
    """Pretty-print a query result."""
    print("\n" + "=" * 70)
    print("  QUESTION")
    print("=" * 70)
    print(textwrap.fill(result["question"], width=68, initial_indent="  ", subsequent_indent="  "))

    plan = result.get("plan", {})
    if plan.get("complexity"):
        print(f"\n  Complexity : {plan['complexity']}")
    if plan.get("sub_queries") and len(plan["sub_queries"]) > 1:
        print("  Sub-queries:")
        for sq in plan["sub_queries"]:
            print(f"    • {sq}")

    print("\n" + "-" * 70)
    print("  ANSWER")
    print("-" * 70)
    print()
    print(textwrap.fill(result["answer"], width=68, initial_indent="  ", subsequent_indent="  "))

    validation = result.get("validation", {})
    if validation:
        conf = validation.get("confidence", "?")
        grounded = "✓" if validation.get("is_grounded") else "✗"
        complete = "✓" if validation.get("is_complete") else "✗"
        print(f"\n  Confidence: {conf}  |  Grounded: {grounded}  |  Complete: {complete}")

        issues = validation.get("issues", [])
        if issues:
            print("  Issues:")
            for issue in issues:
                print(f"    ⚠ {issue}")

    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <pdf_path>")
            sys.exit(1)
        pdf_path = sys.argv[2]
        engine = CogInfera()
        engine.ingest(pdf_path)

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query \"<your question>\"")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        engine = CogInfera()
        result = engine.query(question)
        print_answer(result)

    elif command == "interactive":
        engine = CogInfera()
        print("\n[CogInfera Interactive Mode]")
        print("Type your questions. Type 'quit' to exit.\n")
        while True:
            try:
                question = input("You: ").strip()
                if not question or question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                result = engine.query(question)
                print_answer(result)
                print()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
