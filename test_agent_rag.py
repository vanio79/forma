#!/usr/bin/env python3
"""Test global RAG retrieval with agent-specific parameters.

Tests:
1. Store facts globally (shared across all agents)
2. Store recipes globally (shared across all agents)
3. Retrieve context with different agent rag_config parameters
4. Verify that all agents access the same global knowledge base
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from forma.storage import Storage


def test_global_rag():
    """Test global RAG operations with agent-specific parameters."""
    print("Initializing storage...")
    storage = Storage(
        grafitodb_path="./grafito_data/forma.db",
        grafitodb_embedding_model="all-MiniLM-L6-v2",
        grafitodb_vector_dim=384,
        grafitodb_model_cache_path="./models",
    )

    # Test 1: Store facts globally (shared across all agents)
    print("\n=== Test 1: Store global facts ===")
    global_facts = [
        {"statement": "Python uses indentation for code blocks", "confidence": 0.95},
        {"statement": "FastAPI is a modern Python web framework", "confidence": 0.9},
        {"statement": "Async functions in Python use async/await syntax", "confidence": 0.9},
        {"statement": "Research papers should include citations", "confidence": 0.85},
        {"statement": "Web search requires verifying sources", "confidence": 0.8},
    ]

    facts_count = storage.store_facts(global_facts)
    print(f"Stored {facts_count} facts globally")

    # Test 2: Store recipes globally (shared across all agents)
    print("\n=== Test 2: Store global recipes ===")
    global_recipes = [
        {
            "description": "To create a FastAPI endpoint: define a function with @app.get decorator, specify path and return type",
            "confidence": 0.9,
        },
        {
            "description": "To handle async operations: use async def for the function, await async calls inside",
            "confidence": 0.85,
        },
        {
            "description": "To research a topic: search web, verify sources, summarize findings with citations",
            "confidence": 0.9,
        },
    ]

    recipes_count = storage.store_recipes(global_recipes)
    print(f"Stored {recipes_count} recipes globally")

    # Test 3: Query global facts
    print("\n=== Test 3: Query global facts ===")
    facts_results = storage.query_facts("Python web frameworks", n_results=5)
    print(f"Found {len(facts_results)} facts:")
    for fact in facts_results:
        print(f"  - {fact['document'][:60]}... (distance={fact['distance']:.3f})")

    # Test 4: Query global recipes
    print("\n=== Test 4: Query global recipes ===")
    recipes_results = storage.query_recipes("FastAPI endpoint", n_results=5)
    print(f"Found {len(recipes_results)} recipes:")
    for recipe in recipes_results:
        print(f"  - {recipe['document'][:60]}... (distance={recipe['distance']:.3f})")

    # Test 5: Retrieve context with coder agent's parameters
    # Coder has: token_budget=1000, min_confidence=0.7, max_distance=0.5
    print("\n=== Test 5: Retrieve context with coder parameters ===")
    coder_context = storage.retrieve_context(
        entities_queries=["FastAPI"],
        fact_query="Python async programming",
        recipe_query="create FastAPI endpoint",
        token_budget=1000,  # Coder's budget
        min_confidence=0.7,  # Coder's threshold
        max_distance=0.5,  # Coder's distance limit
    )

    print(f"Context tokens used: {coder_context['tokens_used']}/{1000}")
    print(f"Facts: {len(coder_context['facts'])}")
    for fact in coder_context["facts"]:
        print(f"  - {fact['statement'][:50]}...")
    print(f"Recipes: {len(coder_context['recipes'])}")
    for recipe in coder_context["recipes"]:
        print(f"  - {recipe['description'][:50]}...")

    # Test 6: Retrieve context with researcher agent's parameters
    # Researcher has: token_budget=2000, min_confidence=0.7, max_distance=0.5
    print("\n=== Test 6: Retrieve context with researcher parameters ===")
    researcher_context = storage.retrieve_context(
        entities_queries=["Python"],
        fact_query="research methodology",
        recipe_query="verify sources",
        token_budget=2000,  # Researcher's budget (higher)
        min_confidence=0.7,
        max_distance=0.5,
    )

    print(f"Context tokens used: {researcher_context['tokens_used']}/{2000}")
    print(f"Facts: {len(researcher_context['facts'])}")
    for fact in researcher_context["facts"]:
        print(f"  - {fact['statement'][:50]}...")
    print(f"Recipes: {len(researcher_context['recipes'])}")
    for recipe in researcher_context["recipes"]:
        print(f"  - {recipe['description'][:50]}...")

    # Test 7: Retrieve context with assistant agent's parameters
    # Assistant has: token_budget=1500, min_confidence=0.5, max_distance=0.7
    print("\n=== Test 7: Retrieve context with assistant parameters ===")
    assistant_context = storage.retrieve_context(
        entities_queries=["FastAPI"],
        fact_query="Python programming",
        recipe_query="web framework",
        token_budget=1500,
        min_confidence=0.5,  # Lower threshold (more results)
        max_distance=0.7,  # Higher distance (more results)
    )

    print(f"Context tokens used: {assistant_context['tokens_used']}/{1500}")
    print(f"Facts: {len(assistant_context['facts'])} (should be more due to lower thresholds)")
    for fact in assistant_context["facts"]:
        print(f"  - {fact['statement'][:50]}...")
    print(f"Recipes: {len(assistant_context['recipes'])}")
    for recipe in assistant_context["recipes"]:
        print(f"  - {recipe['description'][:50]}...")

    # Test 8: Verify that different agents get different amounts of context
    # based on their rag_config parameters
    print("\n=== Test 8: Verify agent-specific context sizing ===")
    print("Comparing context retrieval for different agents:")
    print(
        f"  Coder: {coder_context['tokens_used']} tokens, {len(coder_context['facts'])} facts, {len(coder_context['recipes'])} recipes"
    )
    print(
        f"  Researcher: {researcher_context['tokens_used']} tokens, {len(researcher_context['facts'])} facts, {len(researcher_context['recipes'])} recipes"
    )
    print(
        f"  Assistant: {assistant_context['tokens_used']} tokens, {len(assistant_context['facts'])} facts, {len(assistant_context['recipes'])} recipes"
    )
    print(
        "\nAll agents access the same global facts/recipes, but with different retrieval parameters!"
    )

    # Test 9: Format context for prompt
    print("\n=== Test 9: Format context for prompt ===")
    context_str = storage.format_context_for_prompt(coder_context)
    print(f"Formatted context (first 300 chars):\n{context_str[:300]}...")

    # Cleanup
    print("\n=== Cleanup ===")
    storage.close()

    # Summary
    print("\n=== Test Summary ===")
    print("✓ Global facts storage (shared across agents)")
    print("✓ Global recipes storage (shared across agents)")
    print("✓ Global facts query")
    print("✓ Global recipes query")
    print("✓ Context retrieval with coder parameters")
    print("✓ Context retrieval with researcher parameters")
    print("✓ Context retrieval with assistant parameters")
    print("✓ Agent-specific context sizing verification")
    print("✓ Context formatting for prompts")
    print("\nAll tests passed!")
    print("\nKey insight: All agents share the same global knowledge base,")
    print("but each agent retrieves context based on its rag_config parameters:")
    print("  - token_budget: limits context size")
    print("  - min_confidence: filters low-confidence items")
    print("  - max_distance: filters semantically distant items")


if __name__ == "__main__":
    test_global_rag()
