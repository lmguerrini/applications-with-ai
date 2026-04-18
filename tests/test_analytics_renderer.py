from rendering.analytics_renderer import (
    build_evaluation_interpretation,
    format_evaluation_case_rows_for_display,
    format_recent_diagnostics_rows_for_display,
)
from rendering.charts import _format_cost_metric


def test_format_cost_metric_uses_readable_cost_labels() -> None:
    assert _format_cost_metric(None) == "Cost unavailable"
    assert _format_cost_metric(0) == "$0.00"
    assert _format_cost_metric(0.000024) == "$0.000024"
    assert _format_cost_metric(0.125678) == "$0.1257"
    assert _format_cost_metric(12.5) == "$12.50"


def test_format_evaluation_case_rows_for_display_uses_readable_labels_and_values() -> None:
    rows = format_evaluation_case_rows_for_display(
        [
            {
                "question": "How should I persist Chroma locally?",
                "source_recall": 0.5,
                "retrieved_chunks": 2,
                "used_fallback": False,
                "context_match": True,
                "keyword_recall": 0.75,
            }
        ]
    )

    assert rows == [
        {
            "Question": "How should I persist Chroma locally?",
            "Source recall": "50.0%",
            "Retrieved chunks": 2,
            "Used fallback search": "No",
            "Context matched expectation": "Yes",
            "Keyword recall": "75.0%",
        }
    ]


def test_format_recent_diagnostics_rows_for_display_uses_readable_labels_and_values() -> None:
    rows = format_recent_diagnostics_rows_for_display(
        [
            {
                "query_preview": "What is the capital of France...",
                "response_type": "No-context fallback",
                "model": "n/a",
                "source_count": 0,
                "total_tokens": None,
                "estimated_cost_usd": None,
            },
            {
                "query_preview": "How should I persist Chroma...",
                "response_type": "Grounded answer",
                "model": "gpt-4.1-mini",
                "source_count": 2,
                "total_tokens": 30,
                "estimated_cost_usd": 0.000024,
            },
        ]
    )

    assert rows == [
        {
            "Query preview": "What is the capital of France...",
            "Response type": "No-context fallback",
            "Model": "No LLM",
            "Source count": 0,
            "Total tokens": "No tracked usage",
            "Estimated cost": "Cost unavailable",
        },
        {
            "Query preview": "How should I persist Chroma...",
            "Response type": "Grounded answer",
            "Model": "gpt-4.1-mini",
            "Source count": 2,
            "Total tokens": "30",
            "Estimated cost": "$0.000024",
        },
    ]


def test_build_evaluation_interpretation_returns_deterministic_status() -> None:
    good = build_evaluation_interpretation(
        {
            "case_count": 4,
            "average_source_recall": 0.8,
            "average_keyword_recall": 0.9,
            "context_match_rate": 1.0,
        }
    )
    acceptable = build_evaluation_interpretation(
        {
            "case_count": 4,
            "average_source_recall": 0.65,
            "average_keyword_recall": 0.9,
            "context_match_rate": 0.75,
        }
    )
    weak = build_evaluation_interpretation(
        {
            "case_count": 4,
            "average_source_recall": 0.2,
            "average_keyword_recall": 0.4,
            "context_match_rate": 0.6,
        }
    )

    assert good["status"] == "Good"
    assert acceptable["status"] == "Acceptable"
    assert weak["status"] == "Needs improvement"
    assert "80.0%" in good["summary"]
    assert "4 cases" in good["summary"]
    assert "Source recall and context match are below" in acceptable["summary"]
    assert "keyword recall" not in acceptable["summary"].split(".")[0]
    assert "source recall and keyword recall" in weak["summary"]


def test_build_evaluation_interpretation_treats_strong_partial_results_as_good() -> None:
    interpretation = build_evaluation_interpretation(
        {
            "case_count": 6,
            "average_source_recall": 0.833,
            "average_keyword_recall": 1.0,
            "context_match_rate": 0.833,
        }
    )

    assert interpretation["status"] == "Good"
    assert "Core evaluation signals are strong" in interpretation["summary"]
    assert "83.3%" in interpretation["summary"]
    assert "100.0%" in interpretation["summary"]
