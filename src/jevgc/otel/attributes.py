"""GenAI semantic-convention attribute name constants (SPEC.md §4.7),
sourced from the OpenTelemetry Semantic Conventions for Generative AI spec:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Kept as plain string constants (not an enum) so `translate.py` can do cheap
`span.attributes.get(ATTR, default)` lookups without an extra indirection.
"""

from __future__ import annotations

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

ERROR_TYPE = "error.type"

#: Non-standard, jev-gc-specific attributes a host application may set to
#: give the pre-filter/scorer richer content without needing the full
#: message payload in a span event.
JEVGC_INPUT_PREVIEW = "jevgc.input_preview"
JEVGC_OUTPUT_PREVIEW = "jevgc.output_preview"
JEVGC_TURN_INDEX = "jevgc.turn_index"
JEVGC_PINNED = "jevgc.pinned"
