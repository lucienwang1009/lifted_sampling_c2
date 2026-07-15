# JSONL model output design

`wfoms --output/-o PATH` writes one sampled structure per line as UTF-8 JSON.
When the option is absent, the same records continue to go to standard output.
The file is opened in overwrite mode and samples are produced through the
existing iterator, so output memory does not grow with the requested count.

Each record has a `domain` array and a `relations` array. A relation contains
`predicate`, `arity`, and `tuples`. This list representation keeps predicates
with the same name but different arities distinct, represents a true nullary
predicate as `[[]]`, and retains empty relations. Domain elements and tuple
terms are serialized with their stable WFOMC string representation.

JSON Lines is preferred over one large JSON array because sampling and parsing
remain streaming, interrupted runs retain complete earlier records, and common
data tools can consume it directly. A custom logical syntax would be smaller
but would require a new parser and could not represent all-false domain elements
without additional metadata.
