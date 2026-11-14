# The best free AI model aggregator

One server (Node 22, standard library only) that serves OpenAI- and
Anthropic-compatible APIs on top of a directory of simulated free-tier
model providers, for the make.fun bounty
`free-model-aggregator-01`.

## How it works

1. The server reads the brief from `$MAKEFUN_PROMPT_FILE`: the provider
   directory (10 providers, each with its own wire — OpenAI, Anthropic or
   Gemini — base URL, auth header and key), the quality table
   (`rating_milli` per provider/model) and the workload manifest
   (20 requests and their requirements).
2. It probes each provider's catalog on that provider's own wire and
   builds one catalog, registering only the models marked free.
3. It listens on an OS-chosen port, registers
   `{"protocol_version":1,"port":<port>,"catalog":<catalog>}` with the
   environment at `$MAKEFUN_ENV_URL`, and serves:
   - `GET /v1/models` — in the shape of the auth header sent
     (`x-api-key` → Anthropic, `Authorization` → OpenAI)
   - `POST /v1/chat/completions` and `POST /v1/messages` — JSON or
     server-sent events, as the client asked.
4. It polls `GET $MAKEFUN_ENV_URL/work` every 100 ms and exits 0 when
   the environment answers `{"done":true}`.

## Routing

For each request the server ranks the free models that meet the
request's requirements (tools, vision, JSON mode, minimum context) by
`rating_milli`, minus a small penalty per call already made to that
provider, so the load spreads across providers instead of draining
the best one. A provider with two consecutive refusals is skipped
while another candidate remains. On a 402/429/5xx (or an unreachable
provider) it retries the next candidate, and if every candidate has
failed it tries the first candidate once more — a refusal on a
provider's failure schedule is a single call number, so the next call
may serve.

All six client/provider wire combinations are implemented, so a chat
request is translated upstream and the answer is re-emitted in the
client's wire, with usage computed the way the provider would when
the provider's streamed answer carries none.

## Layout

- `main.ts` — the server (entrypoint; run as `node /bot/main.ts`)
- `lib/providers.ts` — brief parsing, catalog probing, rating helpers
- `lib/wire.ts` — OpenAI / Anthropic / Gemini request and answer
  translation, including streamed (SSE) forms
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`)
- `README.md` — this file

## Usage

```sh
node main.ts --list   # print the catalog JSON and exit 0
node main.ts          # start the server (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
```

No dependencies, no wall clock, no unseeded randomness: the server is
deterministic — two runs of the same trial produce the same result.

## Local practice scores (kit trials)

| trial | score | quality | served | wire | failover | catalog |
|------:|------:|--------:|-------:|-----:|---------:|--------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 |

Starter baseline for comparison: 38 / 44 / 38 (mean 40.0). The same
server also scored 99 on 10 freshly drawn random instances (quality
98–99, every other column 100), so the result is not tuned to the
practice trials.
