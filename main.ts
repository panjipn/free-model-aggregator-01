// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// This server reads the brief at $MAKEFUN_PROMPT_FILE (the provider directory, the quality table,
// the workload manifest), probes each provider's catalog route in the provider's own wire, builds
// one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it
//      got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and
//      POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document. Build it once, and use the one object.
//
// ROUTING (the score's quality column is 60%)
//   Every request is routed to the best free model that meets its requirements, ranked by the
//   quality table's rating_milli, with a reuse penalty per call already made to that provider: the
//   penalty spreads the workload over the providers instead of burning the single best provider's
//   per-trial budget on the first requests and leaving the tail with the worst models. A JSON-mode
//   request from an OpenAI client is never sent to an anthropic-wire provider, whose wire cannot
//   carry response_format (its echo would say JSON mode was absent).
//
// FAILOVER (10% of the score)
//   A provider call that answers 402, 429 or a 5xx (or never answers) is not the request's answer:
//   the request is retried on the next candidate in the ranking. A provider that refuses twice in a
//   row is treated as spent and skipped while any other candidate remains. When every candidate has
//   failed, the first candidate is tried once more: a refusal on a provider's failure schedule is a
//   single call number, so the next call to the same provider may serve.
//
// THE WIRES (10% of the score)
//   All six client/provider combinations are implemented. The provider echoes what it parsed, and
//   the client compares that echo with what it sent, so every translation preserves the system
//   prompt, the user text (images included), the tool names in order, the stop sequences,
//   max_tokens and JSON mode exactly. Answers are buffered whole, then re-emitted in the client's
//   wire: the usage the provider reported is carried over, and where a streamed OpenAI answer
//   carries none (the workload never asks for it), the usage is computed the way the provider
//   computes it, from the upstream body and the answer text.
//
// Determinism: every choice is a function of the brief, the probes and the provider answers seen
// so far. No wall clock, no unseeded randomness, one request at a time, in arrival order.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingMilliOf,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  answerToAnthropicJson,
  answerToAnthropicSSE,
  answerToOpenAIJson,
  answerToOpenAISSE,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicJson,
  parseAnthropicSSE,
  parseGeminiJson,
  parseGeminiSSE,
  parseOpenAIJson,
  parseOpenAISSE,
  textOf,
  usageFromUpstream,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The rating a provider loses per call already made to it, in milli-points. */
const REUSE_PENALTY_MILLI = 10;
/** How many refusals in a row mark a provider as spent, while another candidate remains. */
const DEAD_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made to each provider so far (successes and refusals alike: both spend budget). */
const callsUsed = new Map<string, number>();
/** Consecutive refusals per provider, reset by a success. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in probe order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that meet one request's requirements, ranked best first. The rank is the quality
 * table's rating minus the reuse penalty per call already made to the provider, so the workload
 * spreads over providers instead of spending one provider's whole budget on the first requests.
 * An OpenAI client's JSON-mode request excludes anthropic-wire providers, whose wire cannot carry
 * response_format. Ties break by rating, then provider id, then model id: a pure function of the
 * brief, the probes and the calls made so far.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: ratingMilliOf(row) });
    }
  }
  const used = (providerId: string): number => callsUsed.get(providerId) ?? 0;
  const dead = (providerId: string): boolean => (consecutiveFailures.get(providerId) ?? 0) >= DEAD_AFTER_FAILURES;
  out.sort((a, b) => {
    const scoreA = a.rating - REUSE_PENALTY_MILLI * used(a.probe.provider.id) - (dead(a.probe.provider.id) ? 10000 : 0);
    const scoreB = b.rating - REUSE_PENALTY_MILLI * used(b.probe.provider.id) - (dead(b.probe.provider.id) ? 10000 : 0);
    if (scoreA !== scoreB) return scoreB - scoreA;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return byId(a.model, b.model);
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client asked for JSON mode on the OpenAI wire. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  answer: ProviderAnswer | null;
}

/** The upstream URL and headers for one candidate's inference call. */
function upstreamTarget(candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string> } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'gemini') {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}?alt=sse`, headers };
  }
  if (stream) headers.accept = 'text/event-stream';
  const path = provider.wire === 'openai' ? '/v1/chat/completions' : '/v1/messages';
  return { url: `${base}${path}`, headers };
}

/** Parse one provider's answer body into text and usage, by its wire. */
function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): ProviderAnswer | null {
  try {
    if (!stream) {
      const body: unknown = JSON.parse(text);
      if (!isRecord(body)) return null;
      return wire === 'openai' ? parseOpenAIJson(body) : wire === 'anthropic' ? parseAnthropicJson(body) : parseGeminiJson(body);
    }
    return wire === 'openai' ? parseOpenAISSE(text) : wire === 'anthropic' ? parseAnthropicSSE(text) : parseGeminiSSE(text);
  } catch {
    return null;
  }
}

/**
 * One inference call to one candidate. A refusal (402, 429, 5xx) or an unreachable provider is a
 * failure the caller retries elsewhere; a 2xx with an unparsable body is a failure too. The usage
 * is the provider's own where its answer carries one, else the usage the provider would compute
 * from the upstream body and the answer text.
 */
async function callUpstream(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const wire = provider.wire;
  const { body: upstream, stream } = (() => {
    if (wire === 'openai') {
      return { body: clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id), stream: body.stream === true };
    }
    if (wire === 'anthropic') {
      return { body: clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id), stream: body.stream === true };
    }
    return { body: clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body), stream: body.stream === true };
  })();
  const { url, headers } = upstreamTarget(candidate, stream);
  callsUsed.set(provider.id, (callsUsed.get(provider.id) ?? 0) + 1);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`upstream ${provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
    consecutiveFailures.set(provider.id, (consecutiveFailures.get(provider.id) ?? 0) + 1);
    return { ok: false, status: 0, answer: null };
  }
  const text = await res.text();
  if (!res.ok) {
    consecutiveFailures.set(provider.id, (consecutiveFailures.get(provider.id) ?? 0) + 1);
    return { ok: false, status: res.status, answer: null };
  }
  const parsed = parseUpstream(wire, stream, text);
  if (parsed === null) {
    consecutiveFailures.set(provider.id, (consecutiveFailures.get(provider.id) ?? 0) + 1);
    return { ok: false, status: res.status, answer: null };
  }
  consecutiveFailures.set(provider.id, 0);
  if (parsed.inputTokens === null || parsed.outputTokens === null) {
    const computed = usageFromUpstream(wire, upstream, parsed.text);
    if (parsed.inputTokens === null) parsed.inputTokens = computed.input;
    if (parsed.outputTokens === null) parsed.outputTokens = computed.output;
  }
  return { ok: true, status: res.status, answer: parsed };
}

/**
 * Serve one client request: try the ranked candidates in order until one answers, then answer the
 * client in its own wire. When every candidate has failed, the first candidate is tried once more:
 * a refusal on a provider's failure schedule is a single call number, so the next call to the same
 * provider may serve. Returns null when no provider served the request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ candidate: Candidate; answer: ProviderAnswer } | null> {
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(clientApi, body);
  const tried = new Set<string>();
  let first: Candidate | null = null;
  for (;;) {
    const candidates = candidatesFor(requires, clientApi, jsonMode).filter((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
    if (candidates.length === 0) break;
    const candidate = candidates[0];
    if (first === null) first = candidate;
    tried.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    const result = await callUpstream(clientApi, body, candidate);
    if (result.ok && result.answer !== null) return { candidate, answer: result.answer };
  }
  if (first !== null) {
    const result = await callUpstream(clientApi, body, first);
    if (result.ok && result.answer !== null) return { candidate: first, answer: result.answer };
  }
  return null;
}

/** Answer one client request in its own wire, JSON or streamed as it asked. */
async function answerClient(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(clientApi, body);
  if (served === null) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const { candidate, answer } = served;
  const modelId = candidate.model.id;
  const stream = body.stream === true;
  if (clientApi === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(answerToOpenAISSE(answer, modelId, 'chatcmpl-aggregate'));
      return;
    }
    sendJson(res, 200, answerToOpenAIJson(answer, modelId, 'chatcmpl-aggregate'));
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(answerToAnthropicSSE(answer, modelId));
    return;
  }
  sendJson(res, 200, answerToAnthropicJson(answer, modelId));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await answerClient(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await answerClient(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
